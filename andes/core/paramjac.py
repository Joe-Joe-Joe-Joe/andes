"""
Jacobians of the DAE equations with respect to parameters.
"""

#  [ANDES] (C)2015-2024 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

import inspect
import logging
from collections import OrderedDict

import sympy as sp
from sympy.core.function import AppliedUndef

from andes.core.service import BaseService, PostInitService, VarService
from andes.core.symprocessor import SymProcessor, select_args_add
from andes.shared import np

logger = logging.getLogger(__name__)


def _expand_helpers(expr):
    """
    Rewrite functions that ANDES evaluates with NumPy but SymPy does not
    know, so that they can be differentiated:

    - ``conj(z)`` becomes ``conjugate(z)``;
    - ``radians(x)`` becomes ``x * pi / 180``;
    - ``safe_div(a, b)`` becomes ``Piecewise((a / b, b != 0), (0, True))``.

    ``Indicator`` (a 0/1 flag) is kept and held constant by :func:`_diff`.
    """
    rules = {
        sp.Function('conj'): lambda z: sp.conjugate(z),
        sp.Function('radians'): lambda x: x * sp.pi / 180,
        sp.Function('safe_div'): lambda a, b: sp.Piecewise((a / b, sp.Ne(b, 0)), (0, True)),
    }
    for func, rewrite in rules.items():
        if expr.has(func):
            expr = expr.replace(func, rewrite)
    return expr


def _diff(expr, sym):
    """
    Differentiate ``expr`` with respect to ``sym``.

    Calls of undefined functions (such as ``Indicator(x > 0)``, a 0/1 flag)
    are piecewise constant in ANDES models and are held fixed.
    """
    calls = {c: sp.Dummy() for c in expr.atoms(AppliedUndef)}
    if not calls:
        return sp.diff(expr, sym)
    d = sp.diff(expr.xreplace(calls), sym)
    return d.xreplace({v: k for k, v in calls.items()})


def _substitute(expr, rule):
    """
    ``expr.xreplace(rule)``, skipping substitutions that SymPy rejects, such
    as non-real expressions inside comparisons. Skipped symbols stay in the
    expression and are differentiated through as constants.
    """
    try:
        return expr.xreplace(rule)
    except TypeError:
        pass
    for sym, value in rule.items():
        try:
            expr = expr.xreplace({sym: value})
        except TypeError:
            continue
    return expr


class _Term:
    """A lambdified expression and the names of its inputs."""

    def __init__(self, expr, inputs_dict, modules, compile=True):
        self.expr = expr
        self.value = None
        known = set(inputs_dict.values())
        syms = sorted((s for s in expr.free_symbols if s in known), key=lambda s: s.name)
        unknown = [s for s in expr.free_symbols if s not in known]
        if unknown:
            raise ValueError(f'Expression "{expr}" contains unknown symbols {unknown}.')
        if not compile:
            self.func = None
            return

        self.args = [s.name for s in syms]
        self.func = sp.lambdify(syms, expr, modules=modules)
        if 'select' in inspect.getsource(self.func):
            # the printer for Piecewise refers to these helper arrays by name
            extra = [sp.Symbol(name) for name in select_args_add]
            self.func = sp.lambdify(syms + extra, expr, modules=modules)
            self.args = self.args + select_args_add

    def __call__(self, inputs, n):
        if self.func is None:
            return self.value          # filled by a _TermBatch
        out = self.func(*[inputs[name] for name in self.args])
        out = np.broadcast_to(np.asarray(out), (n,))
        if np.iscomplexobj(out):
            out = out.real
        return out.astype(float)


def _model_symbols(mdl):
    """
    Symbols and an expression parser for one model.

    Complex services ``c`` are written as ``c__re + I * c__im`` with two real
    symbols, so that expressions are real and SymPy can differentiate
    ``re``, ``im``, ``conj``, ``abs`` and ``arg`` of them.

    Returns
    -------
    known : OrderedDict
        Name -> symbol, including the ``__re``/``__im`` symbols.
    modules : list
        ``modules`` argument for ``lambdify``.
    split : OrderedDict
        Complex service name -> ``(re name, im name)``.
    parse : callable
        Parses an equation or service string.
    """
    syms = SymProcessor(mdl)
    syms.generate_symbols()
    syms.generate_subs_expr()
    known = OrderedDict(syms.inputs_dict)

    split = OrderedDict()
    to_real = {}
    for name in known.copy():
        instance = mdl.__dict__.get(name)
        if isinstance(instance, BaseService) and getattr(instance, 'vtype', float) == complex:
            re_s = sp.Symbol(f'{name}__re', real=True)
            im_s = sp.Symbol(f'{name}__im', real=True)
            known[re_s.name], known[im_s.name] = re_s, im_s
            to_real[known[name]] = re_s + sp.I * im_s
            split[name] = (re_s.name, im_s.name)

    def parse(string):
        expr = sp.sympify(string, locals=syms.inputs_dict)
        expr = _expand_helpers(syms._do_substitute(expr))
        return _substitute(expr, to_real)

    return known, syms.lambdify_func, split, parse


def _split_complex(expr):
    """Real and imaginary parts of an expression in real symbols."""
    return sp.expand_complex(sp.re(expr)), sp.expand_complex(sp.im(expr))


class _TermBatch:
    """
    All derivative terms of one model in a single lambdified function, so
    that one evaluation per step fills ``term.value`` of every term.
    """

    def __init__(self, terms, known, modules):
        self.terms = list(terms)
        if not self.terms:
            self.func, self.args = None, []
            return
        syms = sorted({s for t in self.terms for s in t.expr.free_symbols}, key=lambda s: s.name)
        exprs = sp.Tuple(*[t.expr for t in self.terms])
        self.args = [s.name for s in syms]
        self.func = sp.lambdify(syms, exprs, modules=modules)
        if 'select' in inspect.getsource(self.func):
            extra = [sp.Symbol(name) for name in select_args_add]
            self.func = sp.lambdify(syms + extra, exprs, modules=modules)
            self.args = self.args + select_args_add

    def evaluate(self, inputs, n):
        if self.func is None:
            return
        out = self.func(*[inputs[name] for name in self.args])
        for t, v in zip(self.terms, out):
            v = np.broadcast_to(np.asarray(v), (n,))
            t.value = (v.real if np.iscomplexobj(v) else v).astype(float)


# Inputs of each model are refreshed once per evaluation round; see
# `input_cache`.
_INPUT_CACHE = None


class input_cache:
    """
    Context manager that reuses model inputs within a block, for evaluating
    several Jacobians at the same point.
    """

    def __enter__(self):
        global _INPUT_CACHE
        self._prev, _INPUT_CACHE = _INPUT_CACHE, {}
        return self

    def __exit__(self, *exc):
        global _INPUT_CACHE
        _INPUT_CACHE = self._prev
        return False


def _model_inputs(mdl, split):
    """Model inputs plus the real and imaginary parts of complex services."""
    if _INPUT_CACHE is not None and id(mdl) in _INPUT_CACHE:
        return _INPUT_CACHE[id(mdl)]
    inputs = dict(mdl.get_inputs(refresh=True))
    for name, (re_n, im_n) in split.items():
        v = np.asarray(mdl.__dict__[name].v)
        inputs[re_n], inputs[im_n] = np.real(v), np.imag(v)
    if _INPUT_CACHE is not None:
        _INPUT_CACHE[id(mdl)] = inputs
    return inputs


class _TermMaker:
    """Builds lambdified derivative terms and records what cannot be differentiated."""

    def __init__(self, mdl, known, modules):
        self.mdl, self.known, self.modules = mdl, known, modules
        self.unsupported = []
        self.terms = []

    def batch(self):
        """A :class:`_TermBatch` of all terms made so far."""
        return _TermBatch(self.terms, self.known, self.modules)

    def __call__(self, expr, sym, where):
        if isinstance(expr, (sp.logic.boolalg.BooleanFunction, sp.logic.boolalg.BooleanAtom,
                             sp.core.relational.Relational)):
            return None       # a true/false flag: piecewise constant
        d = _diff(expr, sym)
        if d == 0:
            return None
        if d.has(sp.Derivative) or d.has(sp.Subs):
            self.unsupported.append(f'{self.mdl.class_name}.{where} w.r.t. {sym}')
            return None
        t = _Term(d, self.known, self.modules, compile=False)
        self.terms.append(t)
        return t


class _ModelParamJac:
    """
    Symbolic parameter derivatives for one model.

    Services form a chain: ``ds/dp = d(expr_s)/dp + sum(d(expr_s)/ds' ds'/dp)``
    over earlier services ``s'``. A complex service ``c`` enters the chain as
    two real nodes ``c__re`` and ``c__im``, so that every expression is real
    and SymPy can differentiate ``re``, ``im``, ``conj``, ``abs`` and ``arg``
    of it.
    """

    def __init__(self, mdl, params):
        self.mdl = mdl
        self.params = list(params)

        known, modules, self.split, parse = _model_symbols(mdl)

        psyms = {p: known[p] for p in self.params}

        # chain nodes in evaluation order: name -> (expression, is VarService)
        nodes = OrderedDict()
        for name, instance in mdl.services.items():
            if isinstance(instance, PostInitService) or instance.v_str is None \
                    or instance.v_numeric is not None:
                continue          # set from variables after init, or numerically
            expr = parse(instance.v_str)
            is_var = isinstance(instance, VarService)
            if name in self.split:
                re_e, im_e = _split_complex(expr)
                re_n, im_n = self.split[name]
                nodes[re_n] = (re_e, is_var)
                nodes[im_n] = (im_e, is_var)
            else:
                nodes[name] = (expr, is_var)

        # equations: rows are the variables whose `e_str` defines them
        eq_exprs = [(var, parse(var.e_str))
                    for var in list(mdl.cache.states_and_ext.values())
                    + list(mdl.cache.algebs_and_ext.values())
                    if var.e_str is not None]
        tconsts = [(var, var.t_const.name) for var in mdl.states.values()
                   if var.t_const is not None]

        # nodes used by the equations, directly or through other nodes ...
        needed = {known[name] for _, name in tconsts if name in known}
        for _, expr in eq_exprs:
            needed |= expr.free_symbols
        for name in reversed(nodes):
            if known[name] in needed:
                needed |= nodes[name][0].free_symbols

        # ... and that depend on the parameters
        self.varying = OrderedDict()   # node name -> is VarService
        reach = set(psyms.values())
        for name, (expr, is_var) in nodes.items():
            if known[name] in needed and expr.free_symbols & reach:
                self.varying[name] = is_var
                reach.add(known[name])

        term = _TermMaker(mdl, known, modules)

        def derivative_terms(expr, where):
            """Terms of d(expr)/dp, split into parameter and node parts."""
            d_p, d_s = OrderedDict(), OrderedDict()
            fs = expr.free_symbols
            for p, ps in psyms.items():
                if ps in fs:
                    t = term(expr, ps, where)
                    if t is not None:
                        d_p[p] = t
            for name in self.varying:
                if known[name] in fs:
                    t = term(expr, known[name], where)
                    if t is not None:
                        d_s[name] = t
            return d_p, d_s

        self.d_services = OrderedDict(
            (name, derivative_terms(nodes[name][0], name)) for name in self.varying)
        self.unsupported = term.unsupported
        self._term_maker = term

        self.eqs = []       # (variable instance, d_p, d_s)
        for var, expr in eq_exprs:
            d_p, d_s = derivative_terms(expr, var.name)
            if d_p or d_s:
                self.eqs.append((var, d_p, d_s))

        # time constants of differential equations
        self.tconst = [(var, name) for var, name in tconsts
                       if name in self.params or name in self.varying]

        self.batch = self._term_maker.batch()
        self.frozen = {}

    def _inputs(self):
        inputs = _model_inputs(self.mdl, self.split)
        self.batch.evaluate(inputs, self.mdl.n)
        return inputs

    def _service_derivatives(self, inputs, n, frozen):
        """
        Total derivatives ``ds/dp`` of the varying nodes, as
        ``{node: {param: array}}``. ``frozen`` holds the values for
        ConstServices computed at ``t=0``; VarServices are evaluated now.
        """
        out = OrderedDict()
        for name, is_var in self.varying.items():
            if not is_var and name in frozen:
                out[name] = frozen[name]
                continue
            d_p, d_s = self.d_services[name]
            total = {}
            for p, term in d_p.items():
                total[p] = term(inputs, n).copy()
            for s, term in d_s.items():
                ds = term(inputs, n)
                for p, v in out.get(s, {}).items():
                    total[p] = total.get(p, 0.0) + ds * v
            out[name] = total
        return out

    def freeze(self):
        """Evaluate and store ``ds/dp`` of ConstServices at the current state."""
        derivs = self._service_derivatives(self._inputs(), self.mdl.n, {})
        self.frozen = {name: v for name, v in derivs.items() if not self.varying[name]}

    def triplets(self, cols):
        """
        Return ``(f, g, tf)`` lists of ``(rows, cols, vals)`` triplets.

        ``cols[p]`` maps device uid to the column in the parameter vector
        (``-1`` for devices not in the vector).
        """
        n = self.mdl.n
        inputs = self._inputs()
        svc = self._service_derivatives(inputs, n, self.frozen)
        out = {'f': [], 'g': [], 'tf': []}

        def add(kind, rows, totals):
            for p, vals in totals.items():
                c = cols[p]
                vals = np.broadcast_to(vals, (n,))
                keep = (c >= 0) & (vals != 0)
                if np.any(keep):
                    out[kind].append((np.asarray(rows)[keep], c[keep], vals[keep]))

        for var, d_p, d_s in self.eqs:
            totals = {p: term(inputs, n) for p, term in d_p.items()}
            for s, term in d_s.items():
                ds = term(inputs, n)
                for p, v in svc[s].items():
                    totals[p] = totals.get(p, 0.0) + ds * v
            add(var.e_code, var.a, totals)

        for var, name in self.tconst:
            if name in self.params:
                add('tf', var.a, {name: np.ones(n)})
            else:
                add('tf', var.a, svc[name])

        return out


class _ModelTotalJac:
    """
    Parameter derivatives for one model, given the total derivatives of its
    constants.

    Every service that is not a ``VarService`` (``ConstService``,
    ``ExtService``, ``PostInitService``, ...) is a constant whose total
    derivative ``dc/dp`` over all columns is supplied by the caller, for
    example from the sensitivity of the initialization. ``VarService``
    values follow the chain at evaluation time. All totals are dense arrays
    of shape ``(n_devices, n_columns)``.
    """

    def __init__(self, mdl, params):
        self.mdl = mdl
        self.params = list(params)
        known, modules, self.split, parse = _model_symbols(mdl)

        # constants: services other than VarService; complex ones split
        self.consts = []          # node names
        for name in known:
            instance = mdl.__dict__.get(name)
            if not isinstance(instance, BaseService) or isinstance(instance, VarService):
                continue
            self.consts.extend(self.split.get(name, (name,)))

        leaves = {name: known[name] for name in self.params + self.consts}

        nodes = OrderedDict()
        for name, instance in mdl.services_var.items():
            if instance.v_str is None or instance.v_numeric is not None:
                continue
            expr = parse(instance.v_str)
            if name in self.split:
                for node, e in zip(self.split[name], _split_complex(expr)):
                    nodes[node] = e
            else:
                nodes[name] = expr

        term = _TermMaker(mdl, known, modules)

        def terms(expr, where, names):
            out = OrderedDict()
            fs = expr.free_symbols
            for name in names:
                if known[name] in fs:
                    t = term(expr, known[name], where)
                    if t is not None:
                        out[name] = t
            return out

        # VarServices that depend on leaves, directly or through others
        self.varying = OrderedDict()
        reach = set(leaves.values())
        for name, expr in nodes.items():
            if expr.free_symbols & reach:
                self.varying[name] = terms(expr, name, list(leaves) + list(self.varying))
                reach.add(known[name])

        self.eqs = []
        for var in list(mdl.cache.states_and_ext.values()) + list(mdl.cache.algebs_and_ext.values()):
            if var.e_str is None:
                continue
            t = terms(parse(var.e_str), var.name, list(leaves) + list(self.varying))
            if t:
                self.eqs.append((var, t))

        self.tconst = [(var, var.t_const.name) for var in mdl.states.values()
                       if var.t_const is not None
                       and (var.t_const.name in leaves or var.t_const.name in self.varying)]
        self.unsupported = term.unsupported
        self.batch = term.batch()

    def triplets(self, totals):
        """
        Return ``(f, g, tf)`` lists of ``(rows, cols, vals)`` triplets.

        ``totals`` maps each parameter and constant node of the model to
        its total derivative, an array of shape ``(n_devices, n_columns)``.
        Missing names count as zero.
        """
        n = self.mdl.n
        inputs = _model_inputs(self.mdl, self.split)
        self.batch.evaluate(inputs, n)
        totals = dict(totals)
        out = {'f': [], 'g': [], 'tf': []}

        def combine(parts):
            acc = None
            for name, t in parts.items():
                tot = totals.get(name)
                if tot is None:
                    continue
                contrib = t(inputs, n)[:, None] * tot
                acc = contrib if acc is None else acc + contrib
            return acc

        for name, parts in self.varying.items():
            acc = combine(parts)
            if acc is not None:
                totals[name] = acc

        def add(kind, rows, mat):
            if mat is None:
                return
            i, j = np.nonzero(mat)
            if i.size:
                out[kind].append((np.asarray(rows)[i], j, mat[i, j]))

        for var, parts in self.eqs:
            add(var.e_code, var.a, combine(parts))
        for var, name in self.tconst:
            add('tf', var.a, totals.get(name))
        return out


class _ModelVarServiceJac:
    """
    Jacobian terms through the ``VarService`` values of one model.

    ANDES evaluates ``VarService`` values from the variables at every
    iteration but treats them as constants in its symbolic Jacobians, which
    is enough for Newton's method. This class gives the missing terms
    ``de/ds * ds/d(x, y)`` for each ``VarService`` ``s``, chained through
    earlier ``VarService`` values.
    """

    def __init__(self, mdl):
        self.mdl = mdl
        known, modules, self.split, parse = _model_symbols(mdl)
        self.vars = OrderedDict((name, var) for name, var in mdl.cache.all_vars.items())

        nodes = OrderedDict()
        for name, instance in mdl.services_var.items():
            if instance.v_str is None or instance.v_numeric is not None:
                continue
            expr = parse(instance.v_str)
            if name in self.split:
                for node, e in zip(self.split[name], _split_complex(expr)):
                    nodes[node] = e
            else:
                nodes[name] = expr

        term = _TermMaker(mdl, known, modules)
        var_syms = {known[name]: name for name in self.vars if name in known}

        # nodes that depend on variables, directly or through earlier nodes
        self.nodes = OrderedDict()     # node -> (var terms, node terms)
        reach = set(var_syms)
        for name, expr in nodes.items():
            fs = expr.free_symbols
            if not fs & reach:
                continue
            d_var = OrderedDict((var_syms[sym], t) for sym in fs & set(var_syms)
                                for t in [term(expr, sym, name)] if t is not None)
            d_node = OrderedDict((k, t) for k in self.nodes if known[k] in fs
                                 for t in [term(expr, known[k], name)] if t is not None)
            self.nodes[name] = (d_var, d_node)
            reach.add(known[name])

        self.eqs = []                  # (equation variable, {node: term})
        if self.nodes:
            for var in list(mdl.cache.states_and_ext.values()) + list(mdl.cache.algebs_and_ext.values()):
                if var.e_str is None:
                    continue
                expr = parse(var.e_str)
                t = OrderedDict((k, tt) for k in self.nodes if known[k] in expr.free_symbols
                                for tt in [term(expr, known[k], var.name)] if tt is not None)
                if t:
                    self.eqs.append((var, t))
        self.unsupported = term.unsupported
        self.batch = term.batch()

    def triplets(self, keep_zeros=False):
        """
        Return ``{'fx': [...], 'fy': [...], 'gx': [...], 'gy': [...]}`` triplet
        lists. With ``keep_zeros``, entries that are zero at this point are
        kept, so the pattern does not change between evaluations.
        """
        out = {'fx': [], 'fy': [], 'gx': [], 'gy': []}
        if not self.eqs:
            return out
        n = self.mdl.n
        inputs = _model_inputs(self.mdl, self.split)
        self.batch.evaluate(inputs, n)

        totals = {}                    # node -> {variable name: ds/dvar per device}
        for name, (d_var, d_node) in self.nodes.items():
            tot = {v: t(inputs, n).copy() for v, t in d_var.items()}
            for k, t in d_node.items():
                dk = t(inputs, n)
                for v, val in totals[k].items():
                    tot[v] = tot.get(v, 0.0) + dk * val
            totals[name] = tot

        for eq, parts in self.eqs:
            acc = {}
            for k, t in parts.items():
                de = t(inputs, n)
                for v, val in totals[k].items():
                    acc[v] = acc.get(v, 0.0) + de * val
            for v, vals in acc.items():
                var = self.vars[v]
                vals = np.broadcast_to(vals, (n,))
                keep = np.ones(n, dtype=bool) if keep_zeros else vals != 0
                if np.any(keep):
                    out[eq.e_code + var.v_code].append(
                        (np.asarray(eq.a)[keep], np.asarray(var.a)[keep], vals[keep]))
        return out


def varservice_jacobian(system, models=None):
    """
    Build an evaluator for the Jacobian terms that pass through
    ``VarService`` values, which ANDES leaves out of ``fx``, ``fy``, ``gx``
    and ``gy``.

    Parameters
    ----------
    system : System
        A system after ``TDS.init()``.
    models : dict, optional
        Models to include. Default: ``system.exist.pflow_tds``.

    Returns
    -------
    callable
        ``evaluate()`` returns ``{'fx', 'fy', 'gx', 'gy'}`` SciPy CSR matrices
        at the current state, to add to the ANDES Jacobians.
        ``evaluate.triplets()`` returns the same terms as ``(rows, cols,
        vals)`` with a fixed pattern, zeros included. The attribute
        ``evaluate.unsupported`` lists terms that could not be differentiated.
    """
    from scipy.sparse import coo_matrix

    models = system.exist.pflow_tds if models is None else models
    parts = [_ModelVarServiceJac(mdl) for mdl in models.values()
             if mdl.n > 0 and len(mdl.services_var) > 0]
    parts = [p for p in parts if p.eqs]
    n, m = system.dae.n, system.dae.m
    shapes = {'fx': (n, n), 'fy': (n, m), 'gx': (m, n), 'gy': (m, m)}

    def triplets():
        """Triplets ``{name: (rows, cols, vals)}`` with a fixed pattern (zeros kept)."""
        trip = {k: [] for k in shapes}
        for part in parts:
            for k, items in part.triplets(keep_zeros=True).items():
                trip[k].extend(items)
        out = {}
        for k in shapes:
            if trip[k]:
                out[k] = tuple(np.concatenate(x) for x in zip(*trip[k]))
            else:
                out[k] = (np.array([], dtype=int), np.array([], dtype=int), np.array([]))
        return out

    def evaluate():
        trip = {k: [] for k in shapes}
        for part in parts:
            for k, items in part.triplets().items():
                trip[k].extend(items)
        out = {}
        for k, shape in shapes.items():
            if trip[k]:
                rows = np.concatenate([r for r, _, _ in trip[k]])
                cols = np.concatenate([c for _, c, _ in trip[k]])
                vals = np.concatenate([v for _, _, v in trip[k]])
                out[k] = coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
            else:
                out[k] = coo_matrix(shape).tocsr()
        return out

    evaluate.triplets = triplets
    evaluate.unsupported = [u for p in parts for u in p.unsupported]
    evaluate.models = [p.mdl.class_name for p in parts]
    return evaluate


class ParamJacobian:
    """
    Jacobians of the DAE residuals with respect to a :class:`ParamVector`.

    With the DAE written as ``Tf(p) * dx/dt = f(x, y, p)`` and
    ``0 = g(x, y, p)``, this class evaluates the sparse matrices ``df/dp``,
    ``dg/dp`` and ``dTf/dp`` at the current state, for use in trajectory
    sensitivity analysis.

    Parameters enter the equations directly and through services. The
    derivatives follow the chain through ``ConstService`` and
    ``VarService`` expressions of the same model. Values that ANDES derives
    from the initial state (``ExtService``, ``PostInitService``) and
    services computed numerically are treated as constants here: their
    dependence on parameters is part of the initial-condition sensitivity.
    ``ConstService`` derivatives are evaluated once, at construction or by
    :meth:`freeze`, because the services themselves are evaluated at
    ``t=0``.

    Parameters
    ----------
    system : System
        A system on which ``TDS.init()`` has been called.
    pv : ParamVector
        The parameters; columns of the Jacobians follow its order.
    const_totals : dict, optional
        Total derivatives of the constants of dynamic models, as
        ``{model: {service: array of shape (n_devices, len(pv))}}``, for
        example from :class:`andes.routines.sensitivity.TrajectorySensitivity`.
        With it, every non-``VarService`` service, including ``ExtService``
        and ``PostInitService`` values set from the initial state, carries
        its given derivative, and constants of models without parameters
        in ``pv`` are included too. Missing entries count as zero.

    Notes
    -----
    The Jacobians describe ``e_str`` equations at fixed ``(x, y)``. Rows of
    states pegged by anti-windup limiters are zeroed, as ANDES overwrites
    those residuals. A pegged state is held at its limit, so a change of a
    limit parameter moves the state itself; that enters a sensitivity
    analysis through the state, not through these Jacobians. Discrete
    flags (limiters, deadbands, lookup-table segments) are held at their
    current values, so at a point exactly on a limit or breakpoint the
    result is the one-sided derivative of the active segment.
    Parts that cannot be differentiated are listed in :attr:`unsupported`
    and taken as zero: numerical equations (``f_numeric``, ``g_numeric``)
    and expressions SymPy leaves unevaluated. Derivatives that are not
    finite at the current point, for example of a saturation coefficient
    at zero, are set to zero; :attr:`nonfinite` lists the affected entries
    of the last evaluation.

    Examples
    --------
    >>> pv = ParamVector.all(ss, pflow=False)
    >>> pj = ParamJacobian(ss, pv)
    >>> fp, gp = pj.evaluate()       # scipy.sparse, (dae.n, len(pv)), (dae.m, len(pv))
    """

    def __init__(self, system, pv, const_totals=None):
        if not system.TDS.initialized:
            raise RuntimeError("ParamJacobian requires TDS.init().")
        if pv.system is not system:
            raise ValueError("The ParamVector belongs to another system.")

        self.system = system
        self.pv = pv
        self.unsupported = []
        self.nonfinite = []
        self.const_totals = None

        if const_totals is not None:
            self._init_totals(const_totals)
            return

        self._models = OrderedDict()   # model name -> (_ModelParamJac, cols)
        for (mname, pname), (mdl, _, uid, sl) in pv._blocks.items():
            if mname not in self._models:
                self._models[mname] = (mdl, OrderedDict())
            col = np.full(mdl.n, -1, dtype=int)
            col[uid] = np.arange(sl.start, sl.stop)
            self._models[mname][1][pname] = col

        for mname, (mdl, cols) in list(self._models.items()):
            for flag in ('f_num', 'g_num'):
                if getattr(mdl.flags, flag, False):
                    self.unsupported.append(f'{mname}.{flag[0]}_numeric')
                    break
            mpj = _ModelParamJac(mdl, cols.keys())
            self.unsupported.extend(mpj.unsupported)
            self._models[mname] = (mpj, cols)

        if self.unsupported:
            logger.warning("ParamJacobian: not differentiated (taken as zero): %s",
                           '; '.join(self.unsupported))

        self.freeze()

    def _init_totals(self, const_totals):
        """Set up the mode where constants have given total derivatives."""
        system, pv = self.system, self.pv
        ncol = len(pv)

        own = {}                           # model -> {param: one-hot totals}
        for (mname, pname), (mdl, _, uid, sl) in pv._blocks.items():
            tot = np.zeros((mdl.n, ncol))
            tot[uid, np.arange(sl.start, sl.stop)] = 1.0
            own.setdefault(mname, {})[pname] = tot

        self._totals = OrderedDict()       # model -> (_ModelTotalJac, totals)
        for mname, mdl in system.exist.tds.items():
            if mdl.n == 0 or mname in system.exist.pflow:
                continue
            given = const_totals.get(mname, {})
            if mname not in own and not given:
                continue
            mtj = _ModelTotalJac(mdl, own.get(mname, {}).keys())
            totals = dict(own.get(mname, {}))
            for name, tot in given.items():
                tot = np.asarray(tot)
                if name in mtj.split:
                    re_n, im_n = mtj.split[name]
                    totals[re_n], totals[im_n] = np.real(tot), np.imag(tot)
                else:
                    totals[name] = np.real(tot)
            for flag in ('f_num', 'g_num'):
                if getattr(mdl.flags, flag, False):
                    self.unsupported.append(f'{mname}.{flag[0]}_numeric')
            self.unsupported.extend(mtj.unsupported)
            self._totals[mname] = (mtj, totals)

        self.const_totals = const_totals
        if self.unsupported:
            logger.warning("ParamJacobian: not differentiated (taken as zero): %s",
                           '; '.join(self.unsupported))

    def freeze(self):
        """
        Evaluate the ``ConstService`` derivatives at the current state.

        Called at construction. Call again at ``t=0`` after
        ``TDS.reinit(keep_params=True)`` with new parameter values.
        """
        if self.const_totals is not None:
            return                 # constants have given totals
        if self.system.dae.t != 0:
            raise RuntimeError("ParamJacobian.freeze() must be called at t=0.")
        for mpj, _ in self._models.values():
            mpj.freeze()

    def evaluate(self):
        """
        Evaluate ``df/dp`` and ``dg/dp`` at the current state.

        Returns
        -------
        fp, gp : scipy.sparse.csr_matrix
            Of shape ``(dae.n, len(pv))`` and ``(dae.m, len(pv))``.
        """
        trip = self._collect()
        fp = self._assemble(trip['f'], self.system.dae.n)
        fp = self._zero_pegged(fp)
        gp = self._assemble(trip['g'], self.system.dae.m)
        self._check_finite(fp, gp)
        return fp, gp

    def _check_finite(self, *mats):
        """Set non-finite entries to zero and record their columns in `nonfinite`."""
        bad = set()
        for m in mats:
            mask = ~np.isfinite(m.data)
            if np.any(mask):
                bad.update(m.tocoo().col[~np.isfinite(m.tocoo().data)].tolist())
                m.data[mask] = 0.0
                m.eliminate_zeros()
        self.nonfinite = [self.pv.names[j] for j in sorted(bad)]
        if self.nonfinite:
            logger.warning("ParamJacobian: derivatives not finite at the current point, "
                           "set to zero: %s", ', '.join(self.nonfinite))

    def tf_p(self):
        """
        Return ``dTf/dp`` as a sparse matrix of shape ``(dae.n, len(pv))``.

        Non-zero for parameters (or services of parameters) used as time
        constants, such as ``GENROU.M``.
        """
        tp = self._assemble(self._collect()['tf'], self.system.dae.n)
        self._check_finite(tp)
        return tp

    def _collect(self):
        out = {'f': [], 'g': [], 'tf': []}
        if self.const_totals is not None:
            for mtj, totals in self._totals.values():
                for kind, items in mtj.triplets(totals).items():
                    out[kind].extend(items)
            return out
        for mpj, cols in self._models.values():
            for kind, items in mpj.triplets(cols).items():
                out[kind].extend(items)
        return out

    def _assemble(self, items, nrows):
        from scipy.sparse import coo_matrix

        if items:
            rows = np.concatenate([np.asarray(r, dtype=int) for r, _, _ in items])
            cols = np.concatenate([c for _, c, _ in items])
            vals = np.concatenate([v for _, _, v in items])
        else:
            rows = cols = np.array([], dtype=int)
            vals = np.array([], dtype=float)
        return coo_matrix((vals, (rows, cols)), shape=(nrows, len(self.pv))).tocsr()

    def _zero_pegged(self, fp):
        keys = [key for item in self.system.antiwindups for key, _, _ in item.x_set]
        if not keys:
            return fp
        from scipy.sparse import diags

        keep = np.ones(fp.shape[0])
        keep[np.concatenate([np.atleast_1d(k) for k in keys]).astype(int)] = 0.0
        fp = (diags(keep) @ fp).tocsr()
        fp.eliminate_zeros()
        return fp
