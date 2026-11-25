"""
Exact sensitivity of the TDS initialization with respect to parameters.
"""

#  [ANDES] (C)2015-2024 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

import logging
from collections import OrderedDict

import sympy as sp

from andes.core.observable import Observable
from andes.core.paramjac import (_diff, _model_inputs, _model_symbols, _split_complex,
                                 _Term)
from andes.core.service import BaseService, PostInitService, VarService
from andes.core.var import BaseVar
from andes.shared import np

logger = logging.getLogger(__name__)


class InitSensitivity:
    """
    Forward-mode (tangent) differentiation of the initialization of
    dynamic models, as done by ``TDS.init()`` and
    ``TDS.reinit(keep_params=True)``.

    The initialization is a sequence: for each dynamic model in order, copy
    external values (``ExtService``), evaluate services, then assign
    variables from ``v_str`` or solve ``v_iter`` groups, and write them to
    the DAE vectors; post-init services come last. This class carries
    ``d/dp`` of every value through the same sequence, with the chain rule
    for assignments and ``dv/dp = -J^-1 dF/dp`` for iterative groups,
    evaluated at the initialized point. The power flow solution is held
    fixed.

    Services computed by a Python function (``v_numeric``) are
    differentiated by a central difference of that function along the
    tangent of its inputs. Model-level ``v_numeric`` hooks are assumed not
    to change values (in ANDES they set addresses or device status).

    Parameters
    ----------
    system : System
        A system after ``TDS.init()``, at its initial point.
    pv : ParamVector
        Parameters of dynamic models.

    Attributes
    ----------
    S0 : np.ndarray
        ``d(x, y)/dp`` at ``t=0``, shape ``(dae.n + dae.m, len(pv))``.
    const_totals : dict
        ``{model: {service: d(service)/dp}}`` for the services of dynamic
        models other than ``VarService``, each of shape
        ``(n_devices, len(pv))``.
    nonfinite : list of str
        Entries of ``pv`` with derivatives that are not finite at the
        initial point (for example a saturation coefficient at zero); they
        are set to zero.
    unsupported : list of str
        Derivatives that could not be taken symbolically (taken as zero).
    """

    def __init__(self, system, pv, numeric_step=1e-7):
        self.system = system
        self.pv = pv
        self.numeric_step = numeric_step
        ss = system
        self.n_p = len(pv)

        self.dyn = OrderedDict((name, mdl) for name, mdl in ss.exist.tds.items()
                               if mdl.n > 0 and name not in ss.exist.pflow)

        self.DX = {'x': np.zeros((ss.dae.n, self.n_p)), 'y': np.zeros((ss.dae.m, self.n_p))}
        self.LT = {}          # (model, name) -> tangent of a non-inplace variable or observable
        self.ST = {}          # (model, name) -> tangent of a service
        self.P1 = {}          # (model, param) -> one-hot tangent of a parameter
        for (mname, pname), (mdl, _, uid, sl) in pv._blocks.items():
            t = np.zeros((mdl.n, self.n_p))
            t[uid, np.arange(sl.start, sl.stop)] = 1.0
            self.P1[(mname, pname)] = t

        self._symbols = {}
        self._initialized = set(name for name in ss.exist.pflow_tds if name not in self.dyn)
        self._run()

    # --- tangent lookup -----------------------------------------------------

    def _var_tangent(self, mdl, var):
        if var.n == 0:
            return None
        if var.v_inplace:
            return self.DX[var.v_code][np.asarray(var.a, dtype=int)]
        key = (mdl.class_name, var.name)
        if key in self.LT:
            return self.LT[key]
        return self.DX[var.v_code][np.asarray(var.a, dtype=int)]

    def _tangent(self, mdl, name):
        """Tangent of the input ``name`` of ``mdl``, or None if it is constant."""
        key = (mdl.class_name, name)
        if key in self.P1:
            return self.P1[key]
        inst = mdl.__dict__.get(name)
        if isinstance(inst, Observable):
            return self.LT.get(key)
        if isinstance(inst, BaseVar):
            return self._var_tangent(mdl, inst)
        if isinstance(inst, BaseService):
            return self.ST.get(key)
        return None

    def _symbol_tangent(self, mdl, sym_name, split_of):
        """Tangent of a symbol, including the real/imaginary parts of complex services."""
        if sym_name in split_of:
            base, part = split_of[sym_name]
            t = self.ST.get((mdl.class_name, base))
            if t is None:
                return None
            return np.real(t) if part == 're' else np.imag(t)
        return self._tangent(mdl, sym_name)

    # --- expression tangents --------------------------------------------------

    def _model_ctx(self, mdl):
        if mdl.class_name not in self._symbols:
            known, modules, split, parse = _model_symbols(mdl)
            split_of = {}
            for base, (re_n, im_n) in split.items():
                split_of[re_n] = (base, 're')
                split_of[im_n] = (base, 'im')
            self._symbols[mdl.class_name] = (known, modules, split, parse, split_of)
        return self._symbols[mdl.class_name]

    def _coefficient(self, mdl, expr, sym, inputs):
        """``d expr / d sym`` evaluated at the current inputs, per device."""
        known, modules, _, _, _ = self._model_ctx(mdl)
        if isinstance(expr, (sp.logic.boolalg.BooleanFunction, sp.logic.boolalg.BooleanAtom,
                             sp.core.relational.Relational)):
            return None
        d = _diff(expr, sym)
        if d == 0:
            return None
        if d.has(sp.Derivative) or d.has(sp.Subs):
            self.unsupported.append(f'{mdl.class_name}: d({expr})/d{sym}')
            return None
        return _Term(d, known, modules)(inputs, mdl.n)

    def _expr_tangent(self, mdl, expr, inputs, skip=()):
        """Tangent of a real expression, from the tangents of its symbols."""
        _, _, _, _, split_of = self._model_ctx(mdl)
        out = None
        for sym in expr.free_symbols:
            if sym.name in skip:
                continue
            t = self._symbol_tangent(mdl, sym.name, split_of)
            if t is None or not np.any(t):
                continue
            c = self._coefficient(mdl, expr, sym, inputs)
            if c is None:
                continue
            t = np.real(t)
            with np.errstate(invalid='ignore'):
                contrib = np.where(t == 0, 0.0, c[:, None] * t)   # NaN * 0 counts as 0
            out = contrib if out is None else out + contrib
        return out

    def _string_tangent(self, mdl, string, inputs, is_complex=False):
        """Tangent of an equation string; complex values give a complex tangent."""
        _, _, _, parse, _ = self._model_ctx(mdl)
        expr = parse(string)
        if is_complex:
            re_e, im_e = _split_complex(expr)
            tr = self._expr_tangent(mdl, re_e, inputs)
            ti = self._expr_tangent(mdl, im_e, inputs)
            if tr is None and ti is None:
                return None
            tr = np.zeros((mdl.n, self.n_p)) if tr is None else tr
            ti = np.zeros((mdl.n, self.n_p)) if ti is None else ti
            return tr + 1j * ti
        return self._expr_tangent(mdl, expr, inputs)

    def _numeric_tangent(self, mdl, func, inputs_names):
        """
        Directional central difference of ``func(**inputs)`` along the
        tangents of the model's inputs, one column at a time.
        """
        changed = []
        for name in inputs_names:
            t = self._tangent(mdl, name)
            inst = mdl.__dict__.get(name)
            if t is None or not np.any(t) or inst is None:
                continue
            if not isinstance(getattr(inst, 'v', None), np.ndarray):
                continue
            changed.append((inst, t))
        if not changed:
            return None

        out = None
        for j in range(self.n_p):
            cols = [(inst, t[:, j]) for inst, t in changed if np.any(t[:, j])]
            if not cols:
                continue
            vals = []
            for sign in (1.0, -1.0):
                saved = [inst.v.copy() for inst, _ in cols]
                for inst, d in cols:
                    inst.v[:] = inst.v + sign * self.numeric_step * d
                vals.append(np.asarray(func(**mdl.get_inputs(refresh=True))))
                for (inst, _), v in zip(cols, saved):
                    inst.v[:] = v
            mdl.get_inputs(refresh=True)
            col = (vals[0] - vals[1]) / (2 * self.numeric_step)
            if out is None:
                out = np.zeros((mdl.n, self.n_p), dtype=col.dtype)
            out[:, j] = col
        return out

    # --- the initialization sequence --------------------------------------------

    def _ext_service(self, mdl, inst):
        """Tangent of an ``ExtService``: the source's tangent, gathered by device."""
        ss = self.system
        if getattr(inst, 'attr', 'v') != 'v':
            return None           # e.g. a per-unit coefficient, which is constant
        src_name = inst.model
        src = ss.__dict__[src_name]
        idx = list(inst.indexer.v)
        out = np.zeros((mdl.n, self.n_p))
        any_nonzero = False
        for i, dev in enumerate(idx):
            if dev is None:
                continue
            owner = src.idx2model(dev) if hasattr(src, 'idx2model') else src
            if owner.class_name in ss.exist.pflow and owner.class_name not in self.dyn:
                continue          # the power flow solution is fixed
            uid = owner.idx2uid(dev)
            t = self._tangent(owner, inst.src)
            if t is None:
                continue
            out[i] = t[uid]
            any_nonzero = True
        return out if any_nonzero else None

    def _services(self, mdl, inputs, kinds):
        for name, inst in mdl.services.items():
            if not isinstance(inst, kinds):
                continue
            is_complex = getattr(inst, 'vtype', float) == complex
            if inst.v_numeric is not None:
                t = self._numeric_tangent(mdl, inst.v_numeric, list(inputs))
            elif inst.v_str is not None:
                t = self._string_tangent(mdl, inst.v_str, inputs, is_complex=is_complex)
            else:
                t = None
            key = (mdl.class_name, name)
            if t is None:
                self.ST.pop(key, None)
            else:
                self.ST[key] = t

    def _assign(self, mdl, var, t):
        """Store the tangent of an assigned variable as ANDES stores the value."""
        key = (mdl.class_name, var.name)
        if t is None:
            t = np.zeros((mdl.n, self.n_p))
        if isinstance(var, Observable):
            self.LT[key] = t
        elif var.v_inplace:
            self.DX[var.v_code][np.asarray(var.a, dtype=int)] = t
        else:
            self.LT[key] = t

    def _iterative(self, mdl, names, inputs):
        """``dv/dp = -J^-1 dF/dp`` for a group solved by Newton's method."""
        _, _, _, parse, _ = self._model_ctx(mdl)
        known = self._model_ctx(mdl)[0]
        F = [parse(mdl.__dict__[v].v_iter) for v in names]
        k, n = len(names), mdl.n
        J = np.zeros((n, k, k))
        R = np.zeros((n, k, self.n_p))
        for a, f in enumerate(F):
            for b, v in enumerate(names):
                c = self._coefficient(mdl, f, known[v], inputs)
                if c is not None:
                    J[:, a, b] = c
            r = self._expr_tangent(mdl, f, inputs, skip=set(names))
            if r is not None:
                R[:, a, :] = r
        # devices with a singular Jacobian keep their v_str tangent, as ANDES
        # keeps the v_str value when the Newton step fails
        ok = np.abs(np.linalg.det(J)) > 1e-12
        if not ok.any():
            return
        dv = -np.linalg.solve(J[ok], R[ok])
        for b, v in enumerate(names):
            var = mdl.__dict__[v]
            t = self._var_tangent(mdl, var) if (mdl.class_name, v) not in self.LT \
                else self.LT[(mdl.class_name, v)]
            t = np.zeros((n, self.n_p)) if t is None else np.array(t)
            t[ok] = dv[:, b, :]
            self._assign(mdl, var, t)

    def _init_model(self, mdl):
        inputs = _model_inputs(mdl, self._model_ctx(mdl)[2])

        for name, inst in mdl.services_ext.items():
            t = self._ext_service(mdl, inst)
            key = (mdl.class_name, name)
            if t is None:
                self.ST.pop(key, None)
            else:
                self.ST[key] = t

        # s_update, then s_update_var
        self._services(mdl, inputs, BaseService)
        self._services(mdl, inputs, VarService)

        for item in mdl.calls.init_seq:
            names = item if isinstance(item, list) else [item]
            iterative = isinstance(item, list) or mdl.__dict__[item].v_iter is not None
            if iterative:
                # as in ANDES: start from v_str, then solve; a device whose
                # Newton step fails (singular Jacobian) keeps the v_str value
                for v in names:
                    inst = mdl.__dict__[v]
                    if inst.v_str is not None:
                        self._assign(mdl, inst, self._string_tangent(mdl, inst.v_str, inputs))
                self._iterative(mdl, names, inputs)
                continue
            inst = mdl.__dict__[item]
            string = inst.e_str if isinstance(inst, Observable) else inst.v_str
            if string is None:
                continue
            self._assign(mdl, inst, self._string_tangent(mdl, string, inputs))

        self._initialized.add(mdl.class_name)
        self._vars_to_dae(mdl)
        self._vars_to_models()

    def _vars_to_dae(self, mdl):
        """As ``System._v_to_dae``: add this model's adders, then apply all setters."""
        for var in mdl.cache.v_adders.values():
            t = self.LT.get((mdl.class_name, var.name))
            if t is not None and var.n:
                np.add.at(self.DX[var.v_code], np.asarray(var.a, dtype=int), t)
        for name in self._initialized:
            owner = self.system.__dict__[name]
            for var in owner.cache.v_setters.values():
                t = self.LT.get((name, var.name))
                if t is not None and var.n:
                    self.DX[var.v_code][np.asarray(var.a, dtype=int)] = t

    def _vars_to_models(self):
        """As ``System.vars_to_models``: non-inplace variables read the DAE values."""
        for name, mdl in self.system.exist.pflow_tds.items():
            for var in mdl.cache.v_getters.values():
                if var.n:
                    self.LT[(name, var.name)] = self.DX[var.v_code][np.asarray(var.a, dtype=int)]

    def _check_supported(self):
        """Models whose services or variables are not one value per device are not supported."""
        for mdl in self.dyn.values():
            for name, inst in list(mdl.services.items()) + list(mdl.services_ext.items()):
                v = getattr(inst, 'v', None)
                if isinstance(v, np.ndarray) and v.ndim == 1 and v.size != mdl.n:
                    raise NotImplementedError(
                        f"{mdl.class_name}.{name} has {v.size} values for {mdl.n} devices; "
                        "the exact initial sensitivity does not support such models.")

    def _run(self):
        self._check_supported()
        self.unsupported = []
        self._vars_to_models()
        for mdl in self.dyn.values():
            self._init_model(mdl)

        # s_update_post
        for mdl in self.dyn.values():
            inputs = _model_inputs(mdl, self._model_ctx(mdl)[2])
            self._services(mdl, inputs, PostInitService)

        self.S0 = np.vstack([self.DX['x'], self.DX['y']])
        bad = ~np.isfinite(self.S0)
        for t in self.ST.values():
            if not np.all(np.isfinite(t)):
                bad_cols = np.flatnonzero(~np.all(np.isfinite(t), axis=0))
                bad[0, bad_cols] = True
        self.nonfinite = [self.pv.names[j] for j in np.flatnonzero(bad.any(axis=0))]
        self.S0[~np.isfinite(self.S0)] = 0.0
        for key, t in self.ST.items():
            if not np.all(np.isfinite(t)):
                t = t.copy()
                t[~np.isfinite(t)] = 0.0
                self.ST[key] = t
        if self.nonfinite:
            logger.warning("InitSensitivity: derivatives not finite at the initial point, "
                           "set to zero: %s", ', '.join(self.nonfinite))
        totals = {}
        for (mname, name), t in self.ST.items():
            inst = self.system.__dict__[mname].__dict__.get(name)
            if isinstance(inst, VarService) or not np.any(t):
                continue
            totals.setdefault(mname, {})[name] = t
        self.const_totals = totals

        if self.unsupported:
            logger.warning("InitSensitivity: not differentiated (taken as zero): %s",
                           '; '.join(sorted(set(self.unsupported))))
