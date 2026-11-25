"""
Forward trajectory sensitivity for time-domain simulation.
"""

#  [ANDES] (C)2015-2024 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

import logging
from collections import OrderedDict

from andes.core.param import NumParam, TimerParam
from andes.core.paramjac import ParamJacobian, input_cache, varservice_jacobian
from andes.core.service import BaseService, VarService
from andes.routines.daeint import ImplicitIter, Trapezoid
from andes.shared import np

logger = logging.getLogger(__name__)


def _copy_sp(m):
    from kvxopt import spmatrix

    return spmatrix(m.V, m.I, m.J, m.size)


def _triplets(m):
    """``(rows, cols, vals)`` of a kvxopt ``spmatrix`` as NumPy arrays."""
    return (np.array(m.I, dtype=int).ravel(), np.array(m.J, dtype=int).ravel(),
            np.array(m.V).ravel())


class _StepMatrix:
    """
    The sensitivity step matrix ``[[Tf - h/2 fx, -h/2 fy], [gx, gy]]`` with a
    fixed sparsity pattern, factorized with KLU.

    The pattern is the union of its blocks and the full diagonal (for rows
    replaced by identity rows). It is built once and rebuilt only if a
    block's pattern changes; each step scatters new values into it and
    reuses the symbolic factorization.
    """

    def __init__(self, n, m, nrhs):
        from kvxopt import matrix

        self.n, self.m = n, m
        self._key = None
        # right-hand side, solved in place; ``rhs`` is a NumPy view of it
        self._B = matrix(0.0, (n + m, nrhs))
        self.rhs = np.asarray(self._B)

    def _build(self, blocks):
        size = self.n + self.m
        rows = np.concatenate([r for r, _ in blocks] + [np.arange(size)])
        cols = np.concatenate([c for _, c in blocks] + [np.arange(size)])
        uniq, inv = np.unique(cols.astype(np.int64) * size + rows, return_inverse=True)
        self.rows = (uniq % size).astype(int)
        self.cols = (uniq // size).astype(int)
        self.inv = inv
        self.diag = inv[-size:]
        self.nnz = uniq.size
        self.F = None

    def solve(self, blocks, values, pinned):
        """
        Assemble from ``blocks`` (list of ``(rows, cols)``) and ``values``,
        replace ``pinned`` rows by identity rows, and solve in place for the
        right-hand side written into :attr:`rhs`, which then holds the
        solution.
        """
        from kvxopt import klu, spmatrix

        key = [np.concatenate([r for r, _ in blocks]), np.concatenate([c for _, c in blocks])]
        if self._key is None or any(a.shape != b.shape or not np.array_equal(a, b)
                                    for a, b in zip(key, self._key)):
            self._build(blocks)
            self._key = key

        data = np.bincount(self.inv[:-self.diag.size], weights=np.concatenate(values),
                           minlength=self.nnz)
        if pinned.any():
            data[pinned[self.rows]] = 0.0
            data[self.diag[pinned]] = 1.0

        A = spmatrix(data, self.rows, self.cols, (self.n + self.m, self.n + self.m))
        if self.F is None:
            self.F = klu.symbolic(A)
        N = klu.numeric(A, self.F)
        klu.solve(A, self.F, N, self._B)
        return self.rhs


class _SensTrapezoid(Trapezoid):
    """Trapezoid method that advances the sensitivities after each converged step."""

    def __init__(self, sens):
        self.sens = sens

    def step(self, tds):
        converged = ImplicitIter.step(tds)
        if converged:
            self.sens._advance(tds)
        return converged


class TrajectorySensitivity:
    """
    Forward sensitivity of the simulated trajectory with respect to a
    :class:`andes.core.ParamVector`.

    For each accepted step of the implicit trapezoidal method, the
    sensitivities ``Sx = dx/dp`` and ``Sy = dy/dp`` are advanced by
    differentiating the discrete step that ANDES solves,

    .. code-block:: text

        Tf (x1 - x0) - h/2 (f(x1, y1) + f0) = 0,     g(x1, y1) = 0,

    which gives one sparse linear system per step with the same matrix as
    the Newton iteration, evaluated at the converged point::

        [Tf - h/2 fx   -h/2 fy] [Sx1]   [Tf Sx0 + h/2 (fx0 Sx0 + fy0 Sy0 + fp0 + fp1) - Tp (x1 - x0)]
        [gx             gy    ] [Sy1] = [-gp1                                                      ]

    ``f0`` is the residual at the previous step, before any event at that
    time, as in ANDES. The result is the exact derivative of the numerical
    trajectory for a given sequence of step sizes. States pinned by
    anti-windup limiters take the derivative of their limit.

    Event times move the step grid. ANDES lands steps exactly on
    ``t* - eps``, ``t*`` and ``t* + eps`` of every event time ``t*``, and
    the event acts at the same step whatever ``t*`` is. So step ``k`` ends
    at ``t_k(p)``, with ``dt_k/dp`` equal to 1 for steps tied to an event
    time ``p`` and those after it, and the step size enters as
    ``+ dh/dp * (f1 + f0) / 2`` on the right-hand side. :attr:`Sx` and
    :attr:`Sy` are derivatives of the values at each step, whose time moves
    with an event time; :meth:`get` with ``fixed_time=True`` converts them
    to derivatives at fixed times.

    The Jacobians are evaluated exactly at each converged point. This
    includes the terms through ``VarService`` values (such as ``ESST3A.VE``),
    which the Jacobians that ANDES uses for Newton's method leave out.

    The initial sensitivities ``S(0)``, and the derivatives of the values
    that ANDES fixes at initialization (setpoints such as ``vref0`` and
    ``pref0``, ``ExtService`` and ``PostInitService`` values), are computed
    by central differences of ``TDS.reinit(keep_params=True)``.

    Parameters
    ----------
    system : System
        A system after ``TDS.init()`` (or a plain ``TDS.reinit()``), at
        ``t = 0``.
    pv : ParamVector
        Parameters of dynamic models, including event times
        (``TimerParam`` such as ``Fault.tf``, ``Fault.tc`` or ``Toggle.t``).
    rel_step : float, optional
        Relative step for the finite differences of the initialization.

    Notes
    -----
    The sensitivities of all variables are stored at every step, which
    takes ``8 * n_steps * (dae.n + dae.m) * len(pv)`` bytes. Parameters
    of power flow models are not supported. Switching of limiters between
    steps is not located in
    time, as in the simulation itself. At a parameter value where the model
    changes structure, such as a zero resistance or time constant, the
    derivative is one-sided.

    Attributes
    ----------
    inconsistent : list of str
        Entries of ``pv`` whose change leaves the initialization out of
        balance with the power flow (for example ``GENROU.gammap`` or
        ``TGOV1.wref0``): ``t=0`` is then not an equilibrium, and their
        sensitivities are not meaningful.
    S0 : np.ndarray
        Initial sensitivities, shape ``(dae.n + dae.m, len(pv))``.
    t : np.ndarray
        Times of the stored sensitivities, after :meth:`run`.
    Sx, Sy : np.ndarray
        Sensitivities of states and algebraic variables, shapes
        ``(len(t), dae.n, len(pv))`` and ``(len(t), dae.m, len(pv))``.
    dt : np.ndarray
        Derivatives of the step times, shape ``(len(t), len(pv))``. Non-zero
        only for event times.
    xdot, ydot : np.ndarray
        Time derivatives of the variables at each step, for converting to
        fixed-time sensitivities (``ydot`` is computed only when ``pv``
        contains event times).

    Examples
    --------
    >>> ss.PFlow.run()
    >>> ss.TDS.config.tf = 5
    >>> ss.TDS.init()
    >>> pv = ParamVector(ss, ['GENROU.M', 'GENROU.D', 'TGOV1.R'])
    >>> sens = TrajectorySensitivity(ss, pv)
    >>> sens.run()
    >>> s = sens.get(ss.GENROU.omega)      # (len(t), n_gen, len(pv))
    """

    def __init__(self, system, pv, rel_step=1e-6):
        tds = system.TDS
        if not tds.initialized:
            raise RuntimeError("Call TDS.init() before TrajectorySensitivity.")
        if system.dae.t != 0:
            raise RuntimeError("TrajectorySensitivity must start at t=0; call TDS.reinit() first.")
        if tds.config.method != 'trapezoid':
            raise ValueError(f"Only the 'trapezoid' method is supported, not {tds.config.method!r}.")
        if pv.system is not system:
            raise ValueError("The ParamVector belongs to another system.")
        if np.any(pv.pflow):
            names = [n for n, f in zip(pv.names, pv.pflow) if f]
            raise ValueError("Parameters of power flow models need a new power flow and are "
                             f"not supported: {', '.join(names[:5])}...")
        self.system = system
        self.pv = pv
        self.rel_step = rel_step

        # columns of event times (TimerParam), which move the step grid
        self._timer = np.zeros(len(pv), dtype=bool)
        for (_, _), (_, inst, _, sl) in pv._blocks.items():
            if isinstance(inst, TimerParam):
                self._timer[sl] = True

        self.S0, self.const_totals = self._initial_sensitivity()
        self.pj = ParamJacobian(system, pv, const_totals=self.const_totals)
        self.vsj = varservice_jacobian(system)
        self.unsupported = list(self.pj.unsupported) + list(self.vsj.unsupported)
        self.tp = self.pj.tf_p().tocsr()
        self._pinned = self._pinned_limits()
        self._pin_index = self._pinned_index()

        self.t = np.array([])
        self.Sx = np.zeros((0, system.dae.n, len(pv)))
        self.Sy = np.zeros((0, system.dae.m, len(pv)))

    # --- initialization ---------------------------------------------------

    def _dynamic_models(self):
        ss = self.system
        return OrderedDict((name, mdl) for name, mdl in ss.exist.tds.items()
                           if mdl.n > 0 and name not in ss.exist.pflow)

    def _constants(self):
        """Copies of all constant service values of dynamic models."""
        out = {}
        for mname, mdl in self._dynamic_models().items():
            vals = {}
            for name, instance in mdl.__dict__.items():
                if not isinstance(instance, BaseService) or isinstance(instance, VarService):
                    continue
                v = getattr(instance, 'v', None)
                if isinstance(v, np.ndarray) and v.shape == (mdl.n,) and v.dtype.kind in 'fc':
                    vals[name] = v.copy()
            out[mname] = vals
        return out

    def _initial_sensitivity(self):
        """
        Central differences of the initialization with respect to each
        parameter: ``S(0)`` and the total derivatives of the constants.
        """
        ss, pv = self.system, self.pv
        dae = ss.dae
        p0 = pv.get()
        n_p = len(pv)

        S0 = np.zeros((dae.n + dae.m, n_p))
        base = self._constants()
        self.inconsistent = []
        totals = {m: {s: np.zeros((v.size, n_p), dtype=v.dtype) for s, v in d.items()}
                  for m, d in base.items()}

        try:
            for j in range(n_p):
                if self._timer[j]:
                    continue        # event times do not change the initialization
                h = self.rel_step * max(1.0, abs(p0[j]))
                runs = []
                for sign in (1.0, -1.0):
                    p = p0.copy()
                    p[j] += sign * h
                    pv.set(p)
                    ss.TDS.reinit(keep_params=True)
                    runs.append((dae.xy.copy(), self._constants(), self._init_residual()))
                (xy_p, c_p, r_p), (xy_m, c_m, r_m) = runs
                S0[:, j] = (xy_p - xy_m) / (2 * h)
                # an equilibrium stays an equilibrium: d(f, g)/dp at t=0 is zero
                if np.max(np.abs(r_p - r_m)) / (2 * h) > 1e-3:
                    self.inconsistent.append(pv.names[j])
                for m, d in totals.items():
                    for s, arr in d.items():
                        arr[:, j] = (c_p[m][s] - c_m[m][s]) / (2 * h)
        finally:
            pv.set(p0)
            ss.TDS.reinit(keep_params=True)

        if self.inconsistent:
            logger.warning("TrajectorySensitivity: after changing these parameters, the "
                           "initialization no longer matches the power flow, so t=0 is not an "
                           "equilibrium and their sensitivities are not meaningful: %s",
                           ', '.join(self.inconsistent))

        # keep only constants that move
        totals = {m: {s: a for s, a in d.items() if np.any(a)} for m, d in totals.items()}
        totals = {m: d for m, d in totals.items() if d}
        return S0, totals

    def _init_residual(self):
        """Residuals at the current (initial) point, as checked by ``TDS.test_init``."""
        ss = self.system
        ss.TDS.fg_update(ss.exist.pflow_tds)
        fg = ss.dae.fg.copy()
        fg[np.asarray(ss.no_check_init, dtype=int)] = 0.0
        return fg

    def _pinned_limits(self):
        """
        For each anti-windup limiter, ``d(limit)/dp`` for the upper and lower
        limits, as arrays of shape ``(n_devices, len(pv))``. Limits that are
        neither parameters in ``pv`` nor moving constants get zeros.
        """
        pv = self.pv
        out = []
        for item in self.system.antiwindups:
            mdl = item.owner
            entry = {}
            for side, sign_name in (('upper', 'sign_upper'), ('lower', 'sign_lower')):
                lim = getattr(item, side, None)
                if lim is None or getattr(item, f'no_{side}', False):
                    continue
                sign = getattr(item, sign_name, None)
                sign = -1.0 if (sign is not None and np.all(np.asarray(sign.v) == -1)) else 1.0
                rows = np.zeros((mdl.n, len(pv)))
                key = (mdl.class_name, lim.name)
                if isinstance(lim, NumParam) and key in pv._blocks:
                    for i, idx in enumerate(mdl.idx.v):
                        try:
                            rows[i, pv.index(key[0], key[1], idx)] = sign
                        except KeyError:
                            pass
                elif lim.name in self.const_totals.get(mdl.class_name, {}):
                    rows = sign * np.real(self.const_totals[mdl.class_name][lim.name])
                entry[side] = rows
            out.append((item, entry))
        return out

    # --- stepping ---------------------------------------------------------

    def _jacobians(self):
        """
        Exact Jacobians at the current point as ``(rows, cols, vals)``
        triplets, including the terms through ``VarService`` values, and the
        parameter Jacobians. ANDES's own Jacobians are left untouched.
        """
        ss = self.system
        dae = ss.dae
        saved = [_copy_sp(m) for m in (dae.fx, dae.fy, dae.gx, dae.gy)]
        ss.j_update(models=ss.exist.pflow_tds)
        jac = dict(zip(('fx', 'fy', 'gx', 'gy'),
                       (_triplets(m) for m in (dae.fx, dae.fy, dae.gx, dae.gy))))
        dae.fx, dae.fy, dae.gx, dae.gy = saved

        with input_cache():
            # terms through VarService values, which ANDES's Jacobians leave out
            extra = self.vsj.triplets()
            fp, gp = self.pj.evaluate()
        jac = {k: tuple(np.concatenate([a, b]) for a, b in zip(jac[k], extra[k])) for k in jac}
        return jac, fp, gp

    def _rhs_operators(self, jac):
        """``fx`` and ``fy`` as SciPy matrices, for the next step's right-hand side."""
        from scipy.sparse import coo_matrix

        n, m = self.system.dae.n, self.system.dae.m
        fx = coo_matrix((jac['fx'][2], (jac['fx'][0], jac['fx'][1])), shape=(n, n)).tocsr()
        fy = coo_matrix((jac['fy'][2], (jac['fy'][0], jac['fy'][1])), shape=(n, m)).tocsr()
        return fx, fy

    def _start(self):
        dae = self.system.dae
        n = dae.n
        self._Sx = self.S0[:n].copy()
        self._Sy = self.S0[n:].copy()
        jac, self._fp0, _ = self._jacobians()
        self._fx0, self._fy0 = self._rhs_operators(jac)
        self._time_map = self._switch_time_map()
        self._dt_prev = np.zeros(len(self.pv))
        self._dt, self._xdot, self._ydot = [self._dt_prev.copy()], [], []
        self._matrix = _StepMatrix(dae.n, dae.m, len(self.pv))
        tp = self.tp.tocoo()
        self._tp = (tp.row, tp.col, tp.data)
        self._t, self._sx, self._sy = [0.0], [self._Sx.copy()], [self._Sy.copy()]

    def _pinned_index(self):
        """
        Stack the limiters' devices once: state addresses, the flag arrays to
        read each step, and the derivative of each device's limit.
        """
        addr, flags, rows = [], [], []
        for item, limits in self._pinned:
            for side, flag_name in (('upper', 'zu'), ('lower', 'zl')):
                if side not in limits:
                    continue
                addr.append(np.asarray(item.state.a, dtype=int))
                flags.append((item, flag_name))
                rows.append(limits[side])
        if not addr:
            return None
        return np.concatenate(addr), flags, np.vstack(rows)

    def _algebraic_rate(self, jac, xdot):
        """``dy/dt`` from ``g(x, y) = 0``: ``gy dy/dt = -gx dx/dt``."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu

        n, m = self.system.dae.n, self.system.dae.m
        (gxr, gxc, gxv), (gyr, gyc, gyv) = jac['gx'], jac['gy']
        gx = coo_matrix((gxv, (gxr, gxc)), shape=(m, n)).tocsr()
        gy = coo_matrix((gyv, (gyr, gyc)), shape=(m, m)).tocsc()
        return splu(gy).solve(-(gx @ xdot))

    def _switch_time_map(self):
        """
        Map each time that steps are clipped to (event switch times and the
        end time) to its derivative with respect to the parameters.

        ANDES lands steps exactly on ``t* - eps``, ``t*`` and ``t* + eps`` for
        every event time ``t*``; these move with ``t*`` and stay fixed for
        all other parameters.
        """
        import inspect

        ss = self.system
        eps = inspect.signature(ss.store_switch_times).parameters['eps'].default
        out = {float(t): np.zeros(len(self.pv)) for t in ss.switch_times}
        out[float(ss.TDS.config.tf)] = np.zeros(len(self.pv))

        p = self.pv.get()
        for j in np.flatnonzero(self._timer):
            v = np.float64(p[j])
            for t in (v - eps, v, v + eps):
                if float(t) in out:
                    out[float(t)][j] = 1.0
        return out

    def _pinned_rows(self, rhs):
        """Mark states pinned at a limit and put the limit's derivative into ``rhs``."""
        pinned = np.zeros(rhs.shape[0], dtype=bool)
        if self._pin_index is None:
            return pinned
        addr, flags, rows = self._pin_index
        on = np.concatenate([np.asarray(getattr(item, name), dtype=bool) if item.x_set
                             else np.zeros(item.owner.n, dtype=bool) for item, name in flags])
        if on.any():
            pinned[addr[on]] = True
            rhs[addr[on]] = rows[on]
        return pinned

    def _advance(self, tds):
        dae = self.system.dae
        n = dae.n
        h = float(tds.h)
        jac, fp1, gp1 = self._jacobians()
        Tf = dae.Tf

        (fxr, fxc, fxv), (fyr, fyc, fyv) = jac['fx'], jac['fy']
        (gxr, gxc, gxv), (gyr, gyc, gyv) = jac['gx'], jac['gy']
        diag = np.arange(n)
        blocks = [(diag, diag), (fxr, fxc), (fyr, fyc + n), (gxr + n, gxc), (gyr + n, gyc + n)]
        values = [Tf, -0.5 * h * fxv, -0.5 * h * fyv, gxv, gyv]

        rhs = self._matrix.rhs
        rhs[:n] = Tf[:, None] * self._Sx + 0.5 * h * (self._fx0 @ self._Sx + self._fy0 @ self._Sy)
        rhs[n:] = 0.0
        fp = (self._fp0 + fp1).tocoo()
        np.add.at(rhs, (fp.row, fp.col), 0.5 * h * fp.data)
        gp = gp1.tocoo()
        np.add.at(rhs, (gp.row + n, gp.col), -gp.data)
        r, c, v = self._tp
        np.add.at(rhs, (r, c), -v * (dae.x - tds.x0)[r])

        # the step grid: a step clipped to a switch time (or to the end time)
        # ends at that time; any other step keeps the previous step's shift
        dt = self._time_map.get(float(dae.t), self._dt_prev)
        dh = dt - self._dt_prev
        cols = np.flatnonzero(dh)
        if cols.size:
            rhs[:n, cols] += 0.5 * (dae.f + tds.f0)[:, None] * dh[cols]

        # states pinned at a limit follow the limit: their rows become
        # identity rows with the derivative of the limit on the right
        pinned = self._pinned_rows(rhs)

        S = self._matrix.solve(blocks, values, pinned)
        self._Sx, self._Sy = S[:n].copy(), S[n:].copy()
        self._fx0, self._fy0 = self._rhs_operators(jac)
        self._fp0 = fp1

        self._t.append(float(dae.t))
        self._sx.append(self._Sx)
        self._sy.append(self._Sy)

        self._dt_prev = dt.copy()
        self._dt.append(self._dt_prev)
        with np.errstate(divide='ignore', invalid='ignore'):
            xdot = np.where(Tf != 0, dae.f / Tf, 0.0)
        self._xdot.append(xdot)
        self._ydot.append(self._algebraic_rate(jac, xdot) if self._timer.any()
                          else np.zeros(dae.m))

    def run(self, **kwargs):
        """
        Run ``TDS.run(**kwargs)`` and compute the sensitivities along the way.

        Returns
        -------
        bool
            The return value of ``TDS.run``.
        """
        tds = self.system.TDS
        if self.system.dae.t != 0:
            raise RuntimeError("Call TDS.reinit() before running the sensitivity again.")

        self._start()
        method = tds.method
        tds.method = _SensTrapezoid(self)
        try:
            ret = tds.run(**kwargs)
        finally:
            tds.method = method

        self.t = np.array(self._t)
        self.Sx = np.array(self._sx)
        self.Sy = np.array(self._sy)
        self.dt = np.array(self._dt)
        n, m = self.system.dae.n, self.system.dae.m
        self.xdot = np.array([np.zeros(n)] + self._xdot)
        self.ydot = np.array([np.zeros(m)] + self._ydot)
        return ret

    def get(self, var, fixed_time=False):
        """
        Sensitivities of a model variable, shape ``(len(t), var.n, len(pv))``.

        Parameters
        ----------
        var : BaseVar
            A model variable, such as ``ss.GENROU.omega``.
        fixed_time : bool, optional
            ``False`` (default): derivatives of the values at each step. For
            event times, the step times move too (by :attr:`dt`).
            ``True``: derivatives at fixed times, ``S - dX/dt * dt``, with
            ``dX/dt`` from the equations (``Tf dx/dt = f`` and
            ``gy dy/dt = -gx dx/dt``). Identical for parameters other than
            event times. Not meaningful right at an event, where the
            variable jumps.
        """
        a = np.asarray(var.a, dtype=int)
        if var.v_code == 'x':
            S, rate = self.Sx[:, a, :], self.xdot[:, a]
        else:
            S, rate = self.Sy[:, a, :], self.ydot[:, a]
        if fixed_time:
            S = S - rate[:, :, None] * self.dt[:, None, :]
        return S
