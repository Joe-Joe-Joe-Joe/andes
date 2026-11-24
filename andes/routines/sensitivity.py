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
from andes.core.paramjac import ParamJacobian, varservice_jacobian
from andes.core.service import BaseService, VarService
from andes.routines.daeint import ImplicitIter, Trapezoid
from andes.shared import np

logger = logging.getLogger(__name__)


def _to_scipy(m):
    """Convert a kvxopt ``spmatrix`` to a SciPy CSC matrix."""
    from scipy.sparse import csc_matrix

    return csc_matrix((np.array(m.V).ravel(), (np.array(m.I).ravel(), np.array(m.J).ravel())),
                      shape=m.size)


def _copy_sp(m):
    from kvxopt import spmatrix

    return spmatrix(m.V, m.I, m.J, m.size)


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
        Parameters of dynamic models, without event times.
    rel_step : float, optional
        Relative step for the finite differences of the initialization.

    Notes
    -----
    The sensitivities of all variables are stored at every step, which
    takes ``8 * n_steps * (dae.n + dae.m) * len(pv)`` bytes. Event times
    (``Fault.tc``, ``Toggle.t``) and parameters of power flow models are
    not supported. Switching of limiters between steps is not located in
    time, as in the simulation itself.

    Attributes
    ----------
    S0 : np.ndarray
        Initial sensitivities, shape ``(dae.n + dae.m, len(pv))``.
    t : np.ndarray
        Times of the stored sensitivities, after :meth:`run`.
    Sx, Sy : np.ndarray
        Sensitivities of states and algebraic variables, shapes
        ``(len(t), dae.n, len(pv))`` and ``(len(t), dae.m, len(pv))``.

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
        timers = [n for (m, p), (_, inst, _, _) in pv._blocks.items()
                  for n in [f'{m}.{p}'] if isinstance(inst, TimerParam)]
        if timers:
            raise ValueError("Event times change the step sequence and are not supported: "
                             + ', '.join(timers))

        self.system = system
        self.pv = pv
        self.rel_step = rel_step

        self.S0, self.const_totals = self._initial_sensitivity()
        self.pj = ParamJacobian(system, pv, const_totals=self.const_totals)
        self.vsj = varservice_jacobian(system)
        self.unsupported = list(self.pj.unsupported) + list(self.vsj.unsupported)
        self.tp = self.pj.tf_p().tocsr()
        self._pinned = self._pinned_limits()

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
        totals = {m: {s: np.zeros((v.size, n_p), dtype=v.dtype) for s, v in d.items()}
                  for m, d in base.items()}

        try:
            for j in range(n_p):
                h = self.rel_step * max(1.0, abs(p0[j]))
                runs = []
                for sign in (1.0, -1.0):
                    p = p0.copy()
                    p[j] += sign * h
                    pv.set(p)
                    ss.TDS.reinit(keep_params=True)
                    runs.append((dae.xy.copy(), self._constants()))
                (xy_p, c_p), (xy_m, c_m) = runs
                S0[:, j] = (xy_p - xy_m) / (2 * h)
                for m, d in totals.items():
                    for s, arr in d.items():
                        arr[:, j] = (c_p[m][s] - c_m[m][s]) / (2 * h)
        finally:
            pv.set(p0)
            ss.TDS.reinit(keep_params=True)

        # keep only constants that move
        totals = {m: {s: a for s, a in d.items() if np.any(a)} for m, d in totals.items()}
        totals = {m: d for m, d in totals.items() if d}
        return S0, totals

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
        """Exact Jacobians at the current point, leaving ANDES's own untouched."""
        ss = self.system
        dae = ss.dae
        saved = [_copy_sp(m) for m in (dae.fx, dae.fy, dae.gx, dae.gy)]
        ss.j_update(models=ss.exist.pflow_tds)
        jac = [_to_scipy(m) for m in (dae.fx, dae.fy, dae.gx, dae.gy)]
        dae.fx, dae.fy, dae.gx, dae.gy = saved

        # terms through VarService values, which ANDES's Jacobians leave out
        extra = self.vsj()
        jac = [j + extra[k] for j, k in zip(jac, ('fx', 'fy', 'gx', 'gy'))]
        fp, gp = self.pj.evaluate()
        return jac, fp.toarray(), gp.toarray()

    def _start(self):
        n = self.system.dae.n
        self._Sx = self.S0[:n].copy()
        self._Sy = self.S0[n:].copy()
        (self._fx0, self._fy0, _, _), self._fp0, _ = self._jacobians()
        self._t, self._sx, self._sy = [0.0], [self._Sx.copy()], [self._Sy.copy()]

    def _advance(self, tds):
        from scipy.sparse import bmat, diags
        from scipy.sparse.linalg import splu

        dae = self.system.dae
        n, m = dae.n, dae.m
        h = float(tds.h)
        (fx, fy, gx, gy), fp1, gp1 = self._jacobians()
        Tf = dae.Tf

        A = bmat([[diags(Tf) - 0.5 * h * fx, -0.5 * h * fy],
                  [gx, gy]], format='csr')

        rhs = np.empty((n + m, len(self.pv)))
        rhs[:n] = (Tf[:, None] * self._Sx
                   + 0.5 * h * (self._fx0 @ self._Sx + self._fy0 @ self._Sy + self._fp0 + fp1)
                   - self.tp.multiply((dae.x - tds.x0)[:, None]).toarray())
        rhs[n:] = -gp1

        # states pinned at a limit follow the limit: replace their rows by
        # identity rows with the derivative of the limit on the right
        pinned = np.zeros(n + m, dtype=bool)
        for item, limits in self._pinned:
            if not item.x_set:
                continue
            for side, flag in (('upper', item.zu), ('lower', item.zl)):
                if side not in limits:
                    continue
                for i in np.flatnonzero(flag):
                    row = int(item.state.a[i])
                    pinned[row] = True
                    rhs[row] = limits[side][i]
        if pinned.any():
            A = diags((~pinned).astype(float)) @ A + diags(pinned.astype(float))

        S = splu(A.tocsc()).solve(rhs)
        self._Sx, self._Sy = S[:n], S[n:]
        self._fx0, self._fy0, self._fp0 = fx, fy, fp1

        self._t.append(float(dae.t))
        self._sx.append(self._Sx.copy())
        self._sy.append(self._Sy.copy())

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
        return ret

    def get(self, var):
        """
        Sensitivities of a model variable, shape ``(len(t), var.n, len(pv))``.

        Parameters
        ----------
        var : BaseVar
            A model variable, such as ``ss.GENROU.omega``.
        """
        a = np.asarray(var.a, dtype=int)
        return self.Sx[:, a, :] if var.v_code == 'x' else self.Sy[:, a, :]
