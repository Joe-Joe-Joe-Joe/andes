"""
Tests for ``andes.core.ParamJacobian``, checked against central finite
differences of the DAE residuals at the initial state.
"""

import unittest

import numpy as np

import andes
from andes.core import ParamJacobian, ParamVector
from andes.utils.paths import get_case


def _load(case):
    ss = andes.load(get_case(case), default_config=True, no_output=True)
    ss.PFlow.run()
    ss.TDS.init()
    return ss


def _residuals(ss):
    x0 = ss.dae.x.copy()
    ss.TDS.fg_update(ss.exist.pflow_tds)
    moved = not np.array_equal(ss.dae.x, x0)    # an anti-windup pinned a state
    return ss.dae.f.copy(), ss.dae.g.copy(), moved


def _restore(ss, mdl, services, x0, y0):
    for name, value in services.items():
        mdl.services[name].v = value.copy()
    mdl.get_inputs(refresh=True)
    mdl.refresh_inputs_arg()
    ss.dae.x[:] = x0
    ss.dae.y[:] = y0
    ss.vars_to_models()


def finite_difference(ss, pv, j, rel=1e-6):
    """
    Central difference of ``(f, g)`` with respect to entry ``j`` of ``pv``,
    at fixed ``(x, y)``. The model's services are recomputed as ANDES does
    at initialization, except those set from variables after init.
    Returns None if the perturbation moved a state pinned by an anti-windup
    limiter, since the Jacobians are defined at fixed ``x``.
    """
    mdl = ss.__dict__[pv.model[j]]
    p0 = pv.get()
    h = rel * max(1.0, abs(p0[j]))
    services = {k: np.array(s.v).copy() for k, s in mdl.services.items()}
    keep = {k: v for k, v in services.items()
            if k in mdl.services_post or mdl.services[k].v_numeric is not None}
    x0, y0 = ss.dae.x.copy(), ss.dae.y.copy()

    out = []
    for sign in (1, -1):
        p = p0.copy()
        p[j] += sign * h
        pv.set(p)
        mdl.s_update()
        _restore(ss, mdl, keep, x0, y0)
        out.append(_residuals(ss))

    pv.set(p0)
    _restore(ss, mdl, services, x0, y0)
    if out[0][2] or out[1][2]:
        return None    # the perturbation moved a pinned state: not at fixed x
    return (out[0][0] - out[1][0]) / (2 * h), (out[0][1] - out[1][1]) / (2 * h)


class ParamJacCase(unittest.TestCase):
    case = None

    @classmethod
    def setUpClass(cls):
        if cls.case is None:
            raise unittest.SkipTest('base class')
        cls.ss = _load(cls.case)
        cls.pv = ParamVector.all(cls.ss, pflow=False)
        cls.pj = ParamJacobian(cls.ss, cls.pv)

    def test_matches_finite_differences(self):
        ss, pv = self.ss, self.pv
        fp, gp = (m.toarray() for m in self.pj.evaluate())
        self.assertEqual(fp.shape, (ss.dae.n, len(pv)))
        self.assertEqual(gp.shape, (ss.dae.m, len(pv)))
        self.assertEqual(self.pj.unsupported, [])

        nonzero = 0
        for j in range(len(pv)):
            fd = finite_difference(ss, pv, j)
            if fd is None or pv.names[j] in self.pj.nonfinite:
                continue          # pinned state moved, or derivative may not exist
            dfj, dgj = fd
            scale = max(1.0, np.abs(dfj).max(), np.abs(dgj).max())
            with self.subTest(pv.names[j]):
                self.assertTrue(np.all(np.isfinite(dfj)) and np.all(np.isfinite(dgj)))
                np.testing.assert_allclose(fp[:, j], dfj, rtol=0, atol=1e-5 * scale)
                np.testing.assert_allclose(gp[:, j], dgj, rtol=0, atol=1e-5 * scale)
            nonzero += bool(np.any(dfj) or np.any(dgj))
        self.assertGreater(nonzero, 10)
        self.assertLess(len(self.pj.nonfinite), len(pv) // 10)


class TestParamJacKundur(ParamJacCase):
    case = 'kundur/kundur_full.xlsx'

    def test_time_constants(self):
        """``GENROU.M`` scales ``omega`` through the mass matrix only."""
        ss, pv = self.ss, self.pv
        tp = self.pj.tf_p().toarray()
        for i, idx in enumerate(ss.GENROU.idx.v):
            j = pv.index('GENROU', 'M', idx)
            col = np.zeros(ss.dae.n)
            col[ss.GENROU.omega.a[i]] = 1.0
            np.testing.assert_array_equal(tp[:, j], col)

        fp, gp = self.pj.evaluate()
        cols = pv.index('GENROU', 'M')
        self.assertEqual(fp[:, cols].nnz + gp[:, cols].nnz, 0)

    def test_subset_columns(self):
        """Columns follow the vector, so a subset gives the matching columns."""
        full_f, full_g = self.pj.evaluate()
        sub = self.pv.subset(self.pv.index('TGOV1', 'R'))
        f, g = ParamJacobian(self.ss, sub).evaluate()
        cols = self.pv.index('TGOV1', 'R')
        np.testing.assert_allclose(f.toarray(), full_f[:, cols].toarray())
        np.testing.assert_allclose(g.toarray(), full_g[:, cols].toarray())

    def test_refreeze_after_reinit(self):
        """After new parameters, ``freeze()`` updates the ConstService derivatives."""
        ss, pv = self.ss, self.pv
        j = pv.index('TGOV1', 'R', 1)
        p0 = pv.get()
        try:
            p = p0.copy()
            p[j] *= 2
            pv.set(p)
            ss.TDS.reinit(keep_params=True)
            pj = ParamJacobian(ss, pv)
            dfj, dgj = finite_difference(ss, pv, j)
            fp, gp = pj.evaluate()
            np.testing.assert_allclose(gp[:, j].toarray().ravel(), dgj, atol=1e-6)
            np.testing.assert_allclose(fp[:, j].toarray().ravel(), dfj, atol=1e-6)
        finally:
            pv.set(p0)
            ss.TDS.reinit(keep_params=True)

    def test_errors(self):
        ss = andes.load(get_case(self.case), default_config=True, no_output=True)
        with self.assertRaises(RuntimeError):
            ParamJacobian(ss, ParamVector(ss, ['GENROU.M']))
        with self.assertRaises(ValueError):
            ParamJacobian(self.ss, ParamVector(ss, ['GENROU.M']))


class TestParamJacNonfinite(unittest.TestCase):

    def test_zero_saturation(self):
        """Saturation off (``S10 = 0``): the derivative does not exist and is reported."""
        ss = _load('kundur/kundur_full.xlsx')
        pj = ParamJacobian(ss, ParamVector(ss, ['GENROU.S10', 'GENROU.D']))
        fp, gp = pj.evaluate()
        self.assertTrue(np.all(np.isfinite(fp.data)) and np.all(np.isfinite(gp.data)))
        self.assertEqual(pj.nonfinite, [f'GENROU.S10[{i}]' for i in ss.GENROU.idx.v])


class TestParamJacIEEE14(ParamJacCase):
    """Exciters, stabilizers and renewables with limiters and complex services."""
    case = 'ieee14/ieee14_full.xlsx'


if __name__ == '__main__':
    unittest.main()
