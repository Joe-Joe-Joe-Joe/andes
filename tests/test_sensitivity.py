"""
Tests for ``andes.routines.sensitivity.TrajectorySensitivity``, checked
against central differences of whole trajectories.
"""

import unittest

import numpy as np

import andes
from andes.core import ParamVector
from andes.routines.sensitivity import TrajectorySensitivity
from andes.utils.paths import get_case


def _load(case, tf=3.0, tol=1e-8, honest=0):
    ss = andes.load(get_case(case), default_config=True, no_output=True)
    c = ss.TDS.config
    c.tf, c.fixt, c.shrinkt, c.tstep, c.tol, c.honest = tf, 1, 0, 0.01, tol, honest
    ss.PFlow.run()
    ss.TDS.init()
    return ss


def _trajectory(ss):
    ss.TDS.run(no_summary=True)
    assert not ss.TDS.busted and ss.dae.t >= ss.TDS.config.tf - 1e-9, ss.TDS.err_msg
    ss.dae.ts.unpack()
    return np.array(ss.dae.ts.t), np.hstack([ss.dae.ts.x, ss.dae.ts.y])


def _fd_trajectory(ss, pv, j, rel=1e-4):
    """Central difference of the whole trajectory with respect to entry ``j``."""
    p0 = pv.get()
    h = rel * max(abs(p0[j]), 1e-2)
    out = []
    for sign in (1, -1):
        p = p0.copy()
        p[j] += sign * h
        ss.TDS.reinit()
        pv.set(p)
        ss.TDS.reinit(keep_params=True)
        out.append(_trajectory(ss)[1])
    ss.TDS.reinit()
    pv.set(p0)
    ss.TDS.reinit(keep_params=True)
    return (out[0] - out[1]) / (2 * h)


class SensitivityCase(unittest.TestCase):
    case = None
    select = None
    load_kwargs = {}
    atol = 2e-3          # relative to the largest sensitivity of each parameter

    @classmethod
    def setUpClass(cls):
        if cls.case is None:
            raise unittest.SkipTest('base class')
        cls.ss = _load(cls.case, **cls.load_kwargs)
        cls.pv = ParamVector(cls.ss, cls.select)
        cls.sens = TrajectorySensitivity(cls.ss, cls.pv)
        cls.sens.run(no_summary=True)
        assert not cls.ss.TDS.busted, cls.ss.TDS.err_msg
        cls.ss.dae.ts.unpack()
        cls.t = np.array(cls.ss.dae.ts.t)
        cls.xy = np.hstack([cls.ss.dae.ts.x, cls.ss.dae.ts.y])

    def test_matches_finite_differences(self):
        ss, pv, sens = self.ss, self.pv, self.sens
        self.assertEqual(sens.pj.unsupported, [])
        idx = np.searchsorted(sens.t, self.t)
        np.testing.assert_allclose(sens.t[idx], self.t)
        S = np.concatenate([sens.Sx, sens.Sy], axis=1)[idx]

        for j in range(len(pv)):
            fd = _fd_trajectory(ss, pv, j)
            scale = max(1e-3, np.abs(fd).max())
            with self.subTest(pv.names[j]):
                self.assertGreater(np.abs(fd).max(), 0)
                np.testing.assert_allclose(S[:, :, j], fd, rtol=0, atol=self.atol * scale)

    def test_trajectory_unchanged(self):
        """Computing sensitivities does not change the simulation."""
        self.ss.TDS.reinit()
        t, xy = _trajectory(self.ss)
        np.testing.assert_array_equal(t, self.t)
        np.testing.assert_allclose(xy, self.xy, rtol=0, atol=1e-9)


class TestSensitivityKundur(SensitivityCase):
    """Line trip at t = 2 s."""
    case = 'kundur/kundur_full.xlsx'
    select = [('GENROU', 'M', [1, 3]), ('GENROU', 'D', 2), ('GENROU', 'xd1', 4),
              ('TGOV1', 'R', 1), ('TGOV1', 'T1', 2), ('EXDC2', 'KA', 3)]

    def test_get(self):
        s = self.sens.get(self.ss.GENROU.omega)
        self.assertEqual(s.shape, (len(self.sens.t), self.ss.GENROU.n, len(self.pv)))
        np.testing.assert_array_equal(s[:, 0, :], self.sens.Sx[:, self.ss.GENROU.omega.a[0], :])

    def test_initial_sensitivity(self):
        """``R`` sets the initial ``pref``; ``M`` changes no initial value."""
        S0 = self.sens.S0
        n = self.ss.dae.n
        j_r = self.pv.index('TGOV1', 'R', 1)
        self.assertGreater(abs(S0[n + self.ss.TGOV1.pref.a[0], j_r]), 1)
        np.testing.assert_allclose(S0[:, self.pv.index('GENROU', 'M', 1)], 0, atol=1e-8)


class TestSensitivityFault(SensitivityCase):
    """Bus fault applied at t = 1 s and cleared at t = 1.1 s."""
    case = 'ieee14/ieee14_fault.xlsx'
    load_kwargs = {'tol': 1e-7, 'honest': 1}     # 1e-8 is not reachable during the fault
    # Governors reach their anti-windup limits after the fault. The perturbed
    # runs of the finite difference reach them at different steps, so the
    # difference averages two branches in those few steps.
    atol = 1e-2
    select = [('GENROU', 'M', 'GENROU_2'), ('GENROU', 'D', 'GENROU_3'),
              ('TGOV1', 'R', 'TGOV1_1'), ('Fault', 'xf', 1)]


class TestSensitivityErrors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ss = _load('ieee14/ieee14_fault.xlsx', tf=0.1)

    def test_rejects(self):
        ss = self.ss
        with self.assertRaises(ValueError):
            TrajectorySensitivity(ss, ParamVector(ss, ['Line.x']))
        with self.assertRaises(ValueError):
            TrajectorySensitivity(ss, ParamVector(ss, ['Fault.tc']))

        ss.TDS.run(no_summary=True)
        with self.assertRaises(RuntimeError):
            TrajectorySensitivity(ss, ParamVector(ss, ['GENROU.M']))
        ss.TDS.reinit()

    def test_method(self):
        ss = _load('ieee14/ieee14_fault.xlsx', tf=0.1)
        ss.TDS.config.method = 'backeuler'
        with self.assertRaises(ValueError):
            TrajectorySensitivity(ss, ParamVector(ss, ['GENROU.M']))


if __name__ == '__main__':
    unittest.main()
