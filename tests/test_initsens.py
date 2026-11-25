"""
Tests for ``andes.routines.initsens.InitSensitivity``, checked against
central differences of ``TDS.reinit(keep_params=True)``.
"""

import unittest

import numpy as np

import andes
from andes.core import ParamVector
from andes.core.param import TimerParam
from andes.routines.initsens import InitSensitivity
from andes.routines.sensitivity import TrajectorySensitivity
from andes.utils.paths import get_case


def _load(case):
    ss = andes.load(get_case(case), default_config=True, no_output=True)
    ss.TDS.config.tol = 1e-8
    ss.PFlow.run()
    ss.TDS.init()
    return ss


def _dynamic_params(ss):
    pv = ParamVector.all(ss, pflow=False)
    keep = [i for i in range(len(pv))
            if not isinstance(pv._blocks[(pv.model[i], pv.param[i])][1], TimerParam)]
    return pv.subset(keep)


def _fd(ss, pv):
    """The finite-difference initial sensitivity of ``TrajectorySensitivity``."""
    fd = TrajectorySensitivity.__new__(TrajectorySensitivity)
    fd.system, fd.pv, fd.rel_step = ss, pv, 1e-6
    fd._timer = np.zeros(len(pv), dtype=bool)
    S0, totals = fd._initial_sensitivity()
    return S0, totals, set(fd.inconsistent)


class InitSensCase(unittest.TestCase):
    case = None

    @classmethod
    def setUpClass(cls):
        if cls.case is None:
            raise unittest.SkipTest('base class')
        cls.ss = _load(cls.case)
        cls.pv = _dynamic_params(cls.ss)
        cls.ex = InitSensitivity(cls.ss, cls.pv)
        cls.S0_fd, cls.totals_fd, skip = _fd(cls.ss, cls.pv)
        skip |= set(cls.ex.nonfinite)       # one-sided or undefined derivatives
        cls.cols = [j for j, name in enumerate(cls.pv.names) if name not in skip]

    def test_matches_finite_differences(self):
        self.assertEqual(self.ex.unsupported, [])
        a, b = self.ex.S0[:, self.cols], self.S0_fd[:, self.cols]
        scale = np.maximum(1.0, np.abs(b).max(axis=0))
        np.testing.assert_array_less(np.abs(a - b).max(axis=0) / scale, 1e-5)
        self.assertGreater(np.count_nonzero(np.abs(b).max(axis=0) > 1e-6), 10)

    def test_constants(self):
        for mname, d in self.totals_fd.items():
            for name, b in d.items():
                if type(self.ss.__dict__[mname].__dict__[name]).__name__ == 'InitChecker':
                    continue          # range checks, not used in equations
                a = self.ex.const_totals.get(mname, {}).get(name, np.zeros_like(b))
                a, b = np.asarray(a)[:, self.cols], b[:, self.cols]
                with self.subTest(f'{mname}.{name}'):
                    # Fault.gf/bf: the finite difference itself is 1e-4 off at xf = 1e-4
                    np.testing.assert_allclose(a, b, rtol=0, atol=2e-4 * max(1.0, np.abs(b).max()))


class TestInitSensKundur(InitSensCase):
    case = 'kundur/kundur_full.xlsx'

    def test_faster_than_finite_differences(self):
        import time

        t0 = time.perf_counter()
        InitSensitivity(self.ss, self.pv)
        t_exact = time.perf_counter() - t0
        t0 = time.perf_counter()
        _fd(self.ss, self.pv)
        self.assertLess(t_exact, time.perf_counter() - t0)


class TestInitSensIterative(InitSensCase):
    """ESST1A initializes ``vref`` and ``vi`` by Newton iterations."""
    case = 'ieee14/ieee14_esst1a.xlsx'


class TestInitSensIEEE39(InitSensCase):
    """IEEEST reads a per-unit coefficient of ``GENROU.M`` through ``ExtService(attr=...)``."""
    case = 'ieee39/ieee39_full.xlsx'


class TestInitSensRenewable(InitSensCase):
    """REECA1 has a service computed by a Python function (``v_numeric``)."""
    case = 'ieee14/ieee14_wt3.xlsx'


class TestInitSensFallback(unittest.TestCase):

    def test_coi_falls_back(self):
        ss = _load('kundur/kundur_coi.xlsx')
        pv = ParamVector(ss, ['GENROU.M', 'TGOV1.R'])
        with self.assertRaises(NotImplementedError):
            InitSensitivity(ss, pv)
        with self.assertLogs('andes.routines.sensitivity', level='WARNING'):
            sens = TrajectorySensitivity(ss, pv)
        S0_fd, _, _ = _fd(ss, pv)
        np.testing.assert_allclose(sens.S0, S0_fd)


if __name__ == '__main__':
    unittest.main()
