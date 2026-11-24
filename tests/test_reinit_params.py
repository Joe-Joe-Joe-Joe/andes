"""
Tests for ``TDS.reinit(keep_params=True)``: re-initialization after
parameter changes, compared against freshly built systems.
"""

import unittest

import numpy as np

import andes
from andes.utils.paths import get_case


def _load(case, changes=(), tf=3.0):
    """Build a system, apply ``(model, param, idx, value)`` changes, init TDS."""
    ss = andes.load(get_case(case), default_config=True, no_output=True)
    for model, param, idx, value in changes:
        ss.__dict__[model].set(param, idx, value)
    ss.PFlow.run()
    ss.TDS.config.tf = tf
    ss.TDS.init()
    return ss


def _run(ss):
    ss.TDS.run(no_summary=True)
    ss.dae.ts.unpack()
    return (np.array(ss.dae.ts.t), np.array(ss.dae.ts.x),
            np.array(ss.dae.ts.y))


class ReinitParamsBase(unittest.TestCase):
    case = 'kundur/kundur_full.xlsx'

    @classmethod
    def setUpClass(cls):
        cls.ss = _load(cls.case)
        cls.base = _run(cls.ss)

    def setUp(self):
        self.ss.TDS.reinit()

    def assert_same(self, a, b, atol=1e-10):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1], rtol=0, atol=atol)
        np.testing.assert_allclose(a[2], b[2], rtol=0, atol=atol)

    def assert_matches_fresh(self, changes):
        """In-place re-init gives the trajectory of a freshly built system."""
        ss = self.ss
        old = [(m, p, idx, ss.__dict__[m].get(p, idx)) for m, p, idx, _ in changes]
        try:
            for model, param, idx, value in changes:
                ss.__dict__[model].set(param, idx, value)
            ss.TDS.reinit(keep_params=True)
            self.assertTrue(ss.TDS.test_ok)
            got = _run(ss)
            self.switch_times = list(ss.switch_dict.keys())
            self.assertEqual(ss.exit_code, 0)
        finally:
            ss.TDS.reinit()
            for model, param, idx, value in old:
                ss.__dict__[model].set(param, idx, value)
            ss.TDS.reinit(keep_params=True)

        ref = _run(_load(self.case, changes))
        self.assert_same(ref, got)

        # the change must be visible, otherwise the test proves nothing
        self.assertGreater(np.max(np.abs(ref[1] - self.base[1])), 1e-4)
        return got


class TestReinitParamsKundur(ReinitParamsBase):

    def test_no_change_is_identity(self):
        self.ss.TDS.reinit(keep_params=True)
        self.assert_same(self.base, _run(self.ss), atol=1e-12)

    def test_governor_droop(self):
        """``R`` enters through ConstServices and the ``pref`` initial value."""
        self.assert_matches_fresh([('TGOV1', 'R', 1, 0.1)])

    def test_inertia_time_constant(self):
        """``M`` enters only through ``t_const`` (the mass matrix)."""
        self.assert_matches_fresh([('GENROU', 'M', 2, 20.0)])

    def test_several_models(self):
        self.assert_matches_fresh([('GENROU', 'D', 1, 2.0),
                                   ('TGOV1', 'T1', 3, 0.3),
                                   ('EXDC2', 'KA', 2, 30.0)])

    def test_new_values_become_baseline(self):
        """A plain ``reinit()`` afterwards keeps the new parameter values."""
        ss = self.ss
        r0 = ss.TGOV1.get('R', 1)
        ss.TGOV1.set('R', 1, 0.1)
        ss.TDS.reinit(keep_params=True)
        first = _run(ss)
        ss.TDS.reinit()
        self.assertEqual(ss.TGOV1.get('R', 1), 0.1)
        second = _run(ss)
        self.assert_same(first, second, atol=1e-12)

        ss.TDS.reinit()
        ss.TGOV1.set('R', 1, r0)
        ss.TDS.reinit(keep_params=True)
        self.assert_same(self.base, _run(ss), atol=1e-12)

    def test_pflow_param_change_is_ignored(self):
        ss = self.ss
        p0 = ss.PQ.get('p0', 'PQ_0')
        ss.PQ.set('p0', 'PQ_0', p0 + 1.0)
        with self.assertLogs('andes.routines.tds', level='WARNING') as cm:
            ss.TDS.reinit(keep_params=True)
        self.assertIn('PQ.p0', '\n'.join(cm.output))
        self.assertEqual(ss.PQ.get('p0', 'PQ_0'), p0)
        self.assert_same(self.base, _run(ss), atol=1e-12)


class TestReinitParamsEvents(ReinitParamsBase):
    """Changed event times are picked up."""

    case = 'ieee14/ieee14_fault.xlsx'

    def test_fault_clearing_time(self):
        got = self.assert_matches_fresh([('Fault', 'tc', 1, 1.15)])
        self.assertIn(1.15, list(got[0]))
        self.assertIn(1.15, self.switch_times)
        self.assertNotIn(1.1, self.switch_times)

        # the original clearing time is back after restoring the parameter
        self.assertIn(1.1, list(self.ss.switch_dict.keys()))
        self.assertNotIn(1.15, list(self.ss.switch_dict.keys()))


class TestReinitParamsStatus(ReinitParamsBase):
    """Device status changed by events during a run is restored."""

    case = 'ieee14/ieee14_gentrip.xlsx'

    def test_tripped_generator_restored(self):
        """Re-initializing ``Toggle`` must not change any device status."""
        ss = self.ss
        _run(ss)
        self.assertIn(0.0, ss.GENROU.u.v)
        ss.TDS.reinit()
        ss.Toggle.set('t', 'Toggler_1', 1.5)
        ss.TDS.reinit(keep_params=True)
        self.assertTrue(np.all(ss.GENROU.u.v == 1))
        self.assertTrue(np.all(ss.GENROU.ue.v == 1))
        got = _run(ss)
        self.assertIn(0.0, ss.GENROU.u.v)

        ref = _run(_load(self.case, [('Toggle', 't', 'Toggler_1', 1.5)]))
        self.assert_same(ref, got)

        ss.TDS.reinit()
        ss.Toggle.set('t', 'Toggler_1', 1.0)
        ss.TDS.reinit(keep_params=True)

    def test_requires_t0(self):
        ss = self.ss
        _run(ss)
        with self.assertRaises(RuntimeError):
            ss.TDS.reinit(keep_params=True)


class TestReinitParamsToggleOn(ReinitParamsBase):
    """A device that starts offline and is switched on by ``Toggle``."""

    case = 'kundur/kundur_motor.xlsx'

    def test_offline_device_stays_offline(self):
        ss = self.ss
        self.assertEqual(ss.Motor5.u.v[0], 0)
        ss.TDS.reinit(keep_params=True)
        self.assertEqual(ss.Motor5.u.v[0], 0)
        self.assertEqual(ss.Motor5.ue.v[0], 0)
        self.assert_same(self.base, _run(ss), atol=1e-12)


class TestReinitParamsPreInit(unittest.TestCase):

    def test_before_init(self):
        ss = andes.load(get_case('kundur/kundur_full.xlsx'),
                        default_config=True, no_output=True)
        with self.assertRaises(RuntimeError):
            ss.TDS.reinit(keep_params=True)


if __name__ == '__main__':
    unittest.main()
