"""
Tests for ``andes.core.ParamVector``.
"""

import unittest

import numpy as np

import andes
from andes.core import ParamVector
from andes.utils.paths import get_case

CASE = 'kundur/kundur_full.xlsx'


def _load(init=True):
    ss = andes.load(get_case(CASE), default_config=True, no_output=True)
    if init:
        ss.PFlow.run()
        ss.TDS.config.tf = 3.0
        ss.TDS.init()
    return ss


class TestParamVectorLayout(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ss = _load()

    def test_selection_forms(self):
        ss = self.ss
        pv = ParamVector(ss, ['GENROU.M', ('TGOV1', 'R'), ('EXDC2', 'KA', [3, 1]),
                              ('Line', 'x', 'Line_2')])
        self.assertEqual(len(pv), ss.GENROU.n + ss.TGOV1.n + 2 + 1)
        self.assertEqual(pv.names[0], 'GENROU.M[1]')
        self.assertEqual(pv.names[-3:], ['EXDC2.KA[3]', 'EXDC2.KA[1]', 'Line.x[Line_2]'])
        self.assertEqual(pv.model[-1], 'Line')
        self.assertEqual(pv.idx[-3:], [3, 1, 'Line_2'])

    def test_pflow_flags(self):
        pv = ParamVector(self.ss, ['GENROU.M', 'Line.x'])
        np.testing.assert_array_equal(pv.pflow, [False] * 4 + [True] * self.ss.Line.n)

    def test_get_reads_model_arrays(self):
        ss = self.ss
        pv = ParamVector(ss, ['GENROU.M', ('EXDC2', 'KA', [3, 1])])
        p = pv.get()
        np.testing.assert_array_equal(p[:4], ss.GENROU.M.v)
        self.assertEqual(p[4], ss.EXDC2.get('KA', 3))
        self.assertEqual(p[5], ss.EXDC2.get('KA', 1))

        # a copy, not a view
        p[:] = 0
        self.assertNotEqual(ss.GENROU.M.v[0], 0)

    def test_device_base(self):
        ss = self.ss
        pv = ParamVector(ss, ['GENROU.M', 'TGOV1.R'], base='device')
        np.testing.assert_allclose(pv.get()[:4], ss.GENROU.M.vin)
        np.testing.assert_allclose(pv.get()[4:], ss.TGOV1.R.vin)
        # system base differs for these parameters
        self.assertFalse(np.allclose(ss.GENROU.M.v, ss.GENROU.M.vin))

    def test_index(self):
        pv = ParamVector(self.ss, ['GENROU.M', ('TGOV1', 'R', [4, 2])])
        np.testing.assert_array_equal(pv.index('GENROU', 'M'), [0, 1, 2, 3])
        self.assertEqual(pv.index('GENROU', 'M', 3), 2)
        self.assertEqual(pv.index('TGOV1', 'R', 2), 5)
        np.testing.assert_array_equal(pv.index('TGOV1', 'R', [2, 4]), [5, 4])
        with self.assertRaises(KeyError):
            pv.index('TGOV1', 'R', 1)
        with self.assertRaises(KeyError):
            pv.index('TGOV1', 'T1')

    def test_as_df(self):
        pv = ParamVector(self.ss, ['GENROU.M', 'Line.x'])
        df = pv.as_df()
        self.assertEqual(list(df.index), pv.names)
        np.testing.assert_array_equal(df['value'].values, pv.get())
        self.assertEqual(list(df.columns),
                         ['model', 'param', 'idx', 'value', 'unit', 'pflow', 'info'])

    def test_invalid_selections(self):
        ss = self.ss
        bad = {
            'unknown model': ['NOPE.M'],
            'unknown param': ['GENROU.nope'],
            'not a NumParam': ['GENROU.bus'],
            'ExtParam': ['TGOV1.Sg'],
            'status': ['GENROU.u'],
            'malformed': ['GENROU'],
            'twice': ['GENROU.M', ('GENROU', 'M', 1)],
            'duplicate idx': [('GENROU', 'M', [1, 1])],
            'missing idx': [('GENROU', 'M', 99)],
        }
        for label, select in bad.items():
            with self.subTest(label), self.assertRaises((KeyError, ValueError)):
                ParamVector(ss, select)

        with self.assertRaises(ValueError):
            ParamVector(ss, ['GENROU.M'], base='system')

    def test_requires_setup(self):
        ss = andes.load(get_case(CASE), setup=False, default_config=True, no_output=True)
        with self.assertRaises(RuntimeError):
            ParamVector(ss, ['GENROU.M'])


class TestParamVectorSet(unittest.TestCase):

    def setUp(self):
        self.ss = _load()

    def test_set_writes_in_place(self):
        ss = self.ss
        arr = ss.GENROU.D.v
        pv = ParamVector(ss, ['GENROU.D', ('TGOV1', 'R', 2)])
        pv.set(np.arange(1.0, 6.0))
        self.assertIs(ss.GENROU.D.v, arr)    # generated code keeps reading this array
        np.testing.assert_array_equal(ss.GENROU.D.v, [1, 2, 3, 4])
        self.assertEqual(ss.TGOV1.get('R', 2), 5.0)
        np.testing.assert_array_equal(pv.get(), np.arange(1.0, 6.0))

    def test_set_updates_time_constants(self):
        ss = self.ss
        pv = ParamVector(ss, ['GENROU.M'])
        pv.set([10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(ss.dae.Tf[ss.GENROU.omega.a], [10, 20, 30, 40])
        teye = [ss.TDS.Teye[int(a), int(a)] for a in ss.GENROU.omega.a]
        np.testing.assert_array_equal(teye, [10, 20, 30, 40])

    def test_set_device_base(self):
        ss = self.ss
        pv = ParamVector(ss, [('GENROU', 'M', 2)], base='device')
        pv.set([7.0])
        self.assertEqual(ss.GENROU.M.vin[1], 7.0)
        self.assertAlmostEqual(ss.GENROU.M.v[1], 7.0 * ss.GENROU.M.pu_coeff[1])
        self.assertEqual(pv.get()[0], 7.0)

    def test_set_wrong_length(self):
        pv = ParamVector(self.ss, ['GENROU.M'])
        with self.assertRaises(ValueError):
            pv.set([1.0, 2.0])

    def test_set_and_reinit_matches_model_set(self):
        """``set`` + ``reinit(keep_params=True)`` gives the same run as ``Model.set``."""
        ss = self.ss
        pv = ParamVector(ss, ['GENROU.M', 'TGOV1.R', ('EXDC2', 'KA', [1, 3])])
        new = pv.get() * np.linspace(0.8, 1.2, len(pv))
        pv.set(new)
        ss.TDS.reinit(keep_params=True)
        self.assertTrue(ss.TDS.test_ok)
        ss.TDS.run(no_summary=True)
        ss.dae.ts.unpack()
        got = np.hstack([ss.dae.ts.x, ss.dae.ts.y])

        ref = _load(init=False)
        for name, value in zip(pv.names, new):
            ref.__dict__[pv.model[pv.names.index(name)]].set(
                pv.param[pv.names.index(name)], pv.idx[pv.names.index(name)], value)
        ref.PFlow.run()
        ref.TDS.config.tf = 3.0
        ref.TDS.init()
        ref.TDS.run(no_summary=True)
        ref.dae.ts.unpack()
        np.testing.assert_allclose(got, np.hstack([ref.dae.ts.x, ref.dae.ts.y]),
                                   rtol=0, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
