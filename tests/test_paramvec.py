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


class TestParamVectorAll(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ss = _load()

    def test_all_contents(self):
        ss = self.ss
        pv = ParamVector.all(ss)
        params = set(zip(pv.model, pv.param))

        expected = 0
        for mname, mdl in ss.models.items():
            if mdl.n == 0:
                continue
            for pname, p in mdl.num_params.items():
                if ParamVector._unusable(mdl, p) is None:
                    expected += mdl.n
                    self.assertIn((mname, pname), params)
        self.assertEqual(len(pv), expected)

        self.assertIn(('GENROU', 'M'), params)
        self.assertIn(('Line', 'x'), params)
        self.assertIn(('Toggle', 't'), params)
        for left_out in [('GENROU', 'u'), ('GENROU', 'ue'), ('TGOV1', 'Sg')]:
            self.assertNotIn(left_out, params)

        np.testing.assert_array_equal(pv.get()[pv.index('GENROU', 'M')], ss.GENROU.M.v)

    def test_all_without_pflow(self):
        pv = ParamVector.all(self.ss, pflow=False)
        self.assertFalse(pv.pflow.any())
        self.assertNotIn('Line', pv.model)
        self.assertIn('GENROU', pv.model)
        self.assertEqual(len(pv), int(np.sum(~ParamVector.all(self.ss).pflow)))

    def test_all_models_filter(self):
        pv = ParamVector.all(self.ss, models=['TGOV1', 'GENROU'])
        self.assertEqual(set(pv.model), {'TGOV1', 'GENROU'})
        self.assertEqual(pv.model[0], 'GENROU')   # System.models order, not argument order
        with self.assertRaises(KeyError):
            ParamVector.all(self.ss, models=['NOPE'])

    def test_subset(self):
        pv = ParamVector.all(self.ss)
        dyn = pv.subset(~pv.pflow)
        self.assertEqual(dyn.names, [n for n, f in zip(pv.names, pv.pflow) if not f])
        np.testing.assert_array_equal(dyn.get(), pv.get()[~pv.pflow])

        pos = [pv.index('TGOV1', 'R', 3), pv.index('GENROU', 'M', 2), pv.index('TGOV1', 'R', 1)]
        sub = pv.subset(pos)
        self.assertEqual(sub.names, ['TGOV1.R[3]', 'TGOV1.R[1]', 'GENROU.M[2]'])
        self.assertEqual(len(pv.subset(np.zeros(len(pv), dtype=bool))), 0)
        with self.assertRaises(ValueError):
            pv.subset([True, False])
        with self.assertRaises(IndexError):
            pv.subset([len(pv)])


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

    def test_all_round_trip(self):
        """Writing back every value of the whole vector changes nothing."""
        ss = self.ss
        ss.TDS.run(no_summary=True)
        ss.dae.ts.unpack()
        base = np.hstack([ss.dae.ts.x, ss.dae.ts.y])
        ss.TDS.reinit()

        pv = ParamVector.all(ss, pflow=False)
        before = {(m, p): ss.__dict__[m].__dict__[p].v.copy() for m, p in set(zip(pv.model, pv.param))}
        pv.set(pv.get())
        for (m, p), v in before.items():
            np.testing.assert_array_equal(ss.__dict__[m].__dict__[p].v, v)

        ss.TDS.reinit(keep_params=True)
        ss.TDS.run(no_summary=True)
        ss.dae.ts.unpack()
        np.testing.assert_allclose(np.hstack([ss.dae.ts.x, ss.dae.ts.y]), base, rtol=0, atol=1e-12)

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
