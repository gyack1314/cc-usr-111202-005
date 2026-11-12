"""资料契约拒绝路径与有限数值口径的测试。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from research_data.io import load
from research_data.validate import validate_material
from research_data.fit import fit_material


def base_material():
    return load('data/normal.json')[0]


class ValidationRejectionTests(unittest.TestCase):
    def _reject(self, mutate):
        material = copy.deepcopy(base_material())
        mutate(material)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            validate_material(material)

    def test_units_must_be_exact_declared_triple(self):
        self._reject(lambda m: m.update(units={'time': 'hour', 'dose': 'mg'}))
        self._reject(lambda m: m.update(units={
            'time': 'h', 'dose': 'mg', 'concentration': 'mg/L'}))

    def test_non_finite_inputs_rejected(self):
        self._reject(lambda m: m['infusions'][0].update(start=float('nan')))
        self._reject(lambda m: m['samples'][0].update(sigma=float('inf')))
        self._reject(lambda m: m['rate_pairs'][0].__setitem__(0, float('-inf')))

    def test_negative_quantified_value_rejected(self):
        self._reject(lambda m: m['samples'][0].update(value=-0.01))

    def test_unknown_observation_kind_rejected(self):
        self._reject(lambda m: m['samples'][0].update(kind='estimated'))

    def test_below_limit_requires_positive_limit(self):
        self._reject(lambda m: next(
            s for s in m['samples'] if s['kind'] == 'below_limit').update(limit=0.0))

    def test_rate_pair_ordering(self):
        self._reject(lambda m: m.update(rate_pairs=[[0.1, 0.4]]))   # alpha<beta
        self._reject(lambda m: m.update(rate_pairs=[[0.4, 0.0]]))   # beta=0
        self._reject(lambda m: m.update(rate_pairs=[[0.4, -0.1]]))  # beta<0
        self._reject(lambda m: m.update(rate_pairs=[]))

    def test_amplitude_rectangle(self):
        self._reject(lambda m: m['amplitude_bounds'].update(A=[-1.0, 1.0]))
        self._reject(lambda m: m['amplitude_bounds'].update(B=[0.5, 0.5]))
        self._reject(lambda m: m['amplitude_bounds'].update(A=[1.0, 0.0]))

    def test_infusion_and_window_geometry(self):
        self._reject(lambda m: m['infusions'][0].update(end=0.0))
        self._reject(lambda m: m['infusions'][0].update(start=-1.0))
        self._reject(lambda m: m.update(auc_windows=[[10.0, 5.0]]))
        self._reject(lambda m: m.update(auc_windows=[]))

    def test_solver_config(self):
        self._reject(lambda m: m['solver'].update(tolerance=0.0))
        self._reject(lambda m: m['solver'].update(max_iterations=0))
        self._reject(lambda m: m['solver'].update(max_iterations=1.5))

    def test_duplicate_identities_rejected(self):
        self._reject(lambda m: m['infusions'][1].update(id=m['infusions'][0]['id']))
        self._reject(lambda m: m['samples'][1].update(id=m['samples'][0]['id']))

    def test_non_finite_json_constants_rejected_at_load(self):
        good = base_material()
        text = json.dumps(good)
        with tempfile.TemporaryDirectory() as d:
            for constant in ('NaN', 'Infinity', '-Infinity'):
                p = Path(d) / 'bad.json'
                p.write_text(text.replace('0.4', constant, 1), encoding='utf8')
                with self.assertRaises(ValueError):
                    load(p)

    def test_fit_rejects_invalid_file(self):
        with self.assertRaises(ValueError):
            fit_material(load('data/invalid-domain.json')[0])

    def test_valid_material_accepted(self):
        # 正常资料与两个业务边界资料都能通过校验
        for name in ('normal.json', 'all-below-limit.json',
                     'near-equal-rates.json'):
            self.assertIsNotNone(validate_material(load(f'data/{name}')[0]))


if __name__ == '__main__':
    unittest.main()
