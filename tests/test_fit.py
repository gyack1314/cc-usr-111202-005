"""删失似然、界约束优化与逐候选拟合编排的测试。"""
import math
import unittest

from research_data.io import load
from research_data.likelihood import (
    log_normal_cdf, mills_ratio, build_observations,
    objective_gradient_hessian, quantified_nll, censored_nll)
from research_data.convolution import design_rows
from research_data.optimize import minimize_boxed
from research_data.diagnostics import design_condition, identifiability
from research_data.fit import fit_material
from research_data.validate import validate_material


def _material(name):
    return load(f'data/{name}')[0]


class CensoringLikelihoodTests(unittest.TestCase):
    def test_log_cdf_matches_erf(self):
        for z in (-6.0, -3.0, -0.0436, 0.0, 0.7, 4.0):
            ref = math.log(0.5 * math.erfc(-z / math.sqrt(2.0)))
            self.assertAlmostEqual(log_normal_cdf(z), ref, places=12)

    def test_deep_tail_finite_and_monotone(self):
        from research_data.likelihood import censor_probability
        prev = math.inf
        for z in (-5.0, -10.0, -20.0, -40.0, -70.0):
            value = log_normal_cdf(z)
            self.assertTrue(math.isfinite(value))
            r = mills_ratio(z)
            self.assertTrue(math.isfinite(r) and r > 0.0)
            self.assertLess(value, prev)
            prev = value
        # 概率即使深尾下溢也夹到最小正浮点，不返回 0
        p = censor_probability(0.3, 0.2, 100.0)
        self.assertGreater(p, 0.0)
        self.assertTrue(math.isfinite(p))

    def test_mills_ratio_positive_and_curvature(self):
        # r(z)>0；删失曲率 r(z)(r(z)+z) 恒非负（r'(z)=r(z-r) 差分验证）
        for z in (-20.0, -3.0, 0.0, 2.0, 10.0):
            r = mills_ratio(z)
            self.assertGreater(r, 0.0)
            self.assertGreaterEqual(r * (r + z), 0.0)

    def test_censored_not_treated_as_half_or_zero(self):
        # 删失 nll 必须随预测浓度上升而增大（高浓度违背 Y<L）
        nll_low = censored_nll(0.3, 0.2, 0.05)
        nll_high = censored_nll(0.3, 0.2, 2.0)
        self.assertGreater(nll_high, nll_low)
        # 与量化观测的点似然不同，且不是把 limit/2 当观测
        q = quantified_nll(0.15, 0.2, 0.15)
        self.assertNotAlmostEqual(censored_nll(0.3, 0.2, 0.15), q)

    def test_gradient_convexity(self):
        v = _material('normal.json')
        obs = build_observations(v['samples'])
        rows, _ = design_rows(v['infusions'],
                              [float(s['time']) for s in v['samples']], 0.4, 0.03)
        for x in ((0.0, 0.0), (0.05, 0.02), (1.0, 1.0), (0.0, 1.0)):
            _, grad, hess = objective_gradient_hessian(x, rows, obs)
            h00, h01, h11 = hess
            self.assertGreaterEqual(h00, 0.0)
            self.assertGreaterEqual(h11, 0.0)
            self.assertGreaterEqual(h00 * h11 - h01 * h01, -1e-12)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        v = _material('normal.json')
        self.obs = build_observations(v['samples'])
        self.rows, _ = design_rows(
            v['infusions'], [float(s['time']) for s in v['samples']], 0.4, 0.03)

    def test_recovers_optimum_from_multiple_starts(self):
        lower, upper = (0.0, 0.0), (1.0, 1.0)
        starts = ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.9, 0.1))
        opt = minimize_boxed(self.rows, self.obs, lower, upper,
                             tol=1e-10, max_iterations=10000, starts=starts)
        self.assertEqual(opt['termination'], 'converged')
        self.assertLess(opt['optimality_residual'], 1e-10)
        # 网格扫描独立核对：凸问题的全局最小
        best_grid = min(
            objective_gradient_hessian((a / 100.0, b / 100.0), self.rows, self.obs)[0]
            for a in range(101) for b in range(101))
        self.assertLessEqual(opt['objective'], best_grid + 1e-3)

    def test_active_bound_reported(self):
        # 极窄上界迫使解贴边，边界必须被报告
        opt = minimize_boxed(self.rows, self.obs, (0.0, 0.0), (0.001, 0.001),
                             tol=1e-10, max_iterations=10000)
        self.assertEqual(opt['termination'], 'converged')
        self.assertTrue(opt['active_bounds'])


class DiagnosticsTests(unittest.TestCase):
    def test_all_censored_unidentifiable(self):
        v = _material('all-below-limit.json')
        obs = build_observations(v['samples'])
        rows, _ = design_rows(v['infusions'],
                              [float(s['time']) for s in v['samples']], 0.4, 0.03)
        status, _ = identifiability(rows, obs)
        self.assertEqual(status, 'all_censored')

    def test_near_collinear_rates_flagged(self):
        v = _material('near-equal-rates.json')
        obs = build_observations(v['samples'])
        rows, _ = design_rows(v['infusions'],
                              [float(s['time']) for s in v['samples']],
                              0.10000001, 0.1)
        status, diag = identifiability(rows, obs)
        self.assertEqual(status, 'near_collinear')
        self.assertGreater(diag['condition_number'], 1e8)

    def test_healthy_design_identifiable(self):
        v = _material('normal.json')
        obs = build_observations(v['samples'])
        rows, _ = design_rows(v['infusions'],
                              [float(s['time']) for s in v['samples']], 0.4, 0.03)
        status, diag = identifiability(rows, obs)
        self.assertEqual(status, 'identifiable')
        self.assertLess(diag['condition_number'], 1e4)


class FitOrchestrationTests(unittest.TestCase):
    def test_normal_fit_report_contract(self):
        report = fit_material(_material('normal.json'))
        self.assertEqual(report['global_status'], 'finite_domain_optimum')
        self.assertEqual(report['winner_index'], 0)
        self.assertEqual(len(report['candidates']), 3)
        for cand in report['candidates']:
            self.assertTrue(cand['converged'])
            self.assertIn(cand['termination'],
                          ('converged', 'max_iterations', 'line_search_failed'))
            self.assertIsNotNone(cand['objective'])
        # 决胜：目标严格非降，并列按编号
        objs = [(c['objective'], c['candidate_index']) for c in report['candidates']]
        self.assertEqual(objs, sorted(objs))
        # 目标与逐点 nll 之和一致
        self.assertLess(report['objective_check']['absolute_difference'], 1e-12)
        # 逐输注贡献分解覆盖全部输注，且和等于预测
        for sample in report['samples']:
            self.assertEqual(set(sample['contributions']),
                             {'dose1', 'dose2', 'overlap'})
            self.assertAlmostEqual(
                sum(sample['contributions'].values()), sample['predicted'], places=12)
        # 三个 AUC 窗，逐输注键完整
        self.assertEqual(len(report['auc']), 3)
        for block in report['auc']:
            self.assertEqual(set(block['per_infusion']),
                             {'dose1', 'dose2', 'overlap'})
        # 近优候选及其预测差异全部保留
        self.assertTrue(report['near_optimal_indices'])
        for item in report['near_optimal_predictions']:
            self.assertEqual(len(item['predicted_concentration']),
                             len(report['samples']))
            self.assertGreaterEqual(item['objective_gap'], -1e-12)

    def test_all_censored_does_not_announce_optimum(self):
        report = fit_material(_material('all-below-limit.json'))
        self.assertEqual(report['global_status'], 'no_converged_candidate')
        self.assertIsNone(report['winner_index'])
        self.assertEqual(len(report['unconverged_candidates']), 3)
        self.assertNotIn('samples', report)

    def test_near_collinear_does_not_announce_optimum(self):
        report = fit_material(_material('near-equal-rates.json'))
        self.assertEqual(report['global_status'], 'no_converged_candidate')
        self.assertEqual(
            report['unconverged_candidates'][0]['termination'], 'unidentifiable')

    def test_unconverged_candidates_are_not_silently_skipped(self):
        import copy
        material = copy.deepcopy(_material('normal.json'))
        material['solver']['max_iterations'] = 1
        report = fit_material(material)
        # 没有候选收敛时不得宣布全域最优，且每个候选的终止原因都保留
        self.assertEqual(report['global_status'], 'no_converged_candidate')
        self.assertIsNone(report['winner_index'])
        self.assertEqual(len(report['unconverged_candidates']), 3)
        for item in report['unconverged_candidates']:
            self.assertEqual(item['termination'], 'max_iterations')
        self.assertNotIn('samples', report)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            fit_material(_material('invalid-domain.json'))

    def test_objective_tie_broken_by_candidate_index(self):
        import copy
        # 两个完全相同的速率对 -> 目标必然并列，编号小者胜出
        material = copy.deepcopy(_material('normal.json'))
        material['rate_pairs'] = [[0.4, 0.03], [0.4, 0.03]]
        report = fit_material(material)
        self.assertEqual(report['winner_index'], 0)
        self.assertEqual(len(report['candidates']), 2)
        gap = abs(report['candidates'][0]['objective']
                  - report['candidates'][1]['objective'])
        self.assertLess(gap, 1e-10)

    def test_reject_undeclared_units_and_bad_values(self):
        good = _material('normal.json')
        bad_units = dict(good, units={'time': 'hour', 'dose': 'mg'})
        with self.assertRaises(ValueError):
            validate_material(bad_units)
        bad_quant = {**good, 'samples': [
            dict(s, value=-1.0) if s['kind'] == 'quantified' else s
            for s in good['samples']]}
        with self.assertRaises(ValueError):
            validate_material(bad_quant)
        bad_pairs = {**good, 'rate_pairs': [[0.1, 0.2]]}  # alpha<beta
        with self.assertRaises(ValueError):
            validate_material(bad_pairs)
        bad_bounds = {**good, 'amplitude_bounds': {'A': [0, -1], 'B': [0, 1]}}
        with self.assertRaises(ValueError):
            validate_material(bad_bounds)

    def test_non_finite_json_constants_rejected_at_load(self):
        import json
        import tempfile
        from pathlib import Path
        good = _material('normal.json')
        with tempfile.TemporaryDirectory() as d:
            for payload in ('NaN', 'Infinity', '-Infinity'):
                text = json.dumps(good).replace('0.4', payload, 1)
                p = Path(d) / 'bad.json'
                p.write_text(text, encoding='utf8')
                with self.assertRaises(ValueError):
                    fit_material(load(p)[0])


if __name__ == '__main__':
    unittest.main()
