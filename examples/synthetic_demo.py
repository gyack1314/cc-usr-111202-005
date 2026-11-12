"""四类合成资料的端到端演示与数值误差界。

场景（全部为虚构数据，无真实受试者信息）：
  1. 单次短输注：解析卷积的基线核对；
  2. 重叠输注：多输注分段与逐输注贡献分解；
  3. 长尾残留：前一剂残留与新一剂峰值分离，不按给药后小时叠加；
  4. 大量删失：多数观测仅给上限，似然使用 Pr(Y<L)。

每个场景都用独立复合 Simpson 数值积分核对浓度与 AUC，并以两级网格的
理查森外推报告经验误差界。运行：

    python -m examples.synthetic_demo
"""
import math

from research_data.fit import fit_material
from research_data.convolution import design_rows, predict, window_auc
from research_data.verification import (numeric_concentration, numeric_auc,
                                        richardson_error)

UNITS = {'time': 'hour', 'dose': 'mg', 'concentration': 'mg/L'}
RATE_PAIRS = [[0.4, 0.03], [0.6, 0.06], [0.9, 0.1]]
BOUNDS = {'A': [0.0, 1.0], 'B': [0.0, 1.0]}
SOLVER = {'tolerance': 1e-8, 'max_iterations': 10000}
# 生成合成观测所用的真实参数（落在候选 0 与矩形内部）
TRUE_ALPHA, TRUE_BETA, TRUE_AB = 0.4, 0.03, (0.05, 0.02)


def _noiseless(infusions, times):
    rows, _ = design_rows(infusions, times, TRUE_ALPHA, TRUE_BETA)
    return predict(rows, TRUE_AB)


def _quant(sid, t, value, sigma=0.2):
    return {'id': sid, 'time': t, 'kind': 'quantified',
            'value': round(value, 4), 'sigma': sigma}


def _censor(sid, t, limit=0.3, sigma=0.2):
    return {'id': sid, 'time': t, 'kind': 'below_limit',
            'limit': limit, 'sigma': sigma}


def scenario_single_short():
    infusions = [{'id': 'bolus', 'start': 0.0, 'end': 1.0, 'dose': 100.0}]
    times = [0.5, 1.0, 2.0, 6.0, 12.0]
    mu = _noiseless(infusions, times)
    samples = [_quant(f'q{k}', t, v) for k, (t, v) in enumerate(zip(times, mu))]
    return _base('single_short', infusions, samples, [(0.0, 12.0)])


def scenario_overlap():
    infusions = [
        {'id': 'first', 'start': 0.0, 'end': 2.0, 'dose': 100.0},
        {'id': 'second', 'start': 1.5, 'end': 3.5, 'dose': 60.0},
    ]
    times = [1.0, 1.75, 2.0, 3.5, 5.0, 10.0]
    mu = _noiseless(infusions, times)
    samples = [_quant(f'q{k}', t, v) for k, (t, v) in enumerate(zip(times, mu))]
    return _base('overlap', infusions, samples, [(0.0, 12.0), (1.5, 3.5)])


def scenario_long_tail():
    # 第二次给药时第一次输注的残留仍显著；错误地按“给药后小时”叠加
    # 会把残留记成新一剂的峰值。拟合以绝对研究时间卷积，二者分离。
    infusions = [
        {'id': 'first', 'start': 0.0, 'end': 2.0, 'dose': 100.0},
        {'id': 'second', 'start': 24.0, 'end': 26.0, 'dose': 100.0},
    ]
    times = [1.0, 2.0, 12.0, 23.5, 24.0, 25.0, 26.0, 36.0, 60.0]
    mu = _noiseless(infusions, times)
    samples = [_quant(f'q{k}', t, v) for k, (t, v) in enumerate(zip(times, mu))]
    return _base('long_tail', infusions, samples,
                 [(0.0, 24.0), (24.0, 48.0), (0.0, 72.0)])


def scenario_heavy_censoring():
    infusions = [{'id': 'only', 'start': 0.0, 'end': 2.0, 'dose': 100.0}]
    quant_times = [1.0, 2.0, 4.0]
    censor_times = [8.0, 16.0, 30.0, 50.0, 80.0]
    mu_q = _noiseless(infusions, quant_times)
    samples = [_quant(f'q{k}', t, v)
               for k, (t, v) in enumerate(zip(quant_times, mu_q))]
    samples += [_censor(f'b{k}', t) for k, t in enumerate(censor_times)]
    return _base('heavy_censoring', infusions, samples, [(0.0, 24.0), (0.0, 96.0)])


def _base(name, infusions, samples, windows):
    return {
        'name': name,
        'material': {
            'units': UNITS,
            'infusions': infusions,
            'samples': samples,
            'rate_pairs': RATE_PAIRS,
            'amplitude_bounds': BOUNDS,
            'auc_windows': [list(w) for w in windows],
            'near_optimal_gap': 2.0,
            'solver': SOLVER,
        },
    }


def _error_bounds(material, report):
    """对胜出候选做两级网格数值核对，返回浓度与 AUC 的最大误差界。"""
    winner = next(c for c in report['candidates']
                  if c['candidate_index'] == report['winner_index'])
    alpha, beta, ab = winner['alpha'], winner['beta'], (winner['A'], winner['B'])
    infusions = material['infusions']

    conc_errors = []
    for sample in report['samples']:
        t = sample['time']
        analytic = sample['predicted']
        coarse = numeric_concentration(t, infusions, alpha, beta, ab, 252)
        fine = numeric_concentration(t, infusions, alpha, beta, ab, 502)
        conc_errors.append((
            abs(analytic - fine),
            richardson_error(coarse, fine),
        ))

    auc_errors = []
    for block in report['auc']:
        lo, hi = block['window']
        analytic = block['auc']
        coarse = numeric_auc((lo, hi), infusions, alpha, beta, ab,
                             ('adaptive', 20))
        fine = numeric_auc((lo, hi), infusions, alpha, beta, ab,
                           ('adaptive', 40))
        auc_errors.append((
            abs(analytic - fine),
            richardson_error(coarse, fine),
        ))
    return {
        'max_concentration_abs_error': max(e[0] for e in conc_errors),
        'max_concentration_richardson_bound': max(e[1] for e in conc_errors),
        'max_auc_abs_error': max(e[0] for e in auc_errors),
        'max_auc_richardson_bound': max(e[1] for e in auc_errors),
    }


def run_all():
    scenarios = [scenario_single_short(), scenario_overlap(),
                 scenario_long_tail(), scenario_heavy_censoring()]
    results = []
    for scenario in scenarios:
        material = scenario['material']
        report = fit_material(material)
        entry = {
            'scenario': scenario['name'],
            'global_status': report['global_status'],
            'winner_index': report['winner_index'],
            'converged_candidates': sum(
                1 for c in report['candidates'] if c['converged']),
            'near_optimal_indices': report['near_optimal_indices'],
        }
        if report['winner_index'] is not None:
            entry['error_bounds'] = _error_bounds(material, report)
            winner = report['candidates'][report['winner_index']]
            entry['winner'] = {
                'alpha': winner['alpha'], 'beta': winner['beta'],
                'A': winner['A'], 'B': winner['B'],
                'objective': winner['objective'],
                'active_bounds': winner['active_bounds'],
                'optimality_residual': winner['optimality_residual'],
            }
            # 长尾场景：显式展示残留与新剂峰值的逐输注分离
            if scenario['name'] == 'long_tail':
                at_25 = next(s for s in report['samples'] if s['time'] == 25.0)
                entry['residual_vs_new_dose_at_t25'] = at_25['contributions']
            entry['n_censored'] = sum(
                1 for s in report['samples'] if s['kind'] == 'below_limit')
        results.append(entry)
    return results


def main():
    import json
    print(json.dumps(
        {'scope': '有限搜索域条件下的合成拟合演示；不输出给药建议',
         'results': run_all()},
        ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
