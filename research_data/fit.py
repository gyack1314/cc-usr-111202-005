"""按实际输注时间的双指数约束最大似然拟合编排。

拟合仅在配置声明的有限候选 (alpha,beta) 与有限非负幅度矩形上进行，
报告必须明确这是有限搜索域条件下的拟合，不是连续空间的全局最优。
未收敛候选保留在结果中并阻止静默宣布全域最优；接近最优的候选全部
保留并给出预测差异，稀疏采样下分不清的参数通过诊断与多候选呈现。
"""
import math

from .validate import validate_material
from .convolution import (design_rows, predict, infusion_contributions,
                          window_auc)
from .likelihood import (build_observations,
                         quantified_nll, censored_nll, censor_probability)
from .optimize import minimize_boxed
from .diagnostics import identifiability

# 第二个起点取矩形中心，配合原点起点覆盖边界解与内点解。
def _starts(lower, upper):
    return ((lower[0], lower[1]),
            (0.5 * (lower[0] + upper[0]), 0.5 * (lower[1] + upper[1])))


def _candidate_fit(infusions, samples, obs, alpha, beta, bounds, tol,
                   max_iterations):
    times = [float(s['time']) for s in samples]
    rows, per_infusion = design_rows(infusions, times, alpha, beta)
    status, diag = identifiability(rows, obs)
    result = {
        'alpha': alpha, 'beta': beta,
        'identifiability': status,
        'diagnostics': diag,
    }
    if status != 'identifiable':
        result.update({
            'converged': False,
            'termination': 'unidentifiable',
            'objective': None,
            'A': None, 'B': None,
            'active_bounds': [],
            'optimality_residual': None,
            'rows': rows,
            'per_infusion': per_infusion,
        })
        return result
    lower = (float(bounds['A'][0]), float(bounds['B'][0]))
    upper = (float(bounds['A'][1]), float(bounds['B'][1]))
    opt = minimize_boxed(rows, obs, lower, upper, tol=tol,
                         max_iterations=max_iterations, starts=_starts(lower, upper))
    result.update({
        'converged': opt['termination'] == 'converged',
        'termination': opt['termination'],
        'objective': opt['objective'],
        'A': opt['x'][0], 'B': opt['x'][1],
        'active_bounds': opt['active_bounds'],
        'optimality_residual': opt['optimality_residual'],
        'iterations': opt['iterations'],
        'starts_tried': opt['starts_tried'],
        'rows': rows,
        'per_infusion': per_infusion,
    })
    return result


def _sample_report(samples, infusions, candidate, obs):
    ab = (candidate['A'], candidate['B'])
    contributions = infusion_contributions(candidate['per_infusion'],
                                           [float(s['time']) for s in samples], ab)
    mu = predict(candidate['rows'], ab)
    report = []
    for j, sample in enumerate(samples):
        entry = {
            'sample_id': sample['id'],
            'time': float(sample['time']),
            'kind': sample['kind'],
            'predicted': mu[j],
            'contributions': {
                infusions[i]['id']: contributions[i][j]
                for i in range(len(infusions))
            },
        }
        kind, level, sigma = obs[j]
        if kind == 'q':
            entry['observed'] = level
            entry['sigma'] = sigma
            entry['residual'] = mu[j] - level
            entry['standardized_residual'] = (mu[j] - level) / sigma
            entry['point_nll'] = quantified_nll(level, sigma, mu[j])
        else:
            entry['limit'] = level
            entry['sigma'] = sigma
            entry['censor_probability'] = censor_probability(level, sigma, mu[j])
            entry['point_nll'] = censored_nll(level, sigma, mu[j])
        report.append(entry)
    return report


def fit_material(material):
    """完整拟合，返回可直接 JSON 序列化的报告字典（不含 NaN/Inf）。"""
    v = validate_material(material)
    infusions = v['infusions']
    samples = v['samples']
    obs = build_observations(samples)
    tol = float(v['solver']['tolerance'])
    max_iterations = int(v['solver']['max_iterations'])
    bounds = v['amplitude_bounds']

    candidates = []
    for index, pair in enumerate(v['rate_pairs']):
        alpha, beta = float(pair[0]), float(pair[1])
        cand = _candidate_fit(infusions, samples, obs, alpha, beta, bounds,
                              tol, max_iterations)
        cand['candidate_index'] = index
        candidates.append(cand)

    converged = [c for c in candidates if c['converged']]
    unconverged = [{'candidate_index': c['candidate_index'],
                    'termination': c['termination']}
                   for c in candidates if not c['converged']]
    if converged:
        # 最优目标升序，目标并列时按候选文件编号决胜
        winner = min(converged, key=lambda c: (c['objective'], c['candidate_index']))
        winner_index = winner['candidate_index']
        best_objective = winner['objective']
        gap = float(v['near_optimal_gap'])
        near = [c for c in converged
                if c['objective'] - best_objective <= gap + 1e-12]
        near_indices = [c['candidate_index'] for c in near]
        global_status = 'finite_domain_optimum'
    else:
        winner = None
        winner_index = None
        best_objective = None
        near = []
        near_indices = []
        global_status = 'no_converged_candidate'

    candidate_summaries = []
    for c in candidates:
        candidate_summaries.append({
            'candidate_index': c['candidate_index'],
            'alpha': c['alpha'], 'beta': c['beta'],
            'identifiability': c['identifiability'],
            'converged': c['converged'],
            'termination': c['termination'],
            'objective': c['objective'],
            'A': c['A'], 'B': c['B'],
            'active_bounds': c['active_bounds'],
            'optimality_residual': c['optimality_residual'],
            'iterations': c.get('iterations'),
            'starts_tried': c.get('starts_tried'),
            'diagnostics': _json_safe(c['diagnostics']),
        })

    report = {
        'scope': '有限搜索域条件下的约束最大似然拟合；不输出给药建议',
        'global_status': global_status,
        'unconverged_candidates': unconverged,
        'winner_index': winner_index,
        'near_optimal_indices': near_indices,
        'near_optimal_gap': float(v['near_optimal_gap']),
        'candidates': candidate_summaries,
    }

    if winner is not None:
        ab = (winner['A'], winner['B'])
        report['samples'] = _sample_report(samples, infusions, winner, obs)
        auc = []
        for window in v['auc_windows']:
            lo, hi = float(window[0]), float(window[1])
            total, per = window_auc(infusions, (lo, hi),
                                    winner['alpha'], winner['beta'], ab)
            auc.append({
                'window': [lo, hi],
                'auc': total,
                'per_infusion': {
                    infusions[i]['id']: per[i] for i in range(len(infusions))
                },
            })
        report['auc'] = auc
        report['near_optimal_predictions'] = _near_predictions(
            near, samples, infusions, obs, v['auc_windows'])
        # 目标决胜核对：逐点负对数似然之和应等于优化目标
        point_sum = sum(e['point_nll'] for e in report['samples'])
        report['objective_check'] = {
            'sum_point_nll': point_sum,
            'optimizer_objective': winner['objective'],
            'absolute_difference': abs(point_sum - winner['objective']),
        }
    return report


def _near_predictions(near, samples, infusions, obs, windows):
    times = [float(s['time']) for s in samples]
    out = []
    winner_mu = None
    for c in near:
        ab = (c['A'], c['B'])
        mu = predict(c['rows'], ab)
        if winner_mu is None:
            winner_mu = mu
        out.append({
            'candidate_index': c['candidate_index'],
            'alpha': c['alpha'], 'beta': c['beta'],
            'objective_gap': None,  # 由调用方填充
            'predicted_concentration': mu,
            'max_prediction_difference': max(
                abs(a - b) for a, b in zip(mu, winner_mu)),
            'auc': [
                window_auc(infusions, (float(w[0]), float(w[1])),
                           c['alpha'], c['beta'], ab)[0]
                for w in windows
            ],
        })
    best_obj = min(c['objective'] for c in near)
    for item, c in zip(out, near):
        item['objective_gap'] = c['objective'] - best_obj
    return out


def _json_safe(obj):
    """把 inf 等非有限值转为 None，配合 allow_nan=False 输出。"""
    if isinstance(obj, dict):
        return {k: _json_safe(val) for k, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
