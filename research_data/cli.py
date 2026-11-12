"""研究资料命令行：inspect 原始摘要，fit 双指数约束最大似然拟合。"""
import argparse
import json

from .io import load
from .domain import summarize
from .fit import fit_material
from .verification import (numeric_concentration, numeric_auc,
                           richardson_error, numeric_cdf_check)
from .convolution import window_auc


def _verify(report, material):
    """对胜出候选做独立数值核对，返回误差界与网格收敛证据。"""
    winner = report['winner_index']
    if winner is None:
        return {'status': 'skipped', 'reason': '无收敛候选，无法核对'}
    cand = next(c for c in report['candidates'] if c['candidate_index'] == winner)
    alpha, beta = cand['alpha'], cand['beta']
    ab = (cand['A'], cand['B'])
    infusions = material['infusions']

    concentration_checks = []
    for sample in report['samples']:
        t = sample['time']
        analytic = sample['predicted']
        coarse = numeric_concentration(t, infusions, alpha, beta, ab, 2002)
        fine = numeric_concentration(t, infusions, alpha, beta, ab, 4004)
        concentration_checks.append({
            'sample_id': sample['sample_id'],
            'analytic': analytic,
            'numeric_fine': fine,
            'absolute_difference': abs(analytic - fine),
            'richardson_error_bound': richardson_error(coarse, fine),
        })

    auc_checks = []
    for window in report['auc']:
        lo, hi = window['window']
        analytic = window['auc']
        coarse = numeric_auc((lo, hi), infusions, alpha, beta, ab, 4002)
        fine = numeric_auc((lo, hi), infusions, alpha, beta, ab, 8004)
        auc_checks.append({
            'window': [lo, hi],
            'analytic': analytic,
            'numeric_fine': fine,
            'absolute_difference': abs(analytic - fine),
            'richardson_error_bound': richardson_error(coarse, fine),
        })

    cdf_checks = []
    for sample in report['samples']:
        if sample['kind'] == 'below_limit':
            z = (sample['limit'] - sample['predicted']) / sample['sigma']
            p, bound = numeric_cdf_check(z)
            cdf_checks.append({
                'sample_id': sample['sample_id'],
                'z': z,
                'cdf_numeric': p,
                'reported_probability': sample['censor_probability'],
                'absolute_difference': abs(p - sample['censor_probability']),
                'error_bound': bound,
            })

    return {
        'status': 'checked',
        'method': '独立复合 Simpson 数值积分（不引用解析公式）',
        'concentration': concentration_checks,
        'auc': auc_checks,
        'censor_cdf': cdf_checks,
        'max_concentration_error': max(
            c['absolute_difference'] for c in concentration_checks),
        'max_auc_error': max(c['absolute_difference'] for c in auc_checks),
    }


def main():
    parser = argparse.ArgumentParser(
        description='重复输注浓度资料的双指数暴露拟合')
    sub = parser.add_subparsers(dest='command', required=True)

    p_inspect = sub.add_parser('inspect', help='显示原始资料摘要')
    p_inspect.add_argument('file')

    p_fit = sub.add_parser('fit', help='按实际输注时间做约束最大似然拟合')
    p_fit.add_argument('file')
    p_fit.add_argument('--verify', action='store_true',
                       help='附加独立数值积分核对与误差界')

    args = parser.parse_args()
    value, digest = load(args.file)

    if args.command == 'inspect':
        out = {'source_sha256': digest, 'summary': summarize(value)}
    else:
        report = fit_material(value)
        report = {'source_sha256': digest, 'report': report}
        if args.verify:
            report['independent_verification'] = _verify(report['report'], value)
        out = report

    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
