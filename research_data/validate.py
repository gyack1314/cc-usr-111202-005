"""拟合入口资料的严格校验。

校验规则与 docs/data-contract.md 一致：单位固定、有限数值、观测只有
quantified/below_limit 两类、速率候选为有限有序对、幅度矩形有限非负。
任何不满足契约的输入都抛出 ValueError，不做静默修正。
"""
from .io import finite, positive, integer, unique

UNITS = {'time': 'hour', 'dose': 'mg', 'concentration': 'mg/L'}


def _pair(row, name):
    if not isinstance(row, (list, tuple)) or len(row) != 2:
        raise ValueError(f'{name}必须是两个数的有序对')
    return finite(row[0], name=f'{name}[0]'), finite(row[1], name=f'{name}[1]')


def validate_material(v):
    """完整校验一份拟合资料，返回规范化后的同结构字典。"""
    if not isinstance(v, dict):
        raise ValueError('顶层需要对象')
    if v.get('units') != UNITS:
        raise ValueError('单位未声明或不匹配：仅支持 hour/mg/(mg/L)')

    infusions = v.get('infusions')
    samples = v.get('samples')
    if not isinstance(infusions, list) or not infusions:
        raise ValueError('infusions 必须是非空列表')
    if not isinstance(samples, list) or not samples:
        raise ValueError('samples 必须是非空列表')
    unique(infusions, 'id')
    unique(samples, 'id')

    for row in infusions:
        start = finite(row.get('start'), name='infusion.start')
        end = finite(row.get('end'), name='infusion.end')
        if start < 0.0 or end <= start:
            raise ValueError('输注时段错误：要求 0<=start<end')
        positive(row.get('dose'), name='infusion.dose')

    for row in samples:
        if finite(row.get('time'), name='sample.time') < 0.0:
            raise ValueError('采样时刻不能为负')
        positive(row.get('sigma'), name='sample.sigma')
        kind = row.get('kind')
        if kind == 'quantified':
            if finite(row.get('value'), name='sample.value') < 0.0:
                raise ValueError('定量浓度不能为负')
        elif kind == 'below_limit':
            positive(row.get('limit'), name='sample.limit')
        else:
            raise ValueError('未知观测类型：仅支持 quantified/below_limit')

    pairs = v.get('rate_pairs')
    if not isinstance(pairs, list) or not pairs:
        raise ValueError('rate_pairs 必须是非空有限候选列表')
    for k, pair in enumerate(pairs):
        alpha, beta = _pair(pair, f'rate_pairs[{k}]')
        if not (alpha > beta > 0.0):
            raise ValueError(f'rate_pairs[{k}] 要求 alpha>beta>0')

    bounds = v.get('amplitude_bounds')
    if not isinstance(bounds, dict):
        raise ValueError('amplitude_bounds 缺失')
    for name in ('A', 'B'):
        lo, hi = _pair(bounds.get(name), f'amplitude_bounds.{name}')
        if lo < 0.0 or hi <= lo:
            raise ValueError(f'amplitude_bounds.{name} 要求 0<=下界<上界')

    windows = v.get('auc_windows')
    if not isinstance(windows, list) or not windows:
        raise ValueError('auc_windows 必须是非空列表')
    for k, window in enumerate(windows):
        lo, hi = _pair(window, f'auc_windows[{k}]')
        if lo < 0.0 or hi < lo:
            raise ValueError(f'auc_windows[{k}] 要求 0<=lo<=hi')

    gap = finite(v.get('near_optimal_gap'), name='near_optimal_gap')
    if gap < 0.0:
        raise ValueError('near_optimal_gap 不能为负')

    solver = v.get('solver')
    if not isinstance(solver, dict):
        raise ValueError('solver 配置缺失')
    positive(solver.get('tolerance'), name='solver.tolerance')
    integer(solver.get('max_iterations'), minimum=1)
    return v
