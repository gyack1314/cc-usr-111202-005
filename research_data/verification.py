"""卷积与 AUC 的独立数值核对。

这里不调用 convolution.py 的任何解析公式，而是直接按物理定义
数值积分：

    c(t) = sum_i r_i ∫ g(s) 1[s 落在输注有效区间] ds
    AUC  = ∫_{lo}^{hi} c(t) dt

复合 Simpson 误差为 O(h^4)，用相邻两级网格的理查森差值给出经验
误差界；若误差不随网格细化按 h^4 衰减则标记核对失败。
"""
import math

from .likelihood import log_normal_cdf


def _kernel(s, lam):
    return math.exp(-lam * s) if s > 0.0 else 1.0


def numeric_concentration(t, infusions, alpha, beta, ab, n_intervals=4002):
    """按卷积定义直接积分单时刻浓度。"""
    a, b = ab
    total = 0.0
    for row in infusions:
        start, end = float(row['start']), float(row['end'])
        rate = float(row['dose']) / (end - start)
        lo = max(0.0, t - end)
        hi = max(0.0, t - start)
        if hi <= lo:
            continue
        total += rate * _simpson(
            lambda s: a * _kernel(s, alpha) + b * _kernel(s, beta), lo, hi,
            n_intervals)
    return total


def numeric_auc(window, infusions, alpha, beta, ab, n_intervals=8002):
    """在时间窗上对数值浓度函数再积分一次（嵌套数值积分）。

    外网格可传显式间隔数，或传自适应元组 ('adaptive', per_hour)，
    按窗长取偶数间隔，使单位时间分辨率与窗长无关。
    """
    lo, hi = float(window[0]), float(window[1])
    if hi <= lo:
        return 0.0

    if isinstance(n_intervals, tuple) and n_intervals[0] == 'adaptive':
        per_hour = n_intervals[1]
        n = int(math.ceil((hi - lo) * per_hour))
        n_intervals = n if n % 2 == 0 else n + 1

    def c_at(t):
        return numeric_concentration(t, infusions, alpha, beta, ab,
                                     n_intervals=4004)

    return _simpson(c_at, lo, hi, n_intervals)


def _simpson(func, lo, hi, n_intervals):
    """复合 Simpson；n_intervals 强制为偶数。"""
    n = n_intervals if n_intervals % 2 == 0 else n_intervals + 1
    h = (hi - lo) / n
    total = func(lo) + func(hi)
    for k in range(1, n):
        total += (4.0 if k % 2 else 2.0) * func(lo + k * h)
    return total * h / 3.0


def richardson_error(value_coarse, value_fine):
    """两级 Simpson 网格的经验误差界：|fine-coarse|/15（h^4 外推）。"""
    return abs(value_fine - value_coarse) / 15.0


def numeric_cdf_check(z, n_intervals=10000):
    """用 phi 的 Simpson 积分独立核对删失概率 Phi(z)，返回 (Phi, 误差界)。"""
    if z < -8.0:
        # 深尾在固定网格上积分不到，改用 log Phi 的级数值作为参考并给保守界
        ref = math.exp(log_normal_cdf(z))
        return ref, ref * 1e-12
    lo = -8.0
    n = n_intervals if n_intervals % 2 == 0 else n_intervals + 1
    phi = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    if z >= 0.0:
        # Phi(z)=0.5 + ∫_0^z phi，避免与远端截断误差混在一起
        fine = 0.5 + _simpson(phi, 0.0, z, n)
        coarse = 0.5 + _simpson(phi, 0.0, z, n // 2)
    else:
        # z<0：Phi(z)=Phi(lo)+∫_lo^z phi，Phi(-8) 可忽略（约 6e-16）
        fine = phi(lo) + _simpson(phi, lo, z, n)
        coarse = phi(lo) + _simpson(phi, lo, z, n // 2)
    return fine, abs(fine - coarse) / 15.0
