"""双指数响应与分段恒速输注的解析卷积。

单位瞬时剂量在经过 s>=0 后的响应为

    g(s) = A*exp(-alpha*s) + B*exp(-beta*s)

每段输注 i 在 [start_i, end_i) 以 r_i = dose_i/(end_i-start_i) 恒速输入，
输注 i 在时刻 t 的浓度贡献为

    c_i(t) = r_i * ∫_{max(0,t-end_i)}^{max(0,t-start_i)} g(s) ds

（尚未开始的输注贡献精确为 0；时刻落在输注边界时按单侧极限取值，
积分公式在 s=0 与空区间上连续，因此边界处无跳变。）

对单个速率常数 lam>0，单输注的积分核为

    K_lam(t; start,end) = r_i * [G_lam(max(0,t-start)) - G_lam(max(0,t-end))]
    G_lam(s) = (1 - exp(-lam*s)) / lam

G 在 lam*s 很小时用指数减一的稳定形式展开，避免 1-exp(-x) 的
灾难性抵消，也避免 alpha、beta 接近时两列设计矩阵近退化却无提示。
"""
import math

# lam*s 小于该阈值时改用泰勒级数（expm1 本身在双精度下已较稳，
# 阈值主要用于 lam 下溢级别与极限连续性测试）。
_SMALL = 1e-4
_SERIES_TERMS = 12


def _exp_int(s, lam):
    """G_lam(s) = (1-exp(-lam*s))/lam，s>=0、lam>0，数值稳定。"""
    x = lam * s
    if x <= 0.0:
        return 0.0
    if x < _SMALL:
        # 1-e^{-x} = x - x^2/2 + x^3/6 - ...，除以 lam 得到 s 的幂级数
        total = 0.0
        term = s
        ratio = -x
        for k in range(1, _SERIES_TERMS + 1):
            total += term
            term *= ratio / (k + 1)
        return total
    return -math.expm1(-x) / lam


def infusion_rate(infusion):
    """恒速输注速率 r=dose/(end-start)，单位 mg/h。"""
    start = float(infusion['start'])
    end = float(infusion['end'])
    return float(infusion['dose']) / (end - start), start, end


def infusion_kernel_value(t, infusion, lam):
    """单输注对 exp(-lam*s) 项的速率加权积分 K_lam(t)。"""
    rate, start, end = infusion_rate(infusion)
    upper = t - start
    lower = t - end
    upper = upper if upper > 0.0 else 0.0
    lower = lower if lower > 0.0 else 0.0
    return rate * (_exp_int(upper, lam) - _exp_int(lower, lam))


def design_rows(infusions, times, alpha, beta):
    """构造设计矩阵行 X_j = (sum_i K_alpha,i(t_j), sum_i K_beta,i(t_j))。

    预测浓度 mu_j = A*X[j,0] + B*X[j,1]。同时返回每个输注各自的
    (K_alpha, K_beta)，用于逐输注贡献分解与 AUC。
    """
    rows = []
    per_infusion = [[] for _ in infusions]
    for t in times:
        xa = 0.0
        xb = 0.0
        for i, infusion in enumerate(infusions):
            ka = infusion_kernel_value(t, infusion, alpha)
            kb = infusion_kernel_value(t, infusion, beta)
            per_infusion[i].append((ka, kb))
            xa += ka
            xb += kb
        rows.append((xa, xb))
    return rows, per_infusion


def predict(rows, ab):
    """设计矩阵行与 (A,B) 的线性预测。"""
    a, b = ab
    return [a * xa + b * xb for xa, xb in rows]


def infusion_contributions(per_infusion, times, ab):
    """返回 {infusion_index: [各采样时刻该输注的浓度贡献]}。"""
    a, b = ab
    return {
        i: [a * ka + b * kb for ka, kb in series]
        for i, series in enumerate(per_infusion)
    }


# ---------------------------------------------------------------------------
# 解析 AUC
# ---------------------------------------------------------------------------

def _auc_primitive(s, lam):
    """G_lam(s) 对 s 的原函数 J_lam(s) = (s - (1-e^{-lam*s})/lam)/lam。

    即 J_lam(s) = ∫_0^s G_lam(u) du，s>=0。小段用级数稳定计算。
    """
    x = lam * s
    if x <= 0.0:
        return 0.0
    if x < _SMALL:
        # J(s) = s^2/2 - lam*s^3/6 + lam^2*s^4/24 - ...
        total = 0.0
        term = s * s
        ratio = -x
        # 首项 s^2 * 1/2!
        term *= 0.5
        for k in range(1, _SERIES_TERMS + 1):
            total += term
            term *= ratio / (k + 2)
        return total
    return (s - _exp_int(s, lam)) / lam


def infusion_window_auc(infusion, window, lam):
    """单输注、单指数项在时间窗 [lo,hi] 上的速率加权解析积分。

    窗外输注不贡献；跨窗输注只截取与窗口相交的部分。
    推导：∫_{lo}^{hi} [G(t-start)-G(t-end)] dt，t<start 项为 0，
    令 s=max(0,t-start)、u=max(0,t-end) 后用原函数 J 分段。
    """
    rate, start, end = infusion_rate(infusion)
    lo, hi = window
    if hi <= lo:
        return 0.0
    clipped_lo = lo if lo > start else start
    clipped_hi = hi if hi > start else start
    if clipped_hi <= clipped_lo:
        return 0.0  # 输注整体落在窗口之后
    # start 项：s = t-start 在交集区间上恒非负
    start_part = _auc_primitive(clipped_hi - start, lam) - \
        _auc_primitive(clipped_lo - start, lam)
    # end 项：u = max(0,t-end)，积分下界平移到 end
    end_lo = lo if lo > end else end
    end_hi = hi if hi > end else end
    end_part = 0.0
    if end_hi > end_lo:
        end_part = _auc_primitive(end_hi - end, lam) - \
            _auc_primitive(end_lo - end, lam)
    return rate * (start_part - end_part)


def window_auc(infusions, window, alpha, beta, ab):
    """窗口内总 AUC 及逐输注分解（A、B 两个指数项分别解析积分）。"""
    a, b = ab
    per = []
    total = 0.0
    for infusion in infusions:
        value = (a * infusion_window_auc(infusion, window, alpha)
                 + b * infusion_window_auc(infusion, window, beta))
        per.append(value)
        total += value
    return total, per
