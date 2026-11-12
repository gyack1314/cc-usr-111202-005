"""设计矩阵近共线与不可识别诊断。

双指数模型的设计矩阵只有两列（alpha 列、beta 列）。当两个速率候选
非常接近时两列近似平行，幅度 A、B 在稀疏采样下无法分别识别；全删失
资料则不含任何定量信息。两类情况都必须显式报告，而不是给出一个
看似精确的点估计。

条件数按列归一化后的 Gram 矩阵最大/最小特征值之比计算，避免量纲
掩盖近线性关系。
"""
import math

# 条件数超过该阈值判为近共线（双精度有效位约 16 位，留 8 位余量）。
COLLINEAR_CONDITION = 1e8


def symmetric_eigenvalues_2x2(h00, h01, h11):
    """2x2 实对称矩阵的两个特征值，闭式稳定计算，升序返回。"""
    trace = h00 + h11
    disc = math.hypot(h00 - h11, 2.0 * h01)
    return 0.5 * (trace - disc), 0.5 * (trace + disc)


def design_condition(rows):
    """列归一化 Gram 矩阵的条件数；出现零列返回 inf。"""
    norms = [0.0, 0.0]
    for xa, xb in rows:
        norms[0] += xa * xa
        norms[1] += xb * xb
    norms = [math.sqrt(v) for v in norms]
    if norms[0] == 0.0 or norms[1] == 0.0:
        return math.inf, norms
    h00 = h11 = 1.0
    h01 = 0.0
    for xa, xb in rows:
        ua, ub = xa / norms[0], xb / norms[1]
        h01 += ua * ub
    lam_min, lam_max = symmetric_eigenvalues_2x2(h00, h01, h11)
    if lam_min <= 0.0:
        return math.inf, norms
    return lam_max / lam_min, norms


def identifiability(rows, obs):
    """返回 (status, details)。

    status 为 'identifiable'、'all_censored' 或 'near_collinear'。
    """
    quantified = sum(1 for kind, _, _ in obs if kind == 'q')
    if quantified == 0:
        return 'all_censored', {
            'reason': '全部观测低于定量限，仅含删失概率，无定量信息',
            'condition_number': None,
        }
    condition, norms = design_condition(rows)
    if not math.isfinite(condition) or condition > COLLINEAR_CONDITION:
        return 'near_collinear', {
            'reason': '设计矩阵近共线，A 与 B 无法从稀疏采样中分别识别',
            'condition_number': condition if math.isfinite(condition) else None,
            'column_norms': norms,
            'threshold': COLLINEAR_CONDITION,
        }
    return 'identifiable', {
        'condition_number': condition,
        'column_norms': norms,
        'threshold': COLLINEAR_CONDITION,
    }
