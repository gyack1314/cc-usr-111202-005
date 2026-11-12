"""双指数卷积、解析 AUC 与边界连续性的测试。"""
import math
import unittest

from research_data.convolution import (
    _exp_int, _auc_primitive, infusion_kernel_value, design_rows, predict,
    infusion_window_auc, window_auc, infusion_contributions)
from research_data.verification import (
    numeric_concentration, numeric_auc, richardson_error)


INFUSIONS = [
    {'id': 'd1', 'start': 0.0, 'end': 2.0, 'dose': 100.0},
    {'id': 'd2', 'start': 24.0, 'end': 26.0, 'dose': 80.0},
    {'id': 'ov', 'start': 25.0, 'end': 27.0, 'dose': 20.0},
]
ALPHA, BETA = 0.4, 0.03
AB = (0.05, 0.02)


class StableExponentialTests(unittest.TestCase):
    def test_exp_int_matches_naive_away_from_zero(self):
        for s, lam in ((1.0, 0.4), (10.0, 0.03), (0.5, 2.0)):
            ref = (1.0 - math.exp(-lam * s)) / lam
            self.assertAlmostEqual(_exp_int(s, lam), ref, places=14)

    def test_exp_int_small_argument_series(self):
        # lam*s 极小：朴素 1-exp(-x) 会抵消，级数应保持精度
        s, lam = 1.0, 1e-10
        ref_series = s - 0.5 * lam * s * s
        self.assertAlmostEqual(_exp_int(s, lam), ref_series, places=12)
        self.assertEqual(_exp_int(0.0, lam), 0.0)

    def test_near_equal_rates_still_resolvable(self):
        # alpha、beta 极接近时两列核的差仍有限、符号正确，不出现 0/0 或 NaN
        inf = {'id': 'x', 'start': 0.0, 'end': 2.0, 'dose': 100.0}
        ka = infusion_kernel_value(5.0, inf, 0.10000001)
        kb = infusion_kernel_value(5.0, inf, 0.1)
        self.assertTrue(math.isfinite(ka) and math.isfinite(kb))
        # 速率越大衰减越快，积分核越小
        self.assertLess(ka, kb)
        self.assertGreater(kb - ka, 0.0)

    def test_auc_primitive_consistency(self):
        # J'(s) 应等于 G(s)；中心差分 h^2 量级的容差
        for s, lam in ((1.0, 0.4), (5.0, 0.03)):
            h = 1e-6
            deriv = (_auc_primitive(s + h, lam) - _auc_primitive(s - h, lam)) / (2 * h)
            self.assertAlmostEqual(deriv, _exp_int(s, lam), places=7)
        self.assertEqual(_auc_primitive(0.0, 0.4), 0.0)


class ConvolutionTests(unittest.TestCase):
    def test_not_started_infusion_is_zero(self):
        # t < start：尚未开始的输注不得凭空贡献
        self.assertEqual(infusion_kernel_value(10.0, INFUSIONS[1], ALPHA), 0.0)
        rows, per = design_rows(INFUSIONS, [10.0], ALPHA, BETA)
        self.assertEqual(per[1][0], (0.0, 0.0))
        self.assertEqual(per[2][0], (0.0, 0.0))

    def test_boundary_continuity(self):
        # 采样时刻落在输注边界 start/end 上必须连续：边界两侧的差商有界
        # （仅允许 O(eps) 的线性爬升/衰减），不存在有限跳变。
        for boundary in (0.0, 2.0, 24.0, 25.0, 26.0, 27.0):
            eps = 1e-7
            left = [infusion_kernel_value(boundary - eps, inf, ALPHA)
                    for inf in INFUSIONS]
            at = [infusion_kernel_value(boundary, inf, ALPHA) for inf in INFUSIONS]
            right = [infusion_kernel_value(boundary + eps, inf, ALPHA)
                     for inf in INFUSIONS]
            for l, a, r in zip(left, at, right):
                self.assertLessEqual(abs(r - a), eps * 60.0)
                self.assertLessEqual(abs(a - l), eps * 60.0)

    def test_start_boundary_zero_then_linear_ramp(self):
        # [start,end) 约定：t=start 时刻该输注贡献恰为 0，
        # 紧邻右侧按 rate*s 线性爬升（g(0)=1）
        inf = INFUSIONS[1]
        self.assertEqual(infusion_kernel_value(24.0, inf, ALPHA), 0.0)
        eps = 1e-8
        value = infusion_kernel_value(24.0 + eps, inf, ALPHA)
        rate = 80.0 / 2.0
        self.assertAlmostEqual(value, rate * eps, places=12)

    def test_dose_scaling_and_overlap_additivity(self):
        # 重叠输注的总浓度等于各自贡献之和；剂量翻倍贡献翻倍
        t = 25.5
        rows, per = design_rows(INFUSIONS, [t], ALPHA, BETA)
        contrib = infusion_contributions(per, [t], AB)
        total = predict(rows, AB)[0]
        self.assertAlmostEqual(total, sum(contrib[i][0] for i in range(3)),
                               places=14)
        doubled = [{'id': 'x', 'start': 25.0, 'end': 27.0, 'dose': 40.0}]
        k1 = infusion_kernel_value(t, INFUSIONS[2], ALPHA)
        k2 = infusion_kernel_value(t, doubled[0], ALPHA)
        self.assertAlmostEqual(k2, 2.0 * k1, places=14)

    def test_analytic_against_numeric_concentration(self):
        for t in (0.0, 1.0, 2.0, 4.0, 12.0, 24.0, 25.0, 25.5, 27.0, 40.0, 80.0):
            rows, _ = design_rows(INFUSIONS, [t], ALPHA, BETA)
            analytic = predict(rows, AB)[0]
            coarse = numeric_concentration(t, INFUSIONS, ALPHA, BETA, AB, 252)
            fine = numeric_concentration(t, INFUSIONS, ALPHA, BETA, AB, 502)
            self.assertLess(abs(analytic - fine), 1e-9)
            self.assertLess(richardson_error(coarse, fine), 1e-10)


class AucTests(unittest.TestCase):
    def test_infusion_outside_window_no_contribution(self):
        # 窗外输注不能凭空贡献
        val = infusion_window_auc(INFUSIONS[1], (0.0, 24.0), ALPHA)
        self.assertEqual(val, 0.0)
        # 输注完全在窗口之前时仅剩指数衰减尾迹；尾迹随窗口后移单调衰减，
        # 且相对窗内暴露占比极小（残留不是新输入）
        tail = infusion_window_auc(INFUSIONS[0], (50.0, 60.0), ALPHA)
        tail_far = infusion_window_auc(INFUSIONS[0], (80.0, 90.0), ALPHA)
        inside = infusion_window_auc(INFUSIONS[0], (0.0, 10.0), ALPHA)
        self.assertGreaterEqual(tail, 0.0)
        self.assertLess(tail, 1e-4 * inside)
        self.assertLess(tail_far, tail)

    def test_clipped_long_infusion(self):
        # 跨窗长输注的正确截取语义：窗内 AUC = 窗内输入的直接暴露
        # （t 落在窗内的部分）+ 窗外输入衰减到窗内的残留，两者均由
        # 解析分段给出，并与独立数值积分一致。截断历史会错误地丢失残留。
        long = {'id': 'long', 'start': 10.0, 'end': 40.0, 'dose': 300.0}
        window = (20.0, 30.0)
        analytic = infusion_window_auc(long, window, BETA)
        numeric = numeric_auc(window, [long], 0.0, BETA, (0.0, 1.0),
                              ('adaptive', 40))
        self.assertLess(abs(analytic - numeric), 1e-9)
        # 残留严格为正：长输注的窗内 AUC 必然高于只在窗内给药的同速率输注
        fresh = {'id': 'fresh', 'start': 20.0, 'end': 30.0, 'dose': 100.0}
        self.assertGreater(analytic, infusion_window_auc(fresh, window, BETA))

    def test_window_partition_additivity(self):
        total, _ = window_auc(INFUSIONS, (0.0, 100.0), ALPHA, BETA, AB)
        p1, per1 = window_auc(INFUSIONS, (0.0, 50.0), ALPHA, BETA, AB)
        p2, per2 = window_auc(INFUSIONS, (50.0, 100.0), ALPHA, BETA, AB)
        self.assertAlmostEqual(p1 + p2, total, places=12)
        for i in range(3):
            self.assertAlmostEqual(per1[i] + per2[i],
                                   window_auc(INFUSIONS, (0.0, 100.0),
                                              ALPHA, BETA, AB)[1][i], places=12)

    def test_analytic_against_numeric_auc(self):
        for window in ((0.0, 24.0), (24.0, 48.0), (0.0, 100.0), (25.0, 26.5)):
            analytic, per = window_auc(INFUSIONS, window, ALPHA, BETA, AB)
            fine = numeric_auc(window, INFUSIONS, ALPHA, BETA, AB,
                               ('adaptive', 20))
            self.assertLess(abs(analytic - fine), 1e-4)
            # 逐输注分解之和等于总量
            self.assertAlmostEqual(sum(per), analytic, places=14)

    def test_point_window_is_zero(self):
        total, _ = window_auc(INFUSIONS, (10.0, 10.0), ALPHA, BETA, AB)
        self.assertEqual(total, 0.0)

    def test_closed_form_total_exposure(self):
        # 单指数单位响应积分 ∫_0^inf exp(-lam s) ds = 1/lam，
        # 故单输注总暴露必为 dose/lam（速率与输注时长抵消）
        inf = {'id': 'd', 'start': 0.0, 'end': 2.0, 'dose': 100.0}
        for lam, dose in ((0.4, 100.0), (0.03, 100.0), (1.1, 50.0)):
            row = inf if dose == 100.0 else {**inf, 'dose': dose}
            total = infusion_window_auc(row, (0.0, 10000.0), lam)
            self.assertAlmostEqual(total, dose / lam, places=8)

    def test_closed_form_in_infusion_window(self):
        # 窗恰好覆盖输注期 [0,2]：AUC = rate * J_lam(2)
        inf = {'id': 'd', 'start': 0.0, 'end': 2.0, 'dose': 100.0}
        lam = 0.4
        rate = 50.0
        j2 = (2.0 - (1.0 - math.exp(-lam * 2.0)) / lam) / lam
        self.assertAlmostEqual(
            infusion_window_auc(inf, (0.0, 2.0), lam), rate * j2, places=12)


if __name__ == '__main__':
    unittest.main()
