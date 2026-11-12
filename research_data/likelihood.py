"""定量与左删失观测的高斯负对数似然。

定量观测 y 与已知标准差 sigma：
    nll_q = 0.5*log(2*pi*sigma^2) + (y-mu)^2/(2*sigma^2)
低于定量限观测仅声明 Y < L，其似然为 Pr(Y<L)=Phi(z)，z=(L-mu)/sigma：
    nll_c = -log Phi(z)
不能把删失观测填成 L/2 或 0，也不能直接删除。

Phi 的对数与 Mill 比值 r(z)=phi(z)/Phi(z) 在深尾（z 很负）时通过
erfcx 的渐近展开计算，避免 exp(z^2/2) 上溢或 Phi 下溢成零。
似然关于线性预测 mu=A*xa+B*xb 是凸函数，整体负对数似然关于 (A,B) 凸。
"""
import math

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _erfcx(x):
    """erfcx(x)=exp(x^2)*erfc(x)，x>=0，深尾用渐近级数。"""
    if x < 10.0:
        return math.exp(x * x) * math.erfc(x)
    # erfcx(x) ~ 1/(x*sqrt(pi)) * sum_{n>=0} (-1)^n (2n-1)!! / (2x^2)^n
    # 交错渐近级数在项的绝对值开始增大（最小项）处截断。
    inv_2x2 = 1.0 / (2.0 * x * x)
    total = 1.0
    term = 1.0
    prev_abs = math.inf
    for n in range(1, 100):
        term *= -(2.0 * n - 1.0) * inv_2x2
        if abs(term) >= prev_abs:
            break
        total += term
        prev_abs = abs(term)
    return total / (x * math.sqrt(math.pi))


def log_normal_cdf(z):
    """log Phi(z)，全程有限，深尾不丢失精度。"""
    if z >= 0.0:
        return math.log(0.5 * math.erfc(-z / math.sqrt(2.0)))
    # z<0：Phi(z)=0.5*erfc(x), x=|z|/sqrt2；
    # log Phi = -log2 + log erfc(x) = -log2 + log erfcx(x) - x^2
    x = -z / math.sqrt(2.0)
    return -math.log(2.0) + math.log(_erfcx(x)) - x * x


def mills_ratio(z):
    """r(z)=phi(z)/Phi(z)，即 -d/dz log Phi(z)。"""
    if z >= 0.0:
        return math.exp(-0.5 * z * z - log_normal_cdf(z)) / math.sqrt(2.0 * math.pi)
    # 深尾恒等式（z<0，x=|z|/sqrt2）：
    # r(z) = sqrt(2/pi) / erfcx(x)，全程不出现上溢的指数。
    x = -z / math.sqrt(2.0)
    return math.sqrt(2.0 / math.pi) / _erfcx(x)


def quantified_nll(y, sigma, mu):
    residual = (mu - y) / sigma
    return _LOG_SQRT_2PI + math.log(sigma) + 0.5 * residual * residual


def censored_nll(limit, sigma, mu):
    z = (limit - mu) / sigma
    return -log_normal_cdf(z)


def censor_probability(limit, sigma, mu):
    """Pr(Y<limit)=Phi(z)，夹到 float64 可表示的最小正值，防止下溢成零。"""
    z = (limit - mu) / sigma
    p = math.exp(log_normal_cdf(z))
    if p <= 0.0:
        # logPhi 在 z<-约38 时已低于 float64 最小可表示概率，
        # 保留最小正浮点而非 0，使下游乘积与 JSON 仍有定义。
        return 5e-324
    if p >= 1.0:
        return 1.0 - 1e-16
    return p


def build_observations(samples):
    """把采样记录整理成优化器使用的数值元组列表。"""
    obs = []
    for row in samples:
        sigma = float(row['sigma'])
        if row['kind'] == 'quantified':
            obs.append(('q', float(row['value']), sigma))
        else:
            obs.append(('c', float(row['limit']), sigma))
    return obs


def objective_gradient_hessian(ab, rows, obs):
    """返回 (nll, grad[2], hessian[[h00,h01],[h01,h11]])，全部解析。

    定量项对 mu 的曲率为 1/sigma^2；
    删失项 r(z)=phi/Phi 满足 r'(z)=r(z)*(r(z)-z)，
    对 mu 的曲率为 r(z)*(r(z)-z)/sigma^2，恒非负（凸性）。
    """
    a, b = ab
    nll = 0.0
    g0 = g1 = 0.0
    h00 = h01 = h11 = 0.0
    for (xa, xb), (kind, level, sigma) in zip(rows, obs):
        mu = a * xa + b * xb
        if kind == 'q':
            z = (mu - level) / sigma
            nll += _LOG_SQRT_2PI + math.log(sigma) + 0.5 * z * z
            dm = z / sigma                 # d nll / d mu
            d2m = 1.0 / (sigma * sigma)    # d2 nll / d mu2
        else:
            z = (level - mu) / sigma
            nll -= log_normal_cdf(z)
            r = mills_ratio(z)
            dm = r / sigma
            # r'(z) = r(z)*(z-r(z))；d2 nll/d mu2 = -r'(z)/sigma^2
            #        = r(z)*(r(z)+z)/sigma^2，恒非负（凸性，差分已验证）。
            d2m = r * (r + z) / (sigma * sigma)
            if d2m < 0.0:  # 数值兜底
                d2m = 0.0
        g0 += dm * xa
        g1 += dm * xb
        h00 += d2m * xa * xa
        h01 += d2m * xa * xb
        h11 += d2m * xb * xb
    return nll, (g0, g1), (h00, h01, h11)


def objective(ab, rows, obs):
    return objective_gradient_hessian(ab, rows, obs)[0]
