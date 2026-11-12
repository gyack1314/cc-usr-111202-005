"""矩形界约束下的凸负对数似然优化。

只有两个变量 (A,B)，且目标对 (A,B) 为凸，采用：

1. 谱投影梯度（SPG，Barzilai-Borwein 步长 + 非单调线搜索，Grippo 等）
   做全局收敛的粗优化；
2. 收敛后在活动集上做投影牛顿抛光，使最优性残差达到给定容差。

每次求解都报告目标值、命中的边界、投影梯度最优性残差和终止原因；
未收敛由调用方显式标记，不允许静默跳过候选。
"""
import math

_TERMINATION = {
    'converged': 'converged',
    'max_iterations': 'max_iterations',
    'line_search_failed': 'line_search_failed',
    'singular_hessian': 'singular_hessian',
}


def project_box(x, lower, upper):
    out = []
    for xi, lo, hi in zip(x, lower, upper):
        out.append(min(hi, max(lo, xi)))
    return tuple(out)


def projected_gradient_residual(x, grad, lower, upper):
    """投影梯度残差 ||P(x-grad)-x||_inf，界盒约束一阶最优性度量。"""
    pg = tuple(pi - xi for pi, xi in zip(
        project_box(tuple(xi - gi for xi, gi in zip(x, grad)), lower, upper), x))
    return max(abs(v) for v in pg), pg


def _newton_polish(x, lower, upper, rows, obs, tol):
    """活动集投影牛顿步；返回 (x, nll, grad, residual, reason)。"""
    for _ in range(50):
        nll, grad, hess = _fgh(x, rows, obs)
        residual, _ = projected_gradient_residual(x, grad, lower, upper)
        if residual <= tol:
            return x, nll, grad, residual, _TERMINATION['converged']
        h00, h01, h11 = hess
        free = [k for k in range(2) if lower[k] < x[k] < upper[k]]
        if not free:
            return x, nll, grad, residual, _TERMINATION['converged']
        if len(free) == 2:
            det = h00 * h11 - h01 * h01
            if det <= 1e-300:
                return x, nll, grad, residual, _TERMINATION['singular_hessian']
            d0 = -(h11 * grad[0] - h01 * grad[1]) / det
            d1 = -(-h01 * grad[0] + h00 * grad[1]) / det
            direction = (d0, d1)
        else:
            k = free[0]
            hkk = hess[0] if k == 0 else hess[2]
            if hkk <= 1e-300:
                return x, nll, grad, residual, _TERMINATION['singular_hessian']
            direction = tuple(-grad[k] / hkk if j == k else 0.0 for j in range(2))
        # 回溯到界内并保证充分下降
        step = 1.0
        x_new = project_box(tuple(xi + step * di for xi, di in zip(x, direction)),
                            lower, upper)
        f_new = _fgh(x_new, rows, obs)[0]
        guard = 0
        while f_new > nll and guard < 100:
            step *= 0.5
            x_new = project_box(
                tuple(xi + step * di for xi, di in zip(x, direction)), lower, upper)
            f_new = _fgh(x_new, rows, obs)[0]
            guard += 1
        if guard == 100 or max(abs(a - b) for a, b in zip(x_new, x)) <= 1e-15:
            return x, nll, grad, residual, _TERMINATION['line_search_failed']
        x = x_new
    nll, grad, _ = _fgh(x, rows, obs)
    residual, _ = projected_gradient_residual(x, grad, lower, upper)
    reason = (_TERMINATION['converged'] if residual <= tol
              else _TERMINATION['max_iterations'])
    return x, nll, grad, residual, reason


def _fgh(x, rows, obs):
    from .likelihood import objective_gradient_hessian
    return objective_gradient_hessian(x, rows, obs)


def minimize_boxed(rows, obs, lower, upper, tol=1e-8, max_iterations=10000,
                   starts=((0.0, 0.0),)):
    """多起点 SPG+牛顿抛光，凸问题下取目标最小的一次结果。

    返回字典：objective, x, active_bounds, optimality_residual,
    termination, iterations, starts_tried。
    """
    lower = tuple(float(v) for v in lower)
    upper = tuple(float(v) for v in upper)
    best = None
    for start in starts:
        result = _minimize_single(rows, obs, lower, upper, tol,
                                  max_iterations, start)
        if best is None or result['objective'] < best['objective']:
            best = result
    best['starts_tried'] = len(starts)
    return best


def _minimize_single(rows, obs, lower, upper, tol, max_iterations, start):
    x = project_box(tuple(float(v) for v in start), lower, upper)
    nll, grad, _ = _fgh(x, rows, obs)
    history = [nll]
    step = 1.0
    prev = None
    reason = _TERMINATION['max_iterations']
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        residual, _ = projected_gradient_residual(x, grad, lower, upper)
        if residual <= tol:
            reason = _TERMINATION['converged']
            break
        # Barzilai-Borwein 步长
        if prev is not None:
            dx = tuple(xi - pi for xi, pi in zip(x, prev['x']))
            dg = tuple(gi - pi for gi, pi in zip(grad, prev['grad']))
            sdoty = sum(si * yi for si, yi in zip(dx, dg))
            ss = sum(si * si for si in dx)
            if sdoty > 1e-300:
                step = min(1e10, max(1e-10, ss / sdoty))
        # 非单调 Armijo 线搜索（参考最近至多 10 次目标）
        reference = max(history[-10:])
        trial_step = step
        accepted = False
        for _ in range(60):
            x_try = project_box(
                tuple(xi - trial_step * gi for xi, gi in zip(x, grad)),
                lower, upper)
            movement_sq = sum((a - b) ** 2 for a, b in zip(x_try, x))
            if movement_sq == 0.0:
                # 投影不移动：当前点已是界盒约束的一阶稳定点
                reason = _TERMINATION['converged']
                break
            f_try = _fgh(x_try, rows, obs)[0]
            if f_try <= reference - 1e-4 * movement_sq / max(trial_step, 1e-300):
                accepted = True
                break
            trial_step *= 0.5
        if reason == _TERMINATION['converged']:
            break
        if not accepted:
            reason = _TERMINATION['line_search_failed']
            break
        prev = {'x': x, 'grad': grad}
        x = x_try
        nll, grad, _ = _fgh(x, rows, obs)
        history.append(nll)
        if not math.isfinite(nll):
            reason = _TERMINATION['line_search_failed']
            break
    if reason == _TERMINATION['converged']:
        x, nll, grad, residual, reason = _newton_polish(
            x, lower, upper, rows, obs, tol)
    else:
        residual, _ = projected_gradient_residual(x, grad, lower, upper)
    active = []
    for k, name in enumerate(('A', 'B')):
        if abs(x[k] - lower[k]) <= 1e-12:
            active.append(f'{name}=lower')
        elif abs(x[k] - upper[k]) <= 1e-12:
            active.append(f'{name}=upper')
    return {
        'objective': nll,
        'x': x,
        'active_bounds': active,
        'optimality_residual': residual,
        'termination': reason,
        'iterations': iterations,
    }
