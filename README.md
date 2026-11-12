# 重复输注浓度资料的双指数暴露拟合

这是研究资料维护用的独立Python项目（纯标准库，Python 3.11+，无第三方依赖）。`data/`保存合成记录，`research_data.io`负责严格JSON读取与SHA-256来源摘要，`research_data.domain`提供原始资料统计，`research_data.cli`输出摘要或完整拟合JSON。字段、时间、计数和量纲见[资料约定](docs/data-contract.md)。所有研究、中心及受试者均为虚构，项目不含真实受试者信息、图片或选中代码。

## 解决的问题

不能把重复输注的浓度曲线按“给药后小时”直接叠加——前一剂残留会被误记为新一剂峰值，低于定量限（BLQ）记录也不能填零或当中点。本项目按**实际输注时间**做双指数响应的约束最大似然拟合：

- 每段输注在`[start,end)`以`dose/(end-start)`恒速输入，输注允许重叠，浓度是全部已开始输注与`A*exp(-alpha*s)+B*exp(-beta*s)`的解析分段卷积；
- 指数差用`expm1`与小段级数稳定计算，采样时刻落在输注边界时连续；
- BLQ记录的似然是`Pr(Y<L)=Phi((L-mu)/sigma)`，深尾经`erfcx`渐近展开保持有限，不删除不当中点；
- `(alpha,beta)`只在配置的有限有序候选对上搜索，`(A,B)`在公开的有限非负矩形内优化，报告明确这是**有限搜索域条件下的拟合**，不输出给药建议；
- 全删失或设计矩阵近共线时给不可识别诊断；未收敛候选不静默跳过，无法宣布全域最优；
- 输出每个采样点的逐输注浓度贡献、定量残差或删失概率，以及时间窗上的解析AUC（逐输注分解，窗外不贡献、跨窗长输注正确截取）；
- 保留全部近优候选及预测差异；另用独立复合Simpson数值积分核对卷积与AUC并给理查森误差界。

## 运行

在项目根目录：

```sh
python -m unittest discover -s tests -v
python -m compileall -q research_data tests examples
python -m research_data.cli inspect data/normal.json
python -m research_data.cli fit data/normal.json
python -m research_data.cli fit data/normal.json --verify
python -m examples.synthetic_demo
```

`inspect`只显示已有摘要，不修改来源。`fit`输出完整拟合报告；`--verify`附加独立数值积分核对与误差界。合成演示覆盖单次短输注、重叠输注、长尾残留、大量删失四类场景并报告数值误差界。文件名以`invalid-`起始的资料包含故意不合法的输入，其他文件是同结构的业务边界（`all-below-limit`与`near-equal-rates`用于演示不可识别诊断）。

## 模块

| 模块 | 职责 |
| --- | --- |
| `research_data/validate.py` | 拟合入口资料的严格契约校验 |
| `research_data/convolution.py` | 分段解析卷积、逐输注贡献、解析AUC |
| `research_data/likelihood.py` | 定量/删失高斯似然、稳定log Phi与Mills ratio、解析梯度与Hessian |
| `research_data/optimize.py` | 界约束谱投影梯度+活动集牛顿抛光，报告残差与终止原因 |
| `research_data/diagnostics.py` | 全删失与近共线不可识别诊断 |
| `research_data/fit.py` | 逐候选MLE、决胜、近优保留、逐点报告与AUC编排 |
| `research_data/verification.py` | 独立Simpson数值积分核对与理查森误差界 |
| `examples/synthetic_demo.py` | 四类合成场景端到端演示 |

数据摘要与来源指纹应同时保留，原始JSON不覆盖。可使用固定镜像容器：`docker build -t research-material .`，随后`docker run --rm research-material`查看资料，`docker run --rm research-material python -m unittest discover -s tests -v`运行检查。全部资料在项目内，运行不需要外部账号或服务。
