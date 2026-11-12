# 浓度与输注记录

## 输入契约

units固定time=hour、dose=mg、concentration=mg/L。infusions的start/end按同一研究起点，end严格大于start，dose为正。samples的time非负；kind为quantified时value为非负定量浓度，为below_limit时只给正limit，sigma是这条观测已知的正测量标准差。各观测独立，未量化表示Y<L而不是区间中点，拟合中其似然为Pr(Y<L)=Phi((L-mu)/sigma)，不能填成L/2、0或删除。

rate_pairs中的每一对按(alpha,beta)解释，alpha>beta>0，候选按文件编号稳定决胜；搜索域就是这份有限有序候选列表，不做连续空间外推。幅度A/B的矩形amplitude_bounds下界有限非负、上界严格大于下界。auc_windows按[lo,hi]积分，端点单点不改变积分。near_optimal_gap施加在负对数似然差上，不是概率覆盖声明。solver.tolerance为正、max_iterations为正整数。

非有限数值（NaN/Infinity，含JSON常量）、布尔冒充数值、负定量值、未知观测类型、未声明或不匹配的单位、重复id、空候选列表均在读取或校验阶段拒绝。

## 模型与卷积

单位瞬时剂量在s>=0后的响应为 g(s)=A*exp(-alpha*s)+B*exp(-beta*s)，A、B非负，alpha>beta>0。输注i在[start_i,end_i)以r_i=dose_i/(end_i-start_i)恒速输入，输注允许重叠；时刻t的浓度是全部已开始输注与g的卷积和，尚未开始的输注贡献精确为0。积分按输注边界解析分段，1-exp(-x)类差值用expm1与小段泰勒级数稳定计算，采样时刻落在边界时按[start,end)单侧极限连续取值。

设计矩阵两列分别为全部输注对alpha项、beta项的速率加权积分；预测mu=A*X_alpha+B*X_beta。列归一化Gram矩阵条件数超过1e8判为近共线，配合全部删失两类不可识别情况输出诊断，不给出伪精确点估计。

## 似然与优化

定量观测给出高斯负对数似然（含log sigma常数项），删失观测给出-log Phi(z)，深尾经erfcx渐近展开保持有限。负对数似然关于(A,B)凸，在幅度矩形上用谱投影梯度（Barzilai-Borwein步长、非单调线搜索）加活动集牛顿抛光求约束MLE，报告目标值、命中边界、投影梯度最优性残差、迭代次数与终止原因。逐点负对数似然之和与优化目标互相核对。

候选按最优目标升序、目标并列按候选编号决胜；未收敛或不可识别候选保留在报告中并阻止宣布全域最优。near_optimal_gap内的全部候选连同预测浓度差与AUC差一起保留。报告明确标注为有限搜索域条件下的拟合，不输出给药建议。

## AUC与数值核对

每个auc_window上的AUC由原函数J_lam(s)=(s-(1-exp(-lam*s))/lam)/lam解析积分，逐输注、逐指数项分解后求和。窗外输注不贡献；跨窗长输注截取与窗口的交集，窗外输入衰减进窗口的残留保留。独立核对使用不引用解析公式的复合Simpson数值积分（O(h^4)），以两级网格理查森外推给经验误差界；删失概率另用密度函数Simpson积分核对。
