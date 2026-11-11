# 浓度与输注记录

units固定time=hour、dose=mg、concentration=mg/L。infusions的start/end按同一研究起点，end严格大于start，dose为正。samples的time非负；kind为quantified时value为非负定量浓度，为below_limit时只给正limit，sigma是这条观测已知的正测量标准差。各观测独立，未量化表示Y<L而不是区间中点。

rate_pairs中的每一对按(alpha,beta)解释，alpha>beta>0，候选按文件编号稳定决胜；幅度A/B的矩形上界有限且正。auc_windows按[lo,hi]积分，端点单点不改变积分。near_optimal_gap施加在负对数似然差上，不是概率覆盖声明。

现有domain只清点剂量与观测类型，不求浓度响应、似然或AUC。
