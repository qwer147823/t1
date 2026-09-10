# TMCN + 邻居对比学习（MNC）

在原始 TMCN 上增加独立、默认关闭的 MNC，保留重建和 AsCL。灵感来自
https://github.com/Lummer-Li/DMVCS ，算法独立实现，未复制其损失代码。
本仓库 TMCN 没有公平/敏感分支：构图使用 commonz，不是 z_fair，不宣称公平或因果保证。

## 方法

对当前 mini-batch 的融合表示归一化，在 no_grad 下选余弦相似度达到阈值的互为 Top-k 邻居。
排除自身；在无向二值图中计算最短距离恰为 1/2/3 的互斥邻居集合，权重为 1/.5/.25。
多正样本交叉熵分母包含所有其他样本；无正邻居的行跳过，整批无邻居时返回可反传的零。
保留 AsCL 原公式和梯度。MNC 在微调前 10 轮关闭，之后 20 轮逐渐增加。
这是 batch 内关系，不是全数据图；多跳可能传播错误关系，增加稠密矩阵乘法开销。

## 运行

在仓库根目录执行：

```bash
cd TMCN-main
pip install -r Requirements.txt
python -m unittest test_mnc -v
python train.py --seed 10 --run_name baseline_s10 --lambda_mnc 0
python train.py --seed 10 --run_name mnc1_s10 --lambda_mnc 0.05 --mnc_hops 1
python train.py --seed 10 --run_name mnc2_s10 --lambda_mnc 0.05 --mnc_hops 2
python train.py --seed 10 --run_name mnc3_s10 --lambda_mnc 0.05 --mnc_hops 3
python test.py --run_dir runs/mnc1_s10
```

数据读取优先使用脚本目录 data/Hdigit.mat，否则读取仓库自带的 Hdigit.mat。
目前数据加载器仅支持 Hdigit。默认训练预算为重建 200 轮、微调 100 轮。
0.05、topk=10、min_sim=.5 是实验起点，不是已验证最优参数。
先比较基线与一跳；若无增益，不要直接堆叠三跳。最终至少比较 3 个相同种子。

每次输出在 runs/<run_name>/：config.json、training.jsonl、metrics.json、model.pth。
同名目录已存在时拒绝覆盖。training.jsonl 记录微调总损失、原始 MNC、当前权重和平均正邻居数。
邻居数长期为零时检查阈值；多跳邻居接近整批时检查错误扩散。
KMeans 使用固定 seed，n_init=100。只评估最终模型，不根据真实标签挑最好 epoch。
网络参数结构未改变。test.py 读取新格式运行目录；旧裸权重没有自动恢复配置功能。

## 验证边界

测试覆盖链式图的精确跳数、互近邻与梯度隔离、空邻居/单样本、解析损失值、
小 batch、权重调度以及原始 TMCN + AsCL + MNC 在 CPU 上的合成数据反向传播。
未执行 Hdigit 训练，未验证 GPU 运行或精度增益。

## 无需重训：拼接各视图投影表示再聚类

```bash
# 在同一个已有 checkpoint 上比较两种聚类特征
python test.py --run_dir runs/baseline_s10 --feature_mode both
python test.py --run_dir runs/mnc1_s10 --feature_mode both
# 或只计算拼接版
python test.py --run_dir runs/mnc1_s10 --feature_mode concat
```

默认 common 使用 commonz；concat 使用 torch.cat([commonz, *hs], dim=1)。
Hdigit 默认由 128 维变成 384 维，支持任意视图数。使用网络本身已归一化的各块，
不额外缩放、不引入权重或后处理，不使用标签选择特征。
每块直接拼接意味着平方欧氏距离为各块距离平方之和：两个视图块合计可能比单个融合块影响更大。
hs 是共享投影层作用于各视图编码的结果，不等于经过 GMAE 解耦的特异因素。

只影响推理/KMeans，不改变训练、AsCL、MNC 构图或 checkpoint 参数格式。
已有模型可以直接复用。训练结束默认仍保存 common 的 metrics.json。
评估分别写 eval_common.json、eval_concat.json；同模式重跑更新对应文件，不覆盖训练指标。
both 使用相同 checkpoint、样本顺序、KMeans seed 和 n_init=100，并打印指标差值。
不同特征空间的初始中心不保证相同，固定 seed 只是使各模式可复现。

建议比较四组：基线/common、基线/concat、MNC/common、MNC/concat。
只有对比 MNC/concat 与基线/concat，才能判断拼接条件下 MNC 的增益。
不保证此前的拼接增益与 MNC 增益可直接相加；不要用测试标签反复挑选特征方案后仍视为无偏评估。
