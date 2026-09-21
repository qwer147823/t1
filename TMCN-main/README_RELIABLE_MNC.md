# TMCN：加权拼接、高阶邻居筛选与融合空间 MNC

本分支基于 `codex/tmcn-complementary`，保留旧模型、旧配置和旧评估模式。
新实验默认关闭，不修改网络参数结构，不自动启用互补分支。
没有进行真实 Hdigit 训练；这些是待验证的改进，不保证超过 98.04%。

## 更新代码

在 PowerShell 中执行（分支首次下载）：

```powershell
cd C:\Users\20322\t1
git fetch origin
git switch --track origin/codex/tmcn-reliable-mnc
cd TMCN-main
conda activate wxh
```

如果本地已有该分支，使用 `git switch codex/tmcn-reliable-mnc` 后
`git pull --ff-only`。若 Git 提示本地修改冲突，先保存/提交修改，不要强制覆盖。
原 `runs/` 和 `Hdigit.mat` 继续使用；无新增训练依赖。

## 1. 无需训练：加权拼接

原 `concat=[c,h1,h2]` 完全保留。新增 `weighted_concat` 使用原模型输出的
L2 归一化块：

`F = [sqrt(alpha)*c, sqrt((1-alpha)/V)*h1, ..., sqrt((1-alpha)/V)*hV]`

其中 V 是视图数，alpha 是共享块在平方欧氏距离中的权重，而不是直接乘特征的系数。
双视图默认 alpha=1/3，等同等权拼接的整体缩放；理论聚类目标等价，但浮点实现
不保证 KMeans 结果逐位一致。alpha=0/1 分别只保留视图块/共享块的距离贡献。
这是推理消融，不是新训练层，hs 也不代表已证实解耦的私有因素。

评估你已有的 98.04% 检查点：

```powershell
python test.py --run_dir runs/complement_c_smalljoint_d1000_s10 --feature_mode concat
python test.py --run_dir runs/complement_c_smalljoint_d1000_s10 --feature_mode weighted_concat --fusion_alpha 0.2 0.3333333333333333 0.5 0.7
```

也可将 run_dir 换为 `runs/mnc3_s10`。不会重训或改动 model.pth/config.json。
KMeans 仍用训练种子、n_init=100，不根据真实标签自动选择 alpha。
每个 alpha 分别保存，如 `eval_weighted_concat_alpha_0p5.json`，不覆盖
`eval_concat.json`；`eval_weighted_concat_sweep.json` 汇总本次调用的权重和指标。
重复评估相同 alpha 会更新对应报告，汇总文件也会更新。

未传 --fusion_alpha 时，若检查点以 weighted_concat MNC 训练则继承保存的训练权重，
否则采用 1/(V+1)。显式参数可覆盖以做消融。common/concat/complement/both/all
维持原含义；all 不自动加权扫描。--fusion_alpha 只接受 weighted_concat。

## 2. 高阶邻居筛选：先跑这一组

原流程：共享表示 c → 互为 KNN + 一跳阈值 → 纯 1/2/3 跳 → 权重 1/0.5/0.25。
新增 `--mnc_endpoint_min_sim`，只对二、三跳候选的两个端点再次检查余弦相似度。
一跳不变；筛掉的候选不会被重分配到更远跳数。图、阈值判断和权重均停止梯度。
当前仍是 batch 内图，不是全数据图，也没有新增跨视图置信度模块。

对照 A（已有同设置 mnc3_s10 可以直接评估，避免重复训练）：

```powershell
python test.py --run_dir runs/mnc3_s10 --feature_mode both
```

需要从头复现 A 时：

```powershell
python train.py --seed 10 --run_name reliable_control_s10 --lambda_mnc 0.05 --mnc_hops 3
python test.py --run_dir runs/reliable_control_s10 --feature_mode both
```

B：只加高阶端点筛选，其余保持 A：

```powershell
python train.py --seed 10 --run_name reliable_filter_s10 --lambda_mnc 0.05 --mnc_hops 3 --mnc_endpoint_min_sim 0.5
python test.py --run_dir runs/reliable_filter_s10 --feature_mode both
```

0.5 是便于与一跳默认阈值对齐的实验起点，不是验证过的最优值。不要同时改 batch、
topk、温度或 MNC 权重。该开关要求 hops>=2、lambda_mnc>0，默认 None 表示关闭。

## 3. MNC 作用空间消融

`--mnc_feature_mode` 可选：

| 模式 | 建图表示 | 对比损失输入 | 对应评估模式 |
|---|---|---|---|
| common（默认） | c.detach() | c | common 和 concat 都应报告 |
| concat | c.detach() | [c,h1,h2] | concat |
| weighted_concat | c.detach() | 加权 F | weighted_concat，使用同一 alpha |

MNC 内部仍对整个输入向量 L2 归一化并计算余弦对比；weighted_concat 使用与评估
相同的块权重。AsCL、重构、Mamba 和调度不变。这里只切换同一项 MNC 的输入，
不会把共享 MNC 与融合 MNC 两项重复相加；损失系数相同不代表梯度强度相同。

C：仅将 MNC 从 c 改到原始 concat，不加端点筛选：

```powershell
python train.py --seed 10 --run_name reliable_concat_s10 --lambda_mnc 0.05 --mnc_hops 3 --mnc_feature_mode concat
python test.py --run_dir runs/reliable_concat_s10 --feature_mode both
```

D：组合 B 和 C，单独验证两者是否兼容：

```powershell
python train.py --seed 10 --run_name reliable_filter_concat_s10 --lambda_mnc 0.05 --mnc_hops 3 --mnc_endpoint_min_sim 0.5 --mnc_feature_mode concat
python test.py --run_dir runs/reliable_filter_concat_s10 --feature_mode both
```

可选 E：在 D 基础上使用预先指定的加权几何（0.5 仅为示例）：

```powershell
python train.py --seed 10 --run_name reliable_weighted_s10 --lambda_mnc 0.05 --mnc_hops 3 --mnc_endpoint_min_sim 0.5 --mnc_feature_mode weighted_concat --fusion_alpha 0.5
python test.py --run_dir runs/reliable_weighted_s10 --feature_mode weighted_concat
python test.py --run_dir runs/reliable_weighted_s10 --feature_mode both
```

建议先推理消融和 B，再按结果决定是否跑 C/D/E。所有 train.py 命令均从头训练，
默认重构预训练 200 轮、微调 100 轮，不会从旧 checkpoint 继续训练。
同名 run_name 已存在时拒绝覆盖。上述训练不启用联合重构或去相关，以隔离改动。
旧互补模型仍可加载；本次没有更改其损失或推荐继续增大去相关权重。

## 4. 日志与比较

新增开关写入 config.json。MNC 激活后，training.jsonl 每轮记录全部批次平均的：

- `neighbors_candidate_hop1/2/3`：筛选前每个样本的各跳邻居数（按配置跳数记录）。
- `neighbors_kept_hop1/2/3`：筛选后数量；一跳应与候选数一致。
- `mnc_active_anchor_fraction`：至少存在一个正邻居的样本比例。
- `neighbors`：筛选后总邻居数，沿用原字段。

MNC 未启用的预热轮不生成各跳诊断字段。邻居数量与聚类质量没有单调关系，
筛选可能过严，也可能仍保留错误邻居；应结合多种子聚类结果判断。

**train.py 的 metrics.json 仍是 common 结果。** concat 应看 eval_concat.json；
加权结果看带 alpha 的文件。不能拿不同评估模式的 ACC 当作单个模块的提升。

固定方案后至少用相同的 5 个种子（例如 10/20/30/40/50）比较 A/B/C/D，
报告均值和标准差。若按真实标签挑权重，要注明调参协议，并在其他数据上验证。
当前 dataloader 仍仅支持 Hdigit，未宣称可直接用上述命令运行其他数据集。

## 验证

```powershell
python -m unittest test_mnc test_inference test_complementary test_fusion_mnc -v
```

新增检查覆盖加权距离解析值、默认等权几何、端点筛选的纯跳数与对称性、空邻居、
梯度路由、旧 checkpoint 配置回退、权重扫描报告隔离，以及合成数据的训练—保存—
加载—聚类。合成数据测试不能证明真实数据精度。
