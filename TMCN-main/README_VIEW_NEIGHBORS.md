# TMCN：各视图的一跳邻域对比实验

在 `codex/tmcn-reliable-mnc` 基础上新增可选视图损失，直接优化最终拼接使用的
`hs[0], hs[1], ...`。支持任意视图数量，不新增网络参数；原来的 Mamba、AsCL、
共享投影层、共享 MNC、辅助互补分支和加权拼接均保留。

本次只做代码与合成数据验证，没有完整训练 Hdigit，不保证超过已报告的 98.09%。

## 1. 更新代码

在 VS Code 的 PowerShell 终端执行：

```powershell
cd C:\Users\20322\t1
git fetch origin
git switch --track origin/codex/tmcn-view-neighbors
cd TMCN-main
conda activate wxh
```

分支已在本地时，使用 `git switch codex/tmcn-view-neighbors` 和 `git pull --ff-only`。
若提示本地修改冲突，先保存或提交修改。无需重新安装依赖，已有 `runs/` 可以继续使用。

## 2. 新增损失做什么

所有邻居均在当前 batch 内建立，使用余弦相似度、互为 KNN 和相似度阈值，固定一跳：

- `shared`：用停止梯度的共享表示 c 建一跳图，每个 h_v 分别使用这张图做视图内对比。
- `consensus`：每个 h_v 也建立一跳图，只保留共享图和当前视图图的交集。
  这不是两个视图相互取交集，也不使用真实标签。

`L_view = mean_v neighbor_contrastive_loss(h_v, W_v)`，
`L_total = L_previous + lambda_view(t) * L_view`。
损失按视图数量取平均，避免视图增多时直接按数量放大新增损失。

建图与交集判断停止梯度；对比损失中的 h_v 保留梯度，更新原投影层和对应编码器。
此项损失不直接回传到 Mamba 或共享投影层，但共同使用的编码器改变后，c 仍可能间接改变。
当前 h_v 不是已证明解耦的私有因素；此项损失也不保证保留所有互补信息。

有正邻居的样本参与该视图的损失平均；无正邻居样本不作为有效 anchor。
整个视图没有正邻居时，该视图贡献可微的零损失，仍计入视图平均的分母。
未选中的非自身样本仍在对比分母中，并不是全部忽略。

新增参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--lambda_view` | 0 | 新损失最大权重；0 完全关闭，保留旧训练计算路径 |
| `--view_neighbor_mode` | shared | shared 或 consensus |
| `--view_topk` | 10 | 新增视图损失所用共享图/视图图的 KNN 数量 |
| `--view_min_sim` | 0.5 | 上述两类图的余弦相似度阈值 |
| `--view_temperature` | 0.5 | 新增损失的温度 |
| `--view_start` | 10 | 微调阶段前 10 轮不启用新损失 |
| `--view_ramp` | 20 | 接下来 20 轮线性增加到最大权重 |

新增图参数独立于原来的 `--mnc_*`。例如共享 MNC 仍可用三跳，新视图损失保持一跳。
`lambda_mnc=0` 时也可单独启用视图损失。不要把 `--mnc_feature_mode weighted_concat`
误当作此项新损失：前者是对整个融合向量计算一次 MNC，后者是逐视图计算后取平均。

## 3. 推荐 A/B/C 对照：沿用你已有模型的配置

`run_view_ablation.py` 读取旧运行的 `config.json`，保留原来的种子、MNC、辅助重构、
去相关、训练轮数和其他参数，只修改新运行名称与新视图损失设置。
**全部从头训练，不加载旧模型权重，不覆盖旧运行。** 源配置只读。
源配置没有的新视图字段使用本节默认值；`--seed` 可覆盖种子用于重复实验。

以下以你之前用于加权拼接评估的运行目录为例。
若你的实际最佳运行在别的目录，只需将三条命令的 `--source_run` 一并替换。

先查看命令，不训练：

```powershell
python run_view_ablation.py --source_run runs/complement_c_smalljoint_d1000_s10 --run_name view_shared_s10 --mode shared --dry_run
```

A：关闭新损失，重新运行同配置对照：

```powershell
python run_view_ablation.py --source_run runs/complement_c_smalljoint_d1000_s10 --run_name view_control_s10 --mode off
```

B：共享一跳邻居分别指导各视图，最大权重 0.01：

```powershell
python run_view_ablation.py --source_run runs/complement_c_smalljoint_d1000_s10 --run_name view_shared_s10 --mode shared --lambda_view 0.01
```

C：共享图与本视图图取交集，其余同 B：

```powershell
python run_view_ablation.py --source_run runs/complement_c_smalljoint_d1000_s10 --run_name view_consensus_s10 --mode consensus --lambda_view 0.01
```

如果旧运行与 A 的配置和代码计算路径一致，可先评估旧模型，避免重复训练 A；
正式重复实验时仍应为每个方法使用相同种子，例如 10/20/30/40/50。
权重 0.01 是待验证的实验起点，不是已验证的最优值。

也可直接使用原训练入口；例如下面命令只包含共享 MNC 和新增视图损失，**未启用辅助分支**：

```powershell
python train.py --seed 10 --run_name view_mnconly_s10 --lambda_mnc 0.05 --mnc_hops 3 --lambda_view 0.01 --view_neighbor_mode shared
```

## 4. 评估融合表示与每个视图

三组都使用相同的推理权重 alpha=0.2：

```powershell
python test.py --run_dir runs/view_control_s10 --feature_mode weighted_concat --fusion_alpha 0.2
python test.py --run_dir runs/view_shared_s10 --feature_mode weighted_concat --fusion_alpha 0.2
python test.py --run_dir runs/view_consensus_s10 --feature_mode weighted_concat --fusion_alpha 0.2
```

新增 `views` 模式分别评估每个原始 h_v，不是辅助 s_v：

```powershell
python test.py --run_dir runs/view_control_s10 --feature_mode views
python test.py --run_dir runs/view_shared_s10 --feature_mode views
python test.py --run_dir runs/view_consensus_s10 --feature_mode views
```

也支持旧检查点，无需重训。输出 `eval_view_1.json`、`eval_view_2.json` 等，以及汇总
`eval_views.json`；编号从 1 开始，`view_1` 对应 `hs[0]`。KMeans 沿用训练种子、n_init=100。
不会改写 `metrics.json` 或原 concat/weighted_concat 报告。
`--feature_mode all` 保持原有 common/concat/complement 含义，不自动运行单视图评估。
训练结束的 `metrics.json` 仍然是 common 指标，不是最终加权拼接指标。

## 5. 日志与结果判断

`training.jsonl` 新增：

- `view_loss`、`view_weight`、`view_weighted`：原始视图平均损失、当前权重、加权贡献。
- `view_loss_1/2/...`：各视图原始损失。
- `view_common_neighbors`：新的一跳共享图中，每个样本平均邻居数。
- `view_neighbors_1/2/...`：各视图实际使用的平均邻居数。
- `view_active_anchor_fraction_1/2/...`：各视图有正邻居的样本比例。
- `view_neighbor_retention_1/2/...`：保留的边数相对于该一跳共享图的比例。

这些诊断取整轮 batch 平均；关闭/预热时前三项为零，其他新增图诊断不计算。
consensus 若长期几乎没有有效 anchor，应先检查筛选是否太严，不应直接增大损失权重。
单视图分数上升而最终融合下降，也不能算总体改进。固定推理权重，比较多种子
ACC/NMI/ARI/PUR 均值和波动后再决定是否保留新项。

## 6. 验证命令

```powershell
python -m unittest test_mnc test_inference test_complementary test_fusion_mnc test_view_neighbors -v
```

测试覆盖图交集、梯度、空邻居、平均尺度、旧检查点评估，以及合成数据上的
预训练、微调、保存、重新加载和聚类。合成数据通过不能证明真实数据上会涨点。
