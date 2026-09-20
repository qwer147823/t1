# TMCN：共享—互补分支实验

本分支基于 `codex/tmcn-mnc` 的 MNC + 拼接版本。新增功能默认关闭；不传
`--complementary` 时仍使用原模型结构，可读取已有 `runs/*/model.pth`。
没有跑真实 Hdigit 训练，不保证精度提高，也不宣称严格解耦、因果或公平性。

## 方法与来源

借鉴 [GMAE](https://github.com/obananas/GMAE) 的共享/特异联合重构和共享端
停止梯度的去相关思路；本实现不是 GMAE 的复现，不包含其判别器或对抗训练。
原 TMCN 的 Mamba、AsCL 和现有 DMVCS-inspired MNC 保留。

- `z_v = E_v(x_v)`：512 维编码；原 `hs[v]` 记作 `a_v`，仍参与 AsCL。
- `c = TMCNF(xs)`：128 维共享表示，仍用于 MNC 建图和对比。
- `s_v = normalize(P_v(z_v))`：新增各视图独立 Linear(512,128)，各头不共享参数。
- `joint_xr_v = D_joint_v([c,s_v])`：输入 256 维，经过 2000/500/500 隐层重构输入。
- `L_dec = sum_v mean(((center(c.detach()).T @ center(s_v))/(B-1))**2)`。
  该项是批次/维度归一化的平方交叉协方差，与 GMAE 的原始 L1 实现不同。
- 微调：`L = old_rec_weight*L_rec + L_AsCL + lambda_mnc(t)*L_MNC
  + lambda_joint*L_joint + lambda_dec(t)*L_dec`。
- 最终表示：`[c,s_1,s_2]`，双视图默认仍是 384 维，各块先 L2 归一化。

`detach` 仅阻止去相关损失通过 c 路径直接回传，不冻结共享编码器。
各分支仍共用底层编码器，不能宣称梯度完全隔离。重构不保证保留的都是聚类语义；
低协方差也不保证独立。s 可能退化为常数，或主要保留噪声，需结合诊断和聚类检查。

## 下载与环境

在已有 T1 根目录（不是 TMCN-main 子目录）执行：

```powershell
git fetch origin
git switch --track origin/codex/tmcn-complementary
cd TMCN-main
```

若该分支已经在本地，使用 `git switch codex/tmcn-complementary`。
若 Git 提示有本地修改，先保存/提交，不要强制覆盖。数据沿用现有 `Hdigit.mat`。
无新训练依赖，使用原来的 PyTorch 环境；没有仓库时：

```powershell
git clone --branch codex/tmcn-complementary https://github.com/qwer147823/t1.git
cd t1/TMCN-main
```

## 三组主实验

每条训练命令会创建新目录；已有同名目录时拒绝覆盖，请更换 run_name。
以下均从头训练；**不是从已有 mnc3 检查点继续训练**。
默认重构预训练 200 轮、微调 100 轮、batch=256、lr=0.0003。
预训练始终只有旧重构；`old_rec_weight` 仅影响微调。

### A：原 MNC3 + 拼接对照

```powershell
python train.py --seed 10 --run_name control_mnc3_s10 --lambda_mnc 0.05 --mnc_hops 3
python test.py --run_dir runs/control_mnc3_s10 --feature_mode both
```

也可以直接评估此前保留的检查点，避免重复训练：

```powershell
python test.py --run_dir runs/mnc3_s10 --feature_mode both
```

### B：独立互补分支 + 联合重构，不加去相关

```powershell
python train.py --seed 10 --run_name complement_b_s10 --lambda_mnc 0.05 --mnc_hops 3 --complementary --lambda_joint 1 --lambda_dec 0
python test.py --run_dir runs/complement_b_s10 --feature_mode all
```

### C：B + 弱去相关

```powershell
python train.py --seed 10 --run_name complement_c_s10 --lambda_mnc 0.05 --mnc_hops 3 --complementary --lambda_joint 1 --lambda_dec 100 --dec_start 20 --dec_ramp 20
python test.py --run_dir runs/complement_c_s10 --feature_mode all
```

**100 只是未经真实数据验证的起始系数，不是最优值，也不自动代表约束强。**
这里的表示先 L2 归一化，损失又按批次和特征元素平均，原始数值可能很小。
应查看 `dec_raw`、`dec_weighted`、方差和结果，不可将此系数与其他损失实现直接比较。
需要调参时保持相同种子和其余设置，分别试 1/100/1000；不要只挑一个最高 ACC 报告。
当前不使用自适应权重，不保证不同数据集上约束强度相同。

B/C 新增层初始化不消耗原主干的全局随机流；同种子的原主干初值与批次随机流尽量对齐。
新增目标仍会改变训练轨迹，跨硬件/库版本不保证逐位一致。

## 评估模式与结果文件

| feature_mode | K-means 输入 | 默认双视图维度 |
|---|---|---|
| common | c | 128 |
| concat | [c, 原 hs[0], 原 hs[1]] | 384 |
| complement | [c, 新 s[0], 新 s[1]] | 384 |
| both | common 和 concat | 分别评估 |
| all | 新模型评估三种；旧模型评估前两种 | 分别评估 |

`concat` 的旧含义不变；新表示必须使用 `complement`。
旧检查点请求 complement 会明确报错，不会用随机初始化的新头冒充训练结果。
`test.py` 根据 config.json 选择架构，strict 加载权重，KMeans n_init=100，使用训练 seed。
`train.py` 的 metrics.json 仍只记录 common；最终拼接指标请看
`eval_concat.json` 和 `eval_complement.json`。三个 eval 文件各含表示类型、维度和指标。
训练完先保存 model.pth，再做默认评估，减少评估失败导致权重丢失的风险。

## 日程、诊断和进一步消融

MNC 沿用已有日程：微调 1–10 轮关闭，11–30 轮线性增加到设定最大值。
去相关默认：1–20 轮关闭，21–40 轮线性增加，41 轮起保持；lambda_dec 默认 0。
联合重构从微调第一轮开始。新分支必须配正的 lambda_joint。

`training.jsonl` 每轮记录：
- 旧重构、AsCL、MNC、联合重构、原始/加权去相关、实际调度系数。
- `common_variance`、`specific_variance_0/1`：特征在样本维度的平均方差。
- `mean_specific_rec_gap`：将各 s 替换为本批均值后，联合重构误差的增加量。
- `mean_common_rec_gap`：将 c 替换为本批均值后的误差增加量。

损失是整轮批次均值，方差和替换诊断仅取每轮第一个批次（更新前），不是全数据统计。
替换差值可能为负，仅衡量解码器敏感性，不能证明语义互补或因果贡献。
如果 s 方差接近零、替换 s 几乎不影响重构且新拼接无收益，需警惕分支退化。
若去相关权重过高出现上述现象，先降低系数；不以总损失降低作为改进证据。

D 组检查微调时移除旧重构：复制 C 命令，使用新 run_name 并增加 `--old_rec_weight 0`。
重构预训练仍保持一致。再比较 A/B/C/D 的同一组种子（例如 10/20/30/40/50）均值与标准差。
A 与 C 不足以单独证明去相关贡献；需要 B 与 C 对照，同时承认新增参数和计算量。

## 验证

```powershell
python -m unittest test_mnc test_inference test_complementary -v
```

新增测试验证解析协方差值、停止梯度、主干初始化/随机流保持、旧权重严格加载、
互补表示拼接顺序、联合重构梯度、分支方差诊断，以及合成数据上的
预训练—微调—保存—三模式 KMeans 全流程。合成测试不代表真实聚类精度。
