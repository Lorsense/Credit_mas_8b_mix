# 纯语义价值预测驱动的熵约束

本文对应 `Credit_mas_8b_mix`。核心配置默认仍为 `entropy_aware`，下述 8B 专用入口默认选择 `semantic_only`。新模式使用语义成功概率决定额外熵约束的强度。这不是预测熵数值，也不向 GRPO 优势、PURE 系数或奖励注入语义分数。

## 推荐：修改脚本顶部，然后一条命令启动

打开 [examples/drmas_trainer/run_math_8b_semantic.sh](../examples/drmas_trainer/run_math_8b_semantic.sh)，在顶部配置区直接填写模型、数据和输出路径，不需要再逐条执行 `export`。

| 顶部配置 | 填写内容 |
| --- | --- |
| `SOLVER_MODEL`、`VERIFIER_MODEL` | 两个策略的实际 8B 初始化模型路径 |
| `VALUE_ENCODER` | 价值预训练使用的同一个 8B 编码器路径 |
| `VALUE_CHECKPOINT` | 已有预训练价值模型的 `prefix_value.pt` 文件 |
| `TRAIN_DATA`、`VAL_DATA` | 主训练及训练期间验证的问题 Parquet 文件，不是历史轨迹 JSON 目录 |
| `RUN_NAME`、`RUN_DIR` | 新实验名称及输出目录 |
| `RESUME_CHECKPOINT` | 需要恢复时填写完整的 `global_step_N` 目录 |
| `EVAL_CHECKPOINT`、`EVAL_DATA` | 独立评测使用的 `global_step_N` 目录及测试集 Parquet |

同一配置区也集中保存节点数、Ray 地址、训练/验证 batch、每题轨迹数、Solver 轮数、价值模式、初始化方式、语义控制强度、熵损失系数、训练 epochs、保存和验证频率。起始设置为 `semantic_only + candidate`、语义强度 `0.25`、熵损失系数 `0.01`、训练 2 epochs、每 50 step 保存、每 10 step 验证。按实际路径修改后，在项目根目录执行：

```bash
bash examples/drmas_trainer/run_math_8b_semantic.sh
```

不传模式即为新的主训练。其他入口如下：

```bash
# 仅检查文件与启动参数并打印命令，不加载模型、不启动 Ray
bash examples/drmas_trainer/run_math_8b_semantic.sh check

# 同模式恢复：先填写脚本顶部 RESUME_CHECKPOINT
bash examples/drmas_trainer/run_math_8b_semantic.sh resume

# 独立评测：先填写脚本顶部 EVAL_CHECKPOINT 和 EVAL_DATA
bash examples/drmas_trainer/run_math_8b_semantic.sh eval
```

`check` 不执行价值模型资格复验，不能据此认定 8B checkpoint 已合格或显存足够。新的训练从 `VALUE_CHECKPOINT` 导入预训练候选语义权重；恢复则从 `RESUME_CHECKPOINT` 还原完整在线状态，两种入口不能混为一谈。

单机 16 卡设置 `NNODES=1`。两机各 8 卡时，在顶部设置 `NNODES=2` 和 `RAY_ADDRESS=auto`（或实际 head 地址），并先在两台机器建立同一个 Ray 集群；模型、数据和 checkpoint 路径必须在相关节点可访问。脚本沿用现有 16 卡入口：15 张 Agent 共享资源池加 1 张价值 GPU，两机布局为 `[7,8]`，不会自动建立集群。

专用脚本顶部的直接赋值覆盖同名外部环境变量，避免旧终端配置悄悄改变实验。需要长期修改参数时编辑顶部；临时 Hydra 覆盖可放在模式参数后，例如：

```bash
bash examples/drmas_trainer/run_math_8b_semantic.sh train \
  trainer.total_training_steps=5 trainer.save_freq=2 trainer.test_freq=2
```

该脚本复用原 `run_math.sh` / `run_math_16gpu.sh` 的训练实现，保留 PURE 两阶段、零优势恢复和原有资源设置。下文的环境变量命令是直接使用底层入口的高级方式，与上述专用脚本二选一即可。

## 预测、资格与在线学习

`semantic_only` 直接调用 `sigmoid(head.semantic(features))`。输入为冻结编码器的前缀表示与原有结构状态，不执行绝对熵或时序熵残差。历史两轮轨迹仍保留两轮预算，在线三轮轨迹使用实际三轮预算。

已有失败预训练文件使用 `VALUE_INIT_MODE=candidate` 导入 **candidate_head.semantic**，不会误用未发布的随机 deployed 语义头。必须使用预训练时相同的编码器身份、序列化规则和结构特征；语义头维度从实际编码器配置读取，不能通过改身份或 reshape 强行加载。

如果文件含已编码 replay，启动时复用缓存进行语义资格复验，使用源文件保存的问题哈希划分和训练侧同范围常数先验。该步骤不重新编码历史轨迹、不额外拟合、不修改源文件。旧格式仅在确认原始 schema 后，从 raw absolute block 中迁移动作有效性、截断及 token 数元数据，熵均值、coverage 和 availability 不作为语义资格条件。缺少可靠动作元数据时明确拒绝该复验样本。

控制资格只评估有效、未截断、非空动作之后的完整非终局前缀。initial 和 terminal 不进入这一评估范围，initial 可以用于动作前预测。Solver 和 Verifier 分别检查：

| 配置字段（均位于 `algorithm.entropy_credit.value`） | 默认值 |
| --- | --- |
| `semantic_min_val_prefixes` | 32 |
| `semantic_min_val_questions` | 8 |
| `semantic_min_val_per_class` | 成功、失败各 8 条不同轨迹 |
| `semantic_min_val_auc` | 0.55 |
| `semantic_min_brier_improvement` | 0.0，要求 Brier 严格优于先验 |

一个角色合格即可使当前模式 ready；另一角色仍可无控制权限。熵增益不参与这些检查，时序熵权限保持为零。复验失败或数据不足仍允许 candidate 主训练启动，在线积累代表性 fresh 轨迹后继续训练和评估。源历史 replay、optimizer、RNG 不迁入新的在线实验。

每个 batch 先由冻结部署头打分并完成 Actor 更新，随后当前真实成败标签用于候选语义头 BCE 训练。优化器只包含 `candidate.semantic.parameters()`；编码器、部署头、两个熵残差和两个对照头不更新。候选与部署在相同留出集合、相同非终局语义范围比较，合格且满足替换条件才发布供后续 batch 使用。数据不足保留现有有效部署；足量验证窗口连续失败时按角色暂停控制权限。

两个角色共享同一个部署语义头。发布新权重时，候选还必须在已有授权角色的足量验证范围内合格且不劣于当前部署；已有授权角色数据不足时，暂缓替换这个共享头，继续使用旧部署。尚未授权的另一角色不会阻止合格角色首次发布。这一限制用于避免另一角色的更新覆盖已经验证有效的预测器。

## Actor 约束

```text
D_sem = p_sem_after - p_sem_before
bad_sem = clip((-D_sem - delta_deadzone) / delta_scale, 0, 1)
gate_sem = semantic_role_permission * role_ramp * entropy_risk * bad_sem * semantic_strength
L_semantic_hinge = mean_valid_unique_actions(stopgrad(gate_sem) * relu(H_actor_full - stopgrad(cap)))
L_total = L_existing + loss_coef * L_semantic_hinge
```

沿用全词表熵风险、固定初始 cap、角色独立 ramp 和原有微批/分布式归一化。PURE 仍使用原 top-16 熵统计，不能把 top-16 熵与这里的全词表 cap 混用。语义预测、gate、cap 均不接收 Actor 梯度，语义 BCE 不加入 Actor 损失。

`semantic_strength=0.25` 是默认起点；在其余因子相同且门未因其他条件关闭时，设为 `1.0` 将 gate 和对应熵损失贡献扩大四倍。这与 `actor_rollout_ref.actor.entropy_control.loss_coef=0.01` 是两个独立系数。

打开模式不保证每步惩罚非零：动作还必须有效、角色获得权限、有足够 cap 校准、存在熵风险和语义负向进展，且 Actor 熵超过 cap。默认 `calibration_steps=1`，某角色/轮次在初始窗口样本不足可能一直没有 cap；应检查校准阻断指标，不通过强制 gate 或放宽 cap 制造信号。

## 8B、16 卡主训练

以下路径必须替换为服务器实际路径。`TRAIN_DATA`、`VAL_DATA` 是主任务题目 Parquet，**不是历史 rollout JSON 目录**。

```bash
cd /实际项目路径/Credit_mas_8b_mix
export SOLVER_MODEL='/models/实际8B策略模型'
export VERIFIER_MODEL='/models/实际8B策略模型'
export VALUE_ENCODER='/models/价值预训练所用的同一个8B编码器'
export VALUE_CHECKPOINT='/checkpoints/8b_value/prefix_value.pt'
export VALUE_PREDICTION_MODE=semantic_only
export VALUE_INIT_MODE=candidate
export SEMANTIC_CONTROL_STRENGTH=0.25
export TRAIN_DATA='/data/math/train.parquet'
export VAL_DATA='/data/math/test_sampled.parquet'
export NNODES=1
export TRAIN_BATCH_SIZE=30
export GROUP_SIZE=8
export MAX_SOLVER_TURNS=3
export VAL_BATCH_SIZE=110
export VAL_GROUP_SIZE=1
export RUN_NAME='mix_8b_semantic_only'
export RUN_DIR='/checkpoints/mix_8b_semantic_only'
export LOGGERS='[console,wandb]'
unset RESUME_FROM
unset DRY_RUN

DRY_RUN=1 bash examples/drmas_trainer/run_math.sh train
bash examples/drmas_trainer/run_math.sh train
```

DRY_RUN 检查枚举、强度、文件存在性和资源布局并打印参数，不加载 checkpoint、不运行语义资格复验，也不证明模型能装入显存。完整配置的一致性检查在 Ray 初始化之前执行；不要用 Hydra 的 `+` 强行绕过不存在的配置项。

默认 16 张物理卡中，Solver/Verifier 共用 15-rank 资源池，独立冻结价值编码器占 1 卡。Solver 与 Verifier 仍为两个独立策略。两机各 8 卡时，先在两台机器建立同一个 Ray 集群，并保证路径两侧一致：

```bash
export NNODES=2
export RAY_ADDRESS=auto
bash examples/drmas_trainer/run_math.sh train
```

此时 Actor 布局为 `[7,8]`。资源布局正确不保证 8B 显存足够；若 Actor 微批导致 OOM，可明确覆盖为每卡 1 条（其余算法参数保持原值）：

```bash
bash examples/drmas_trainer/run_math.sh train \
  '+agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=[1,1]' \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
```

这不解决所有可能的 rollout、长序列或编码器 OOM，应按实际报错定位。默认启动前验证，随后每 10 step 验证、每 50 step 保存，训练 2 epochs；总步数取决于实际训练数据。

## 同模式恢复与独立评测

保留上述同一模型、模式、强度及数据配置，在实际完整 checkpoint 目录恢复：

```bash
export RESUME_FROM="$RUN_DIR/global_step_50"
bash examples/drmas_trainer/run_math.sh train
```

恢复完整 Actor、价值候选/部署、optimizer、RNG、replay、角色权限、版本和坏窗口，以及 controller 的校准/ramp 状态。临时 ready=False 不妨碍同模式恢复。恢复不执行离线初始化、不清空缓存、不重置 ramp。不同 prediction_mode 的 exact resume 明确拒绝；换模式必须新建实验并显式导入权重。

独立评测：

```bash
export RESUME_FROM="$RUN_DIR/global_step_实际步数"
export VAL_DATA='/data/math/test.parquet'
export VAL_BATCH_SIZE=60
export VAL_GROUP_SIZE=16
bash examples/drmas_trainer/run_math.sh eval
```

评测只加载 Actor 完成验证，不创建价值训练 worker、不更新 Actor 或价值头。现有入口仍检查 TRAIN_DATA 文件，因此保留有效的 TRAIN_DATA 设置。评测后的训练应恢复训练用 VAL_DATA/VAL_GROUP_SIZE。

未来获得合格的熵感知 checkpoint 时，可在新实验中恢复旧行为：

```bash
export VALUE_PREDICTION_MODE=entropy_aware
export VALUE_INIT_MODE=qualified
export VALUE_CHECKPOINT='/checkpoints/qualified_entropy_value/prefix_value.pt'
export RUN_NAME='mix_8b_entropy_aware'
export RUN_DIR='/checkpoints/mix_8b_entropy_aware'
unset RESUME_FROM
bash examples/drmas_trainer/run_math.sh train
```

## 如何确认它在工作

必须依次区分训练、质量、部署、gate 和实际损失，不能仅凭训练 loss 下降判断 Actor 已受约束。

| 观察对象 | 主要指标 |
| --- | --- |
| 在线语义训练 | `value_model/update/semantic_train_loss`、`semantic_train_prefixes` |
| 非终局语义质量 | `value_model/update/candidate_sem_nonterminal_brier`、`candidate_sem_nonterminal_prior_brier`、`candidate_sem_nonterminal_auc` |
| 分角色质量 | `value_model/update/candidate_solver_sem_brier`、`candidate_solver_sem_prior_brier`、`candidate_solver_sem_auc`、`candidate_solver_sem_prefixes`；Verifier 对应字段 |
| 当前实际部署 | `value_model/score/semantic_only`、`semantic_ready`、`semantic_reliability_solver`、`semantic_reliability_verifier`、`version` |
| Actor 接收到权重 | `actor/{wg_id}/entropy_control/mean_gate` |
| 实际触发约束 | `actor/{wg_id}/entropy_control/active_fraction`、`hinge_loss`、`semantic_hinge_loss` |

表内省略前缀的指标沿用同一行第一个指标的目录前缀。启动复验详细结果见 `Value initialization report` 和 `value_model/startup/*`。保留原 `candidate_sem_brier` 的总体前缀口径，它不能替代新的控制相关范围指标。

同时查看 `entropy_control/*` 的语义覆盖率、负向进展比例、权限/ramp、风险、超过 cap 的比例和阻断原因。`semantic_hinge_loss` 是纯语义模式下现有 hinge 的标识，不会重复加入损失。Actor 日志记录真实 `loss_coef` 与 `semantic_strength`。小量使用 W&B/原始日志精度，不把控制台三位小数的 `0.000` 当成严格零。

本地合成测试不等于真实集群验证，实际 8B checkpoint 能否启动通过资格须以服务器复验报告为准；语义质量合格也不能保证熵约束提高最终任务成功率。集群验证执行说明见 [semantic_only_cluster_validation_prompt.md](semantic_only_cluster_validation_prompt.md)。

## 本次修改与本地验证

2026-09-29 完成相关 CPU 回归测试：**496 passed**，包含真实 PyTorch Actor 反向传播和 Bash/Hydra 配置检查。使用小型编码器与合成缓存，没有下载 8B 模型或启动 Ray/GPU 集群。递归的全库测试依赖本机没有的 Ray/GPU 环境，不在本次已通过范围内。

在装有项目测试依赖的 Linux 环境中，可复现此次测试集合：

```bash
python -m pytest tests/utils/test_*.py \
  tests/workers/test_credit_value.py \
  tests/workers/test_credit_value_absolute_init.py \
  tests/workers/test_credit_value_curriculum.py \
  tests/workers/test_credit_value_warm_start.py \
  tests/workers/test_semantic_value_worker.py \
  tests/trainer/ppo/test_mix_entropy_value_integration.py \
  tests/trainer/ppo/test_sparse_entropy_credit_integration.py \
  tests/trainer/ppo/test_advantage_recovery_integration.py \
  tests/trainer/ppo/test_recovery_checkpoint.py -q -p no:cacheprovider
```

修改范围：

| 文件 | 作用 |
| --- | --- |
| `verl/workers/credit_value.py` | 语义前向、缓存复验、独立资格、在线训练和状态保存恢复 |
| `verl/utils/value_credit.py` | 无熵依赖的记录与预测附加、显式控制来源和可用性 |
| `verl/utils/entropy_control.py` | 语义门控、阻断诊断、控制器恢复校验 |
| `verl/workers/actor/dp_actor.py` | 语义 hinge 指标与实际系数；保留原损失归一化 |
| `verl/trainer/ppo/ray_trainer.py` | 模式、角色权限、预测、Actor 元信息及启动报告接入 |
| `verl/trainer/main_ppo.py` | 分配 Ray 资源前校验配置一致性 |
| `verl/trainer/config/ppo_trainer.yaml` | 两个模式字段、语义强度与独立资格阈值 |
| `examples/drmas_trainer/run_math_16gpu.sh` | 环境变量接入与校验，保持 16 卡布局 |

新增 `tests/workers/test_semantic_value_worker.py` 和 `tests/utils/test_semantic_control.py`；扩展 `test_entropy_actor.py`、`test_mix_16gpu_launch.py`、`test_mix_entropy_value_integration.py`、`test_advantage_recovery_integration.py`。新增本文与集群验证 prompt。原 `Credit_mas_mix` 的 645 个非缓存、非 Git 文件在操作前后的内容摘要一致。
