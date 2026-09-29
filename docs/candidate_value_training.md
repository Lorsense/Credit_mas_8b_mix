# 从未通过资格的价值 checkpoint 启动 16 卡主训练

入口为 `examples/drmas_trainer/run_math.sh`，显式设置 `VALUE_INIT_MODE=candidate`。
适用于离线训练已完成、语义预测有效但熵增益未通过的当前 `prefix_value.pt`。
不需要重新执行离线预训练，不需要重新读取、编码全部历史轨迹。

## 初始化与每一步的顺序

1. 从文件的 `candidate_head` 加载已训练参数并继承固定 scaler。不能使用失败文件中从未发布的 `deployed_head`，不能手动修改 `ready`。两个对照从同一候选权重开始。
2. 初始化新运行的优化器、随机数状态和在线缓存；不把离线 replay 或动量当作主训练断点恢复。编码器始终冻结。启动为 `warm_started=1, ready=0, version=0`，Solver/Verifier 基础和时序控制权限均为零。
3. 当前 Actor 生成最多三轮 Solver 的新轨迹，记录现有归一化 Top-16 action 平均熵。pure 第一阶段按熵排序，第二阶段做同角色轨迹信用分配，零优势恢复/题目课程继续运行。价值模型不替代这两阶段、不改真实终局标签。
4. 在 Actor 更新前，用上一步已发布且冻结的价值头计算控制信息；没有已合格模型时，条件熵约束权重严格为零。控制器仍统计初始全词表熵 cap 和在线趋势。
5. Solver、Verifier 都完成本 batch 的 Actor 更新后，候选价值头才学习新轨迹。仅随机保护的 fresh 样本进入价值训练/留出缓存；课程采样的轨迹只在模型可用时评分，不改变价值训练分布。虚拟最高奖励不作为价值标签。
6. 新训练协议每轮先更新一次语义分支，复制给两个对照，再冻结语义基底。三个熵残差实验使用相同的 minibatch 顺序和 dropout 随机性。无熵对照仅屏蔽熵数值，保留 coverage、长度、有效性和历史可用性。
7. 熵残差训练、验证和实际输出统一限定在“非终局、当前动作所属角色的熵可用”范围。终局和不可用位置回退纯语义预测并关闭该动作熵控制，避免使用未在该范围训练/验证的残差。
8. 在线留出验证通过总体门及相应角色门，候选才发布到部署头，最早从下一 batch 使用。绝对熵资格和时序熵资格分别管理；未获时序资格的角色继续使用绝对分支。资格阈值不降低。
9. Solver、Verifier 各自以默认 20 个有可用预测的合格 batch 渐进启用约束；晚通过的角色从自己的第一个 batch 开始。失去资格重置该角色 ramp，缺少可用动作暂停计数。通过资格并不意味着每个 action 都有惩罚：还需熵风险、价值负进展等条件成立。

价值输出是成功率预测及控制依据，不是熵对成功率的因果估计。若始终没有通过资格，主任务继续 pure 和优势恢复，条件熵约束始终不启用。

## 单机 16 卡命令

在 Linux 服务器中同步更新后的 `Credit_mas_mix` 项目。以下项目、模型、parquet 和 checkpoint 路径需换成实际路径；`VALUE_ENCODER` 须与离线预训练一致。

```bash
cd /workspace/Credit_mas_mix

export SOLVER_MODEL='/models/Qwen3-4B'
export VERIFIER_MODEL='/models/Qwen3-4B'
export VALUE_ENCODER='/models/Qwen3-4B'
export VALUE_CHECKPOINT='/checkpoints/baseline_value_init/prefix_value.pt'
export VALUE_INIT_MODE=candidate

export TRAIN_DATA='/data/drmas_math/train.parquet'
export VAL_DATA='/data/drmas_math/test_sampled.parquet'
export NNODES=1
export TRAIN_BATCH_SIZE=30
export GROUP_SIZE=8
export MAX_SOLVER_TURNS=3
export RUN_NAME='mix_candidate_online_16gpu'
export RUN_DIR="/checkpoints/$RUN_NAME"
export LOGGERS='[console]'
unset RESUME_FROM

# 只检查路径、布局并打印 Hydra 命令，不加载 checkpoint 或启动 Ray。
DRY_RUN=1 bash examples/drmas_trainer/run_math.sh train

# 正式训练：15 个共享双 Agent rank + 1 个独立价值 GPU。
bash examples/drmas_trainer/run_math.sh train
```

两 Agent 共用同一 15 卡资源池但模型参数不共享，不是分别占 15 卡。默认每步 30 题、每题 8 条，共 240 条 rollout；`TRAIN_BATCH_SIZE` 必须为 15 的正整数倍。

`VALUE_CHECKPOINT` 指向已有 `.pt` 文件；`RUN_DIR` 保存新主训练产物，不应指向旧离线文件。`SOLVER_MODEL`/`VERIFIER_MODEL` 指定 Actor 起点；价值 checkpoint 不包含两 Agent 的 Actor 权重。

历史轨迹目录 `/opt/huawei/dataset/RL_Collab/Msc_data/DrMAS-4bnoshare-7a7p5/verl/rollout_trajectories` 本次不需要传入。`TRAIN_DATA` 是主任务问题集 parquet，不能用这批轨迹 JSON 代替。

默认在线价值学习率 `1e-4`、每次更新语义与熵阶段各 1 epoch、batch size 256、在线轨迹缓存 2048。可按需覆盖，例如：

```bash
bash examples/drmas_trainer/run_math.sh train \
  algorithm.entropy_credit.value.learning_rate=0.0001 \
  algorithm.entropy_credit.value.train_epochs=1
```

保持 `algorithm.entropy_credit.value.enable=True`、`algorithm.entropy_credit.control.enabled=True` 和 `actor_rollout_ref.actor.entropy_control.enabled=True`。这些开关保留学习/门控流程，是否施加熵惩罚由运行时资格决定。不需要设置 `require_pretrained=False`。

## 两机各 8 卡

两台机器使用同一环境、同一代码，并能访问相同路径的模型、数据和保存目录。在 head 启动 Ray，再在另一台连接（替换示例 IP）：

```bash
# head，需可用 8 张卡，且给 Ray 留足 CPU。
ray start --head --node-ip-address=10.0.0.1 --port=6379 --num-gpus=8

# 第二台执行。
ray start --address=10.0.0.1:6379 --num-gpus=8

# 回到 head，保留前述训练环境变量，只改：
export NNODES=2
export RAY_ADDRESS=auto
bash examples/drmas_trainer/run_math.sh train
```

资源为 head 的 7 个 Agent rank + 1 张价值卡，第二台 8 个 Agent rank。不要在两台机器上各启动一遍训练入口。

## 日志与资格检查

启动时应看到：

```text
value_model/warm_started_at_training_start = 1
value_model/ready_at_training_start = 0
value_model/prepare/paired_entropy_training = 1
```

开始的 `insufficient_data_retained_deployment` 或 `candidate_rejected` 可以是正常待资格状态，并不意味着没有加载预训练权重。样本达到训练/验证的独立题目和类别覆盖条件后才训练候选；不能保证在固定第几步获得资格。

重点关注：

- `value_model/update/train_trajectories`、`val_trajectories`、`semantic_train_loss`、`train_loss`、`control_train_loss`。
- `value_model/update/candidate_absolute_gain`、`candidate_abs_matched_control_gain`、各角色 absolute/matched/temporal gain。
- `value_model/update/ready`、`version`、`reliability_solver`、`reliability_verifier`、`temporal_reliability_solver`、`temporal_reliability_verifier`。
- `entropy_control/Solver Agent/ramp`、`entropy_control/Verifier Agent/ramp`、对应 `qualified_updates` 和每个 turn 的 `cap`、`risk`、`mean_weight`。

`ready=1` 并不表示两个角色和时序分支都获准；要结合角色权限。每次 `update` 的新资格用于下一轮 Actor。训练 loss 仍是训练阶段均值，不能单凭它宣称熵学到了额外信息。

默认熵 cap 仅用第一个训练 step 标定。某个角色/轮次在初始窗口没有足量动作时，该 scope 会保持关闭；不能把后期已经膨胀的熵重新当作健康上限。资格通过和 cap 标定均需满足，才可能产生对应动作的约束。

## 保存、恢复与评测

默认训练开始前验证一次，之后每 10 step 验证、每 50 step 保存，训练 2 epoch。启动时的验证是主任务答题评测，和价值网络的在线留出资格检查是两套流程。主任务步数可用 `trainer.total_training_steps=...` 覆盖，但不会承诺效果在某一步改善。

恢复完整主训练，不再从离线候选重置：

```bash
export RESUME_FROM="$RUN_DIR/global_step_50"
unset VALUE_CHECKPOINT
bash examples/drmas_trainer/run_math.sh train
```

断点目录包含 Actor、`prefix_value.pt`、`entropy_controller.json`、`advantage_recovery.json` 和数据进度。即使保存时价值仍为 `ready=0`，也能恢复候选训练、缓存、优化器和随机状态。保持模型、资源布局、价值训练和控制配置一致。新 controller v2 保存各角色 ramp；兼容读取旧 v1，但旧文件无法恢复各角色资格历史，因此保留 cap/EMA 并重新渐进启用。

离线评测选定主训练 checkpoint：

```bash
export RESUME_FROM="$RUN_DIR/global_step_200"
export VAL_DATA='/data/drmas_math/test.parquet'
export VAL_GROUP_SIZE=16
export VAL_BATCH_SIZE=60
bash examples/drmas_trainer/run_math.sh eval
```

评测不启动价值 GPU worker，不更新 Actor 或价值头。沿用 15 个 Agent rank 的布局以匹配 Actor checkpoint；默认每题采样 16 条、temperature 0.6、top_p 0.95。结果进入 console/W&B；当前验证代码不导出逐条评测回答，虽然脚本配置了 validation 目录。`TRAIN_DATA` 仍需指向存在的 parquet，因为入口共用数据初始化和路径检查。

默认 `VALUE_INIT_MODE=qualified` 的旧启动方式继续要求已通过离线资格。`candidate` 是新运行的显式初始化方式；恢复训练始终使用整个 `RESUME_FROM` 目录，不能将修改 ready 标志或伪造断点目录当作替代。
