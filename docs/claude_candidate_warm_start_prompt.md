你是一名负责多 Agent 强化学习训练框架的高级工程师。请直接在服务器上的 Credit_mas_mix 项目实现下面的修改，完成代码、配置、启动脚本、测试与文档。请先读取当前代码确认已有接口，已经存在的功能核对后保留，缺失部分再补齐。不要只给设计建议。

项目根目录：以当前服务器工作目录中的 Credit_mas_mix 仓库为准，不要把 Windows 路径写进代码。先确认存在 verl/workers/credit_value.py 和 examples/drmas_trainer/run_math_16gpu.sh，避免修改 baseline、pure、sup 或其他副本。

本任务是复刻一次范围明确的增量修改：允许使用未通过熵资格的离线候选 checkpoint 初始化主训练，候选在线继续学习，经过原有资格检查才发布并逐步参与熵控制。不是重写此前的 pure、优势恢复或价值预训练方案。

一、背景与必须达到的行为

已有离线 prefix_value.pt，来自约 185318 条 baseline 轨迹；历史 Solver 最大预算 2 轮，主训练预算 3 轮。离线模式是 absolute，语义分支有预测能力，但熵增益未通过，报告为 ready=0、version=0、candidate_rejected。示例指标：candidate_brier=0.149733，candidate_sem_brier=0.149700，candidate_absolute_gain=-0.00005345，candidate_matched_control_gain=-0.00192019。

该 checkpoint 中 candidate_head 已训练；deployed_head 只有在资格检查通过后才会收到 candidate 的参数。version=0 的失败文件通常仍保存初始化时的 deployed_head。因此必须从 candidate_head 导入已有知识，绝不能只修改 ready/pretrained 标志后沿用旧 fresh load。

新增显式候选初始化入口，预期命令为：

    VALUE_INIT_MODE=candidate VALUE_CHECKPOINT=/实际路径/prefix_value.pt \
      bash examples/drmas_trainer/run_math.sh train

启动后必须满足：warm_started=True、ready=False、version=0，Solver 和 Verifier 的基础及时间序列熵控制权限均为零。主任务正常 rollout 和 Actor 更新，价值候选继续学习；未取得资格时，价值控制的 Actor 熵惩罚严格为零。通过原有总体和对应角色门后，最早下一 batch 使用新发布的模型。

默认旧入口保持 VALUE_INIT_MODE=qualified，仍要求已通过离线资格。不要把“准许开始在线学习”混同为“允许立即控制 Actor”。

二、修改文件范围

核心文件：

    verl/workers/credit_value.py
    verl/utils/entropy_control.py
    verl/trainer/ppo/ray_trainer.py
    verl/trainer/config/ppo_trainer.yaml
    examples/drmas_trainer/run_math_16gpu.sh

测试文件：

    新增 tests/workers/test_credit_value_warm_start.py
    扩展 tests/utils/test_entropy_control.py
    扩展 tests/utils/test_mix_16gpu_launch.py
    扩展 tests/trainer/ppo/test_mix_entropy_value_integration.py

文档：新增 docs/candidate_value_training.md；从 docs/entropy_value_mix.md 和 docs/baseline_value_pretrain.md 链接到新说明。

如果服务器文件结构略有差异，按实际职责定位等价位置并在最终报告中说明。不要覆盖仓库中与本任务无关的现有修改。

三、价值 worker：显式候选初始化与兼容

将加载接口扩展为：

    load(self, path, *, resume=False, warm_start=False)

两个标志必须为 bool；resume=True 与 warm_start=True 同时出现应报错。三种模式必须分清：普通 qualified fresh load、候选 warm start、完整运行 resume。

新增 worker 状态：

    training_protocol = "legacy"
    warm_started = False

新增协议常量：

    _PAIRED_TRAINING_PROTOCOL = "paired_nonterminal_v1"

维持价值 checkpoint_version=2 和原 FEATURE_SCHEMA，不改变头的参数形状。旧 checkpoint 缺少新增字段时，解释为 legacy 和 False；未知 training_protocol 或非 bool 的 warm_started 应报错。正常旧文件加载、旧运行 resume 不能被悄悄升级成新训练协议。

warm_start=True 时仍执行原来的兼容性校验：checkpoint 版本、schema、encoder_identity、头维度、holdout_salt、validation_fraction、min_entropy_coverage、scaler_clip/scaler_floor、特征顺序/维度、scaler 有限值和正 scale、角色权限及部署阶段一致性等。旧 sup 单头文件仍不兼容。只豁免普通 fresh load 对 ready && pretrained 的要求，不能取消其他校验。

候选初始化必须：

1. 从 state["candidate_head"] 加载 candidate_head 的完整参数，包括语义、绝对熵和时序熵分支。
2. 两个对照 control_head、temporal_control_head 也从同一份 candidate_head 参数初始化，避免继承旧离线对照各自不同的训练起点。
3. 深拷贝原固定 scaler，不重新拟合，不对 replay 特征重复缩放。
4. encoder 保持 eval、requires_grad=False；deployed_head 保持冻结且不参与当前预测。候选与对照保持可训练。只有正常资格发布路径能让候选成为部署模型。
5. 设置 training_protocol="paired_nonterminal_v1"、warm_started=True、deployment_mode="absolute_bootstrap"；保留源文件 pretraining_mode，当前文件应为 absolute。
6. 设置 ready=False、pretrained=False，version=step=bad_windows=0；清零 solver/verifier 的 reliability、temporal_reliability、role_bad_windows、temporal_bad_windows。
7. last_validation_fingerprint=None，_absolute_pretrain_active=False，清空 _pending、_replay，重置 last_metrics 为新运行状态。
8. 按当前主训练配置的 seed 重建 worker 私有 CPU torch.Generator；按当前 learning_rate、weight_decay 重建三个 AdamW 优化器。不要加载离线 optimizer、replay、pending 或 RNG 进度。允许主训练使用不同于离线的学习率和缓存容量。
9. 不修改输入 .pt，也不重新运行历史数据编码。

load() 返回值和 prepare() 顶层返回值都必须包含 bool warm_started。特别是 prepare([]) 也必须提供，不能只放进 metrics。部署 metrics 中增加数值 warm_started 和 paired_entropy_training。

save() 保存 training_protocol 和 warm_started。resume=True 必须恢复全部原运行状态，包括候选/部署/两个对照、三个优化器、scaler、缓存、RNG、step、version、资格、坏窗口、验证指纹及新增协议字段。保留原严格恢复配置检查；不能调用 warm-start 重置逻辑。

保留 pretrained 的原有“离线已合格”含义：候选初始化时为 False，不需要为了允许主训练而改为 True。在线发布通过 ready/version 和角色权限表达。完整主训练恢复允许 pretrained=False、ready=False。此任务不新增“将在线产物标为离线合格文件”的转换流程。

四、新协议的在线学习：共享语义与配对随机性

仅 training_protocol="paired_nonterminal_v1" 使用新的在线训练路径；legacy 保持原行为。可新增 _train_paired_online(rows, epochs)，由 update() 在样本达到原 _enough 条件后调用。

每次在线更新依次进行：

1. 仅 candidate 的 semantic 分支训练一次，使用现有真实终局二元标签和 BCE。
2. 将刚更新的 semantic 参数复制到两个对照的 semantic 分支。
3. 冻结这三套相同的 semantic 参数，并把 semantic 置于 eval 模式，开始熵残差训练。
4. 候选、无熵对照、无时序熵对照使用同样的数据、epoch、minibatch 顺序与 dropout 随机性。仅输入消融不同。
5. 三个残差头及各自优化器保留在线学习历史，不要每步重置它们；语义基础每次重新同步。

私有 RNG 配对实现应等价于：

    semantic_loss = self._train(rows, epochs, "semantic")
    # copy candidate semantic to both controls
    paired_rng = self._rng.get_state().clone()
    for flag in (False, True, "temporal"):
        self._rng.set_state(paired_rng)
        loss = self._train(rows, epochs, "entropy", control=flag)
        # 保存第一组训练结束后的 RNG 状态
    # 最后把 self._rng 设置为第一组训练结束后的状态

必须配对实际 randperm 和 dropout，不能只是三个模型使用相同初始 seed。沿用 _train 中 fork_rng 对 CPU/CUDA 随机状态的隔离，不能扰动 Actor/其他组件 RNG。记录 semantic_train_loss、train_loss、control_train_loss、temporal_control_train_loss。

五、无熵对照保留辅助元数据

当前 schema 为 causal-role-mean-top16-v2。每个角色 absolute block 为：

    [mean, coverage, log_tokens, valid, truncated, available]

每个角色 temporal block 为：

    [previous_mean, delta, past_drift, residual, log_history, available]

Solver 和 Verifier block 各 6 维，总 absolute 12 维、temporal 12 维。不要扩大输入维度或引入新特征 schema。

EntropyValueHead.forward 可增加 preserve_entropy_metadata=False，默认 False 保持旧行为。新协议在训练和预测时传 True：

- no_entropy=True：将标准化后的 absolute 索引 [0,6] 置零，将 temporal 索引 [0,1,2,3,6,7,8,9] 置零。保留两角色的 coverage、log_tokens、valid、truncated、available、log_history 及所有结构语义输入。
- no_temporal=True：只将 temporal 的上述 8 个熵数值置零，完整保留 absolute 和时序辅助元数据。
- 在 clone 后的 tensor 上操作，不要原地污染缓存或其他实验分支。
- legacy 的 no_entropy/no_temporal 仍按原逻辑工作，避免旧 checkpoint 行为变化。

三分支结构仍为 semantic logit、semantic+absolute residual、semantic+absolute+temporal residual；不扩大编码器，也不加入熵重构等新的训练目标。

六、训练、资格和实际输出的前缀范围一致

新协议新增类似 _control_training_data(data) 的统一范围处理，在 _train_impl 和 _predict 中都使用。支持 flatten 后的 roles 与单条记录的 prefix_roles。

依据原始特征中的 availability 和 prefix_terminal 构造：

    abs_mask = 非终局 AND 当前已完成动作所属角色的 absolute_available
    temp_mask = 非终局 AND 当前角色的 absolute_available AND temporal_available

必须使用当前角色自己的 availability，不能因为另一个角色的旧熵还存在就判定可用。initial prefix 不属于 Solver/Verifier 动作，因此这两个 mask 为零。

语义 BCE 仍可使用全部合法前缀，包括 initial 和 terminal。只是熵残差的训练、输出范围收紧。保留真实轮次预算、因果前缀编码、原题哈希留出、前缀等权，不增加未来信息。

terminal 或当前角色无可靠 absolute 熵的位置，应通过实际 forward mask 得到 sem=abs=full，而不是只在评价指标里回退。prepare() 输出的 absolute_available、temporal_available、action_absolute_available、action_temporal_available 同步反映上述范围，使这些位置不会错误进入熵控制。

保留 absolute_bootstrap 的按角色时序路由：某角色没有时序资格，提供给控制器的 full 回退 abs；有非终局时序样本也不代表已经获得时序权限。

不降低 min_entropy_brier_gain、min_temporal_brier_gain、AUC/Brier/类别/独立题目等原资格要求。候选无增益就继续待资格，不能为了让程序启动而制造资格。

七、Trainer、配置与 Bash 接入

ppo_trainer.yaml 的 algorithm.entropy_credit.value 下增加：

    initialization_mode: qualified

保留 require_pretrained: True。新增 helper validate_value_initialization(config, state)，行为为：

- mode 仅接受 qualified 或 candidate，未知值报错。
- candidate 必须检查实际 state["warm_started"] 为真，不能只因为配置选择 candidate 就放行随机初始化。
- qualified 且 require_pretrained=True 时继续检查 ready=True。

在 _validate_value_control_config 中检查 mode 枚举；保留 value/control/Actor control 开关一致性检查。

_init_value_scorer() 有 initial_checkpoint 时，按模式调用：

    load(initial, resume=False, warm_start=(mode == "candidate"))

加载成功后调用上述 helper。fit() 仍先恢复 _load_checkpoint()，然后通过 prepare([]) 检查实际状态：只有 global_steps==0 的新启动执行初始化资格检查；正常 global_steps>0 的完整恢复保留暂时未合格/停用状态。

增加启动指标：

    value_model/ready_at_training_start
    value_model/warm_started_at_training_start

run_math_16gpu.sh 增加：

    VALUE_INIT_MODE=${VALUE_INIT_MODE:-qualified}

枚举检查 qualified|candidate，无效值退出 1。把参数传入 Hydra：

    algorithm.entropy_credit.value.initialization_mode=$VALUE_INIT_MODE

新训练仍需存在的 VALUE_CHECKPOINT；报错提示说明未合格文件需要 candidate 模式。不要通过 require_pretrained=False 实现此功能。

RESUME_FROM 非空时维持 INITIAL_CHECKPOINT=null，使用完整断点恢复，不能重复加载离线候选。eval 不创建 value worker、不加载离线初始化、不更新参数。保留 run_math.sh 现有转发入口以及 DRY_RUN。

八、熵控制器：角色独立渐进启用

旧全局 ready_step/ramp 会让晚获得资格的 Verifier 直接沿用 Solver 已跑满的 ramp。本次修复为每角色独立状态：

    role_ramps[role] = {"qualified_updates": 0, "qualified": False}

角色 qualified 要求整体 ready=True 且角色 reliability>0。每次 prepare：

1. 未合格或整体 ready=False，该角色 qualified_updates 清零、ramp=0。
2. 合格且该 batch 至少有一个可用于控制的有效唯一动作，计数增加一次，上限为 ramp_steps。
3. 可用于计数的动作须在已标定 cap、scope 动作数充足的组内，并有可用价值预测及正角色可靠性；不要求实际惩罚非零，熵风险为零的合格 batch 也可推进 ramp。
4. 同一角色有多个 turn、多个 action 或 padding 副本时，每 batch 只能增加一次。
5. 角色缺席、缺少预测、未满足 scope 样本数时暂停推进，不按 global step 差值补足。
6. 若可靠性映射或标量能够表明某个缺席角色已失资格，也要重置该角色；逐行数组无法描述缺席角色时保留之前状态并暂停。
7. 支持 worker 的 solver/verifier 资格键与 rollout 的 Solver Agent/Verifier Agent 名称映射，保留原 _role_confidence 的角色/轮次映射支持。

计算：ramp_role = qualified_updates / ramp_steps，默认 ramp_steps=20。其他角色已满强度不能影响新合格角色；重新合格从 1/20 开始。

保留原门控公式和 Actor 损失：

    weight = reliability * ramp_role * entropy_risk * bad_progress
             * (kappa + (1-kappa) * temporal_harm)

最终仍显式要求 ready、可靠价值/absolute 特征可用。未取得时序资格时 temporal_harm=0；kappa=0.25 不是未合格模型的兜底刹车。gate 和 cap detach，只有 Actor 当前全词表熵参与求导。

保留初始 cap 标定、fast/slow EMA、有效性/截断检查、padding 去重及步数单调检查。即使 ready=False，初始标定和趋势统计也继续。默认仅首个 step 标定；初始样本不足的角色/轮次不能拿后期膨胀熵补作健康 cap。

EntropyController.VERSION 从 1 升为 2；state_dict 保存 role_ramps，可使用按角色排序的列表序列化。继续保存原 config、start_step、ready_step、last_step、groups。加载 v2 验证角色不重复、qualified_updates 为合法整数且在 [0,ramp_steps]，未合格状态不能带正计数。

允许加载旧 v1：保留 cap、EMA、step 和配置，因旧文件不能确定各角色首次合格时间，重置独立 role ramp，重新渐进启用。其他未知版本拒绝，原配置兼容检查继续保留。

新增角色指标：

    entropy_control/{role}/qualified
    entropy_control/{role}/qualified_updates
    entropy_control/{role}/ramp
    entropy_control/{role}/ramp_paused
    entropy_control/{role}/ramp_reset
    entropy_control/{role}/eligible_actions

role/turn 指标中也记录 ramp；原 entropy_control/ramp 改为已知角色 ramp 的均值，不能再用于统一决定所有角色强度。

九、必须保留的主训练行为

- 16 张物理卡：单机 Agent layout=[15] 加独立 value GPU；两机各 8 卡，Agent layout=[7,8] 加 head 上的 value GPU。两个 Agent 共用 15-rank 池但不共享模型参数。
- 默认 TRAIN_BATCH_SIZE=30、GROUP_SIZE=8，每步 240 条 rollout；训练问题 batch 必须为 15 的正整数倍。MAX_SOLVER_TURNS=3。
- 编码器冻结；在线价值默认 lr=1e-4、每阶段 1 epoch、batch 256、replay 2048。旧离线默认值不必随之修改。
- 两阶段 pure 信用、零优势恢复、成功/失败组治理和题目课程保留。价值模型不替代 advantage、不重写真实 reward/returns，不另乘价值信用系数。
- 仅受保护的随机 fresh 轨迹进入价值训练/留出；课程轨迹保留原 score-only 行为。虚拟奖励不是价值监督标签。
- 顺序必须是：rollout → 用之前已部署头准备预测及控制 → 完成所有 Actor 更新 → 候选价值训练与验证 → 可能发布 → 下一 batch 才使用。
- 部署后候选仍继续学习；原失准停用、按角色基础/时序资格机制保留。
- 原 action Top-16 统计是每 token Top-16 归一化熵除以 ln(16) 后按 action 平均，不能重复归一化。Actor 惩罚使用可微全词表熵，单位 nats，不能把旧 Top-16 标量直接拿去求 Actor 梯度或与全词表 cap 比较。
- 保留历史 Solver 连续重复落盘去重功能；本次不修改轨迹适配器、文本序列化或原始文件。
- 本次不新增 std/分位数特征、不更换 schema、不扩大编码器、不重跑全部历史编码，也不把语义预测成功称为熵控制已经有效。

十、测试与验收

优先使用已有 TinyScorer/小型因果编码器 fixture，在 CPU 上验证实际代码路径，不下载 4B 模型。测试至少覆盖：

1. 构造 candidate 与 deployed 参数不同的失败 checkpoint，验证 warm-start 确实导入 candidate，三个头初始相同，输入文件未改。
2. warm-start 后新优化器无动量、缓存为空、RNG/step 重置、encoder 冻结、ready/权限为零；prepare 仍能收集在线记录但不输出部署预测。
3. 失败文件普通 fresh load 仍拒绝；resume 与 warm_start 同时为真拒绝；编码器/schema/scaler/维度不匹配拒绝。
4. 新字段缺失时按 legacy/False 兼容；未知协议或错误 warm_started 类型拒绝。
5. 实际成功 warm-start 才能通过 Trainer 新启动检查；仅配置 candidate、没有加载成功标志必须失败。qualified 模式不能把 warm_started 当作 ready。
6. 共享语义参数，真实标准化熵输入全零时，即使 dropout 非零，配对训练各头结果一致、matched gain 为零；不扰动 worker 外的 RNG。
7. 无熵对照对被遮熵数值不敏感，但仍使用长度等辅助元数据；无时序对照保留 absolute 输入。
8. 终局/当前角色熵不可用的实际输出 sem=abs=full、availability 为 False；terminal-only 数据不能更新熵残差。
9. 无熵增益时候选确实更新，但门仍关闭。
10. 使用独立题目划分的合成数据，仅让熵携带标签信号，真实优化后应获得正 absolute/matched gain 并正常发布。不能 monkeypatch ready 来替代该学习性测试。
11. 保存再 resume，下一次训练的参数、loss、RNG、缓存、资格和协议应与连续运行一致；允许 ready=False 的完整恢复。
12. Solver 已满 ramp 时 Verifier 晚通过从自己的 1/20 开始；单角色失资格只重置该角色；缺席/缺预测/样本不足暂停；多 turn 和 padding 不多计。
13. controller v2 精确恢复；v1 保留 cap/EMA 但重启角色 ramp；未合格时最终熵权重为零。
14. 用真实 Bash DRY_RUN 输出做 Hydra compose，覆盖 1/2 节点、qualified/candidate、fresh/resume/eval，确认 16 卡布局、参数传递、resume 不重新初始化。未知 VALUE_INIT_MODE 拒绝。
15. 保留 score-before-Actor、value-update-after-all-Actors 的 Trainer 集成测试。

重点运行：

    python -B -m pytest -q -p no:cacheprovider \
      tests/workers/test_credit_value_warm_start.py \
      tests/utils/test_entropy_control.py \
      tests/utils/test_mix_16gpu_launch.py \
      tests/trainer/ppo/test_mix_entropy_value_integration.py

然后运行现有 credit_value、absolute_init、curriculum、value_credit、entropy_actor、pure、advantage_recovery、baseline adapter、pretrain CLI/Bash、资源布局和 checkpoint 测试，确认 legacy/离线流程无回归。运行 Bash 语法检查。测试临时文件放临时目录，禁用项目字节码/pytest 缓存并清理本次创建的临时产物，保留测试源码，不删除其他原有文件。

参考实现曾通过 416 项 CPU 测试，但服务器必须报告实际运行数量、失败/跳过及环境限制，不能照抄参考数字。CPU 测试不能冒充 16 卡真实训练或真实轨迹资格通过。可以对可访问的真实 checkpoint 做只读加载核验；本任务不自动启动完整长跑或重新离线编码。

十一、最终交付

直接完成实现后报告修改文件、关键行为、实际测试结果及尚未验证部分。提供以下可执行启动示例，路径使用服务器实际位置或明确占位符：

    cd /实际路径/Credit_mas_mix
    export SOLVER_MODEL=/models/Qwen3-4B
    export VERIFIER_MODEL=/models/Qwen3-4B
    export VALUE_ENCODER=/models/Qwen3-4B
    export VALUE_CHECKPOINT=/checkpoints/baseline_value_init/prefix_value.pt
    export VALUE_INIT_MODE=candidate
    export TRAIN_DATA=/data/drmas_math/train.parquet
    export VAL_DATA=/data/drmas_math/test_sampled.parquet
    export NNODES=1
    export TRAIN_BATCH_SIZE=30
    export GROUP_SIZE=8
    export MAX_SOLVER_TURNS=3
    export RUN_NAME=mix_candidate_online_16gpu
    export RUN_DIR=/checkpoints/mix_candidate_online_16gpu
    export LOGGERS='[console]'
    unset RESUME_FROM
    DRY_RUN=1 bash examples/drmas_trainer/run_math.sh train
    bash examples/drmas_trainer/run_math.sh train

说明 VALUE_ENCODER 必须匹配预训练编码器；TRAIN_DATA 是主任务问题 parquet，不是历史轨迹 JSON 目录；VALUE_CHECKPOINT 只初始化价值模型，不初始化 Actor。新输出目录不得覆盖源离线 .pt。

还要给出两台 8 卡 Ray 启动方式、RESUME_FROM=完整 global_step_N 目录的恢复方式、eval 指定已保存 Actor checkpoint 的评测方式。默认每 10 step 验证、每 50 step 保存、训练 2 epoch；eval 不更新价值网络，默认每题采样 16 条、temperature=0.6、top_p=0.95。当前验证只输出指标，不应声称已导出逐条回答。

文档明确：启动日志 warm_started=1、ready=0 是预期状态；候选何时合格由真实在线留出数据决定，不保证固定步数开启。首次 step cap 标定不足的角色/轮次可能保持关闭。若始终无熵增益，主任务仍可继续，但价值熵约束不会启用。
