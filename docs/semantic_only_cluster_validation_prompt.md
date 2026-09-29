# 交给服务器 Claude Code 的集群验证 prompt

请只验证服务器上的 `Credit_mas_8b_mix` 纯语义价值控制改动，不修改原 `Credit_mas_mix` 或其他项目。这里是实际集群验证任务；本地尚未运行 8B/16 卡训练，不能把本地 CPU 测试当成集群通过。

先阅读 `docs/semantic_only_value_control.md` 和相关实现。使用用户提供的真实 8B checkpoint、预训练同一编码器、主任务 train/val Parquet，所有测试输出写入新的目录。保留源 checkpoint 不变，记录运行配置、git diff 和源文件 hash。

1. **配置与路径**：检查 `VALUE_PREDICTION_MODE=semantic_only`、`VALUE_INIT_MODE=candidate`、`SEMANTIC_CONTROL_STRENGTH=0.25`。确认编码器身份/维度匹配。先执行 DRY_RUN，再用 Hydra compose 核对 value/control 模式一致、强度有限、单机 16 卡或双机 8+8 的布局。DRY_RUN 不加载模型，不能当成显存验证。
2. **启动资格复验**：启动前从源 checkpoint 的 candidate semantic 权重和已编码 replay 复验。记录每角色非终局合法前缀数、不同问题数、不同成功/失败轨迹数、AUC、Brier、训练侧先验 Brier、权限和拒绝原因。确认不重新编码 185k 历史轨迹、不增加拟合、不覆盖源文件。合格角色可立即 semantic_ready，熵资格不应参与该决策。未通过时真实报告失败，不放宽阈值强行过关。
3. **16 卡短程训练**：建立用户指定 Ray 集群后使用文档命令，建议独立短程目录并覆盖 `trainer.total_training_steps=5 trainer.save_freq=2 trainer.test_freq=2`。记录 Ray 节点、各 GPU 显存、Actor 15-rank 布局、价值 GPU 和吞吐；如 OOM，先定位阶段，可将 Actor/old-log-prob micro batch 降为 1，不暗改学习率、信用分配、恢复或强度。
4. **真实数据链路**：确认无 top-16 熵或低 coverage 的有效动作仍可有语义预测，initial 可用于 before，terminal/invalid/truncated 不得控制。统计 `value_semantic_available`、语义负向进展、cap 校准、风险、角色权限/ramp、gate 和超过 cap 的交集；不能仅用训练 loss 下降判断控制生效。
5. **更新与梯度**：对一个短程更新前后核对候选 semantic 参数变化，冻结编码器、absolute/temporal、两个对照头未变，部署 semantic 仅发布时变化。Actor 使用的是本批开始的部署版，在线更新在所有 Actor 更新后进行。熵损失非零时记录全精度 hinge、coef、strength 及梯度证据；价值参数不能获得 Actor 梯度。自然 batch 未触发时报告阻断原因，使用独立合成诊断验证正 hinge，不强迫实际轨迹 gate=1。
6. **完整恢复**：从短程 `global_step_2` 同模式同强度恢复，比较 value version、replay、optimizer/RNG、角色权限/坏窗口，以及 controller cap/ramp 是否保持；暂时 ready=False 也可恢复。与连续运行比较下一次固定输入的输出/状态。跨模式 exact resume 必须明确报错，不应默默重置；不要在恢复时重做离线资格复验。
7. **独立评测**：对短程 checkpoint 执行 eval，确认不创建价值训练 worker、不更新 Actor/价值参数，验证输出与配置一致。
8. **旧模式回归**：运行本项目现有 CPU 测试和新增语义测试。对适当合格的 entropy_aware checkpoint 验证旧模式可用；对同一熵失败文件的 entropy_aware+candidate 验证仍遵循旧资格逻辑。不得把 semantic ready 解释成 entropy-qualified。
9. **公平效果实验**：仅在短程验证通过后，按用户授权启动完整 run。建议同数据顺序/模型/种子/PURE/优势恢复设置下比较控制关闭与纯语义控制，分别报告任务成功率、熵、长度、零方差比例和控制触发比例。语义 Brier 优于先验不能证明最终任务收益。

最终提交：实际命令、配置、测试结果、源 checkpoint hash 前后对比、每角色启动资格报告、训练和控制指标、短程资源情况、恢复/评测结果及未通过项。遇到失败先报告具体证据，不通过强改 ready、伪造 availability 或删掉资格条件让验证看似通过。
