#!/usr/bin/env bash
# Edit the configuration below, then launch with:
#   bash examples/drmas_trainer/run_math_8b_semantic.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# ==================== 在这里修改配置 ====================
# 以下是示例路径，请替换成服务器实际路径。
SOLVER_MODEL='/models/Qwen3-8B'
VERIFIER_MODEL='/models/Qwen3-8B'
# 必须与价值网络预训练使用的编码器一致。
VALUE_ENCODER='/models/Qwen3-8B'
VALUE_CHECKPOINT='/checkpoints/8b_value/prefix_value.pt'

# 主任务 Parquet 数据，不是历史轨迹 JSON 目录。
TRAIN_DATA='/data/math/train.parquet'
TRAIN_VAL_DATA='/data/math/test_sampled.parquet'
EVAL_DATA='/data/math/test.parquet'

RUN_NAME='mix_8b_semantic_only'
RUN_DIR="$PROJECT_DIR/checkpoints/$RUN_NAME"
LOGGERS='[console,wandb]'

# 总计 16 卡：单机 16 卡设为 1；两机各 8 卡设为 2。
# 双机需先建立同一个 Ray 集群，并将 RAY_ADDRESS 改为 'auto' 或 head:port。
NNODES=1
RAY_ADDRESS=''

VALUE_PREDICTION_MODE='semantic_only'
VALUE_INIT_MODE='candidate'
SEMANTIC_CONTROL_STRENGTH=0.25
ENTROPY_LOSS_COEF=0.01

TRAIN_BATCH_SIZE=30
GROUP_SIZE=8
MAX_SOLVER_TURNS=3
TRAIN_VAL_BATCH_SIZE=110
TRAIN_VAL_GROUP_SIZE=1
EVAL_BATCH_SIZE=60
EVAL_GROUP_SIZE=16
TOTAL_EPOCHS=2
SAVE_FREQ=50
TEST_FREQ=10
VAL_BEFORE_TRAIN=True

# 仅在 resume / eval 模式使用，填写实际存在的完整 checkpoint 目录。
RESUME_CHECKPOINT="$RUN_DIR/global_step_50"
EVAL_CHECKPOINT="$RUN_DIR/global_step_50"
# ==================== 配置区结束 ========================

usage() {
  cat <<'EOF'
Usage: bash examples/drmas_trainer/run_math_8b_semantic.sh [mode] [Hydra overrides...]
  train (default)       Start a new main-training run from VALUE_CHECKPOINT.
  resume                Resume from RESUME_CHECKPOINT in the configuration above.
  eval                  Evaluate EVAL_CHECKPOINT on EVAL_DATA.
  check [train|resume|eval]
                        Check paths and print the launch command without loading models.

Edit this script's configuration block first; no external exports are required.
Script settings replace inherited environment variables. Explicit Hydra overrides
are appended last, e.g. train trainer.total_epochs=3.
EOF
}

ACTION=${1:-train}
if [[ $# -gt 0 ]]; then shift; fi
DRY_RUN=0
if [[ "$ACTION" == check ]]; then
  DRY_RUN=1
  ACTION=${1:-train}
  if [[ $# -gt 0 ]]; then shift; fi
fi

# Explicit assignments prevent an old shell's RESUME_FROM / VAL_DATA / DRY_RUN
# from silently changing a fresh run or reusing training validation for eval.
RESUME_FROM=''
VAL_DATA="$TRAIN_VAL_DATA"
VAL_BATCH_SIZE="$TRAIN_VAL_BATCH_SIZE"
VAL_GROUP_SIZE="$TRAIN_VAL_GROUP_SIZE"
case "$ACTION" in
  train) LAUNCH_MODE=train ;;
  resume)
    : "${RESUME_CHECKPOINT:?Set RESUME_CHECKPOINT in this script}"
    RESUME_FROM="$RESUME_CHECKPOINT"
    LAUNCH_MODE=train
    ;;
  eval)
    : "${EVAL_CHECKPOINT:?Set EVAL_CHECKPOINT in this script}"
    RESUME_FROM="$EVAL_CHECKPOINT"
    VAL_DATA="$EVAL_DATA"
    VAL_BATCH_SIZE="$EVAL_BATCH_SIZE"
    VAL_GROUP_SIZE="$EVAL_GROUP_SIZE"
    LAUNCH_MODE=eval
    ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 1 ;;
esac

export SOLVER_MODEL VERIFIER_MODEL VALUE_ENCODER VALUE_CHECKPOINT
export VALUE_PREDICTION_MODE VALUE_INIT_MODE SEMANTIC_CONTROL_STRENGTH
export TRAIN_DATA VAL_DATA NNODES RAY_ADDRESS TRAIN_BATCH_SIZE GROUP_SIZE MAX_SOLVER_TURNS
export VAL_BATCH_SIZE VAL_GROUP_SIZE RUN_NAME RUN_DIR LOGGERS RESUME_FROM DRY_RUN

exec bash "$SCRIPT_DIR/run_math.sh" "$LAUNCH_MODE" \
  "actor_rollout_ref.actor.entropy_control.loss_coef=$ENTROPY_LOSS_COEF" \
  "trainer.total_epochs=$TOTAL_EPOCHS" \
  "trainer.save_freq=$SAVE_FREQ" \
  "trainer.test_freq=$TEST_FREQ" \
  "trainer.val_before_train=$VAL_BEFORE_TRAIN" \
  "$@"
