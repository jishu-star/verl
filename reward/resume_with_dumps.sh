#!/usr/bin/env bash
# Resume the table-RL run from its latest checkpoint, with rollout dumping on.
#
# Safe to run ONLY once a checkpoint exists (save_freq=10, so step 10, 20, ...).
# resume_mode=auto finds the latest global_step_N in trainer.default_local_dir and
# restores weights, Adam state, the step counter AND the StatefulDataLoader position,
# so no sample is re-seen.
#
# Do NOT change TRAIN_BATCH on resume: the saved dataloader offset is in units of the
# old batch size, so steps-per-epoch would no longer line up.
set -euo pipefail

CKPT_DIR=/home/ubuntu/ckpts/table-rl/qwen3_5-9b-grpo-fullft-nokl
DUMP_DIR=${DUMP_DIR:-/home/ubuntu/rollout_dumps}

if [[ ! -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "ERROR: no checkpoint yet in ${CKPT_DIR} -- wait for step 10." >&2
    exit 1
fi
echo "resuming from global_step_$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt")"
mkdir -p "${DUMP_DIR}"

cd /home/ubuntu/verl
export VIRTUAL_ENV=/home/ubuntu/verl/.venv
export PATH=/home/ubuntu/verl/.venv/bin:$PATH

MODEL_PATH=~/models/Qwen3.5-9B TRAIN_BATCH=256 \
bash reward/run_qwen3_5_9b_table_fullft_grpo.sh \
    trainer.rollout_data_dir="${DUMP_DIR}" \
    trainer.val_before_train=False \
    "$@"
