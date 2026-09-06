#!/usr/bin/env bash
# Archive the table-RL run to S3, under the same dataset prefix.
#
# Needs credentials in the environment first:  aws configure
# Dry run:   DRYRUN=1 bash upload_to_s3.sh
set -euo pipefail

BASE=${BASE:-s3://form-checkboxes-annotation/SyntheticTableData/2026-09-05-new-table-rl-data}
DEST="${BASE}/rl-run-20260905"
CKPT_DIR=/home/ubuntu/ckpts/table-rl/qwen3_5-9b-grpo-fullft-nokl
DRY=""
[[ "${DRYRUN:-0}" == "1" ]] && DRY="--dryrun"

aws sts get-caller-identity >/dev/null || { echo "ERROR: no AWS credentials. Run: aws configure" >&2; exit 1; }
echo "destination: ${DEST}"

# 1. checkpoints (the big one: 2 x 87G, model + optimizer, fully resumable)
echo "[1/6] checkpoints (174G) ..."
aws s3 sync "${CKPT_DIR}" "${DEST}/checkpoints/" $DRY --only-show-errors

# 2. rollout dumps -- every generation with its score, per step
echo "[2/6] rollout dumps (848M) ..."
aws s3 sync /home/ubuntu/rollout_dumps "${DEST}/rollout_dumps/" $DRY --only-show-errors

# 3. the parquets carrying the v2 prompt (regenerating them needs this exact text)
echo "[3/6] rewritten parquets (97M) ..."
aws s3 cp /home/ubuntu/table_exps/stage5/train.parquet "${DEST}/data-v2-prompt/train.parquet" $DRY --only-show-errors
aws s3 cp /home/ubuntu/table_exps/stage5/val.parquet   "${DEST}/data-v2-prompt/val.parquet"   $DRY --only-show-errors

# 4. code that defines the run: reward + launch scripts + prompt rewriters
echo "[4/6] code ..."
aws s3 sync /home/ubuntu/verl/reward "${DEST}/code/reward/" $DRY --only-show-errors \
    --exclude "__pycache__/*" --exclude "*.pyc"
aws s3 cp /home/ubuntu/table_exps/stage5/rewrite_prompt.py "${DEST}/code/rewrite_prompt.py" $DRY --only-show-errors
aws s3 cp /home/ubuntu/table_exps/stage5/prompt_v3.py      "${DEST}/code/prompt_v3.py"      $DRY --only-show-errors

# 5. training logs -- the full metric history
echo "[5/6] logs ..."
aws s3 sync /home/ubuntu/verl/logs "${DEST}/logs/" $DRY --only-show-errors --exclude "*" --include "*.log"

# 6. manifest
echo "[6/6] manifest ..."
aws s3 cp /home/ubuntu/verl/RUN_MANIFEST.md "${DEST}/RUN_MANIFEST.md" $DRY --only-show-errors

echo
echo "done. verifying:"
aws s3 ls "${DEST}/" --recursive --summarize | tail -3
