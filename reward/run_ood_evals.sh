#!/usr/bin/env bash
# Run all three OOD conditions sequentially, then score them.
#
#   A  base Qwen3.5-9B      + v2 plan prompt
#   B  base Qwen3.5-9B      + html-only prompt
#   C  step-60 finetuned    + v2 plan prompt
#
# Each takes all 8 GPUs (tp=8), so they cannot overlap. Skips any condition whose
# output already exists, so it is safe to re-run after an interruption.
set -uo pipefail

BASE=${BASE:-/home/ubuntu/models/Qwen3.5-9B}
FT=${FT:-/home/ubuntu/models/qwen3_5-9b-table-rl-step60}
OUT=${OUT:-/home/ubuntu/ood_eval}
N=${N:-0}                      # 0 = all 1998 rows
LOGD=${LOGD:-/home/ubuntu/ood_eval/logs}

mkdir -p "$OUT" "$LOGD"
cd /home/ubuntu/verl
export VIRTUAL_ENV=/home/ubuntu/verl/.venv PATH=/home/ubuntu/verl/.venv/bin:$PATH

run () {                        # run <tag> <model> <prompt>
    local tag=$1 model=$2 prompt=$3
    local out="$OUT/${tag}.jsonl"
    if [[ -s "$out" ]]; then
        echo "[$(date +%H:%M:%S)] $tag already done ($(wc -l < "$out") rows) — skipping"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] START $tag  model=$(basename "$model")  prompt=$prompt"
    python -u reward/eval_ood.py --model "$model" --prompt "$prompt" \
           --tp 8 --n "$N" --out "$out" > "$LOGD/${tag}.log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 || ! -s "$out" ]]; then
        echo "[$(date +%H:%M:%S)] FAILED $tag rc=$rc — last lines:"
        grep -vE "^INFO|it/s\]|Warning|deprecated" "$LOGD/${tag}.log" | tail -12
        return 1
    fi
    echo "[$(date +%H:%M:%S)] DONE  $tag  ($(wc -l < "$out") rows)"
}

run A_base_plan      "$BASE" plan || exit 1
run B_base_html      "$BASE" html || exit 1
run C_finetuned_plan "$FT"   plan || exit 1

echo "[$(date +%H:%M:%S)] all three generations complete — scoring"
python -u reward/score_ood.py "$OUT" 2>&1 | tail -40
echo "[$(date +%H:%M:%S)] ALL_OOD_EVALS_COMPLETE"
