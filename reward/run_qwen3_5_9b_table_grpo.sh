#!/usr/bin/env bash
# GRPO | Qwen3.5-9B | table image -> <plan> + HTML
#
#   LoRA everywhere, the last N decoder layers additionally trained in full,
#   and a KL term against a real frozen copy of the base model.
#
# Run from the verl repo root:   bash reward/run_qwen3_5_9b_table_grpo.sh
# Every setting below is overridable from the environment, e.g.
#   UNFREEZE_LAST_N=4 LORA_RANK=64 bash reward/run_qwen3_5_9b_table_grpo.sh
#
# dependency: vllm==0.18.0 -- Qwen3.5 is 24/32 linear-attention layers, whose recurrent
# state the rollout engine has to manage; older vLLM either crashes or trains on garbage.
# dependency: pip install -r reward/requirements.txt  (apted, lxml -- the reward runs
# inside the trainer process, so they must be in the SAME environment as verl).

set -xeuo pipefail

########################### user-adjustable ###########################
PROJECT_NAME=${PROJECT_NAME:-table-rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5-9b-grpo-lora-last${UNFREEZE_LAST_N:-8}}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
DATA_DIR=${DATA_DIR:-/home/ubuntu/table_exps/stage5}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
TEST_FILE=${TEST_FILE:-${DATA_DIR}/val.parquet}
CKPTS_DIR=${CKPTS_DIR:-${HOME}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}}

NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
GEN_TP=${GEN_TP:-2}                       # 9B is dense; 2-way TP is plenty for rollout
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-${NDEVICES_PER_NODE}}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}

# --- what gets trained ---
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
UNFREEZE_LAST_N=${UNFREEZE_LAST_N:-8}     # last N decoder layers, fully trainable
KL_COEF=${KL_COEF:-0.01}

# --- reward weights (no code change needed to retune) ---
W_HTML=${W_HTML:-0.75}
W_SCHEMA=${W_SCHEMA:-0.05}
W_CONTENT=${W_CONTENT:-0.20}

# --- reward parallelism ---
# MEASURED on the real corpus: compute_score is 3.7 s mean / 15 s p90 per call, because
# APTED runs twice per table (structure-only 0.63 s, then with-text 3.2 s).  One GRPO step
# is TRAIN_BATCH * ROLLOUT_N calls -- 512 by default, i.e. ~1900 core-seconds.
# reward.num_workers is real process parallelism: RewardLoopManager spawns this many Ray
# actors and static-chunks the batch across them (experimental/reward_loop/reward_loop.py
# :324-352).  Set it near the box's vCPU count or the reward, not the GPUs, sets step time.
REWARD_WORKERS=${REWARD_WORKERS:-$(( $(nproc) * 3 / 4 ))}
if (( REWARD_WORKERS < 8 )); then REWARD_WORKERS=8; fi

# --- sequence budget, from the measured corpus ---
# visual tokens are capped at 4096 in data prep (Qwen3.5: patch 16, merge 2 -> 32x32 px
# per token); target p99 is 3231 tokens, max 6295.
# 4608 was sized for a 32-token prompt.  The plan grammar is now specified in the prompt
# itself (389 tokens), so a 4096-visual-token row needs 4096+389+~20 template = ~4505 --
# only ~100 tokens of slack, and data.truncation=error turns any overflow into a dead
# step rather than a warning.  The visual cap is what actually bounds prompt size, so the
# headroom is free.  Measured worst case over the 40 largest images: 4534.
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-5120}
# 4096 truncated 62% of held-out rollouts, and a cut-off table scores ~0 on TEDS however
# good the plan was.  Targets (plan + canonical HTML) measured over all 63,816 rows:
# p50 2176, p95 4340, p99 5529, max 10550.  8192 covers >99.9%; at 4096 -> 8192 the
# held-out truncated rate went 0.307 -> 0.016 and has_table 0.910 -> 1.000.
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-5120}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-4096}
TRAIN_BATCH=${TRAIN_BATCH:-64}
ROLLOUT_N=${ROLLOUT_N:-8}
# --- PPO update granularity ---
# ppo_mini_batch_size is in PROMPTS: ray_trainer.py:1351 multiplies it by rollout.n
# before dispatch, so TRAIN_BATCH / PPO_MINI_BATCH is the number of gradient updates
# per step.  Hardcoding it meant overriding TRAIN_BATCH alone failed validation
# (train_batch_size must be >= ppo_mini_batch_size, workers/config/actor.py:226).
# Derive it instead, keeping the 4-updates-per-step ratio at any batch size, then
# floor it so mini * rollout_n still gives every data-parallel rank a sequence.
PPO_MINI_BATCH=${PPO_MINI_BATCH:-$(( TRAIN_BATCH / 4 ))}
(( PPO_MINI_BATCH < 1 )) && PPO_MINI_BATCH=1
MIN_MINI=$(( (NDEVICES_PER_NODE * NNODES + ROLLOUT_N - 1) / ROLLOUT_N ))
(( PPO_MINI_BATCH < MIN_MINI )) && PPO_MINI_BATCH=${MIN_MINI}
(( PPO_MINI_BATCH > TRAIN_BATCH )) && PPO_MINI_BATCH=${TRAIN_BATCH}

########################### end user-adjustable ###########################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs "${CKPTS_DIR}"

# ---- preflight -------------------------------------------------------------------
for f in "${TRAIN_FILE}" "${TEST_FILE}"; do
    if [[ ! -f "${f}" ]]; then
        echo "MISSING: ${f}" >&2
        echo "Build it with:  python stage5/build_parquet.py   (in table_exps/)" >&2
        echo "It emits an 'images' column of absolute file paths -- fine on one node or a" >&2
        echo "shared filesystem.  For multi-node without one, rebuild with" >&2
        echo "--image-mode bytes so the images travel inside the parquet." >&2
        exit 1
    fi
done
PYBIN=${PYBIN:-python3}
"${PYBIN}" -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from reward.table_reward import compute_score
r = compute_score('t', '<plan>\nheaders: none\nrow groups: none\nmerged: none\nempty: none\n</plan>\n<table><tr><td>a</td></tr></table>',
                  '<table><tr><td>a</td></tr></table>', {'plan': '<plan>\nheaders: none\nrow groups: none\nmerged: none\nempty: none\n</plan>'})
assert r['error'] == 0.0 and r['score'] > 0.9, r
print('reward preflight ok:', {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
"

start_time=$(date +%Y%m%d_%H%M%S)

########################### parameter arrays ###########################
ALGO=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False        # KL goes in the loss, not the reward
)

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH}
    data.max_prompt_length=${MAX_PROMPT_LEN}
    data.max_response_length=${MAX_RESPONSE_LEN}
    data.image_key=images
    data.image_patch_size=16                # must match vision_config.patch_size
    data.shuffle=True
    # OFF deliberately. filter_overlong_prompts silently DROPS rows over the limit, and
    # the drop is not random: big image -> tall table -> exactly the hard examples
    # stage 2 selected. Resolution is capped in data prep instead, so nothing vanishes.
    data.filter_overlong_prompts=False
    data.truncation=error
)

REWARD=(
    # NOTE the leading + on reward_kwargs: reward.yaml declares only path and name
    # under custom_reward_function, so hydra struct mode rejects the weights as
    # undeclared keys.  They ARE read -- trainer/ppo/reward.py:81 does
    # reward_fn_config.get("reward_kwargs", {}) -- so + attaches them where the
    # loader looks, rather than silently dropping them.
    reward.custom_reward_function.path=reward/table_reward.py
    reward.custom_reward_function.name=compute_score
    +reward.custom_reward_function.reward_kwargs.w_html=${W_HTML}
    +reward.custom_reward_function.reward_kwargs.w_schema=${W_SCHEMA}
    +reward.custom_reward_function.reward_kwargs.w_content=${W_CONTENT}
    reward.num_workers=${REWARD_WORKERS}    # TEDS is CPU-bound; this is the parallelism
)

# LoRA + full-rank tail + a real reference model.
# All three are required together and verl refuses the run without them:
#   lora.merge=True   otherwise only adapter tensors reach vLLM and the unfrozen layers
#                     train in the actor but never in the sampler
#   ref_in_actor=False otherwise the KL reference is "actor with adapters off", which now
#                     contains the base-weight training the KL is meant to measure against
MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.model.exclude_modules='.*visual.*'
    actor_rollout_ref.model.lora.merge=True
    actor_rollout_ref.model.unfreeze_last_n_layers=${UNFREEZE_LAST_N}
    actor_rollout_ref.model.ref_in_actor=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.ref.use_torch_compile=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LEN}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LEN}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False   # weights change every step
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console','wandb']
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=True              # token counts vary widely across tables
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.val_before_train=True           # zero-shot baseline; every later claim needs it
    trainer.save_freq=20
    trainer.test_freq=20
    trainer.total_epochs=2
)

########################### launch ###########################
"${PYBIN}" -m verl.trainer.main_ppo \
    "${ALGO[@]}" "${DATA[@]}" "${REWARD[@]}" "${MODEL[@]}" \
    "${ACTOR[@]}" "${REF[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" \
    "$@" 2>&1 | tee "logs/${EXPERIMENT_NAME}-${start_time}.log"
