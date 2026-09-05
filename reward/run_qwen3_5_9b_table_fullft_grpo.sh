#!/usr/bin/env bash
# GRPO | Qwen3.5-9B | table image -> <plan> + HTML
#
#   Full-rank language model, frozen vision tower, NO KL term.
#
# Differs from run_qwen3_5_9b_table_grpo.sh (LoRA + last-8 + KL) in three ways:
#
#   1. no LoRA at all (lora_rank=0), the decoder trains at full rank
#   2. model.freeze_patterns=['visual'] is the ONLY thing frozen
#   3. use_kl_loss=False and use_kl_in_reward=False, which makes
#      need_reference_policy() (trainer/ppo/utils.py:79) return False -- verl then
#      never constructs the ref worker at all, so a whole forward pass per step
#      disappears along with the KL.
#
# Read the consequences before running: with no KL and no frozen base, the PPO clip
# ratio is the only thing bounding how far the policy can move in one step.
#
# Run from the verl repo root:  bash reward/run_qwen3_5_9b_table_fullft_grpo.sh

set -xeuo pipefail

########################### user-adjustable ###########################
PROJECT_NAME=${PROJECT_NAME:-table-rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5-9b-grpo-fullft-nokl}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
DATA_DIR=${DATA_DIR:-/home/ubuntu/table_exps/stage5}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
TEST_FILE=${TEST_FILE:-${DATA_DIR}/val.parquet}
CKPTS_DIR=${CKPTS_DIR:-${HOME}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}}

NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
GEN_TP=${GEN_TP:-2}
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-${NDEVICES_PER_NODE}}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}   # lower than LoRA: optimizer needs the room

# --- what gets trained ---
# freeze_patterns is re.search against parameter names, and a pattern matching NOTHING
# raises (utils/param_groups.py) -- so these are self-checking, not hopeful.
#
#   'visual'        -> 0.43 B, the whole vision tower (333 tensors, verified in the
#                      checkpoint weight map). Never trains.
#   'embed_tokens'  -> 1.02 B. vocab is 248,320 x 4096; AdamW keeps dense fp32 moments
#   'lm_head'       -> 1.02 B. for every row of both, which is ~32 GB of optimizer state
#                      spent on a task that introduces no new vocabulary. Freezing lm_head
#                      also anchors the output distribution, which is worth something now
#                      that there is no KL doing it.
#
# FREEZE_EMBEDDINGS=0 trains them anyway: 9.36 B trainable instead of 7.32 B.
FREEZE_EMBEDDINGS=${FREEZE_EMBEDDINGS:-1}
if [[ "${FREEZE_EMBEDDINGS}" == "1" ]]; then
    FREEZE_PATTERNS="['visual','embed_tokens','lm_head']"
else
    FREEZE_PATTERNS="['visual']"
fi

# --- reward weights ---
W_HTML=${W_HTML:-0.75}
W_SCHEMA=${W_SCHEMA:-0.05}
W_CONTENT=${W_CONTENT:-0.20}

# --- reward parallelism ---
# MEASURED: compute_score is 3.7 s mean / 15 s p90 per call (APTED runs twice per table).
# One step is TRAIN_BATCH * ROLLOUT_N = 512 calls, ~1900 core-seconds. reward.num_workers
# is real process parallelism (experimental/reward_loop/reward_loop.py:324-352).
REWARD_WORKERS=${REWARD_WORKERS:-$(( $(nproc) * 3 / 4 ))}
if (( REWARD_WORKERS < 8 )); then REWARD_WORKERS=8; fi

# --- memory placement ---
# 8xH100 (640 GB): ~7.3 B trainable x ~16 B/param = 117 GB of FSDP state, 15 GB/GPU.
#   That fits on device, so keep offload OFF -- CPU offload of optimizer state is the
#   single largest avoidable cost in a full fine-tune.
# 2xH100 (160 GB): it does not fit. Offload is mandatory and the step gets much slower.
if (( NDEVICES_PER_NODE >= 8 )); then
    OFFLOAD=${OFFLOAD:-0}
else
    OFFLOAD=${OFFLOAD:-1}
fi
if [[ "${OFFLOAD}" == "1" ]]; then OFF=True; else OFF=False; fi

# --- optimisation ---
# 1e-6 was already a full-fine-tune LR (it was conservative for LoRA). Halved here
# because removing the KL removes the only term that pulled a bad step back.
LR=${LR:-5e-7}

# --- sequence budget, from the measured corpus ---
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4608}
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
    algorithm.use_kl_in_reward=False
)

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH}
    data.max_prompt_length=${MAX_PROMPT_LEN}
    data.max_response_length=${MAX_RESPONSE_LEN}
    data.image_key=images
    data.image_patch_size=16
    data.shuffle=True
    data.filter_overlong_prompts=False       # would silently DROP tall tables
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
    reward.num_workers=${REWARD_WORKERS}
)

# No LoRA. freeze_patterns is the whole selective-training story, and because nothing
# is unfrozen on top of it, _validate_selective_training (utils/config.py:228) returns
# early -- the lora.merge / ref_in_actor requirements do not apply here.
MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank=0
    actor_rollout_ref.model.freeze_patterns="${FREEZE_PATTERNS}"
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=False        # <-- no ref worker is built at all
    actor_rollout_ref.actor.entropy_coeff=0
    # Compute entropy for LOGGING without adding it to the loss.  trainer_base.py:1733
    # gates it on `actor.calculate_entropy or entropy_coeff != 0`, so with the coeff at 0
    # the actor/entropy panel is simply absent -- and with no KL term, a falling entropy
    # is the earliest sign the rollouts are collapsing to identical samples, which zeroes
    # the GRPO advantage and stalls learning while every other metric still looks fine.
    actor_rollout_ref.actor.calculate_entropy=True
    actor_rollout_ref.actor.clip_ratio=0.2           # now the ONLY leash on step size
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=${OFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFF}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFF}
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
)

# (no REF array -- there is no reference policy)

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
    actor_rollout_ref.rollout.enable_prefix_caching=False
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
    trainer.balance_batch=True
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.val_before_train=True
    trainer.save_freq=10                     # halved: no KL means a bad run diverges fast
    trainer.test_freq=10
    # A full-rank checkpoint is ~108 GB (bf16 model 19.6 + fp32 master 29.3 + Adam m,v
    # 58.6), against ~2 GB for a LoRA one.  Retention defaults to null = keep every
    # single save, so save_freq=10 fills a 1 TB disk after ~90 steps.
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT:-3}
    trainer.total_epochs=2
)

########################### launch ###########################
"${PYBIN}" -m verl.trainer.main_ppo \
    "${ALGO[@]}" "${DATA[@]}" "${REWARD[@]}" "${MODEL[@]}" \
    "${ACTOR[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" \
    "$@" 2>&1 | tee "logs/${EXPERIMENT_NAME}-${start_time}.log"
