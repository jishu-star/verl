# Table RL — GRPO on Qwen3.5-9B

Table image → `<plan>` trace + HTML, trained with GRPO in verl.

Two launch scripts live here. They share the dataset, the reward, and every tunable
below; they differ only in what trains and whether there is a KL term.

| | `run_qwen3_5_9b_table_grpo.sh` | `run_qwen3_5_9b_table_fullft_grpo.sh` |
|---|---|---|
| Adapter | LoRA r=32 + last 8 layers full-rank | none (`lora_rank=0`) |
| Frozen | `visual` | `visual`, `embed_tokens`, `lm_head` |
| KL | yes, `kl_coef=0.01`, real ref worker | **no** — no ref worker is built at all |
| Rollout GPU mem | 0.45 | 0.35 (optimizer needs the room) |
| Checkpoint size | ~2 GB | ~108 GB |

With no KL and no frozen base, the PPO clip ratio is the only thing bounding how far
the policy moves in one step. Read the header of the full-FT script before running it.

---

## Extra modules beyond the original runbook

The original setup was four `uv pip install` lines: `-e .`, `vllm==0.18.0`, `flash-attn`,
and `reward/requirements.txt`. That is **not enough to reach step 1** — the run reaches
the first training step and dies. Three things are missing, each with a different
failure signature.

| # | What's missing | Why the original missed it | How it fails |
|---|---|---|---|
| 1 | `transformers==5.5.3` | dropped by an extra-gated uv override; uv reports success | `ModuleNotFoundError: transformers` at import, before Ray starts |
| 2 | the `verl-core` extra | `-e .` installs base deps only; `qwen_vl_utils` lives in the extra | every rollout errors, then `RuntimeError: ... no materializable trajectories` |
| 3 | `flash-linear-attention` | not a declared dep; Qwen3.5 is 24/32 GDN layers | no crash — silently runs the slow torch path |

### The full corrected install

Replace step 3 of the original runbook with this. Order matters.

```bash
cd ~/verl && source .venv/bin/activate

uv pip install -e ".[verl-core]"                    # (1) NOT plain `-e .`
(cd /tmp && uv pip install "transformers==5.5.3")   # (2) MUST be outside the repo dir
uv pip install vllm==0.18.0
uv pip install flash-attn --no-build-isolation      # ~11 min compile on 240 vCPU
uv pip install flash-linear-attention               # (3) GDN kernels
uv pip install -r reward/requirements.txt           # apted, lxml -- SAME env as verl
```

What each of the three adds:

```
(1) .[verl-core] ->  qwen-vl-utils 0.0.14   REQUIRED - vision preprocessing, hard blocker
                     mathruler 0.1.0        other reward fns
                     nvtx 0.2.16            profiling
                     torchcodec 0.16.0      video only; see "Known-harmless breakage"
                     av 18.1.0              torchcodec dep
                     pytest / pytest-asyncio / pytest-rerunfailures / pluggy / iniconfig

(2) transformers 5.5.3  (also downgrades tokenizers 0.23.2 -> 0.22.2, which still
                         satisfies vllm's >=0.21.1)

(3) flash-linear-attention 0.5.2 + fla-core 0.5.2
```

Optional, perf only — see "Optional: `causal-conv1d`":

```bash
uv pip install causal-conv1d --no-build-isolation
```

### Confirm it took

```bash
python -c "import verl, vllm, flash_attn, fla, qwen_vl_utils, apted, lxml.html; print('all ok')"
python -c "from qwen_vl_utils import process_vision_info; print('vision ok')"
uv pip list | grep -E "^(torch|vllm|transformers|flash-attn|flash-linear-attention) "
```

Expected — and the pins must **not** have moved after any of the installs above:

```
flash-attn                 2.8.3.post1
flash-linear-attention     0.5.2
torch                      2.10.0
transformers               5.5.3
vllm                       0.18.0
```

If `torch` shows 2.11.0, something pulled the `vllm 0.24` line of deps and your
flash-attn build is now ABI-stale — reinstall flash-attn or roll torch back.

---

## 0 · Host requirements

```bash
echo $HOME; nproc; df -h $HOME; nvidia-smi --query-gpu=name,memory.total --format=csv
```

- **`$HOME` must be `/home/ubuntu`.** The parquet stores absolute image paths under
  `/home/ubuntu/table_exps`. If yours differs, rebase them (step 2).
- **Disk ≥ 400 GB** for the full-FT variant: data 7 + model 20 + three checkpoints at
  108 GB = 324. The LoRA variant needs far less.
- **vCPU count sets step time.** The reward is CPU-bound — `REWARD_WORKERS` defaults to
  `nproc * 3/4`.
- 8 × 80 GB GPUs for the defaults below.

---

## 1 · Data

```bash
mkdir -p ~/table_exps/stage5 && cd ~/table_exps
B=s3://form-checkboxes-annotation/SyntheticTableData/2026-09-05-new-table-rl-data
aws s3 cp $B/train.parquet    stage5/
aws s3 cp $B/val.parquet      stage5/
aws s3 cp $B/rebase_paths.py  stage5/
aws s3 cp $B/renders.tar - | tar -xf - -C .      # streams, no local tarball
```

## 2 · Verify before renting GPU hours on a broken copy

```bash
find stage3/renders        -type f | wc -l   # 65137
find stage5/renders_capped -type f | wc -l   # 1195

python -c "
import pyarrow.parquet as pq, os
p=[r[0] for r in pq.read_table('stage5/train.parquet',columns=['images']).column('images').to_pylist()]
print(len(p),'rows;',sum(map(os.path.exists,p)),'resolve')"   # 63816 rows; 63816 resolve
```

`val.parquet` is 1000 rows and should also resolve 1000/1000.

If `$HOME` wasn't `/home/ubuntu` this is where it shows up as zeros:

```bash
python stage5/rebase_paths.py --old /home/ubuntu/table_exps --new $HOME/table_exps --check-exists
export DATA_DIR=$HOME/table_exps/stage5
```

---

## 3 · Code and environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone -b table-rl https://github.com/jishu-star/verl.git ~/verl && cd ~/verl
uv venv --python 3.12 && source .venv/bin/activate
```

Then, **in this order** — the three annotated lines are not optional and not obvious:

```bash
# (a) NOT plain `-e .`  -- qwen_vl_utils, mathruler, nvtx, torchcodec live in the
#     verl-core extra, and a bare `-e .` installs base dependencies only.
uv pip install -e ".[verl-core]"

# (b) transformers must be installed from OUTSIDE the repo directory.  See "The two
#     install traps" below.  5.5.3 is what the vllm extra pins.
(cd /tmp && uv pip install "transformers==5.5.3")

uv pip install vllm==0.18.0            # pinned: Qwen3.5 is 24/32 linear-attention
uv pip install flash-attn --no-build-isolation
uv pip install flash-linear-attention  # GDN kernels for the 24 linear-attention layers
uv pip install -r reward/requirements.txt   # apted, lxml -- SAME env as verl
```

The last line matters: the reward runs **inside the trainer process**, so `apted` in a
different venv means the run dies at the first reward call, not at import. The launch
scripts preflight `compute_score` before Ray starts, so a miss fails in ~1 s.

Verify:

```bash
python -c "import verl, vllm, flash_attn, fla, qwen_vl_utils, apted, lxml.html; print('ok')"
uv pip list | grep -E "^(torch|vllm|transformers|flash-attn|fla) "
```

Expected: `torch 2.10.0`, `vllm 0.18.0`, `transformers 5.5.3`, `flash-attn 2.8.3.post1`.

### The two install traps

**(a) `uv pip install -e .` silently omits `transformers`.** `pyproject.toml` declares

```toml
override-dependencies = [
  "transformers==5.5.3 ; extra == 'vllm'",
  "transformers>=5.3.0,<=5.5.3 ; extra == 'verl-core'",
  ...
]
```

Every transformers override is gated on an `extra ==` marker. A plain `uv pip install`
run *from inside the repo* activates no extra, no override matches — and a uv override
matching nothing **drops the requirement entirely**. uv reports success; the run dies
much later at `verl/utils/torch_functional.py:30` with
`ModuleNotFoundError: No module named 'transformers'`. Worse, `uv pip install --dry-run
transformers` inside the repo says *"Would make no changes"*. Running from `/tmp` sidesteps
it because pyproject isn't read there.

**(b) `qwen_vl_utils` is in the `verl-core` extra, not the base deps.** Without it every
vision prompt fails in `rl_dataset.py:491`, all rollouts return nothing, and the step dies with

```
RuntimeError: Sync replay buffer selected terminal groups with no materializable trajectories.
```

which names the buffer, not the missing module. Check the rollout workers' tracebacks
(`Error in _run_prompt`) for the real cause.

### Optional: `causal-conv1d`

`flash-linear-attention` alone leaves this warning:

> The fast path is not available because one of the required library is not installed.

That gate is all-or-nothing:

```python
is_fast_path_available = all(
    (causal_conv1d_fn, causal_conv1d_update, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
)
```

but the bindings fall back **independently**. `fla` supplies the two gated-delta-rule
kernels — the expensive part — so installing it is the win. Only `causal_conv1d_fn`
stays unbound, and its fallback is one depthwise conv + SiLU
(`F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])`). Correctness is unaffected. Add it
only if you want that last bit of throughput on the 24 GDN layers:

```bash
uv pip install causal-conv1d --no-build-isolation   # CUDA compile, can take a while
```

### Known-harmless breakage

`torchcodec` installs at 0.16.0, built against torch 2.11, and fails to load on torch
2.10 (`OSError: Could not load this library: libtorchcodec_image.so`). It is imported
lazily and **only for video**; this dataset is images, so it never runs. But
`is_torchcodec_available()` uses `find_spec`, which returns True while the real import
raises — so a video sample would die there rather than fall back.

---

## 4 · Model and logging

```bash
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ~/models/Qwen3.5-9B   # ~20 GB
wandb login
```

---

## 5 · Running

```bash
cd ~/verl && source .venv/bin/activate
MODEL_PATH=~/models/Qwen3.5-9B bash reward/run_qwen3_5_9b_table_fullft_grpo.sh
```

Everything is env-overridable, and extra Hydra overrides are forwarded after `"$@"`.
A 2-step smoke test:

```bash
MODEL_PATH=~/models/Qwen3.5-9B TRAIN_BATCH=8 ROLLOUT_N=2 REWARD_WORKERS=16 \
bash reward/run_qwen3_5_9b_table_fullft_grpo.sh \
  trainer.total_training_steps=2 trainer.val_before_train=False "trainer.logger=[console]"
```

`PYBIN` defaults to `python3`, resolved off `PATH` — so the venv must be **activated**,
or set `PYBIN=~/verl/.venv/bin/python` explicitly (matters for cron / bare shells).

### Knobs

| Variable | Default | Notes |
|---|---|---|
| `MODEL_PATH` | `Qwen/Qwen3.5-9B` | local dir avoids a re-download |
| `DATA_DIR` | `/home/ubuntu/table_exps/stage5` | |
| `NDEVICES_PER_NODE` / `NNODES` | 8 / 1 | |
| `TRAIN_BATCH` | 64 | prompts per step |
| `ROLLOUT_N` | 8 | completions per prompt |
| `PPO_MINI_BATCH` | `TRAIN_BATCH / 4` | 4 updates/step; floored so every rank gets a sequence |
| `GEN_TP` | 2 | rollout tensor parallel |
| `REWARD_WORKERS` | `nproc * 3/4`, min 8 | reward is CPU-bound |
| `W_HTML` / `W_SCHEMA` / `W_CONTENT` | 0.75 / 0.05 / 0.20 | |
| `FREEZE_EMBEDDINGS` | 1 | 0 also trains `embed_tokens` + `lm_head` |
| `MAX_CKPT` | 3 | full-rank ckpt is ~108 GB; unset retention fills 1 TB in ~90 steps |

`PPO_MINI_BATCH` is derived, not hardcoded. verl multiplies it by `rollout.n`
(`ray_trainer.py:1351`), so a step dispatches `PPO_MINI_BATCH * ROLLOUT_N` sequences
across the data-parallel ranks; the floor is `ceil(world_size / ROLLOUT_N)` so no rank
goes empty. At the defaults it evaluates to 16 — the value it was previously hardcoded to.

### Reward weights need a `+`

In the launch scripts the three weights carry a leading `+`:

```
    reward.custom_reward_function.path=reward/table_reward.py       # no +
    +reward.custom_reward_function.reward_kwargs.w_html=${W_HTML}   # +
```

`reward.yaml` declares only `path` and `name` under `custom_reward_function`, so under
Hydra struct append the key. Declaring
`reward_kwargs: {}` in the yaml does **not** help — an empty dict in struct mode still
rejects new keys. `verl/trainer/ppo/reward.py:82` reads it via `.get("reward_kwargs", {})`.

---

## 6 · Verifying a run

**The freeze report.** Emitted once by rank 0 as the FSDP actor is built:

```bash
grep -A15 "selective training" logs/qwen3_5-9b-grpo-fullft-nokl-*.log
```

Measured on `Qwen3.5-9B` with the default `FREEZE_EMBEDDINGS=1`:

```
selective training: 6.92 B / 9.41 B trainable (73.5%)
  layers.0-31                      6.92 B   trainable
  visual                         456.01 M   frozen
  language_model                   1.02 B   mixed
  lm_head                          1.02 B   frozen
```

`language_model` reads "mixed" because the group holds both the frozen `embed_tokens`
(1.02 B = 248k vocab × 4096) and the trainable final norm.

> The script header quotes **7.32 B / 9.79 B (74.8%)**. The measured checkpoint is
> 9.41 B total / 6.92 B trainable. The freeze *targeting* is correct — every pattern
> matched real parameters — but the counts in that comment are stale. Trust the report.

`freeze_patterns` is `re.search` against parameter names and a pattern matching **nothing
raises** (`verl/utils/param_groups.py`), so these flags are self-checking rather than
hopeful — a silent no-op is impossible.

Worker `print()` reaches the tee'd driver log only after a flush lag; if the grep comes
up empty on a live run, look in `/tmp/ray/session_latest/logs/worker-*.out`.

**Metrics to watch.** `actor/entropy` is logged because the full-FT script sets
`actor.calculate_entropy=True` — with no KL term it is the one signal that says whether
rollouts are collapsing. Also watch `critic/score/mean` and the reward's per-component
curves (`html`, `schema`, `content`, `error`); `compute_score` returns a dict precisely
so a stalled run shows *which* half it lost.

---

## 7 · The reward

`table_reward.py` → three components over two disjoint halves of the completion:

| Component | Weight | Source |
|---|---|---|
| `html` | 0.75 | TEDS against ground-truth table — `teds.py` |
| `schema` | 0.05 | are the four section keys extractable — `trace_reward.py` |
| `content` | 0.20 | the K:V pairs inside each section — `trace_reward.py` |

The halves cannot contaminate each other: `teds.py` strips `<plan>` before looking for a
table, and `extract_trace` stops at `<table` before reading trace lines. `schema` is
deliberately tiny — it is a format check, and paying much for four literal words invites
the policy to farm it.

`compute_score` **never raises**; an exception would kill the training step and early
rollouts are arbitrary text. Failures surface as the `error` curve instead.

**Data contract** — the image is *not* passed to the reward, so the parquet row must carry:

- `reward_model.ground_truth` — the minified canonical HTML
- `extra_info["plan"]` — the ground-truth `<plan>` text

Cost, measured on the real corpus: 3.7 s mean / 15 s p90 per call (APTED runs twice per
table — structure-only 0.63 s, then with-text 3.2 s). A default step is
`TRAIN_BATCH * ROLLOUT_N` = 512 calls ≈ 1900 core-seconds, which is why `REWARD_WORKERS`
scales with `nproc`.
