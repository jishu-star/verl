# Table RL — GRPO on Qwen3.5-9B

Table image → `<plan>` structure trace + HTML, trained with GRPO in verl.

**Start here if you are on a fresh machine:** [§1 Quickstart](#1-quickstart-fresh-machine).
Everything needed is in this repo; the only external inputs are the S3 dataset, the model
weights, and your credentials.

---

## 0 · Results

Held-out, 1000 rows of `val.parquet`, greedy (`temperature 0, do_sample False`).

| | reward | html | schema | content | teds | has_table |
|---|---|---|---|---|---|---|
| base Qwen3.5-9B, html-only prompt † | — | — | — | — | **0.8281** | 1.000 |
| step 0 (untrained, v2 prompt) | 0.2790 | 0.2862 | 0.5318 | 0.1887 | 0.2082 | 0.554 |
| step 10 | 0.7885 | 0.8497 | 0.9975 | 0.5064 | 0.8449 | 0.992 |
| **step 20** | **0.8402** | 0.8805 | 0.9993 | 0.6491 | 0.8761 | 1.000 |

† `reward/baseline_openrouter.py`, 1002 rows, 0 errors — the base model asked only for
HTML, with reasoning disabled. **Training is worth +0.048 TEDS on the table** (0.828 →
0.876), plus the entire plan half, which the base model does not attempt.

`reward = 0.75·html + 0.05·schema + 0.20·content`.

---

## 1 · Quickstart (fresh machine)

Assumes 8×80 GB GPUs, ≥400 GB disk, and `$HOME=/home/ubuntu` (the parquet stores absolute
image paths; see [§3](#3-data) to rebase if yours differs).

```bash
# --- code ---
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone -b table-rl https://github.com/jishu-star/verl.git ~/verl && cd ~/verl
uv venv --python 3.12 && source .venv/bin/activate

uv pip install -e ".[verl-core]"                    # (1) NOT plain `-e .`
(cd /tmp && uv pip install "transformers==5.5.3")   # (2) MUST run outside the repo
uv pip install vllm==0.18.0                         # pinned: Qwen3.5 is 24/32 GDN layers
uv pip install flash-attn --no-build-isolation      # ~11 min compile on 240 vCPU
uv pip install flash-linear-attention               # (3) GDN kernels
uv pip install -r reward/requirements.txt           # apted, lxml — SAME env as verl

# --- data + model ---
mkdir -p ~/table_exps/stage5 && cd ~/table_exps
B=s3://form-checkboxes-annotation/SyntheticTableData/2026-09-05-new-table-rl-data
aws s3 cp $B/rl-run-20260905/data-v2-prompt/train.parquet stage5/   # already v2-prompted
aws s3 cp $B/rl-run-20260905/data-v2-prompt/val.parquet   stage5/
aws s3 cp $B/renders.tar - | tar -xf - -C .
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ~/models/Qwen3.5-9B
wandb login

# --- verify, then launch ---
cd ~/verl && python reward/verify_setup.py
MODEL_PATH=~/models/Qwen3.5-9B TRAIN_BATCH=256 \
  bash reward/run_qwen3_5_9b_table_fullft_grpo.sh \
    trainer.rollout_data_dir=/home/ubuntu/rollout_dumps
```

The three numbered lines are not optional. Each fails differently and none is obvious —
see [§2](#2-the-three-install-traps).

**If you pull `train.parquet` from the original dataset prefix rather than
`rl-run-20260905/data-v2-prompt/`, it carries the OLD prompt** and `schema`/`content`
(0.25 of the reward) will score ~0. Apply the current prompt with:

```bash
cp reward/prompts/rewrite_prompt.py ~/table_exps/stage5/
cd ~/table_exps/stage5 && python rewrite_prompt.py train.parquet val.parquet
```

---

## 2 · The three install traps

**(1) `uv pip install -e .` silently omits `transformers` and `qwen_vl_utils`.**
`pyproject.toml` declares `override-dependencies` gated on `extra ==` markers. A plain
`uv pip install` from inside the repo activates no extra, so no override matches — and a
uv override matching nothing *drops the requirement entirely*. uv reports success. The run
dies later at `verl/utils/torch_functional.py:30`. `qwen_vl_utils` (along with
`mathruler`, `nvtx` and `torchcodec`) is separately only in the `verl-core` extra; without it every vision rollout fails and the step dies with
`RuntimeError: ... no materializable trajectories`, which names the buffer, not the cause.

**(2) transformers must be installed from OUTSIDE the repo directory** — `cd /tmp` first,
so `pyproject.toml` isn't read. Inside the repo, even `uv pip install --dry-run
transformers` reports "Would make no changes".

**(3) `flash-linear-attention` is undeclared** but Qwen3.5 is 24/32 linear-attention
layers; without it GDN silently runs the torch fallback. `causal-conv1d` is optional —
`fla` supplies the expensive gated-delta-rule kernels, and only the cheap depthwise conv
falls back. The "fast path is not available" warning persists either way because the gate
is `all()` of four bindings.

**Known-harmless:** `torchcodec` installs at 0.16.0 (built for torch 2.11) and fails to
load on torch 2.10. It is imported lazily and only for video; this dataset is images.

---

## 3 · Data

`$HOME` must be `/home/ubuntu` or the parquet's absolute image paths won't resolve:

```bash
python stage5/rebase_paths.py --old /home/ubuntu/table_exps --new $HOME/table_exps --check-exists
export DATA_DIR=$HOME/table_exps/stage5
```

Verify before renting GPU hours:

```bash
find stage3/renders -type f | wc -l          # 65137
python -c "
import pyarrow.parquet as pq, os
p=[r[0] for r in pq.read_table('stage5/train.parquet',columns=['images']).column('images').to_pylist()]
print(len(p),'rows;',sum(map(os.path.exists,p)),'resolve')"   # 63816 rows; 63816 resolve
```

---

## 4 · Running

```bash
MODEL_PATH=~/models/Qwen3.5-9B TRAIN_BATCH=256 \
  bash reward/run_qwen3_5_9b_table_fullft_grpo.sh \
    trainer.rollout_data_dir=/home/ubuntu/rollout_dumps
```

`PYBIN` defaults to `python3` off `PATH`, so **activate the venv** or set
`PYBIN=~/verl/.venv/bin/python` (matters for cron / bare shells).

| Variable | Default | Notes |
|---|---|---|
| `TRAIN_BATCH` / `ROLLOUT_N` | 256 / 8 | 2048 sequences per step |
| `PPO_MINI_BATCH` | `TRAIN_BATCH/4` | derived; floored so no rank goes empty |
| `MAX_PROMPT_LEN` | 5120 | largest real prompt is 4534 (v2 prompt + 4 MP image) |
| `MAX_RESPONSE_LEN` | 8192 | 4096 truncated 62% of rollouts |
| `REWARD_WORKERS` | `nproc*3/4` | reward is CPU-bound |
| `W_HTML`/`W_SCHEMA`/`W_CONTENT` | .75/.05/.20 | |
| `FREEZE_EMBEDDINGS` | 1 | freezes `visual`, `embed_tokens`, `lm_head` |
| `MAX_CKPT` | 3 | a full-rank checkpoint is 87 GB |

Step cost at these settings: ~33 min, ~11 days for all 498 steps. `test_freq=10` adds a
1000-row validation (~25 min) and `save_freq=10` an 87 GB write.

### Reward weights need a leading `+`

In the launch scripts the three weights carry a `+` and `path`/`name` do not:

```
    reward.custom_reward_function.path=reward/table_reward.py       # no +
    +reward.custom_reward_function.reward_kwargs.w_html=${W_HTML}   # +
```

`reward/reward.yaml` declares only `path` and `name` under `custom_reward_function`, so
under Hydra struct mode adding `reward_kwargs` needs `+` to append the key. Declaring
`reward_kwargs: {}` in the yaml does **not** help — an empty dict in struct mode still
rejects new keys. `verl/trainer/ppo/reward.py:82` reads it via `.get("reward_kwargs", {})`.

**Resume** — `resume_mode: auto` finds the latest `global_step_N` and restores weights,
Adam state, step counter and dataloader offset:

```bash
bash reward/resume_with_dumps.sh                          # adds dumps, skips re-validation
bash reward/resume_with_dumps.sh trainer.total_training_steps=150
```

Do **not** change `TRAIN_BATCH` on resume — the saved dataloader offset is in units of the
old batch size.

---

## 5 · Reading a run

```bash
grep -A15 "selective training" logs/*.log        # 6.92 B / 9.41 B trainable (73.5%)
grep -ao "step:20 - .*" logs/*.log | tr ' ' '\n' | grep val-   # held-out metrics
bash reward/inspect_rollouts.sh                  # per-step table from the dumps
bash reward/inspect_rollouts.sh 23 plan          # the <plan> blocks at step 23
python reward/baseline_openrouter.py 1000        # base-model comparison (needs OPENROUTER_API_KEY)
```

Worker `print()` reaches the tee'd log only after a flush lag; if a grep comes up empty on
a live run, look in `/tmp/ray/session_latest/logs/worker-*.out`. Validation metrics for
step N are buffered and print alongside step N's training metrics.

`actor/entropy` is the metric to watch: there is no KL term, so nothing anchors the policy.
It fell 0.256 → 0.056 over 25 steps while score rose; entropy falling *while score is flat*
is the signal to stop.

---

## 6 · The reward

`table_reward.py` → three components over two disjoint halves of the completion.

| | weight | what | source |
|---|---|---|---|
| `html` | 0.75 | TEDS vs ground truth, `0.5·struct + 0.5·with-text` | `teds.py` |
| `schema` | 0.05 | are the four plan keys present (0/.25/.5/.75/1) | `trace_reward.py` |
| `content` | 0.20 | per-section F1 of what's inside them | `trace_reward.py` |

The halves can't contaminate each other: `teds.py` strips `<plan>` before looking for a
table, and `extract_trace` stops at `<table`. `compute_score` never raises — failures
surface as the `error` curve.

**Data contract** — the image is not passed to the reward, so the row must carry
`reward_model.ground_truth` (minified canonical HTML) and `extra_info["plan"]`.

Cost: 3.7 s mean / 15 s p90 per call (APTED runs twice per table).

### Plan format (what `content` actually checks)

```
<plan>
headers:
  Stub
  GroupA [3 cols]
    SubA [2 cols]
      ColOne
row groups:
  SectOne [4 cols] (section)
    RowA [2 rows]
merged: none
empty:
  SectOne (section)  /  GroupA › SubA › ColTwo
</plan>
```

- keys are lowercase, at column 0, in this order; `key: none` when empty
- 2-space indent per nesting level
- spans as ` [3 cols]` / ` [4 rows]` / ` [2r x 3c]`; **omit entirely for 1×1**
- `empty` uses `  /  ` (two spaces, slash, two spaces) **once**, between row and column;
  ` › ` (U+203A) joins a nested **column** path only
- over 40 entries: `key: <total> (first 40)` then only the first 40

`schema_reward` is lenient (case-insensitive, tolerates `row_groups`); `parse_trace`,
which gates all of `content`, is strict.

---

## 7 · Known issues

**Per-section `content` at step 23** — `content` averages over sections live on either
side, so the weak ones cap the term:

| section | F1 | diagnosis |
|---|---|---|
| `headers` | 0.900 | fine |
| `row groups` | 0.681 | under-lists nested children by 27% |
| `merged` | 0.216 | 20.7% of entries are plain 1×1 cells; under-lists by 26%; over-reports rowspan (34% vs GT 15%) |
| `empty` | 0.005 | **35% of entries use ` › ` where `  /  ` belongs** — the cell is identified correctly but can never match |

The `empty` failure is a prompt-clarity problem, not perception: the prompt never says how
to render a *nested row*, and the model resolves that by using ` › ` throughout. Worth
fixing in `reward/prompts/rewrite_prompt.py` before the next run.

**Fixed here:** `_SPAN_RE` accepted only plural `cols`/`rows`, so a model writing the
grammatical `[1 col]` lost both the span *and* the text (the bracket stayed glued on).
That silently zeroed 8.9% of `merged` entries and a share of `row groups`.

**Prompt is load-bearing.** v0 never stated the four-key format, so `schema`/`content`
scored ~0. v1 added the format but framed the plan as "your reasoning" — Qwen3.5 is a
reasoning model, and 99.9% of rollouts then opened with prose, 39.4% exhausting the token
budget before emitting any tag. v2 (current) forbids the preamble. Note that **no prompt
wording moved `starts-with-<plan>` off 0.0%** — training did, taking plan+table from 35% to
100% in seven steps.

**Intermittent mRoPE crash** — `RuntimeError: The size of tensor a (N) must match the size
of tensor b (4) at non-singleton dimension 2` at `modeling_qwen3_5.py:573`, in
`compute_log_prob`. Seen once in three early 2-step runs, never since. A conditional probe
in `verl/workers/engine/fsdp/transformer_impl.py` prints `[DBG-MROPE-ANOMALY]` only on a
malformed layout; it has never fired in a full run.

---

## 8 · Archive

```
s3://form-checkboxes-annotation/SyntheticTableData/2026-09-05-new-table-rl-data/rl-run-20260905/
  checkpoints/global_step_10, _20     87 GB each, FSDP-sharded, resumable
  rollout_dumps/N.jsonl               every rollout with its score
  data-v2-prompt/                     the parquets carrying the v2 prompt
  code/ logs/ RUN_MANIFEST.md
```

Re-archive with `bash reward/upload_to_s3.sh` (`DRYRUN=1` to preview).
