"""One-shot preflight for a fresh machine. Run before launching a training job.

    python reward/verify_setup.py

Checks every trap that cost time on the first setup: silently-missing deps, the prompt
version baked into the parquet, data path resolution, prompt-length headroom, GPUs, disk.
Exits non-zero if anything would break the run.
"""

import importlib
import os
import shutil
import subprocess
import sys

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "
fails = []
warns = []


def check(label, cond, detail="", fatal=True):
    tag = OK if cond else (BAD if fatal else WARN)
    print(f"[{tag}] {label}{('  — ' + detail) if detail else ''}")
    if not cond:
        (fails if fatal else warns).append(label)
    return cond


print("=" * 72)
print("imports")
print("=" * 72)
for mod, why in [("verl", "the trainer"), ("vllm", "rollout engine"),
                 ("transformers", "dropped by the uv override trap"),
                 ("qwen_vl_utils", "lives in the verl-core extra"),
                 ("flash_attn", "attention kernels"),
                 ("fla", "GDN kernels for 24/32 linear-attention layers"),
                 ("apted", "TEDS — must be in the SAME env as verl"),
                 ("lxml", "HTML parsing")]:
    try:
        importlib.import_module(mod)
        check(f"import {mod}", True)
    except Exception as e:
        check(f"import {mod}", False, f"{why} — {type(e).__name__}")

print()
print("=" * 72)
print("versions")
print("=" * 72)
try:
    import torch
    import transformers
    import vllm
    check("torch 2.10.x", torch.__version__.startswith("2.10"), torch.__version__, fatal=False)
    check("transformers 5.5.3", transformers.__version__ == "5.5.3", transformers.__version__, fatal=False)
    check("vllm 0.18.0", vllm.__version__ == "0.18.0", vllm.__version__, fatal=False)
    check("CUDA visible", torch.cuda.is_available(), f"{torch.cuda.device_count()} GPUs")
except Exception as e:
    check("version probe", False, str(e))

print()
print("=" * 72)
print("reward")
print("=" * 72)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from reward.table_reward import compute_score
    gt = "<table><tr><td>a</td></tr></table>"
    plan = "<plan>\nheaders:\n  a\nrow groups: none\nmerged: none\nempty: none\n</plan>"
    r = compute_score("t", plan + gt, gt, {"plan": plan})
    check("compute_score runs", r["error"] == 0.0)
    check("scores a perfect answer as 1.0", r["score"] > 0.99, f"got {r['score']:.3f}")
    from reward.trace_reward import _split_span
    check("singular [1 col] parses", _split_span("Foo [1 col]") == ("Foo", 1, 1),
          "the _SPAN_RE plural-only bug", fatal=False)
except Exception as e:
    check("reward import/run", False, f"{type(e).__name__}: {e}")

print()
print("=" * 72)
print("data")
print("=" * 72)
DATA = os.environ.get("DATA_DIR", os.path.expanduser("~/table_exps/stage5"))
train = os.path.join(DATA, "train.parquet")
val = os.path.join(DATA, "val.parquet")
if check(f"{train} exists", os.path.exists(train)):
    import pyarrow.parquet as pq
    t = pq.read_table(train, columns=["prompt", "images"])
    prompts = {m[0]["content"] for m in t.column("prompt").to_pylist()}
    p = next(iter(prompts))
    check("one prompt across all rows", len(prompts) == 1, f"{len(prompts)} distinct")
    v2 = "must begin with" in p and "row groups:" in p
    check("parquet carries the v2 prompt", v2,
          "OLD prompt — schema/content will score ~0; see README §1" if not v2 else f"{len(p)} chars")
    paths = [r[0] for r in t.column("images").to_pylist()]
    n_ok = sum(map(os.path.exists, paths))
    check("image paths resolve", n_ok == len(paths), f"{n_ok}/{len(paths)}")
if os.path.exists(val):
    import pyarrow.parquet as pq
    check("val.parquet rows", pq.ParquetFile(val).metadata.num_rows == 1000,
          str(pq.ParquetFile(val).metadata.num_rows), fatal=False)

print()
print("=" * 72)
print("budgets")
print("=" * 72)
MODEL = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen3.5-9B"))
check(f"{MODEL} exists", os.path.isdir(MODEL))
free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
check("disk >= 400 GB free", free_gb >= 400, f"{free_gb:.0f} GB", fatal=False)
check("$HOME is /home/ubuntu", os.path.expanduser("~") == "/home/ubuntu",
      "parquet stores absolute paths; rebase if not — README §3", fatal=False)
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20).stdout.split("\n")
    used = [int(x.split()[0]) for x in out if x.strip()]
    check("GPUs idle", all(u < 1000 for u in used), f"used: {used}", fatal=False)
except Exception:
    pass

print()
print("=" * 72)
if fails:
    print(f"FAILED: {len(fails)} blocking issue(s)")
    for f in fails:
        print("   -", f)
    sys.exit(1)
print("all blocking checks passed" + (f" ({len(warns)} warning(s))" if warns else ""))
for w in warns:
    print("   warn:", w)
