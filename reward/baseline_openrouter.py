"""Baseline: does the RL training actually help?

Runs the same 1000 held-out val rows through the *base* Qwen3.5-9B via OpenRouter with a
plain "emit only the HTML" prompt, then scores with the same TEDS reward the run uses.

Comparable to the trained model's step-20 held-out numbers because both use greedy
decoding (verl val_kwargs: temperature 0, do_sample False, n 1). Only the `html` / `teds`
terms are comparable — this prompt asks for no <plan>, so schema/content are 0 by design.

  OPENROUTER_API_KEY=... python reward/baseline_openrouter.py [N]

Resumable: rows already in the output JSONL are skipped.
"""

import base64
import json
import mimetypes
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import requests

sys.path.insert(0, "/home/ubuntu/verl")
from reward.teds import compute_score as teds_score  # noqa: E402

MODEL = os.environ.get("OR_MODEL", "qwen/qwen3.5-9b")
OUT = os.environ.get("OUT", "/home/ubuntu/baseline_base_model.jsonl")
KEY = os.environ.get("OPENROUTER_API_KEY")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
CONC = int(os.environ.get("CONC", "12"))

PROMPT = ("Convert this table image to HTML. Output only the HTML table, "
          "starting with <table> and ending with </table>. No explanation, no markdown "
          "fences, no other text.")

if not KEY:
    sys.exit("ERROR: OPENROUTER_API_KEY not set (put it in /home/ubuntu/verl/.env)")

t = pq.read_table("/home/ubuntu/table_exps/stage5/val.parquet", columns=["images", "reward_model"])
images = t.column("images").to_pylist()
gts = [r["ground_truth"] for r in t.column("reward_model").to_pylist()]
rows = list(enumerate(zip(images, gts)))[:N]

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            done.add(json.loads(line)["i"])
        except Exception:
            pass
todo = [r for r in rows if r[0] not in done]
print(f"{len(rows)} rows, {len(done)} already done, {len(todo)} to fetch  |  model={MODEL}")

lock = threading.Lock()
fh = open(OUT, "a")
counter = {"n": 0, "err": 0}


def one(item):
    i, (img, gt) = item
    path = img[0]
    mime = mimetypes.guess_type(path)[0] or "image/webp"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 8192,
        # Qwen3.5 is a reasoning model: without this it returns content=null and spends
        # the whole budget in `reasoning`. Same behaviour as the CoT preamble seen locally.
        "reasoning": {"enabled": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
    }
    text, err = "", None
    for attempt in range(3):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers={"Authorization": f"Bearer {KEY}"},
                              json=body, timeout=300)
            if r.status_code == 200:
                msg = r.json()["choices"][0]["message"]
                text = msg.get("content") or msg.get("reasoning") or ""
                err = None
                break
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    rec = {"i": i, "output": text, "gts": gt, "error": err}
    with lock:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        counter["n"] += 1
        counter["err"] += bool(err)
        if counter["n"] % 25 == 0:
            print(f"  {counter['n']}/{len(todo)} done, {counter['err']} errors", flush=True)


if todo:
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        list(ex.map(one, todo))
fh.close()

# ---- score ----
recs = [json.loads(l) for l in open(OUT)]
recs = [r for r in recs if r["i"] < N]
ok = [r for r in recs if not r.get("error")]
print(f"\nscored {len(ok)} rows ({len(recs) - len(ok)} errored)")

res = [teds_score("t", r["output"] or "", r["gts"], None, w_struct=0.5) for r in ok]


def m(k):
    v = [x[k] for x in res if isinstance(x.get(k), (int, float))]
    return sum(v) / len(v) if v else float("nan")


print("\n=== BASE Qwen3.5-9B via OpenRouter, html-only prompt, greedy ===")
for k in ("score", "teds", "teds_struct", "has_table", "truncated", "n_tables"):
    print(f"  {k:12} {m(k):.4f}")
print("\n=== trained model, step-20 held-out (same 1000 rows, greedy) ===")
print("  html         0.8805\n  teds         0.8761\n  teds_struct  0.8849\n  has_table    1.0000")
