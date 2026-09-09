"""Evaluate an OpenRouter model on the OOD set with the html-only prompt.

    OPENROUTER_API_KEY=... python reward/eval_ood_openrouter.py --model google/gemini-3.1-pro-preview \
        --out /home/ubuntu/ood_eval/D_gemini_html.jsonl --n 25

Reports running token spend so a small pilot can size the full run before committing.
Resumable: rows already in the output JSONL are skipped.
"""

import argparse
import base64
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import requests

HTML_ONLY = ("Convert this table image to HTML. Output only the HTML table, "
             "starting with <table> and ending with </table>. No explanation, no markdown "
             "fences, no other text.")


def img_b64(cell):
    """OOD stores {bytes, path}; train/val store a plain path string."""
    if isinstance(cell, dict) and cell.get("bytes"):
        raw = cell["bytes"]
    else:
        p = cell["path"] if isinstance(cell, dict) else cell
        raw = open(p, "rb").read()
    return base64.b64encode(raw).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--data", default="/home/ubuntu/table_exps/ood/ood_2k.parquet")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--no-reasoning", action="store_true",
                    help="send reasoning:{enabled:false} (Qwen needs it; Gemini rejects it)")
    a = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")

    t = pq.read_table(a.data, columns=["images", "reward_model", "extra_info"])
    imgs = t.column("images").to_pylist()
    gts = [r["ground_truth"] for r in t.column("reward_model").to_pylist()]
    eis = t.column("extra_info").to_pylist()
    rows = list(range(a.n or len(gts)))

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["i"])
            except Exception:
                pass
    todo = [i for i in rows if i not in done]
    print(f"{len(rows)} rows, {len(done)} done, {len(todo)} to fetch | model={a.model}", flush=True)

    lock = threading.Lock()
    fh = open(a.out, "a")
    acc = {"n": 0, "err": 0, "pt": 0, "ct": 0, "cost": 0.0}

    def one(i):
        body = {
            "model": a.model,
            "temperature": 0,
            "max_tokens": a.max_tokens,
            # Gemini 3.1 Pro rejects {"enabled": False} with
            # "Reasoning is mandatory for this endpoint"; Qwen needs it or returns
            # content=null. Only send it when the model accepts it.
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64(imgs[i][0])}"}},
                {"type": "text", "text": HTML_ONLY},
            ]}],
        }
        if a.no_reasoning:
            body["reasoning"] = {"enabled": False}
        text, err, usage = "", None, {}
        for _ in range(3):
            try:
                r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                  headers={"Authorization": f"Bearer {key}"},
                                  json=body, timeout=300)
                if r.status_code == 200:
                    j = r.json()
                    m = j["choices"][0]["message"]
                    text = m.get("content") or m.get("reasoning") or ""
                    usage = j.get("usage", {}) or {}
                    err = None
                    break
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        with lock:
            fh.write(json.dumps({"i": i, "output": text, "gts": gts[i],
                                 "plan": (eis[i] or {}).get("plan") or "",
                                 "source": (eis[i] or {}).get("pattern") or "",
                                 "error": err, "usage": usage}) + "\n")
            fh.flush()
            acc["n"] += 1
            acc["err"] += bool(err)
            acc["pt"] += usage.get("prompt_tokens", 0)
            acc["ct"] += usage.get("completion_tokens", 0)
            acc["cost"] += float(usage.get("cost", 0) or 0)
            if acc["n"] % 25 == 0 or acc["n"] == len(todo):
                print(f"  {acc['n']}/{len(todo)} err={acc['err']} "
                      f"prompt_tok={acc['pt']} completion_tok={acc['ct']} "
                      f"cost=${acc['cost']:.4f}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=a.conc) as ex:
            list(ex.map(one, todo))
    fh.close()

    n = max(acc["n"], 1)
    print(f"\nper row: prompt {acc['pt']/n:.0f} tok, completion {acc['ct']/n:.0f} tok, "
          f"cost ${acc['cost']/n:.5f}")
    print(f"projected for 2000 rows: ${acc['cost']/n*2000:.2f}")


if __name__ == "__main__":
    main()
