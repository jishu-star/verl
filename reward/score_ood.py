"""Score the OOD eval JSONLs and print the three-condition comparison.

    python reward/score_ood.py /home/ubuntu/ood_eval

Reads A_base_plan.jsonl / B_base_html.jsonl / C_finetuned_plan.jsonl, scores each row
with the same compute_score the training run used, and reports overall plus per-source.
Condition B asks for no plan, so only its html/teds terms are meaningful.
"""

import glob
import importlib.util
import json
import os
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/ubuntu/verl")
_spec = importlib.util.spec_from_file_location("tr", "/home/ubuntu/verl/reward/table_reward.py")
_tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr)


def work(r):
    d = _tr.compute_score("t", r.get("output") or "", r["gts"], {"plan": r.get("plan") or ""})
    d["_source"] = r.get("source", "")
    o = r.get("output") or ""
    d["_has_plan"] = float("<plan>" in o)
    d["_has_table"] = float("<table" in o)
    d["_chars"] = len(o)
    d["_preamble"] = float(not o.lstrip().startswith(("<plan>", "<table")))
    return d


def mean(rows, k):
    v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return st.fmean(v) if v else float("nan")


LABEL = {"A_base_plan": "A  base + plan prompt",
         "B_base_html": "B  base + html-only",
         "C_finetuned_plan": "C  finetuned + plan prompt"}

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/ood_eval"
    results = {}
    for tag in ("A_base_plan", "B_base_html", "C_finetuned_plan"):
        p = os.path.join(d, tag + ".jsonl")
        if not os.path.exists(p):
            print(f"[skip] {p} missing")
            continue
        rows = [json.loads(l) for l in open(p)]
        with ProcessPoolExecutor(max_workers=48) as ex:
            results[tag] = list(ex.map(work, rows, chunksize=4))
        print(f"[ok] {tag}: {len(rows)} rows scored", flush=True)

    if not results:
        sys.exit("nothing to score")

    cols = ["score", "html", "schema", "content", "teds", "teds_struct",
            "_has_plan", "_has_table", "_preamble", "truncated"]
    print("\n" + "=" * 96)
    print(f"OOD — real document tables ({len(next(iter(results.values())))} rows), greedy")
    print("=" * 96)
    hdr = f"{'condition':<30}" + "".join(f"{c.replace('_',''):>11}" for c in cols[:7])
    print(hdr)
    print("-" * 96)
    for tag, rows in results.items():
        line = f"{LABEL[tag]:<30}"
        for c in cols[:7]:
            line += f"{mean(rows, c):>11.4f}"
        print(line)
    print()
    hdr2 = f"{'condition':<30}" + "".join(f"{c.replace('_',''):>13}" for c in cols[7:])
    print(hdr2)
    print("-" * 96)
    for tag, rows in results.items():
        line = f"{LABEL[tag]:<30}"
        for c in cols[7:]:
            line += f"{mean(rows, c):>13.4f}"
        print(line)

    print("\n" + "=" * 96)
    print("teds by source")
    print("=" * 96)
    srcs = sorted({r["_source"] for rows in results.values() for r in rows})
    print(f"{'source':<18}{'n':>6}" + "".join(f"{t.split('_')[0]:>12}" for t in results))
    for s in srcs:
        n = sum(1 for r in next(iter(results.values())) if r["_source"] == s)
        line = f"{s:<18}{n:>6}"
        for tag, rows in results.items():
            sub = [r for r in rows if r["_source"] == s]
            line += f"{mean(sub, 'teds'):>12.4f}" if sub else f"{'-':>12}"
        print(line)
