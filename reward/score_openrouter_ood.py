"""Score an OpenRouter OOD eval JSONL (html-only conditions).

    python reward/score_openrouter_ood.py /home/ubuntu/ood_eval/E_qwen27b_html.jsonl

Only html/teds are meaningful: the html-only prompt asks for no <plan>, so schema and
content are zero by construction.
"""

import importlib.util
import json
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/ubuntu/verl")
_spec = importlib.util.spec_from_file_location("tr", "/home/ubuntu/verl/reward/table_reward.py")
_tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr)


def work(r):
    d = _tr.compute_score("t", r.get("output") or "", r["gts"], {"plan": r.get("plan") or ""})
    o = r.get("output") or ""
    return {
        "score": d["score"], "html": d["html"],
        "teds": d.get("teds", 0.0), "teds_struct": d.get("teds_struct", 0.0),
        "truncated": d.get("truncated", 0.0),
        "src": r.get("source", ""),
        "has_table": float("<table" in o),
        "starts_table": float(o.lstrip().startswith("<table")),
        "chars": len(o),
    }


def main():
    path = sys.argv[1]
    rows = [json.loads(l) for l in open(path)]
    ok = [r for r in rows if not r.get("error")]
    with ProcessPoolExecutor(max_workers=24) as ex:
        res = list(ex.map(work, ok, chunksize=8))

    def m(k):
        return st.fmean(x[k] for x in res)

    print(f"{path.split('/')[-1]}  (n={len(ok)}, {len(rows) - len(ok)} errors)")
    for k in ("score", "html", "teds", "teds_struct", "has_table", "starts_table", "truncated"):
        print(f"  {k:<14}{m(k):.4f}")
    print(f"  {'median chars':<14}{int(st.median(x['chars'] for x in res))}")
    print("\n  teds by source")
    for s in sorted({x["src"] for x in res}):
        sub = [x for x in res if x["src"] == s]
        print(f"    {s:<16}{len(sub):>5}  {st.fmean(x['teds'] for x in sub):.4f}")


if __name__ == "__main__":
    main()
