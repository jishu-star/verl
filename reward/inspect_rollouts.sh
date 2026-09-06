#!/usr/bin/env bash
# Inspect the rollout dumps written by trainer.rollout_data_dir.
#
#   bash inspect_rollouts.sh            # summarise every step dumped so far
#   bash inspect_rollouts.sh 3          # show sample generations from step 3
#   bash inspect_rollouts.sh 3 plan     # show just the <plan> blocks from step 3
set -euo pipefail
DUMP_DIR=${DUMP_DIR:-/home/ubuntu/rollout_dumps}
PY=/home/ubuntu/verl/.venv/bin/python

if [[ $# -eq 0 ]]; then
    "${PY}" - "${DUMP_DIR}" <<'EOF'
import glob, json, os, re, sys
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "*.jsonl")), key=lambda p: int(re.findall(r"(\d+)\.jsonl", p)[0]))
if not files:
    print(f"no dumps yet in {d} (first appears after step 1 completes)"); raise SystemExit
print(f"{'step':>5} {'n':>6} {'score':>7} {'html':>7} {'schema':>7} {'content':>8} {'has_plan':>9}")
for f in files:
    rows = [json.loads(l) for l in open(f)]
    def m(k):
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return sum(v) / len(v) if v else float("nan")
    plan = sum(1 for r in rows if "<plan>" in (r.get("output") or "")) / max(len(rows), 1)
    step = re.findall(r"(\d+)\.jsonl", f)[0]
    print(f"{step:>5} {len(rows):>6} {m('score'):>7.3f} {m('html'):>7.3f} "
          f"{m('schema'):>7.3f} {m('content'):>8.3f} {plan:>9.1%}")
EOF
    exit 0
fi

STEP=$1
MODE=${2:-full}
"${PY}" - "${DUMP_DIR}/${STEP}.jsonl" "${MODE}" <<'EOF'
import json, re, sys
path, mode = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(path)]
rows.sort(key=lambda r: r.get("score", 0), reverse=True)
picks = [("BEST", rows[0]), ("MEDIAN", rows[len(rows) // 2]), ("WORST", rows[-1])]
for label, r in picks:
    out = r.get("output") or ""
    comp = {k: round(v, 3) for k, v in r.items()
            if k in ("score", "html", "schema", "content", "teds", "truncated", "has_table")}
    print(f"\n{'=' * 78}\n{label}  {comp}\n{'=' * 78}")
    if mode == "plan":
        m = re.search(r"<plan>.*?(</plan>|$)", out, re.S)
        print(m.group(0)[:2000] if m else "(no <plan> block)")
    else:
        print(out[:2500])
EOF
