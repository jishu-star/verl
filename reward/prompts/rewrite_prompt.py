"""Rewrite the `prompt` column in train/val parquet.

v2. The v1 rewrite specified the plan format correctly (schema 0.0015 -> 0.477,
key_f1 0 -> 0.409) but framed the plan as "work out ... this is your reasoning", and
the model took that literally: 99.9% of rollouts opened with chain-of-thought prose,
0.0% began with <plan>, and 39.4% burned the whole 8192-token budget before emitting
either tag. html collapsed 0.567 -> 0.211.

v2 keeps the same format spec but forbids the preamble: output must START with <plan>.

Nothing else is touched: images, reward_model.ground_truth and extra_info.plan are
independent of the prompt, and compute_score() never receives it.
"""

import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

PROMPT = """<image>
Convert the table image to HTML.

Your reply must begin with `<plan>`. Do not write anything before it, and do not think
out loud in prose — the plan itself is where you set out the structure, and the HTML you
emit after it must match.

Inside <plan></plan>, write exactly these four keys, lowercase, in this order, each
starting a line:

  headers:     every column header, one per line, indented 2 spaces; indent 2 more per
               nesting level under its parent.
  row groups:  labels that group rows, same nesting. Append ` (section)` to a heading
               that spans the full width.
  merged:      the text of each cell spanning more than one row or column.
  empty:       one blank body cell per line, as `<row>  /  <column>` — two spaces, slash,
               two spaces. Join a nested column path with ` › `. Keep the ` (section)`
               suffix on a row label that has one. Skip a cell whose row or column you
               cannot name.

After any entry that spans, append its size: ` [3 cols]`, ` [4 rows]`, ` [2r x 3c]`.

A key with nothing to list is `key: none` on one line. A key with more than 40 entries is
`key: <total> (first 40)`, followed by only the first 40.

Example of the required shape:

<plan>
headers:
  Stub
  GroupA [3 cols]
    SubA [2 cols]
      ColOne
      ColTwo
    ColThree
row groups:
  SectOne [4 cols] (section)
    RowA [2 rows]
    RowB
merged: none
empty:
  SectOne (section)  /  GroupA › SubA › ColTwo
</plan>

Immediately after `</plan>`, emit the complete table as HTML. Both the plan and the
table are required."""


def rewrite(path: str) -> None:
    t = pq.read_table(path)
    prompts = t.column("prompt").to_pylist()
    print(f"{path}: {t.num_rows} rows, {len({p[0]['content'] for p in prompts})} distinct prompt(s)")

    new = []
    for p in prompts:
        msgs = [dict(m) for m in p]
        assert msgs[0]["role"] == "user", msgs[0]["role"]
        msgs[0]["content"] = PROMPT
        new.append(msgs)

    idx = t.schema.get_field_index("prompt")
    t2 = t.set_column(idx, t.schema.field(idx), pa.array(new, type=t.schema.field(idx).type))
    pq.write_table(t2, path + ".new")

    chk = pq.read_table(path + ".new")
    assert chk.num_rows == t.num_rows, (chk.num_rows, t.num_rows)
    assert {p[0]["content"] for p in chk.column("prompt").to_pylist()} == {PROMPT}
    assert chk.column("images").to_pylist() == t.column("images").to_pylist(), "images changed"
    assert chk.column("reward_model").to_pylist() == t.column("reward_model").to_pylist(), "gt changed"
    assert chk.column("extra_info").to_pylist() == t.column("extra_info").to_pylist(), "plan changed"

    shutil.move(path + ".new", path)
    print(f"  -> rewritten, {chk.num_rows} rows verified")


if __name__ == "__main__":
    for f in sys.argv[1:] or ["train.parquet", "val.parquet"]:
        rewrite(f)
