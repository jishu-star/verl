"""v3 prompt — READY, NOT APPLIED. Only use if v2's step-0 validation falls short.

History:
  v0 "First describe the table's structure inside <plan></plan>"
     -> reward 0.4255, html 0.567, has_table 0.910, schema 0.0015, content 0.0
     The four-key format was never stated, so the plan half could not score.

  v1 added the format spec, framed as "work out the structure -- this is your reasoning"
     -> reward 0.2106, html 0.211, has_table 0.554, schema 0.477, content 0.141
     Format spec worked. But 99.9% of rollouts opened with chain-of-thought prose and
     0.0% began with <plan>; 39.4% burned the whole budget before emitting either tag.

  v2 kept the spec, replaced the reasoning framing with "must begin with <plan>, do not
     think out loud in prose".

  v3 (this file) is v2 with the prohibition made absolute: the reply is the two blocks
     and nothing else, stated first, restated last, with the failure named explicitly.
"""

PROMPT = """<image>
Convert the table image to HTML.

Output format — this is strict. Your reply must consist of exactly two things, in order:

  1. one <plan>...</plan> block, in the format below
  2. the complete HTML table

Nothing else. No analysis, no commentary, no step-by-step reasoning, no "the table has
N columns" narration, no notes before, between or after the two blocks. The first
character of your reply must be `<` of `<plan>`. Writing any prose before the plan is
a failed answer, even if the table that follows is correct.

The plan is not scratch work — it is the declared structure that the HTML after it must
match.

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

A complete reply looks exactly like this, with no other text:

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
<table>...</table>

Begin now with `<plan>`."""
