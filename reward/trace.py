# Copyright 2026 jishu-star
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parse a <plan> reasoning trace into typed records.

The trace is four sections of indented lines. This turns each line into a record whose
KEY is its identity and whose VALUE is its attributes, so the two can be scored apart:

    headers      key (path,)                 value (colspan, rowspan, depth)
    row groups   key (path,)                 value (colspan, rowspan, depth, is_section)
    merged       key (text,)                 value (colspan, rowspan)
    empty        key (row, column path)      value ()   -- the pair IS the assertion

`path` is the tuple of ancestor texts down to the node, so two groups legitimately
sharing a label (`Day Shift` under `Line A` and under `Line B`) stay distinct.

Parsing is total: a malformed line degrades to a record with whatever could be read
rather than raising, because this runs on model output.
"""

import re

__all__ = ["parse_trace", "render_trace", "SECTIONS"]

SECTIONS = ("headers", "row groups", "merged", "empty")

_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.S)
_SEC_RE = re.compile(r"^(headers|row groups|merged|empty):\s*(.*)$")
_SPAN_RE = re.compile(r"\s\[(?:(\d+)r x (\d+)c|(\d+) cols|(\d+) rows)\]$")
_SECTION_TAG = " (section)"
_PATH_SEP = " › "
_EMPTY_SEP = "  /  "


def _split_span(line):
    """'Peak Hours [4 cols]' -> ('Peak Hours', 1, 4)."""
    m = _SPAN_RE.search(line)
    if not m:
        return line, 1, 1
    text = line[: m.start()]
    if m.group(1):
        return text, int(m.group(1)), int(m.group(2))
    if m.group(3):
        return text, 1, int(m.group(3))
    return text, int(m.group(4)), 1


def parse_trace(text):
    """-> {section: [record]}. Accepts a full completion or a bare trace body.

    Returns None if there is no <plan> block at all; a present-but-empty section yields
    an empty list, which is a real answer ('none') and distinct from a missing trace.
    """
    if text is None:
        return None
    m = _PLAN_RE.search(text)
    body = m.group(1) if m else (text if _SEC_RE.match(text.strip()[:20] or "x") else None)
    if body is None:
        return None

    out = {s: [] for s in SECTIONS}
    seen = set()
    cur = None
    stack = []                      # (depth, text) ancestors of the current line
    for raw in body.split("\n"):
        if not raw.strip():
            continue
        head = _SEC_RE.match(raw.strip()) if not raw.startswith(" ") else None
        if head:
            cur = head.group(1)
            seen.add(cur)
            stack = []
            continue                # 'none' / 'N (first 40)' carry no records
        if cur is None:
            continue

        line = raw[2:] if raw.startswith("  ") else raw.lstrip()
        depth = (len(line) - len(line.lstrip(" "))) // 2
        line = line.strip()
        if not line:
            continue

        if cur == "empty":
            row, _, path = line.partition(_EMPTY_SEP)
            out["empty"].append({
                "row": row.strip(),
                "path": tuple(p.strip() for p in path.split(_PATH_SEP)) if path else (),
            })
            continue

        is_section = line.endswith(_SECTION_TAG)
        if is_section:
            line = line[: -len(_SECTION_TAG)]
        text_, rowspan, colspan = _split_span(line)
        text_ = text_.strip()

        del stack[depth:]           # ancestors are everything shallower than this line
        stack.append(text_)
        rec = {"text": text_, "depth": depth, "rowspan": rowspan, "colspan": colspan,
               "path": tuple(stack)}
        if cur == "row groups":
            rec["is_section"] = is_section
        out[cur].append(rec)

    out["_sections_present"] = tuple(s for s in SECTIONS if s in seen)
    return out


# ---------------------------------------------------------------- inverse, for testing
def _fmt_span(rowspan, colspan):
    if rowspan > 1 and colspan > 1:
        return f" [{rowspan}r x {colspan}c]"
    if colspan > 1:
        return f" [{colspan} cols]"
    if rowspan > 1:
        return f" [{rowspan} rows]"
    return ""


def render_trace(parsed, truncated=(), none=()):
    """Rebuild the trace text. Exists so parse_trace can be checked by round-tripping:
    a parser for model output is only trustworthy if it is provably lossless on the
    labels it was designed against."""
    lines = []
    for s in SECTIONS:
        recs = parsed.get(s, [])
        if s in none:
            lines.append(f"{s}: none")
            continue
        if s in truncated:
            lines.append(f"{s}: {truncated[s]} (first {len(recs)})")
        else:
            lines.append(f"{s}:")
        for r in recs:
            if s == "empty":
                lines.append("  " + r["row"] + _EMPTY_SEP + _PATH_SEP.join(r["path"]))
            else:
                tag = _SECTION_TAG if r.get("is_section") else ""
                lines.append("  " + "  " * r["depth"] + r["text"]
                             + _fmt_span(r["rowspan"], r["colspan"]) + tag)
    return "\n".join(lines)
