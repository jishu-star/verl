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
"""Extract and parse the <plan> reasoning trace from a completion.

Counterpart to reward/teds.py, which takes the <table> half of the same completion.
Between them the two halves are disjoint: teds.py strips <plan> before looking for a
table, and extract_trace here stops at <table before looking for trace lines, so
neither can score the other's text.

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

__all__ = ["extract_trace", "find_section_keys", "parse_trace", "render_trace",
           "schema_reward", "SECTIONS"]

SECTIONS = ("headers", "row groups", "merged", "empty")

_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.S)
_PLAN_OPEN_RE = re.compile(r"<plan>", re.I)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_TABLE_OPEN_RE = re.compile(r"<table[^>]*>", re.I)
_SEC_RE = re.compile(r"^(headers|row groups|merged|empty):\s*(.*)$")
_SPAN_RE = re.compile(r"\s\[(?:(\d+)r x (\d+)c|(\d+) cols|(\d+) rows)\]$")
_SECTION_TAG = " (section)"
_PATH_SEP = " › "
_EMPTY_SEP = "  /  "


def extract_trace(completion):
    """-> (trace body, n_plan_blocks, truncated). Mirrors teds.extract_table.

    <think> is dropped first. A well-formed completion has one <plan>...</plan>; a
    rollout cut off by max_response_length has an opening tag and no closing one, and
    its partial trace is still worth scoring -- discarding it would give every truncated
    rollout in a GRPO group the same score, hence zero advantage and no gradient.

    The recovered body stops at the first <table so a missing </plan> cannot swallow the
    HTML into the trace. Returns (None, 0, False) when there is no <plan> at all, which
    is distinct from an empty trace: one is a missing answer, the other is an answer of
    'nothing to report'.
    """
    completion = _THINK_RE.sub("", completion or "")
    closed = _PLAN_RE.findall(completion)
    if closed:
        return closed[0], len(closed), False
    opened = _PLAN_OPEN_RE.search(completion)
    if not opened:
        return None, 0, False
    body = completion[opened.end():]
    table = _TABLE_OPEN_RE.search(body)
    if table:
        body = body[: table.start()]
    return body, 0, True


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


# ---------------------------------------------------------------- schema
# `row groups` is the only two-word key; accept an underscore for it, since that is a
# spelling a model will reach for and it identifies the section just as unambiguously.
_KEY_RES = {
    "headers":    re.compile(r"^[ \t]*headers[ \t]*:", re.I | re.M),
    "row groups": re.compile(r"^[ \t]*row[ _]groups[ \t]*:", re.I | re.M),
    "merged":     re.compile(r"^[ \t]*merged[ \t]*:", re.I | re.M),
    "empty":      re.compile(r"^[ \t]*empty[ \t]*:", re.I | re.M),
}


def find_section_keys(completion):
    """-> {section: bool} for the four keys, searched in the trace half of a completion.

    Searched inside <plan> when there is one, otherwise in the completion with any
    <table> removed. The table is removed because cell text is arbitrary and a table with
    a row reading `Empty:` would otherwise mint a section key out of data.

    Whatever follows the colon is ignored -- that is content, scored elsewhere. Matching
    is case-insensitive and tolerates leading whitespace, so this measures whether the
    four keys are RECOVERABLE, not whether they are typeset exactly.
    """
    body, _, _ = extract_trace(completion)
    if body is None:
        body = _TABLE_RE_ANY.sub("", completion or "")
    return {name: bool(rx.search(body)) for name, rx in _KEY_RES.items()}


_TABLE_RE_ANY = re.compile(r"<table[^>]*>.*?(?:</table>|$)", re.S | re.I)


def schema_reward(completion):
    """Fraction of the four section keys that can be extracted from the output.

    0, 0.25, 0.5, 0.75 or 1.0 -- graded rather than all-or-nothing, because a rollout
    that produces three of four keys must outrank one that produces none. Under GRPO an
    all-or-nothing term collapses to a constant across a group that all fail, and a
    constant reward is an advantage of zero.
    """
    found = find_section_keys(completion)
    return sum(found.values()) / len(SECTIONS)
