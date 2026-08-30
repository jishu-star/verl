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

__all__ = ["content_reward", "extract_trace", "find_section_keys", "match_section",
           "parse_trace", "render_trace",
           "schema_reward", "SECTIONS"]

SECTIONS = ("headers", "row groups", "merged", "empty")

_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.S)
_PLAN_OPEN_RE = re.compile(r"<plan>", re.I)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_TABLE_OPEN_RE = re.compile(r"<table[^>]*>", re.I)
_TABLE_RE_ANY = re.compile(r"<table[^>]*>.*?(?:</table>|$)", re.S | re.I)
_SEC_RE = re.compile(r"^(headers|row groups|merged|empty):\s*(.*)$")
_ANY_SECTION_RE = re.compile(r"^[ \t]*(headers|row[ _]groups|merged|empty)[ \t]*:", re.I | re.M)
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
    body, _, _ = extract_trace(text)
    if body is None:
        # No <plan> wrapper. This never happens for the ground truth, which is emitted
        # through structure_plan.wrap(), but a MODEL may write the four sections and omit
        # the tags, and those sections are still a trace worth scoring. The table is
        # stripped first so cell text cannot be read as trace lines.
        stripped = _TABLE_RE_ANY.sub("", text)
        body = stripped if _ANY_SECTION_RE.search(stripped) else None
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


def schema_reward(completion):
    """Fraction of the four section keys that can be extracted from the output.

    0, 0.25, 0.5, 0.75 or 1.0 -- graded rather than all-or-nothing, because a rollout
    that produces three of four keys must outrank one that produces none. Under GRPO an
    all-or-nothing term collapses to a constant across a group that all fail, and a
    constant reward is an advantage of zero.
    """
    found = find_section_keys(completion)
    return sum(found.values()) / len(SECTIONS)


# ---------------------------------------------------------------- content
# Text similarity is duplicated from teds.py rather than imported: verl loads a reward
# file by path with load_extern_object, so relative imports inside the package are not
# guaranteed to resolve.
def _norm_levenshtein(a, b, cap=200):
    a, b = a[:cap], b[:cap]
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def _text_sim(a, b):
    return 1.0 - _norm_levenshtein(a, b)


# Below this, two texts are different cells rather than one cell spelled badly, and
# pairing them would let an unrelated prediction absorb a ground-truth record.
MATCH_FLOOR = 0.5
ALPHA = 0.5          # weight on "found the right cell" vs "described it correctly"


def _span_sim(a, b):
    """Graded, per axis. Predicting colspan 3 when the truth is 4 really is closer than
    predicting 1, and exact match throws that gradient away."""
    r = 1.0 - abs(a["rowspan"] - b["rowspan"]) / max(a["rowspan"], b["rowspan"])
    c = 1.0 - abs(a["colspan"] - b["colspan"]) / max(a["colspan"], b["colspan"])
    return 0.5 * r + 0.5 * c


def _parent(rec):
    return rec["path"][-2] if len(rec["path"]) > 1 else None


def _value_sim(section, a, b):
    if section == "empty":
        return 1.0                                  # key-only section
    s = _span_sim(a, b)
    if section == "merged":
        return s                                    # flat: no parent to compare
    pa, pb = _parent(a), _parent(b)
    if pa is None and pb is None:
        p = 1.0
    elif pa is None or pb is None:
        p = 0.0
    else:
        p = _text_sim(pa, pb)
    return 0.5 * s + 0.5 * p


def _key_sim(section, a, b):
    if section == "empty":
        row = _text_sim(a["row"], b["row"])
        pa, pb = a["path"], b["path"]
        if not pa and not pb:
            path = 1.0
        elif not pa or not pb:
            path = 0.0
        else:
            path = _text_sim(" › ".join(pa), " › ".join(pb))
        return 0.5 * row + 0.5 * path
    s = _text_sim(a["text"], b["text"])
    if section == "row groups" and a.get("is_section") != b.get("is_section"):
        return 0.0            # a banner and a stub label are different kinds of node
    return s


def _pair_sim(section, a, b):
    k = _key_sim(section, a, b)
    if k < MATCH_FLOOR:
        return 0.0, 0.0, 0.0
    v = _value_sim(section, a, b)
    return k * (ALPHA + (1.0 - ALPHA) * v), k, v


def _hungarian(cost):
    """Minimum-cost assignment, O(n^2 m). Rows must not outnumber columns.

    Exact rather than greedy: 31.5% of tables repeat a header text, and greedy pairing
    can lock an early near-tie into a choice that blocks a better global one. n is capped
    at MAXLIST=40 records per section, so the exact algorithm costs nothing.
    """
    n, m = len(cost), len(cost[0])
    INF = float("inf")
    u, v, p, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv, used = [INF] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]


def _identity(section, r):
    if section == "empty":
        return ("empty", r["row"], r["path"])
    return (section, r["text"], r["rowspan"], r["colspan"],
            r.get("is_section"), _parent(r))


def _drain_identical(section, gt, pred, parts):
    """Remove record pairs identical on both sides, appending a perfect match for each."""
    from collections import Counter
    pool = Counter(_identity(section, r) for r in pred)
    gt_left, matched = [], Counter()
    for r in gt:
        k = _identity(section, r)
        if pool[k] > matched[k]:
            matched[k] += 1
            trivial = (section == "empty" or (r["rowspan"] == 1 and r["colspan"] == 1))
            parts.append((1.0, 1.0, None if trivial else 1.0))
        else:
            gt_left.append(r)
    pred_left, used = [], Counter()
    for r in pred:
        k = _identity(section, r)
        if used[k] < matched[k]:
            used[k] += 1
        else:
            pred_left.append(r)
    return gt_left, pred_left


def match_section(section, gt, pred):
    """-> (matched mass, [(key_sim, value_sim, span_sim_or_None) per matched pair]).

    span_sim is None for pairs where neither side carries a span, so the caller can
    restrict to the records that actually test span understanding."""
    if not gt or not pred:
        return 0.0, []

    # Fast path: pull out records that are IDENTICAL on both sides before doing any
    # fuzzy work. Such a pair scores the maximum 1.0, so an exchange argument says some
    # optimal assignment contains it -- removing it first cannot lose optimality. Most
    # records in a good prediction match exactly, and this avoids running an edit
    # distance over every candidate pair, which was 90% of the cost.
    total, parts = 0.0, []
    gt, pred = _drain_identical(section, gt, pred, parts)
    if not gt or not pred:
        return float(len(parts)), parts
    total = float(len(parts))

    flip = len(gt) > len(pred)
    a, b = (pred, gt) if flip else (gt, pred)
    cost = [[-_pair_sim(section, x, y)[0] for y in b] for x in a]
    for i, j in _hungarian(cost):
        x, y = (a[i], b[j]) if not flip else (b[j], a[i])
        s, k, val = _pair_sim(section, x, y)
        if s > 0:
            total += s
            trivial = (section == "empty" or
                       (x["rowspan"] == 1 and x["colspan"] == 1
                        and y["rowspan"] == 1 and y["colspan"] == 1))
            parts.append((k, val, None if trivial else _span_sim(x, y)))
    return total, parts


def content_reward(completion, gt_plan, detail=False):
    """Score the K:V pairs inside each section against the ground-truth trace.

    Returns the macro-average F1 over the sections that carry records on at least one
    side. Sections empty on BOTH sides are excluded rather than scored 1.0: `merged` is
    empty in 46.2% of tables and `empty` in 71.3%, so crediting them would put a large
    unearned floor under every trace including a wrong one.
    """
    gt = parse_trace(gt_plan)
    pred = parse_trace(completion)
    out = {"content": 0.0, "key_f1": 0.0, "value_acc": 0.0, "span_acc": 0.0}
    if gt is None or pred is None:
        return out          # no trace emitted at all is a zero, not a vacuous match
    per, keys, vals, spans = {}, [], [], []
    for s in SECTIONS:
        g, p = gt[s], pred[s]
        if not g and not p:
            per[s] = None
            continue
        if not g or not p:
            per[s] = 0.0
            continue
        mass, parts = match_section(s, g, p)
        precision, recall = mass / len(p), mass / len(g)
        per[s] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        keys += [k for k, _, _ in parts]
        vals += [v for _, v, _ in parts]
        spans += [sp for _, _, sp in parts if sp is not None]
    live = [x for x in per.values() if x is not None]
    if not live:
        # Every section 'none' on both sides -- one table in 50,000. Nothing was asserted
        # and nothing was missed, so the diagnostics are vacuously perfect too; leaving
        # them at 0 would plot a perfect match as a total failure.
        return {"content": 1.0, "key_f1": 1.0, "value_acc": 1.0, "span_acc": 1.0}
    out["content"] = sum(live) / len(live)
    out["key_f1"] = sum(keys) / len(keys) if keys else 0.0
    out["value_acc"] = sum(vals) / len(vals) if vals else 0.0
    # Restricted to records where a span actually exists on either side. 63.1% of header
    # cells are plain 1x1, so an unrestricted value score is 89% satisfied by predicting
    # "no span" everywhere; this one cannot be guessed.
    out["span_acc"] = sum(spans) / len(spans) if spans else 1.0
    if detail:
        out["per_section"] = per
    return out
