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
"""TEDS reward for table image -> HTML.

Tree-Edit-Distance-based Similarity (Zhong et al., PubTabNet):

    TEDS(a, b) = 1 - EditDistance(a, b) / max(|a|, |b|)

over the table parsed as a tree of tr -> td/th, where rowspan and colspan are node
attributes. One number covers structure, cell content and completeness at once: a
flattened span is a rename, a missing row is a subtree deletion, a truncated table is a
run of deletions.

Two variants, both returned so each is visible as its own curve:

    teds_struct   cell text removed -- pure structure
    teds          cell text included, with a GRADED rename cost

GRADED, not binary. Canonical TEDS charges a substitution of 1 whenever two cells differ
at all, so "Total Revenue" against "Total Revenu" costs exactly as much as against "".
That is fine for a leaderboard and bad for a reward: it flattens the surface the policy
has to climb. Here a cell whose tag and spans match is charged the normalised edit
distance of its text, so getting closer scores higher.

COST CONTROL. APTED is O(n^2)-ish in practice and its tail is brutal on large tables --
measured on this corpus, a few hundred nodes runs in milliseconds and a thousand does not
return. A reward is on the critical path of every GRPO step, so a tree above MAX_NODES
falls back to the structure-only variant, and above HARD_MAX to a cheap grid comparison.
Both fallbacks are reported in the returned dict, never silently.
"""

import re
from collections import deque

from apted import APTED, Config
from lxml import etree

__all__ = ["teds", "build_tree", "tree_size", "compute_score"]

# Tables above this many nodes skip the text-bearing variant (it is the expensive one).
MAX_NODES = 400
# Above this, skip APTED entirely and fall back to grid agreement.
HARD_MAX = 1200

_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.S | re.I)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------- tree building
class Node:
    __slots__ = ("tag", "colspan", "rowspan", "text", "children")

    def __init__(self, tag, colspan=1, rowspan=1, text=None):
        self.tag, self.colspan, self.rowspan, self.text = tag, colspan, rowspan, text
        self.children = []

    def shape(self):
        """Everything except the text: what must match before text is even compared."""
        return (self.tag, self.rowspan, self.colspan)


def build_tree(html, with_text):
    """<table> -> Node tree. Presentation attributes are ignored; spans are not."""
    dom = etree.HTML(html or "", parser=etree.HTMLParser())
    if dom is None:
        return None
    table = dom.find(".//table")
    if table is None:
        return None
    root = Node("table")
    for tr in table.xpath(".//tr"):
        row = Node("tr")
        for cell in tr.xpath("td|th"):
            def span(key):
                try:
                    return max(1, int(cell.get(key, "1")))
                except Exception:
                    return 1
            text = _WS_RE.sub(" ", "".join(cell.itertext())).strip() if with_text else None
            row.children.append(Node(cell.tag, span("colspan"), span("rowspan"), text))
        root.children.append(row)
    return root


def tree_size(node):
    n, queue = 0, deque([node])
    while queue:
        cur = queue.popleft()
        n += 1
        queue.extend(cur.children)
    return n


# ---------------------------------------------------------------- edit distance
def _norm_levenshtein(a, b, cap=200):
    """Normalised edit distance in [0,1]. Long strings are truncated: cell text past a
    couple of hundred characters says nothing more about whether the cell is right, and
    the O(len^2) tail would otherwise dominate a large table."""
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


class _Cfg(Config):
    """APTED costs. Insert and delete are 1, the TEDS default."""

    def __init__(self, with_text):
        self.with_text = with_text

    def rename(self, a, b):
        if a.shape() != b.shape():
            return 1.0
        if not self.with_text or a.text is None or b.text is None:
            return 0.0
        return _norm_levenshtein(a.text, b.text)

    def children(self, node):
        return node.children


def teds(gt_html, pred_html, with_text=True):
    """TEDS in [0,1]. 0 when either side has no parseable table."""
    a = build_tree(gt_html, with_text)
    b = build_tree(pred_html, with_text)
    if a is None or b is None:
        return 0.0
    size = max(tree_size(a), tree_size(b))
    if size == 0:
        return 0.0
    distance = APTED(a, b, _Cfg(with_text)).compute_edit_distance()
    return max(0.0, 1.0 - distance / size)


# ---------------------------------------------------------------- cheap fallback
def _grid(html):
    """(cell-id per occupied position, text by id, rows, cols) after span expansion."""
    dom = etree.HTML(html or "", parser=etree.HTMLParser())
    if dom is None:
        return None
    table = dom.find(".//table")
    if table is None:
        return None
    occ, texts, r, cid = {}, {}, -1, 0
    for tr in table.xpath(".//tr"):
        r += 1
        c = 0
        for cell in tr.xpath("td|th"):
            def span(key):
                try:
                    return max(1, int(cell.get(key, "1")))
                except Exception:
                    return 1
            rs, cs = span("rowspan"), span("colspan")
            while (r, c) in occ:
                c += 1
            texts[cid] = _WS_RE.sub(" ", "".join(cell.itertext())).strip()
            for dr in range(rs):
                for dc in range(cs):
                    occ[(r + dr, c + dc)] = cid
            c += cs
            cid += 1
    if not occ:
        return None
    return occ, texts, max(x for x, _ in occ) + 1, max(y for _, y in occ) + 1


def _grid_agreement(gt_html, pred_html):
    """O(rows x cols) stand-in for TEDS on tables too large for APTED.

    Labels every grid position with the logical cell covering it and asks, for each
    adjacent pair, "same cell?". A colspan=3 answers 'same' twice where three separate
    cells answer 'different' twice, so the bit vector is the merge structure. Positions
    outside a table's grid count as mismatches, which penalises wrong dimensions directly.
    """
    a, b = _grid(gt_html), _grid(pred_html)
    if a is None or b is None:
        return 0.0
    (oa, _, ra, ca), (ob, _, rb, cb) = a, b
    rows, cols = max(ra, rb), max(ca, cb)
    agree = total = 0
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if r2 >= rows or c2 >= cols:
                    continue
                total += 1
                a1, a2 = oa.get((r, c)), oa.get((r2, c2))
                b1, b2 = ob.get((r, c)), ob.get((r2, c2))
                in_a, in_b = a1 is not None and a2 is not None, b1 is not None and b2 is not None
                if in_a != in_b:
                    continue
                if not in_a or (a1 == a2) == (b1 == b2):
                    agree += 1
    return agree / total if total else 0.0


# ---------------------------------------------------------------- verl entry point
def extract_table(completion):
    """The first complete <table>...</table>, or a truncated one recovered from <table.

    A rollout that hits max_response_length has no closing tag. Returning nothing for
    those makes every truncated rollout in a GRPO group score identically, which is an
    advantage of exactly zero and no gradient from the samples that most need one. lxml
    parses the recovered fragment; the missing rows then show up as deletions, which is
    what TEDS is for.
    """
    completion = _THINK_RE.sub("", completion or "")
    found = _TABLE_RE.findall(completion)
    if found:
        return found[0], len(found), False
    start = re.search(r"<table[^>]*>", completion, re.I)
    if start:
        return completion[start.start():], 0, True
    return None, 0, False


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None,
                  w_struct=0.5, **kwargs):
    """verl reward entry point. Returns a dict so every part is logged separately.

    Never raises: an exception here kills the training step, and early rollouts are
    arbitrary text.
    """
    out = {"score": 0.0, "teds": 0.0, "teds_struct": 0.0,
           "has_table": 0.0, "truncated": 0.0, "n_tables": 0.0,
           "fallback_struct_only": 0.0, "fallback_grid": 0.0, "error": 0.0}
    try:
        pred, n_tables, truncated = extract_table(solution_str)
        out["n_tables"] = float(n_tables)
        out["truncated"] = float(truncated)
        if pred is None:
            return out
        gt_tree = build_tree(ground_truth, with_text=False)
        pred_tree = build_tree(pred, with_text=False)
        if gt_tree is None or pred_tree is None:
            return out
        out["has_table"] = 1.0
        size = max(tree_size(gt_tree), tree_size(pred_tree))

        if size > HARD_MAX:
            out["fallback_grid"] = 1.0
            struct = full = _grid_agreement(ground_truth, pred)
        else:
            struct = teds(ground_truth, pred, with_text=False)
            if size > MAX_NODES:
                out["fallback_struct_only"] = 1.0
                full = struct
            else:
                full = teds(ground_truth, pred, with_text=True)

        out["teds_struct"], out["teds"] = struct, full
        out["score"] = w_struct * struct + (1.0 - w_struct) * full
    except Exception:
        out["error"] = 1.0
    return out
