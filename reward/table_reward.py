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
"""The reward verl calls: table image -> <plan> trace + HTML.

    reward.custom_reward_function.path=reward/table_reward.py
    reward.custom_reward_function.name=compute_score

Three components over two disjoint halves of the completion:

    html      TEDS against the ground-truth table          reward/teds.py
    schema    are the four section keys extractable        reward/trace_reward.py
    content   the K:V pairs inside each section            reward/trace_reward.py

The halves cannot contaminate each other: teds.py strips <plan> before looking for a
table, and extract_trace stops at <table before reading trace lines.

DATA CONTRACT. The parquet must carry
    reward_model.ground_truth   the minified canonical HTML
    extra_info["plan"]          the ground-truth <plan> text
The image is NOT passed to a reward function, so everything needed must be in the row.

Weights are overridable from the launch script without touching this file:

    reward.custom_reward_function.reward_kwargs.w_content=0.30
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward.teds import compute_score as teds_score          # noqa: E402
from reward.trace_reward import content_reward, schema_reward  # noqa: E402

__all__ = ["compute_score"]

# The HTML is the objective; the trace is the reasoning that should produce it. Schema is
# deliberately tiny -- it is a format check, and paying much for four literal words
# invites the policy to farm it.
W_HTML = 0.75
W_SCHEMA = 0.05
W_CONTENT = 0.20


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None,
                  w_html=W_HTML, w_schema=W_SCHEMA, w_content=W_CONTENT,
                  w_struct=0.5, **kwargs):
    """Never raises. An exception here kills the training step, and early rollouts are
    arbitrary text. Returns a dict so every component is its own logged curve: a single
    scalar cannot say whether a stalled run lost the table or lost the trace."""
    out = {"score": 0.0, "html": 0.0, "schema": 0.0, "content": 0.0, "error": 0.0}
    try:
        gt_plan = (extra_info or {}).get("plan") or ""

        h = teds_score(data_source, solution_str, ground_truth, extra_info, w_struct=w_struct)
        c = content_reward(solution_str, gt_plan) if gt_plan else {}
        s = schema_reward(solution_str)

        out.update({k: v for k, v in h.items() if k != "score"})
        out.update({k: v for k, v in c.items() if k != "content"})
        out["html"] = h["score"]
        out["schema"] = s
        out["content"] = c.get("content", 0.0)
        out["score"] = w_html * out["html"] + w_schema * out["schema"] + w_content * out["content"]
    except Exception:
        out["error"] = 1.0
    return out
