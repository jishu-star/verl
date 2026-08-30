# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Selective freezing and unfreezing of model parameters by name.

Backs ``model.freeze_patterns`` / ``model.unfreeze_last_n_layers`` /
``model.unfreeze_patterns``. Applied in that order, so

    freeze_patterns: ['visual']
    unfreeze_last_n_layers: 8

reads as "freeze the vision tower, keep the last eight decoder layers trainable".

EVERY PATTERN MUST MATCH SOMETHING. verl already carries an
``actor.freeze_vision_tower`` flag that is declared in the config and read by no code
at all -- set it and the vision tower trains anyway, with nothing in the logs to say so.
A selective-training switch that silently does nothing is worse than no switch, because
the run looks healthy while optimising the wrong parameters. So a pattern matching zero
parameters raises here rather than passing quietly.

Depends only on ``named_parameters()``, so it can be unit-tested on CPU against a mock
module, and run against a ``device_map='meta'`` model to check patterns before a run.
"""

import re
from collections import OrderedDict

__all__ = ["apply_param_groups", "format_report", "num_hidden_layers_of"]

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def num_hidden_layers_of(hf_config) -> int:
    """Decoder layer count, looking through the nested text config first.

    Multimodal checkpoints put the language model under ``text_config`` (Qwen3.5) or
    ``llm_config``, and the top-level ``num_hidden_layers``, when present at all, may
    describe the vision tower instead.
    """
    for attr in ("text_config", "llm_config", "language_config"):
        sub = getattr(hf_config, attr, None)
        n = getattr(sub, "num_hidden_layers", None) if sub is not None else None
        if n:
            return int(n)
    n = getattr(hf_config, "num_hidden_layers", None)
    if not n:
        raise ValueError(
            "cannot determine num_hidden_layers from the model config; set "
            "model.unfreeze_patterns explicitly instead of unfreeze_last_n_layers"
        )
    return int(n)


def _layer_index(name: str):
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _group_of(name: str) -> str:
    """A coarse bucket for the report: 'visual', 'layers.24', or the leading path."""
    if "visual" in name or "vision" in name:
        return "visual"
    i = _layer_index(name)
    if i is not None:
        return f"layers.{i}"
    parts = [p for p in name.split(".") if p not in ("base_model", "model", "weight", "bias")]
    return ".".join(parts[:1]) or "other"


def apply_param_groups(
    module,
    hf_config=None,
    freeze_patterns=(),
    unfreeze_last_n_layers: int = 0,
    unfreeze_patterns=(),
) -> dict:
    """Set ``requires_grad`` per parameter. Returns a report dict.

    Must be called AFTER any PEFT wrapping -- ``get_peft_model`` freezes every base
    parameter, so an unfreeze applied before it is undone -- and BEFORE FSDP wrapping,
    after which parameters are ``DTensor`` shards.
    """
    freeze_patterns = list(freeze_patterns or [])
    unfreeze_patterns = list(unfreeze_patterns or [])
    if not freeze_patterns and not unfreeze_patterns and unfreeze_last_n_layers <= 0:
        return {}

    named = list(module.named_parameters())
    if not named:
        raise ValueError("module has no parameters")

    hits = {p: 0 for p in freeze_patterns + unfreeze_patterns}

    def _matched(patterns, name):
        found = False
        for pat in patterns:
            if re.search(pat, name):
                hits[pat] += 1
                found = True
        return found

    # ---- which layer indices count as "the last N" ---------------------------------
    last_layers: set[int] = set()
    if unfreeze_last_n_layers > 0:
        present = {i for i in (_layer_index(n) for n, _ in named) if i is not None}
        if not present:
            raise ValueError(
                "unfreeze_last_n_layers is set but no parameter name matches "
                r"'\.layers\.<int>\.'; this model does not use the usual decoder layer "
                "naming, so use unfreeze_patterns instead"
            )
        n_layers = num_hidden_layers_of(hf_config) if hf_config is not None else max(present) + 1
        if unfreeze_last_n_layers > n_layers:
            raise ValueError(f"unfreeze_last_n_layers={unfreeze_last_n_layers} exceeds {n_layers} decoder layers")
        last_layers = set(range(n_layers - unfreeze_last_n_layers, n_layers))
        missing = last_layers - present
        if missing:
            raise ValueError(
                f"decoder layers {sorted(missing)} have no parameters in this module; "
                f"num_hidden_layers={n_layers} disagrees with the checkpoint"
            )

    # ---- apply, in order ------------------------------------------------------------
    for name, param in named:
        if freeze_patterns and _matched(freeze_patterns, name):
            param.requires_grad_(False)
        if last_layers and _layer_index(name) in last_layers:
            param.requires_grad_(True)
        if unfreeze_patterns and _matched(unfreeze_patterns, name):
            param.requires_grad_(True)

    # An unfreeze on a model where nothing is frozen is a no-op: without LoRA or
    # freeze_patterns every parameter is already trainable, so `unfreeze_last_n_layers=8`
    # silently yields a FULL fine-tune rather than the last eight layers. Same failure
    # class as a freeze flag nothing reads, so it is refused rather than logged.
    if (last_layers or unfreeze_patterns) and all(p.requires_grad for _, p in named):
        raise ValueError(
            "unfreeze_* was requested but no parameter ends up frozen, so it had no "
            "effect and the run would be a full fine-tune. Unfreezing only makes sense "
            "on top of something that freezes: enable LoRA (model.lora_rank > 0), or "
            "name what to freeze in model.freeze_patterns."
        )

    dead = [p for p, n in hits.items() if n == 0]
    if dead:
        raise ValueError(
            f"pattern(s) {dead} matched no parameter names. Check them against the "
            f"checkpoint (e.g. {[n for n, _ in named[:3]]}) rather than assuming a naming "
            f"convention -- this is exactly the failure mode that makes a freeze flag a no-op."
        )

    # ---- report ---------------------------------------------------------------------
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for name, param in named:
        g = groups.setdefault(_group_of(name), [0, 0])
        g[0 if param.requires_grad else 1] += param.numel()
    trainable = sum(g[0] for g in groups.values())
    total = trainable + sum(g[1] for g in groups.values())
    return {"groups": groups, "trainable": trainable, "total": total,
            "pattern_hits": hits, "last_layers": sorted(last_layers)}


def _fmt(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return str(n)


def format_report(report: dict) -> str:
    """Human-readable summary, with contiguous decoder layers collapsed into ranges."""
    if not report:
        return "selective training: disabled (all parameters trainable)"
    rows, run = [], None
    for name, (train, frozen) in report["groups"].items():
        state = "trainable" if train and not frozen else ("frozen" if frozen and not train else "mixed")
        i = int(name.split(".")[1]) if name.startswith("layers.") else None
        if run and i is not None and run[3] == state and i == run[2] + 1:
            run[2], run[4], run[5] = i, run[4] + train, run[5] + frozen
            continue
        if run:
            rows.append(run)
        run = ["layers" if i is not None else name, i, i, state, train, frozen]
    if run:
        rows.append(run)

    out = [f"selective training: {_fmt(report['trainable'])} / {_fmt(report['total'])} trainable "
           f"({100 * report['trainable'] / max(report['total'], 1):.1f}%)"]
    for label, lo, hi, state, train, frozen in rows:
        name = label if lo is None else (f"layers.{lo}" if lo == hi else f"layers.{lo}-{hi}")
        out.append(f"  {name:<28} {_fmt(train + frozen):>10}   {state}")
    return "\n".join(out)
