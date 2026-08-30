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
"""Tests for selective freeze/unfreeze of parameters.

Uses mock parameters rather than torch modules: ``apply_param_groups`` only needs
``named_parameters()``, and keeping the test torch-free means the patterns can be
checked anywhere, including on a machine with no GPU stack installed.
"""

import unittest

from verl.utils.param_groups import apply_param_groups, num_hidden_layers_of


class MockParam:
    def __init__(self, numel=1000, requires_grad=True):
        self.requires_grad = requires_grad
        self._numel = numel

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def numel(self):
        return self._numel


class MockModule:
    def __init__(self, names, requires_grad=True):
        self._params = [(n, MockParam(requires_grad=requires_grad)) for n in names]

    def named_parameters(self):
        return list(self._params)


class MockTextConfig:
    num_hidden_layers = 32


class MockHFConfig:
    """Multimodal config: the decoder count lives under text_config, as in Qwen3.5."""

    text_config = MockTextConfig
    num_hidden_layers = 27  # the VISION depth -- must not be picked up


LM = [f"model.language_model.layers.{i}.self_attn.q_proj.weight" for i in range(32)]
VISION = [f"model.visual.blocks.{i}.attn.qkv.weight" for i in range(27)]
NAMES = LM + VISION + ["lm_head.weight"]


def trainable(module):
    return {n for n, p in module.named_parameters() if p.requires_grad}


class TestParamGroups(unittest.TestCase):
    def test_reads_layer_count_from_text_config(self):
        self.assertEqual(num_hidden_layers_of(MockHFConfig), 32)

    def test_lora_case_unfreezes_only_the_last_n(self):
        """The hybrid: PEFT has frozen everything, the last 8 layers come back."""
        m = MockModule(NAMES, requires_grad=False)
        report = apply_param_groups(m, MockHFConfig, unfreeze_last_n_layers=8)
        self.assertEqual(report["last_layers"], list(range(24, 32)))
        self.assertEqual(trainable(m), {f"model.language_model.layers.{i}.self_attn.q_proj.weight"
                                        for i in range(24, 32)})

    def test_freeze_patterns_only(self):
        m = MockModule(NAMES)
        apply_param_groups(m, MockHFConfig, freeze_patterns=["visual"])
        self.assertEqual(trainable(m), set(LM) | {"lm_head.weight"})

    def test_freeze_then_unfreeze_order(self):
        """freeze_patterns runs first, so a later unfreeze can override it."""
        m = MockModule(NAMES)
        apply_param_groups(m, MockHFConfig, freeze_patterns=["visual"],
                           unfreeze_patterns=[r"visual\.blocks\.26\."])
        self.assertIn("model.visual.blocks.26.attn.qkv.weight", trainable(m))
        self.assertNotIn("model.visual.blocks.25.attn.qkv.weight", trainable(m))

    def test_layer_index_survives_peft_name_mangling(self):
        peft_names = [f"base_model.model.model.layers.{i}.self_attn.q_proj.base_layer.weight"
                      for i in range(32)]
        m = MockModule(peft_names, requires_grad=False)
        report = apply_param_groups(m, MockHFConfig, unfreeze_last_n_layers=4)
        self.assertEqual(report["last_layers"], [28, 29, 30, 31])

    def test_disabled_by_default(self):
        self.assertEqual(apply_param_groups(MockModule(NAMES), MockHFConfig), {})

    # ---- the failure modes this module exists to make loud -------------------------
    def test_pattern_matching_nothing_raises(self):
        with self.assertRaisesRegex(ValueError, "matched no parameter names"):
            apply_param_groups(MockModule(NAMES), MockHFConfig, freeze_patterns=["vision_tower"])

    def test_unfreeze_without_anything_frozen_raises(self):
        """Otherwise `unfreeze_last_n_layers=8` alone silently means a full fine-tune."""
        with self.assertRaisesRegex(ValueError, "no parameter ends up frozen"):
            apply_param_groups(MockModule(NAMES), MockHFConfig, unfreeze_last_n_layers=8)

    def test_too_many_layers_raises(self):
        with self.assertRaisesRegex(ValueError, "exceeds 32 decoder layers"):
            apply_param_groups(MockModule(NAMES, requires_grad=False), MockHFConfig,
                               unfreeze_last_n_layers=99)

    def test_missing_layer_naming_raises(self):
        m = MockModule(["encoder.block.0.weight"], requires_grad=False)
        with self.assertRaisesRegex(ValueError, "does not use the usual decoder layer"):
            apply_param_groups(m, MockHFConfig, unfreeze_last_n_layers=2)


if __name__ == "__main__":
    unittest.main()
