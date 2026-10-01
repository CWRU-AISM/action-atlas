#!/usr/bin/env python3
"""
Per-model prompt tokenization and instruction spans for residual-stream edits.

A ModelSpec tokenizes a prompt the way the policy sees it, picks the residual
site to hook, and maps instruction tokens to positions in that site's sequence.
Fillers are function-word prompts matched to an instruction's token count.
"""

import copy
from typing import List

import torch

from experiments.hooks import ResidualSite, SmolVLAResidualSite

FILLER_WORDS = ("the", "of", "and", "to", "in", "a", "is", "it", "on", "at",
                "by", "for", "with", "as", "or", "an", "be", "this", "that", "from")

# Prompt format of openvla_oft's get_vla_action
OFT_PROMPT = "In: What action should the robot take to {}?\nOut:"


def filler(n_words: int) -> str:
    return " ".join(FILLER_WORDS[i % len(FILLER_WORDS)] for i in range(n_words))


def match_filler(count_tokens, target: int, max_words: int = 40) -> str:
    # Shortest filler whose attended-token count equals the instruction's
    for n in range(1, max_words + 1):
        text = filler(n)
        if count_tokens(text) == target:
            return text
    raise ValueError(f"no filler of up to {max_words} words has {target} tokens")


def differing_positions(ids_a: torch.Tensor, ids_f: torch.Tensor) -> List[int]:
    return torch.nonzero(ids_a != ids_f).flatten().tolist()


class ModelSpec:
    # Tokenizes a prompt as the policy sees it and maps instruction tokens to residual positions

    default_span = "instruction"
    # OFT's first policy call in a process differs from every later call
    warmup = False

    def __init__(self, adapter):
        self.adapter = adapter

    def site(self) -> ResidualSite:
        raise NotImplementedError

    def tokenize(self, env, seed):
        # Returns prompt -> (language token ids, attended token count) on one fixed observation
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        obs, _ = env.reset(seed=seed)

        def run(prompt):
            batch = self.adapter.prepare_batch(copy.deepcopy(obs), prompt)
            ids = batch[OBS_LANGUAGE_TOKENS][0].cpu()
            mask = batch.get(OBS_LANGUAGE_ATTENTION_MASK)
            count = int(mask[0].sum()) if mask is not None else int((ids != ids[-1]).sum())
            return ids, count

        return run

    def text_offset(self, seq_len: int, n_text: int) -> int:
        raise NotImplementedError

    def episode_kwargs(self, env_info: dict) -> dict:
        return {}

    def span(self, mode, diff, seq_len, n_text):
        offset = self.text_offset(seq_len, n_text)
        if mode == "instruction":
            return [offset + p for p in diff]
        if mode == "tail":
            return list(range(offset + diff[0], offset + n_text))
        raise ValueError(f"span mode {mode!r} not supported for {type(self).__name__}")


class Pi05Spec(ModelSpec):
    # Prefix = image tokens then the padded text block

    def site(self):
        return ResidualSite(self.adapter.get_layer_groups()["paligemma"])

    def text_offset(self, seq_len, n_text):
        return seq_len - n_text


class SmolVLASpec(ModelSpec):
    # Prefix = image tokens, the language block, then one state token

    def site(self):
        vlm = self.adapter.policy.model.vlm_with_expert.vlm
        return SmolVLAResidualSite(list(vlm.model.text_model.layers))

    def text_offset(self, seq_len, n_text):
        return seq_len - n_text - 1


class XVLASpec(ModelSpec):
    # Transformer sequence = [action tokens | Florence encoder output (image + text) |
    # aux views | soft prompts]

    default_span = "vlm_block"

    def __init__(self, adapter):
        super().__init__(adapter)
        self.layout = {}

        # Recorded on every forward: the Florence block length follows the prompt
        def record(module, args, kwargs):
            self.layout["n_actions"] = kwargs["action_with_noise"].shape[1]
            self.layout["n_vlm"] = kwargs["vlm_features"].shape[1]

        adapter.policy.model.transformer.register_forward_pre_hook(record, with_kwargs=True)

    def site(self):
        return ResidualSite(self.adapter.get_layer_groups()["transformer"])

    def text_offset(self, seq_len, n_text):
        return self.layout["n_actions"] + self.layout["n_vlm"] - n_text

    def span(self, mode, diff, seq_len, n_text):
        if mode == "vlm_block":
            start = self.layout["n_actions"]
            return list(range(start, start + self.layout["n_vlm"]))
        return super().span(mode, diff, seq_len, n_text)


class GR00TSpec(ModelSpec):
    # Eagle chat template: image tokens, the instruction, then the assistant suffix

    default_span = "tail"

    def site(self):
        return ResidualSite(self.adapter.get_layer_groups()["eagle"])

    def tokenize(self, env, seed):
        from experiments.groot_common import build_groot_inputs, get_libero_state_groot
        from experiments.libero_utils import get_libero_images
        obs = env.reset()
        images = get_libero_images(obs, target_size=(256, 256))
        state = get_libero_state_groot(obs, env)

        def run(prompt):
            inputs = build_groot_inputs(images, state, prompt, self.adapter.stats,
                                        self.adapter.eagle_processor, torch.device("cpu"))
            ids = inputs["eagle_input_ids"][0]
            return ids, len(ids)

        return run

    def text_offset(self, seq_len, n_text):
        return 0


class OFTSpec(ModelSpec):
    # One Llama stack with bidirectional attention over [BOS | image + proprio patches |
    # prompt | action placeholders], so the instruction reaches every position and the
    # whole sequence is edited

    default_span = "sequence"
    warmup = True

    def site(self):
        return ResidualSite(self.adapter.get_layer_groups()["llm"])

    def tokenize(self, env, seed):
        tokenizer = self.adapter.components["processor"].tokenizer

        def run(prompt):
            ids = torch.tensor(tokenizer(OFT_PROMPT.format(prompt.lower()))["input_ids"])
            return ids, len(ids)

        return run

    def span(self, mode, diff, seq_len, n_text):
        if mode == "sequence":
            return list(range(seq_len))
        raise ValueError(f"span mode {mode!r} not supported for OFTSpec")

    def episode_kwargs(self, env_info):
        return {"init_states": env_info["init_states"]}


SPECS = {"pi05": Pi05Spec, "smolvla": SmolVLASpec, "xvla": XVLASpec,
         "groot": GR00TSpec, "oft": OFTSpec}
