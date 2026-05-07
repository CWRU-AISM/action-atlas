# Reusable forward hooks for activation capture, ablation, and injection

from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn


class ZeroAblationHook:

    def __init__(self):
        self.enabled = True
        self.call_count = 0

    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        self.call_count += 1
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)


class MeanAblationHook:

    def __init__(self):
        self.enabled = True
        self.running_mean = None
        self.count = 0
        self.call_count = 0

    def update_mean(self, output):
        h = output[0] if isinstance(output, tuple) else output
        h_mean = h.detach().mean(dim=0, keepdim=True)
        if self.running_mean is None:
            self.running_mean = h_mean
        else:
            self.running_mean = 0.9 * self.running_mean + 0.1 * h_mean
        self.count += 1

    def __call__(self, module, input, output):
        if not self.enabled:
            self.update_mean(output)
            return output
        self.call_count += 1
        if self.running_mean is not None:
            replacement = self.running_mean.expand_as(
                output[0] if isinstance(output, tuple) else output
            )
            if isinstance(output, tuple):
                return (replacement,) + output[1:]
            return replacement
        return output


class ActivationCaptureHook:
    """
    Capture all forward pass activations for replay/analysis.

    Unlike ActivationCollector, captures every call without gating,
    suitable for replaying into ActivationInjectionHook.
    """

    def __init__(self):
        self.activations: List[torch.Tensor] = []
        self.enabled = True

    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        h = output[0] if isinstance(output, tuple) else output
        self.activations.append(h.detach().clone().cpu())
        return output

    def reset(self):
        self.activations = []


class ActivationInjectionHook:

    def __init__(self, stored: List[torch.Tensor], device: str = "cuda"):
        self.stored = stored
        self.device = device
        self.step = 0
        self.enabled = True
        self.injection_count = 0
        self.shape_mismatches = 0

    def __call__(self, module, input, output):
        if not self.enabled or self.step >= len(self.stored):
            return output
        actual = output[0] if isinstance(output, tuple) else output
        injected = self.stored[self.step].to(device=self.device, dtype=actual.dtype)
        self.step += 1
        if injected.shape != actual.shape:
            self.shape_mismatches += 1
            return output
        self.injection_count += 1
        if isinstance(output, tuple):
            return (injected,) + output[1:]
        return injected

    def reset(self):
        self.step = 0
        self.injection_count = 0
        self.shape_mismatches = 0


class NullInjectionHook:

    def __init__(self):
        self.enabled = True
        self.call_count = 0
        self.original_norms: List[float] = []

    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        self.call_count += 1
        h = output[0] if isinstance(output, tuple) else output
        if self.call_count <= 10:
            self.original_norms.append(h.detach().norm().item())
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)


class ActivationCollector:
    """
    Collect activations from named layers with optional denoising gating.

    Supports per-token and mean-pooled modes. Gating prevents duplicate
    collection from flow-matching denoising steps (GR00T DiT).
    """

    def __init__(self, per_token: bool = True, subsample_every: int = 1):
        self.activations = defaultdict(list)
        self.handles: List[torch.utils.hooks.RemovableHook] = []
        self.per_token = per_token
        self.subsample_every = subsample_every
        self._step_counter = 0
        self._collected_steps: dict = {}

    def new_step(self):
        self._step_counter += 1

    def get_hook(self, name: str, gated: bool = False):
        """
        Return a forward hook for the given layer name.

        Args:
            name: Key under which activations are stored.
            gated: If True, collect at most once per env step (for DiT layers
                   where flow matching causes multiple forward passes).
        """
        self._collected_steps[name] = -1

        def hook_fn(module, input, output):
            if gated and self._step_counter == self._collected_steps.get(name, -1):
                return
            if self.subsample_every > 1 and self._step_counter % self.subsample_every != 0:
                return
            h = output[0] if isinstance(output, tuple) else output
            if self.per_token:
                self.activations[name].append(h.detach().cpu().to(torch.bfloat16))
            else:
                self.activations[name].append(
                    h.detach().mean(dim=1).cpu().to(torch.bfloat16)
                )
            self._collected_steps[name] = self._step_counter

        return hook_fn

    def register(self, module: nn.Module, name: str, gated: bool = False):
        handle = module.register_forward_hook(self.get_hook(name, gated=gated))
        self.handles.append(handle)

    def clear(self):
        self.activations = defaultdict(list)
        self._step_counter = 0
        self._collected_steps = {k: -1 for k in self._collected_steps}

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def get_activations(self):
        return {name: torch.stack(acts, dim=0) for name, acts in self.activations.items() if acts}
