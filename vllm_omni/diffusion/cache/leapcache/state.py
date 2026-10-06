# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Skip-or-run estimator and held outputs for one request."""

from __future__ import annotations

import torch


class CacheState:
    """EasyCache's skip-or-run rule on the prompt pass, plus one held output per pass.

    The rule predicts how much the model output would move from how much the input moved,
    using the ratio of the two seen at the last two model runs, and sums the predictions
    since the last run. Only prompt-pass observations update the ratio. Held outputs are the
    model's own tensors; nothing downstream writes into them.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.previous_input: torch.Tensor | None = None
        self.last_input: torch.Tensor | None = None
        self.last_output: torch.Tensor | None = None
        self.k: float | None = None
        self.accumulated = 0.0
        self.held: dict[int, torch.Tensor] = {}

    def decide(self, signal: torch.Tensor, threshold: float, *, force: bool) -> bool:
        """Return True when the model must run at this step."""
        if self.threshold == 0:
            return True
        if force or self.k is None or self.previous_input is None or self.last_output is None:
            run = True
        else:
            change = (signal - self.previous_input).abs().mean()
            norm = self.last_output.abs().mean().clamp_min(1e-8)
            self.accumulated += float((self.k * change / norm).item())
            run = self.accumulated >= threshold
        if run:
            self.accumulated = 0.0
        self.previous_input = signal
        return run

    def observe(self, branch: int, signal: torch.Tensor, output: torch.Tensor) -> None:
        """Hold a fresh output for ``branch``; the prompt pass (branch 0) also updates the ratio."""
        if self.threshold == 0:
            return
        self.held[branch] = output.detach()
        if branch != 0:
            return
        if self.last_input is not None and self.last_output is not None:
            dx = (signal - self.last_input).abs().mean().clamp_min(1e-8)
            dy = (output - self.last_output).abs().mean()
            self.k = float((dy / dx).item())
        self.last_input = signal
        self.last_output = output

    def held_output(self, branch: int) -> torch.Tensor:
        return self.held[branch]
