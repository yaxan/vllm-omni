# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-request LeapCache runtime behind the hooks in ``Wan22Pipeline.diffuse``."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.leapcache.config import (
    SIGMA_LATE,
    SIGMA_PHASE,
    SIGMA_WARMUP,
    TOLERANCE_RATIO,
    LeapCacheConfig,
)
from vllm_omni.diffusion.cache.leapcache.state import CacheState
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler

logger = init_logger(__name__)


@dataclass
class LeapCacheStats:
    """Work done by the last request."""

    forwards: int = 0
    trial_solver_steps: int = 0
    replayed_solver_steps: int = 0


@dataclass
class _Step:
    index: int
    passes: int
    force: bool
    run: bool = True
    seen: int = 0


@dataclass
class _Checkpoint:
    """Solver state, latent and model output at the last model run of the late phase."""

    index: int
    scheduler: FlowUniPCMultistepScheduler
    latents: torch.Tensor
    noise_pred: torch.Tensor


@dataclass
class _Request:
    scheduler: FlowUniPCMultistepScheduler
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    warmup: int
    phase_boundary: int
    normalizer: float
    state: CacheState
    visited: list[int] = field(default_factory=list)
    late: bool = False
    step: _Step | None = None
    checkpoint: _Checkpoint | None = None

    @property
    def num_steps(self) -> int:
        return len(self.timesteps)


def tolerance_factor(sigma: float) -> float:
    """1 down to SIGMA_PHASE, rising to TOLERANCE_RATIO at SIGMA_LATE and below."""
    position = max(0.0, min(1.0, (SIGMA_PHASE - sigma) / (SIGMA_PHASE - SIGMA_LATE)))
    return math.exp(math.log(TOLERANCE_RATIO) * position)


class LeapCacheRuntime:
    """Serves one request at a time through the hooks in the Wan denoising loop.

    ``request`` owns the per-request state. Per step, ``begin_step`` tells the runtime which step
    starts and how many passes it runs, ``predict_noise`` holds or runs each pass, ``next_step``
    picks the next step and rebinds the solver's schedule, and ``before_scheduler_step`` replays
    held steps once the model has run again.
    """

    def __init__(self, config: LeapCacheConfig) -> None:
        self.config = config
        self.stats = LeapCacheStats()
        self._request: _Request | None = None

    def reset(self) -> None:
        self._request = None
        self.stats = LeapCacheStats()

    @contextmanager
    def request(self, scheduler: FlowUniPCMultistepScheduler, latents: torch.Tensor) -> Iterator[None]:
        """Own one request's state; drop it when the request ends, also on error."""
        if self._request is not None:
            raise RuntimeError("LeapCache serves one request at a time; a request is already active")
        if not isinstance(scheduler, FlowUniPCMultistepScheduler):
            raise ValueError("LeapCache requires the unipc sample_solver")
        if latents.shape[0] != 1:
            raise ValueError(f"LeapCache serves one video per batch, got a batch of {latents.shape[0]}")
        sigmas, num_steps = scheduler.sigmas, len(scheduler.timesteps)
        warmup = min(num_steps, max(2, int((sigmas[:-1] > SIGMA_WARMUP).sum())))
        phase_boundary = next((i for i, s in enumerate(sigmas[:-1].tolist()) if s <= SIGMA_PHASE), num_steps - 1)
        # Normalize so the mean tolerance over the steps that may be skipped equals the knob.
        skippable = sigmas[warmup:-2].tolist()
        normalizer = sum(tolerance_factor(s) for s in skippable) / len(skippable) if skippable else 1.0
        self.stats = LeapCacheStats()
        self._request = _Request(
            scheduler=scheduler,
            sigmas=sigmas,
            timesteps=scheduler.timesteps,
            warmup=warmup,
            phase_boundary=phase_boundary,
            normalizer=normalizer,
            state=CacheState(self.config.threshold),
        )
        try:
            yield
        finally:
            self._request = None
            logger.debug(
                "LeapCache request done: forwards=%d trial_solver_steps=%d replayed_solver_steps=%d",
                self.stats.forwards,
                self.stats.trial_solver_steps,
                self.stats.replayed_solver_steps,
            )

    def begin_step(self, step_index: int, do_true_cfg: bool) -> None:
        """Start step ``step_index``; it runs two passes with CFG, one without."""
        request = self._active()
        # Every step before the late phase runs the model, and so does the step that enters it.
        force = not request.late
        request.late = request.late or step_index >= request.phase_boundary
        request.step = _Step(index=step_index, passes=2 if do_true_cfg else 1, force=force)

    def predict_noise(self, hidden_states: torch.Tensor, run_model: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Hold or run one pass. The prompt pass decides for the whole step."""
        request, step = self._current()
        branch, step.seen = step.seen, step.seen + 1
        if branch == 0:
            force = step.force or step.index == request.num_steps - 1 or len(request.state.held) < step.passes
            step.run = request.state.decide(hidden_states, self._threshold(request, step.index), force=force)
        if not step.run:
            return request.state.held_output(branch)
        output = run_model()
        self.stats.forwards += 1
        request.state.observe(branch, hidden_states, output)
        return output

    def next_step(self, noise_pred: torch.Tensor, latents: torch.Tensor) -> int:
        """Return the next step; the solver's schedule becomes the steps visited so far plus that one."""
        request, step = self._current()
        request.visited.append(step.index)
        next_index = step.index + 1
        if not request.late and step.index >= request.warmup - 1:
            next_index = self._leap(request, step, noise_pred, latents)
        self._bind(request, request.scheduler, next_index)
        return next_index

    def before_scheduler_step(self, noise_pred: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """In the late phase, replay the held steps once the model runs again, then checkpoint the run."""
        request, step = self._current()
        if not request.late or not step.run:
            return latents
        checkpoint = request.checkpoint
        if checkpoint is not None and step.index > checkpoint.index + 1:
            latents = self._replay(request, checkpoint, step.index, noise_pred)
        # The replay steps the checkpoint's solver in place, so a checkpoint is used once; copy a fresh one.
        request.checkpoint = _Checkpoint(step.index, copy.deepcopy(request.scheduler), latents, noise_pred)
        return latents

    def _active(self) -> _Request:
        if self._request is None:
            raise RuntimeError("LeapCache hooks were called outside a request")
        return self._request

    def _current(self) -> tuple[_Request, _Step]:
        request = self._active()
        if request.step is None:
            raise RuntimeError("LeapCache hooks were called before begin_step")
        return request, request.step

    def _threshold(self, request: _Request, index: int) -> float:
        return self.config.threshold * (tolerance_factor(float(request.sigmas[index])) / request.normalizer)

    def _bind(self, request: _Request, scheduler: FlowUniPCMultistepScheduler, next_index: int) -> None:
        indices = request.visited + ([next_index] if next_index < request.num_steps else [])
        scheduler.sigmas = torch.cat([request.sigmas[indices], request.sigmas[-1:]])
        scheduler.timesteps = request.timesteps[indices]
        scheduler.num_inference_steps = len(indices)

    def _leap(self, request: _Request, step: _Step, noise_pred: torch.Tensor, latents: torch.Tensor) -> int:
        """Trial solver steps find the next step where the rule says run.

        Each trial runs on its own copy of the solver because ``step`` rewrites the solver's
        history lists and counters.
        """
        state = request.state
        if state.k is None or state.last_input is None or state.last_output is None:
            return step.index + 1
        previous = state.last_input
        norm = state.last_output.abs().mean().clamp_min(1e-8)
        accumulated = 0.0
        for candidate in range(step.index + 1, request.num_steps - 1):
            trial = copy.deepcopy(request.scheduler)
            self._bind(request, trial, candidate)
            predicted = trial.step(noise_pred, request.timesteps[step.index], latents, return_dict=False)[0]
            self.stats.trial_solver_steps += 1
            signal = predicted.to(previous.dtype)
            accumulated += float((state.k * (signal - previous).abs().mean() / norm).item())
            if accumulated >= self._threshold(request, candidate):
                return candidate
            previous = signal
        return request.num_steps - 1

    def _replay(self, request: _Request, checkpoint: _Checkpoint, end: int, noise_pred: torch.Tensor) -> torch.Tensor:
        """Redo the steps from the checkpoint to ``end`` with a blend of the two real outputs; no model calls."""
        replay, live = checkpoint.scheduler, request.scheduler
        replay.sigmas, replay.timesteps = live.sigmas, live.timesteps
        replay.num_inference_steps = live.num_inference_steps
        start = checkpoint.index
        sigma_start, sigma_end = float(request.sigmas[start]), float(request.sigmas[end])
        latents = checkpoint.latents
        for index in range(start, end):
            if index == start:
                estimate = checkpoint.noise_pred
            else:
                weight = (float(request.sigmas[index]) - sigma_start) / (sigma_end - sigma_start)
                estimate = torch.lerp(checkpoint.noise_pred.float(), noise_pred.float(), weight).to(noise_pred.dtype)
            latents = replay.step(estimate, request.timesteps[index], latents, return_dict=False)[0]
            self.stats.replayed_solver_steps += 1
        live.__dict__.update(replay.__dict__)
        return latents
