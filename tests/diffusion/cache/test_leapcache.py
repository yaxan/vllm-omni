# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the LeapCache backend on the real Wan denoising loop and UniPC solver.

The small transformer is nonlinear in the latent and in time, so the skip-or-run rule
leaps, holds and replays at the default threshold. No checkpoint or GPU is needed.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.cache.leapcache import (
    GUIDANCE_INTERVAL,
    LEAP_CACHE_CONFIG_ATTR,
    LEAP_CACHE_RUNTIME_ATTR,
    CacheState,
    LeapCacheBackend,
    LeapCacheConfig,
    LeapCacheRuntime,
    LeapCacheStats,
    get_leapcache_runtime,
)
from vllm_omni.diffusion.cache.leapcache.config import SIGMA_PHASE
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.data import DiffusionCacheConfig, OmniDiffusionConfig
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline, build_wan_scheduler
from vllm_omni.engine.omni_engine_base import OmniEngineBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PROMPT = torch.tensor([1.0, 1.0])
NEGATIVE = torch.tensor([1.0, 0.0])
PARALLEL_DEGREES = (
    "cfg_parallel_size",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "ulysses_degree",
    "ring_degree",
    "allgather_degree",
)


@pytest.fixture(autouse=True)
def _single_device(monkeypatch):
    # The pipeline-parallel wrapper around diffuse needs a process group of one.
    monkeypatch.setattr("vllm_omni.diffusion.distributed.parallel_state._PP", SimpleNamespace(world_size=1))


class _Transformer(nn.Module):
    """Nonlinear in the latent and in time; records (timestep, input, output, is_prompt_pass) per forward."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[float, torch.Tensor, torch.Tensor, bool]] = []
        self.fail_below: float | None = None

    def forward(self, hidden_states, timestep, encoder_hidden_states, attention_kwargs=None, return_dict=False):
        if self.fail_below is not None and float(timestep[0]) < self.fail_below:
            raise RuntimeError("injected failure")
        amplitude, condition = encoder_hidden_states.tolist()
        sigma = timestep[0].float() / 1000
        x = hidden_states.float()
        value = 0.35 + 0.08 * x + 0.02 * x.square() + amplitude * 0.1 * torch.sin(sigma * 10) + condition * 0.007
        value = value.to(hidden_states.dtype)
        self.calls.append((float(timestep[0]), hidden_states, value, condition == 1.0))
        return (value,)


@contextmanager
def _noop_progress_bar(*args, **kwargs):
    del args, kwargs
    yield SimpleNamespace(update=lambda n=1: None)


def _make_pipeline() -> Wan22Pipeline:
    pipeline = object.__new__(Wan22Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = _Transformer()
    pipeline.transformer_2 = None
    pipeline.od_config = OmniDiffusionConfig()
    pipeline.is_dmd = False
    pipeline.expand_timesteps = False
    pipeline._cache_dit_requires_paired_cfg = False
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.scheduler = build_wan_scheduler("unipc", 5.0)
    pipeline.progress_bar = _noop_progress_bar
    return pipeline


def _enable(pipeline, threshold: float = 0.044) -> LeapCacheRuntime:
    LeapCacheBackend(DiffusionCacheConfig(leap_threshold=threshold)).enable(pipeline)
    runtime = get_leapcache_runtime(pipeline)
    assert runtime is not None
    return runtime


def _diffuse(
    pipeline,
    *,
    num_steps: int = 40,
    guidance: float = 4.0,
    flow_shift: float = 5.0,
    guidance_interval: tuple[float, float] | None = None,
    negative: bool = True,
    latents: torch.Tensor | None = None,
):
    """Run the real loop. Returns the output, the schedule, the transformer calls and the solver inputs."""
    pipeline.scheduler.set_timesteps(num_steps, device="cpu", shift=flow_shift)
    pipeline._num_timesteps = num_steps
    timesteps, sigmas = pipeline.scheduler.timesteps.tolist(), pipeline.scheduler.sigmas.tolist()
    first_call = len(pipeline.transformer.calls)
    solver_steps: list[tuple[float, torch.Tensor, torch.Tensor]] = []
    step = pipeline.scheduler_step_maybe_with_cfg

    def record(noise_pred, t, current_latents, do_true_cfg):
        solver_steps.append((float(t), current_latents, noise_pred))
        return step(noise_pred, t, current_latents, do_true_cfg)

    pipeline.scheduler_step_maybe_with_cfg = record
    try:
        output = pipeline.diffuse(
            latents=torch.linspace(-2, 2, 16 * 3 * 2 * 2).reshape(1, 16, 3, 2, 2) if latents is None else latents,
            timesteps=pipeline.scheduler.timesteps,
            prompt_embeds=PROMPT,
            negative_prompt_embeds=NEGATIVE if negative else None,
            guidance_low=guidance,
            guidance_high=guidance,
            boundary_timestep=875.0,
            dtype=torch.bfloat16,
            attention_kwargs={},
            guidance_interval=guidance_interval,
        )
    finally:
        del pipeline.scheduler_step_maybe_with_cfg
    calls = pipeline.transformer.calls[first_call:]
    return SimpleNamespace(
        output=output,
        timesteps=timesteps,
        sigmas=sigmas,
        calls=calls,
        solver_steps=solver_steps,
        visited=[timesteps.index(int(t)) for t, _, _ in solver_steps],
        fresh=sorted({timesteps.index(int(t)) for t, _, _, _ in calls}),
    )


def _assert_same_calls(actual, expected) -> None:
    assert len(actual) == len(expected)
    for (t, x, y, prompt), (other_t, other_x, other_y, other_prompt) in zip(actual, expected):
        assert t == other_t and prompt == other_prompt
        assert torch.equal(x, other_x) and torch.equal(y, other_y)


def _assert_same_solver_state(actual, expected) -> None:
    for name in ("model_outputs", "timestep_list", "last_sample", "_step_index", "lower_order_nums", "this_order"):
        left, right = getattr(actual, name), getattr(expected, name)
        if isinstance(left, list):
            assert len(left) == len(right), name
        else:
            left, right = [left], [right]
        for a, b in zip(left, right):
            assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b, name


def _estimator_snapshot(state: CacheState):
    return (state.k, state.accumulated, state.previous_input, state.last_input, state.last_output, dict(state.held))


def _assert_estimator_untouched(state: CacheState, snapshot) -> None:
    assert (state.k, state.accumulated) == snapshot[:2]
    assert all(a is b for a, b in zip((state.previous_input, state.last_input, state.last_output), snapshot[2:5]))
    assert state.held.keys() == snapshot[5].keys() and all(state.held[b] is snapshot[5][b] for b in state.held)


# --- config, backend, selector -------------------------------------------------------------------


class TestLeapCacheConfig:
    def test_threshold_comes_from_leap_threshold(self):
        assert GUIDANCE_INTERVAL == (600.0, 1000.0)
        assert LeapCacheConfig.from_diffusion_cache_config(DiffusionCacheConfig()).threshold == 0.044
        assert LeapCacheConfig.from_diffusion_cache_config(DiffusionCacheConfig(leap_threshold=0.1)).threshold == 0.1

    def test_unknown_cache_config_keys_are_rejected(self):
        with pytest.raises(ValueError, match="leap_tolerance"):
            LeapCacheConfig.from_diffusion_cache_config(DiffusionCacheConfig.from_dict({"leap_tolerance": 0.1}))

    @pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf"), "0.1", True])
    def test_rejects_invalid_threshold(self, threshold):
        with pytest.raises(ValueError, match="leap_threshold"):
            LeapCacheConfig.from_diffusion_cache_config(DiffusionCacheConfig(leap_threshold=threshold))


class TestLeapCacheBackend:
    def test_selector_returns_backend_with_config(self):
        backend = get_cache_backend("leap_cache", {"leap_threshold": 0.1})
        assert isinstance(backend, LeapCacheBackend)
        assert backend.config.leap_threshold == 0.1
        assert get_cache_backend("leap_cache", {}).config.leap_threshold == 0.044

    def test_engine_default_cache_config(self):
        assert OmniEngineBase._get_default_cache_config("leap_cache") == {"leap_threshold": 0.044}

    def test_enable_attaches_config_and_runtime(self):
        pipeline = _make_pipeline()
        backend = LeapCacheBackend(DiffusionCacheConfig(leap_threshold=0.1))
        backend.enable(pipeline)
        assert backend.enabled
        assert getattr(pipeline, LEAP_CACHE_CONFIG_ATTR) == LeapCacheConfig(threshold=0.1)
        runtime = get_leapcache_runtime(pipeline)
        assert isinstance(runtime, LeapCacheRuntime) and runtime.config.threshold == 0.1

    @pytest.mark.parametrize(
        "kind", ["use_hsdp", "other_pipeline", "transformer_2", "is_dmd", "expand_timesteps", *PARALLEL_DEGREES]
    )
    def test_enable_rejects_unsupported_pipelines_before_attaching_anything(self, kind, monkeypatch):
        pipeline = _make_pipeline()
        if kind == "other_pipeline":
            pipeline = SimpleNamespace()
        elif kind == "transformer_2":
            pipeline.transformer_2 = _Transformer()
        elif kind in ("is_dmd", "expand_timesteps"):
            setattr(pipeline, kind, True)
        elif kind == "use_hsdp":
            monkeypatch.setattr(pipeline.od_config.parallel_config, "use_hsdp", True)
        else:
            monkeypatch.setattr(pipeline.od_config.parallel_config, kind, 2)
        with pytest.raises(ValueError, match="leap_cache"):
            LeapCacheBackend(DiffusionCacheConfig()).enable(pipeline)
        assert not hasattr(pipeline, LEAP_CACHE_CONFIG_ATTR) and not hasattr(pipeline, LEAP_CACHE_RUNTIME_ATTR)

    def test_refresh_resets_request_state_and_stats(self):
        pipeline = _make_pipeline()
        backend = LeapCacheBackend(DiffusionCacheConfig())
        backend.enable(pipeline)
        runtime = get_leapcache_runtime(pipeline)
        assert _diffuse(pipeline).calls and runtime.stats.forwards > 0
        latents = torch.zeros(1, 16, 1, 1, 1)
        pending = runtime.request(pipeline.scheduler, latents)
        pending.__enter__()  # a request the loop never finished
        assert runtime._request is not None
        backend.refresh(pipeline, num_inference_steps=40, verbose=False)
        assert runtime._request is None and runtime.stats == LeapCacheStats()
        pending.__exit__(None, None, None)  # close it, or its teardown would clear a later request
        with runtime.request(pipeline.scheduler, latents):
            pass
        with pytest.raises(RuntimeError, match="not enabled"):
            backend.refresh(_make_pipeline(), num_inference_steps=40)


# --- disabled path -------------------------------------------------------------------------------


@pytest.mark.parametrize("guidance_interval", [None, (600.0, 1000.0)])
@pytest.mark.parametrize(
    "num_steps,guidance,flow_shift,negative",
    [(40, 4.0, 5.0, True), (30, 5.0, 5.0, True), (50, 3.0, 3.0, True), (1, 4.0, 5.0, True), (2, 1.0, 3.0, False)],
)
def test_disabled_zero_threshold_gives_the_plain_pipeline_frames(
    num_steps, guidance, flow_shift, negative, guidance_interval
):
    settings = dict(
        num_steps=num_steps,
        guidance=guidance,
        flow_shift=flow_shift,
        negative=negative,
        guidance_interval=guidance_interval,
    )
    plain = _diffuse(_make_pipeline(), **settings)
    pipeline = _make_pipeline()
    runtime = _enable(pipeline, threshold=0.0)
    cached = _diffuse(pipeline, **settings)
    assert torch.equal(cached.output, plain.output)
    _assert_same_calls(cached.calls, plain.calls)
    assert cached.visited == plain.visited == list(range(num_steps))
    assert runtime.stats == LeapCacheStats(forwards=len(plain.calls))


# --- hit path ------------------------------------------------------------------------------------


def test_hit_held_steps_reuse_the_latest_outputs_of_both_passes():
    pipeline = _make_pipeline()
    runtime = _enable(pipeline)
    result = _diffuse(pipeline)
    held = [index for index in result.visited if index not in result.fresh]
    assert held and runtime.stats.forwards == len(result.calls) == 2 * len(result.fresh)
    fields = {index: field for index, (_, _, field) in zip(result.visited, result.solver_steps)}
    for index in held:
        latest_run = max(run for run in result.fresh if run < index)
        assert torch.equal(fields[index], fields[latest_run])


def test_hit_tail_holds_and_refreshes_the_prompt_pass_only():
    pipeline = _make_pipeline()
    _enable(pipeline)
    result = _diffuse(pipeline, guidance_interval=(600.0, 1000.0))
    calls_at: dict[int, list[tuple[bool, torch.Tensor]]] = {index: [] for index in result.visited}
    for t, _, output, prompt in result.calls:
        calls_at[result.timesteps.index(int(t))].append((prompt, output))
    tail = [index for index in result.visited if result.timesteps[index] < 600]
    assert tail and any(calls_at[index] for index in tail) and not all(calls_at[index] for index in tail)
    for index in result.visited:
        passes = [prompt for prompt, _ in calls_at[index]]
        assert passes in ([], [True]) if index in tail else passes in ([], [True, False])
    last_prompt_output = None
    for index, (_, _, field) in zip(result.visited, result.solver_steps):
        if calls_at[index]:
            last_prompt_output = calls_at[index][0][1]
        if index in tail:
            assert torch.equal(field, last_prompt_output)


# --- miss path -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_steps,guidance,flow_shift,warmup", [(40, 4.0, 5.0, 7), (30, 5.0, 5.0, 6), (50, 3.0, 3.0, 6)]
)
def test_miss_warmup_and_final_step_run_and_early_leaps_drop_steps(num_steps, guidance, flow_shift, warmup, mocker):
    pipeline = _make_pipeline()
    runtime = _enable(pipeline)
    record_denoise_step = mocker.spy(pipeline, "record_denoise_step")
    result = _diffuse(pipeline, num_steps=num_steps, guidance=guidance, flow_shift=flow_shift)
    assert torch.isfinite(result.output).all()
    assert result.fresh[:warmup] == list(range(warmup)) and result.fresh[-1] == result.visited[-1] == num_steps - 1
    assert len(result.visited) < num_steps
    phase_boundary = next(index for index, sigma in enumerate(result.sigmas[:-1]) if sigma <= SIGMA_PHASE)
    first_late = next(index for index in result.visited if index >= phase_boundary)
    early = result.visited[: result.visited.index(first_late)]
    assert early and all(index in result.fresh for index in early)
    assert result.visited[len(early) :] == list(range(first_late, num_steps))
    # The dropped steps left the loop, and the replays put them back: the run ends on the full schedule.
    assert pipeline.scheduler.timesteps.tolist() == result.timesteps
    assert pipeline.scheduler.step_index == num_steps
    assert [call.args[0] for call in record_denoise_step.call_args_list] == result.visited
    assert pipeline._num_timesteps == num_steps
    # Each skipped step, dropped or held, is redone once, plus the checkpoint node of every replay.
    gaps = sum(1 for a, b in zip(result.fresh, result.fresh[1:]) if b > a + 1)
    assert gaps > 0 and runtime.stats.trial_solver_steps > 0
    assert runtime.stats.replayed_solver_steps == (num_steps - len(result.fresh)) + gaps


def test_miss_trial_steps_leave_the_live_solver_and_estimator_untouched(monkeypatch):
    pipeline = _make_pipeline()
    runtime = _enable(pipeline)
    next_step = runtime.next_step
    leaps = []

    def checked_next_step(noise_pred, latents):
        solver, state = pipeline.scheduler, runtime._request.state
        before, snapshot = copy.deepcopy(solver), _estimator_snapshot(state)
        trials, calls = runtime.stats.trial_solver_steps, len(pipeline.transformer.calls)
        next_index = next_step(noise_pred, latents)
        if runtime.stats.trial_solver_steps > trials:
            leaps.append(next_index)
        assert pipeline.scheduler is solver and len(pipeline.transformer.calls) == calls
        _assert_same_solver_state(solver, before)
        _assert_estimator_untouched(state, snapshot)
        return next_index

    monkeypatch.setattr(runtime, "next_step", checked_next_step)
    _diffuse(pipeline)
    assert leaps


def test_miss_replay_blends_the_two_real_outputs_and_never_observes_them(monkeypatch):
    pipeline = _make_pipeline()
    runtime = _enable(pipeline)
    before_scheduler_step = runtime.before_scheduler_step
    replays = []

    def checked_before_scheduler_step(noise_pred, latents):
        request = runtime._request
        step, checkpoint, state = request.step, request.checkpoint, request.state
        snapshot, calls = _estimator_snapshot(state), len(pipeline.transformer.calls)
        if checkpoint is None or not step.run or step.index <= checkpoint.index + 1:
            result = before_scheduler_step(noise_pred, latents)
            assert result is latents
        else:
            start, end, next_index = checkpoint.index, step.index, request.next_index
            dropped = start + 1 not in request.visited  # a leap dropped the interval; else the late phase held it
            # Expected: from the checkpoint, redo every step of the interval on the dense schedule with the
            # straight-line blend in sigma of the two real outputs.
            dense = sorted(set(request.visited) | set(range(start + 1, end)))
            nodes = dense + ([next_index] if next_index < request.num_steps else [])
            assert dense == list(range(end + 1))
            expected_solver = copy.deepcopy(checkpoint.scheduler)
            expected_solver.sigmas = torch.cat([request.sigmas[nodes], request.sigmas[-1:]])
            expected_solver.timesteps = request.timesteps[nodes]
            expected_solver.num_inference_steps = len(nodes)
            assert expected_solver.step_index == start
            # A held stretch is already in the schedule next_step bound; a dropped one is added to it here.
            assert torch.equal(expected_solver.sigmas, pipeline.scheduler.sigmas) == (not dropped)
            # The model saw the latent as it was before the replay.
            assert torch.equal(pipeline.transformer.calls[-1][1], latents.to(torch.bfloat16))
            expected = checkpoint.latents
            sigma_start, sigma_end = request.sigmas[start].item(), request.sigmas[end].item()
            for index in range(start, end):
                weight = (request.sigmas[index].item() - sigma_start) / (sigma_end - sigma_start)
                estimate = torch.lerp(checkpoint.noise_pred.float(), noise_pred.float(), weight).to(noise_pred.dtype)
                field = checkpoint.noise_pred if index == start else estimate
                expected = expected_solver.step(field, request.timesteps[index], expected, return_dict=False)[0]
            result = before_scheduler_step(noise_pred, latents)
            assert torch.equal(result, expected) and not torch.equal(result, latents)
            # The live solver continues from the dense past: its schedule, step index and history are the copy's.
            _assert_same_solver_state(pipeline.scheduler, expected_solver)
            assert torch.equal(pipeline.scheduler.sigmas, expected_solver.sigmas)
            assert torch.equal(pipeline.scheduler.timesteps, expected_solver.timesteps)
            assert pipeline.scheduler.num_inference_steps == expected_solver.num_inference_steps
            assert pipeline.scheduler.step_index == end
            replays.append((start, end, dropped))
        if step.run and step.index < request.num_steps - 1:
            assert request.checkpoint.index == step.index and request.checkpoint.latents is result
        else:
            # A held step takes no checkpoint, and neither does the last step: nothing could read it.
            assert request.checkpoint is checkpoint
        assert request.visited == list(range(step.index + 1))
        assert len(pipeline.transformer.calls) == calls
        _assert_estimator_untouched(state, snapshot)
        return result

    monkeypatch.setattr(runtime, "before_scheduler_step", checked_before_scheduler_step)
    result = _diffuse(pipeline)
    assert torch.isfinite(result.output).all()
    # Leaps drop steps before the phase boundary and the late phase holds them after it; both are replayed.
    phase_boundary = next(index for index, sigma in enumerate(result.sigmas[:-1]) if sigma <= SIGMA_PHASE)
    assert any(dropped and start < phase_boundary for start, _, dropped in replays)
    assert any(not dropped and start >= phase_boundary for start, _, dropped in replays)
    assert all(dropped == (start < phase_boundary) for start, _, dropped in replays)
    assert runtime.stats.replayed_solver_steps == sum(end - start for start, end, _ in replays)


# --- invalidation path ---------------------------------------------------------------------------


def test_invalidation_next_request_with_another_schedule_or_guidance_starts_clean():
    pipeline = _make_pipeline()
    runtime = _enable(pipeline)
    first = _diffuse(pipeline, num_steps=30, guidance=5.0)
    first_stats = runtime.stats
    other = _diffuse(pipeline, num_steps=50, guidance=1.0, flow_shift=3.0, negative=False)
    again = _diffuse(pipeline, num_steps=30, guidance=5.0)
    assert len(other.calls) < len(other.visited) <= 50
    assert torch.equal(first.output, again.output)
    _assert_same_calls(first.calls, again.calls)
    assert runtime.stats == first_stats


# --- concurrent request path ---------------------------------------------------------------------


def test_concurrent_request_is_rejected_before_any_model_call():
    pipeline = _make_pipeline()
    runtime = _enable(pipeline)
    pipeline.scheduler.set_timesteps(40, device="cpu", shift=5.0)
    with runtime.request(pipeline.scheduler, torch.zeros(1, 16, 1, 1, 1)):
        with pytest.raises(RuntimeError, match="already active"):
            _diffuse(pipeline)
    assert not pipeline.transformer.calls
    assert torch.isfinite(_diffuse(pipeline).output).all()


def test_concurrent_batch_above_one_is_rejected_before_any_model_call():
    pipeline = _make_pipeline()
    _enable(pipeline)
    with pytest.raises(ValueError, match="one video per batch"):
        _diffuse(pipeline, latents=torch.zeros(2, 16, 3, 2, 2))
    assert not pipeline.transformer.calls
    assert torch.isfinite(_diffuse(pipeline).output).all()


def test_request_rejects_the_euler_solver():
    runtime = _enable(_make_pipeline())
    with pytest.raises(ValueError, match="unipc"):
        with runtime.request(build_wan_scheduler("euler", 5.0), torch.zeros(1, 16, 1, 1, 1)):
            pass


# --- teardown path -------------------------------------------------------------------------------


def test_teardown_drops_request_state_after_every_request_and_on_error():
    reference_pipeline = _make_pipeline()
    _enable(reference_pipeline)
    reference = _diffuse(reference_pipeline)
    pipeline = _make_pipeline()
    runtime = _enable(pipeline)
    pipeline.transformer.fail_below = 700.0
    with pytest.raises(RuntimeError, match="injected failure"):
        _diffuse(pipeline)
    with pytest.raises(RuntimeError, match="outside a request"):
        runtime.begin_step(0, True)
    pipeline.transformer.fail_below = None
    result = _diffuse(pipeline)
    assert torch.equal(result.output, reference.output)
    with pytest.raises(RuntimeError, match="outside a request"):
        runtime.next_step(result.output, result.output)


# --- estimator -----------------------------------------------------------------------------------


def test_state_decides_on_the_prompt_pass_and_only_it_updates_the_ratio():
    def tensor(value):
        return torch.full((1, 16, 1, 1, 1), value, dtype=torch.bfloat16)

    state = CacheState(0.032)
    for x, prompt, negative in [(0.0, 10.0, -3.0), (1.0, 12.0, -7.0)]:
        assert state.decide(tensor(x), 0.032, force=True)
        state.observe(0, tensor(x), tensor(prompt))
        k = state.k
        state.observe(1, tensor(x), tensor(negative))
        assert state.k == k
    assert state.k == 2.0  # (12 - 10) / (1 - 0)
    assert torch.equal(state.held_output(0), tensor(12.0)) and torch.equal(state.held_output(1), tensor(-7.0))
    assert state.held_output(0).dtype == torch.bfloat16 and state.held_output(0) is state.held[0]
    assert not state.decide(tensor(1.0625), 0.032, force=False)  # predicted 2 * 0.0625 / 12 is below the tolerance
    assert state.accumulated > 0 and state.k == 2.0
    assert state.decide(tensor(1.25), 0.032, force=False)  # the running total crosses the tolerance
    assert state.accumulated == 0.0
    state.observe(0, tensor(1.25), tensor(99.0))
    assert state.k == 87.0 / 0.25  # measured between model runs, not against the held step

    disabled = CacheState(0.0)
    assert disabled.decide(tensor(1.0), 0.0, force=False)
    disabled.observe(0, tensor(1.0), tensor(2.0))
    assert disabled.held == {} and disabled.k is None
