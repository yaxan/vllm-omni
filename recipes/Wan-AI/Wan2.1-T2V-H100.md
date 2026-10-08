# Wan2.1 Text-to-Video with LeapCache on one H100 80GB

> Single-GPU text-to-video for Wan2.1 14B and 1.3B with the LeapCache
> step-skipping cache: 2.3x lower latency per request at the default setting,
> at a mean LPIPS of 0.10 against the uncached clip

## Summary

- Vendor: Wan-AI
- Model: `Wan-AI/Wan2.1-T2V-14B-Diffusers` (measured, revision `38ec498c`) and
  `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- Task: Text-to-video generation
- Mode: Offline `Omni` through the shared text-to-video example, and online
  serving with the OpenAI-compatible Videos API
- Hardware: 1x NVIDIA H100 80GB
- Recommended deployment: one GPU, `--cache-backend leap_cache` at the default
  `leap_threshold` of 0.044, `max_num_seqs` at its default of 1
- Maintainer: Community

## When to use this recipe

Use it when one Wan2.1 text-to-video clip per request on one H100 should take
about 2.3x less time and a small drift from the uncached clip is acceptable.
Run without LeapCache when the output must match the uncached clip exactly.
LeapCache does not run with parallelism. For multi-GPU Wan serving, see the
[Wan2.2 text-to-video recipe](./Wan2.2-T2V.md). The method is described in the
[LeapCache guide](../../docs/user_guide/diffusion/cache_acceleration/leapcache.md).

## Supported model contract

| Task | Input | Output | Entry points |
| --- | --- | --- | --- |
| Text-to-video | Text prompt, optional negative prompt | MP4 clip of 81 frames at 832x480 for both checkpoints, or 1280x720 for the 14B. The model card recommends 480p for the 1.3B | Offline `Omni`, `POST /v1/videos` and `POST /v1/videos/sync` |

Profiles on this hardware at 81 frames, 40 steps, guidance 4, flow shift 5 and
the default `leap_threshold`. LPIPS is against the uncached clip from the same
seed, the mean over the four development prompts and the worst single prompt:

| Profile | Checkpoint | Speed-up | LPIPS mean / worst | Status |
| --- | --- | ---: | ---: | --- |
| 14B 480p | `Wan-AI/Wan2.1-T2V-14B-Diffusers`, 832x480 | 2.30x | 0.099 / 0.149 | Measured with the research harness. The shipped backend gave the same frames on the four development prompts (Qualification evidence) |
| 14B 720p | `Wan-AI/Wan2.1-T2V-14B-Diffusers`, 1280x720 | 2.67x | 0.102 / 0.141 | Research harness only |
| 1.3B 480p | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, 832x480 | 2.07x | 0.053 / 0.102 | Speed from the research harness. The shipped backend passed the GPU test on one prompt at LPIPS 0.046, bound 0.10 |

With LeapCache on: one request per batch (`max_num_seqs=1`, the default), one
GPU, and `guidance_interval=[600, 1000]` unless the request sets its own
interval.

## References

- Model cards: <https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers>,
  <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>
- LeapCache guide:
  [`docs/user_guide/diffusion/cache_acceleration/leapcache.md`](../../docs/user_guide/diffusion/cache_acceleration/leapcache.md)
- Design and measurements: [RFC #8522](https://github.com/vllm-project/vllm-omni/issues/8522).
  The `guidance_interval` option: [PR #8446](https://github.com/vllm-project/vllm-omni/pull/8446)
- Shared examples:
  [`examples/offline_inference/text_to_video`](../../examples/offline_inference/text_to_video),
  [`examples/online_serving/text_to_video`](../../examples/online_serving/text_to_video)
- Videos API: [`docs/serving/videos_api.md`](../../docs/serving/videos_api.md)
- Support tables: [Supported Models](../../docs/models/supported_models.md),
  [feature matrix](../../docs/user_guide/diffusion_features.md#supported-models)

## Hardware

- Accelerator model and per-device memory: NVIDIA H100 80GB HBM3
- Number of devices: 1
- Device interconnect: not applicable to this single-GPU profile
- Host memory: not relevant, no CPU offload
- Qualification scope: the checkpoints in BF16 as shipped, one request at a
  time, regional `torch.compile` with dynamic shapes, the three profiles above

## Software environment

Environment 1, where every number below was measured unless stated:

- OS: Linux
- Python: 3.12.3
- Driver: NVIDIA 580.82.07
- PyTorch: 2.13.0+cu130
- diffusers: 0.40.0 (cache-dit 1.5.0 for the comparison method only)
- vLLM: 0.29.0
- vLLM-Omni: 0.1.dev3119, with the research harness described under
  Qualification evidence

Environment 2, a cross-check: torch 2.13.0+cu132, vLLM 0.30.0, vLLM-Omni at
upstream `3bc3f1a7` with PR #8446. On the development prompts LeapCache gave
2.31x at LPIPS 0.082 there, against 2.30x at 0.099 in Environment 1. Frame
hashes change with the toolchain. The speed-ups move by 0.01x.

## Command

Online serving. The online path with LeapCache was exercised in unit tests,
not run on hardware:

```bash
vllm serve Wan-AI/Wan2.1-T2V-14B-Diffusers --omni --port 8091 \
  --cache-backend leap_cache \
  --cache-config '{"leap_threshold": 0.044}'
```

Request one clip with the measured settings. The server uses an empty negative
prompt when the request sends none, which is the measured condition:

```bash
curl --fail-with-body -X POST http://localhost:8091/v1/videos/sync \
  -F "prompt=a ballerina practicing in the dance studio" \
  -F "width=832" -F "height=480" -F "num_frames=81" \
  -F "num_inference_steps=40" -F "guidance_scale=4.0" -F "flow_shift=5.0" \
  -F "seed=179961516" \
  --output ballet_leapcache.mp4
```

Offline, with the shared example:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --prompt "a ballerina practicing in the dance studio" \
  --negative-prompt "" \
  --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 40 --guidance-scale 4.0 --flow-shift 5.0 \
  --seed 179961516 \
  --cache-backend leap_cache \
  --output ballet_leapcache.mp4
```

Run the same command without `--cache-backend leap_cache`, writing to
`ballet_uncached.mp4`, to get the same-seed reference clip. For the 720p
profile use `--height 720 --width 1280`. For the 1.3B use
`--model Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.

## Verification

```bash
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,nb_read_frames \
  -of default=noprint_wrappers=1 ballet_leapcache.mp4
```

Expected:

```text
width=832
height=480
nb_read_frames=81
```

Compare the wall-clock time of the cached and the uncached command. On the
development prompts the default gives 2.30x against the uncached run of 206 s
per clip. For the ballet prompt in the commands above the default gave 2.6x at
LPIPS 0.025 in the research harness.

To reproduce an LPIPS number, score the cached clip against the same-seed
uncached clip:

```bash
python benchmarks/diffusion/quantization_quality.py --compare ballet_uncached.mp4 ballet_leapcache.mp4
```

The script needs the `lpips` package (`pip install lpips`, part of the `dev`
extra). It prints the mean and the worst frame. The default measured 0.099 mean
on the development prompts. The example script run with its own defaults
(seed 42, a model-specific negative prompt and guidance) on the scarf
development prompt gave 204.2 s uncached and 72.3 s cached, mean LPIPS 0.085
(worst frame 0.114), with peak reserved memory 44.4 GiB in both runs. The
example script's defaults differ from the harness protocol, so this run differs
from the scarf row under Qualification evidence.

## Notes

- Memory usage: peak allocated memory at the default, in GiB. Per-request
  cache state is 64 MiB and is released after every request.

  | Profile | Uncached | LeapCache |
  | --- | ---: | ---: |
  | 14B 480p | 39.96 to 40.00 | 40.05 |
  | 14B 720p | 43.21 to 43.25 | 43.36 |
  | 1.3B 480p | 15.17 to 15.21 | 15.26 |

- Key flags: `--cache-backend leap_cache` turns the cache on.
  `--cache-config '{"leap_threshold": 0.044}'` is the one knob at its default.
  The documented range is 0.011 to 0.088 (table below). Leave `--max-num-seqs`
  at its default of 1.
- Guidance interval: with LeapCache on, a request that sets no
  `guidance_interval` runs with `[600, 1000]`: no negative-prompt pass below
  timestep 600, the last 9 of 40 steps. See the
  [Videos API](../../docs/serving/videos_api.md#wan-text-to-video-guidance-interval).
- Negative prompt: the measurements used an empty negative prompt unless
  stated. The standard Wan negative prompt drifts less: LPIPS 0.066 / 0.074 at
  2.28x.
- Known limitations: Wan2.1 text-to-video only, one GPU, one video per batch
  (what fails and how is in the guide). Quantized checkpoints, CUDA-graph
  capture, throughput and concurrency were not measured. The two H100s of the
  test machine do not produce the same frames as each other, so compare cached
  and uncached clips from the same GPU.

## Supported features

| Feature | Status for this profile | Shared guide |
| --- | --- | --- |
| LeapCache | Measured at the default on the 14B at 480p (Qualification evidence) | [LeapCache](../../docs/user_guide/diffusion/cache_acceleration/leapcache.md) |
| OpenAI-compatible Videos API | Configuration-only with LeapCache: unit tests, not run on hardware | [Videos API](../../docs/serving/videos_api.md) |
| `guidance_interval` | On by default with LeapCache (`[600, 1000]`) | [Videos API](../../docs/serving/videos_api.md#wan-text-to-video-guidance-interval) |
| Cache-DiT, TeaCache | Not combinable with LeapCache, only one cache backend can be active | [Cache-DiT](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) |
| SP, CFG, tensor and pipeline parallelism, HSDP | Not supported with LeapCache, fails at startup | [Parallelism overview](../../docs/user_guide/diffusion/parallelism/overview.md) |
| Request-level batching (`max_num_seqs>1`) | Not supported with LeapCache | [Execution modes](../../docs/user_guide/diffusion/execution_modes.md) |
| CPU offload | Not needed on 80GB, not measured with LeapCache | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| Quantization | Not measured with LeapCache | [Quantization](../../docs/user_guide/quantization/overview.md) |

## Qualification evidence

Setup: Wan2.1-T2V-14B-Diffusers (revision `38ec498c`), BF16, 832x480, 81
frames, 40-step FlowUniPC, guidance scale 4, flow shift 5, empty negative
prompt unless stated, seed 179961516, Environment 1, one request at a time,
one untimed warm-up per process, then one request timed by wall clock on an
otherwise idle GPU. Prompt sets: 7 development prompts, the 4 tuned on in the
tables below, 8 held-out prompts, and 4 final prompts never tuned on. LPIPS is
the mean over all 81 frames against the uncached clip from the same seed on the
same GPU. "Mean / worst" is the mean over a prompt set and the worst single
prompt.

The knob was redefined after these runs. The setting labelled `0.064` then is
the default `0.044` today: the same tolerance, which the old definition counted
as 1.457 times today's value. At the default, the new definition makes the same
skip decisions on the 40-step, flow-shift-5 schedule. The 50-step, flow-shift-3
row below was measured at 0.040 in today's definition.

**The shipped backend against the research harness.** The tables below come
from a research harness with its own controller. The shipped backend at commit
`e84a14400`, before the knob was redefined, was run in Environment 2 on the
four development prompts, each on the same GPU as its uncached reference.
Every prompt gave the same frame hash as the harness, and wall times were
within 0.6 s of it:

| Prompt | Model runs (of 80) | Wall time | Uncached | LPIPS vs uncached |
| --- | ---: | ---: | ---: | ---: |
| hummingbird | 28 | 74.7 s | 206.3 s | 0.036 |
| pumpkin | 40 | 104.8 s | 205.4 s | 0.110 |
| scarf | 32 | 84.4 s | 205.8 s | 0.096 |
| train | 37 | 97.1 s | 205.4 s | 0.085 |

**The knob**, development prompts. The last column is the cached clip's VBench
aesthetic score minus the uncached clip's. A paired change within 0.03 is
noise: two uncached clips of one prompt from different seeds differ by 0.04 to
0.17.

| `leap_threshold` | Earlier label | Speed-up | LPIPS mean / worst | VBench aesthetic, paired |
| ---: | ---: | ---: | ---: | ---: |
| 0.011 | 0.016 | 1.20x | 0.017 / 0.036 | -0.002 |
| 0.022 | 0.032 | 1.68x | 0.041 / 0.086 | -0.002 |
| 0.044 (default) | 0.064 | 2.30x | 0.099 / 0.149 | -0.002 |
| 0.066 | 0.096 | 2.74x | 0.147 / 0.228 | -0.014 |
| 0.088 | 0.128 | 2.99x | 0.168 / 0.235 | -0.012 |

At the same speed, Cache-DiT at its default threshold gives 2.31x at LPIPS
0.194 / 0.282 and a paired aesthetic change of -0.020. The other comparison
methods are in the RFC. All ran on the four development prompts only.

**LeapCache at the default under other conditions:**

| Condition | Speed-up | LPIPS mean / worst |
| --- | ---: | ---: |
| Development, 4 prompts | 2.30x | 0.099 / 0.149 |
| Held-out, 8 prompts | 2.45x | 0.099 / 0.173 |
| Held-out, 8 prompts, second seed | 2.48x | 0.117 / 0.260 |
| Final, 4 prompts never tuned on | 2.30x | 0.164 / 0.229 |
| Standard Wan negative prompt | 2.28x | 0.066 / 0.074 |
| 30 steps, guidance 5 | 1.87x | 0.080 / 0.111 |
| 50 steps, guidance 3, flow shift 3 (0.040 today) | 3.05x | 0.062 / 0.074 |
| 40 steps, guidance 6 | 2.20x | 0.103 / 0.122 |
| 1280x720, 81 frames (uncached 784 s) | 2.67x | 0.102 / 0.141 |
| Wan2.1-T2V-1.3B, 480p (uncached 42 s) | 2.07x | 0.053 / 0.102 |

- Paired VBench over the 48 prompt-condition pairs of the table above:
  aesthetic -0.002, range -0.064 to +0.028. Subject, background and motion
  scores move by 0.002 or less. Dynamic degree agrees on all 48 pairs. 44 of 48
  pairs are within noise of the uncached clip.
- Cost: 32.5 model runs per clip at the default against 80. The solver-only
  work of the leap and the replay is below timing noise. The wall-clock
  speed-up is lower than the run count suggests because the timed request
  includes text encoding and video decoding, which the cache does not touch.
- Repeatability: 56 jobs re-run in fresh processes gave the same frame hash
  for every job. Wall times agreed within 0.65%.
- Not run: the online `/v1/videos` path with the cache on hardware, the 720p
  profile with the shipped backend, the shipped backend after the knob was
  redefined, the comparison methods on the held-out and final prompts, and a
  blinded human review.
