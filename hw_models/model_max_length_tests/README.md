# Max-context regression suite

Max-context-length regression tests for QAIC, covering four models chosen to span the
axes that change how a case has to be built and sized: vision-language vs. text-only,
dense vs. MoE, and quantized vs. unquantized.

| Model | Modalities | MoE | Quantization |
| --- | --- | --- | --- |
| `openai/gpt-oss-20b` | text | yes (32 experts, top-4) | mxfp4 |
| `Qwen/Qwen3-32B` | text | no | none |
| `Qwen/Qwen2.5-VL-32B-Instruct` | text, image, video, mixed | no | none |
| `Qwen/Qwen3-VL-32B-Instruct` | text, image, video, mixed | no | none |

These four are the checked-in defaults, not the limit of what the suite can run --
see "Model coverage" below for what a new model needs to plug in, and "Extending it"
for how.

The other VLM suites in `hw_models/` pin `max_model_len` to 14K–16K, so the top ~87% of
a 128K-context model's window is never exercised — and never through the vision path at
all (every existing `limit_mm_per_prompt` sets `video.count = 0` and routes video frames
through the *image* key). This suite fills the context window from ~8K up to each
model's true ceiling, at whichever input shapes and batch sizes its capabilities allow,
and checks four things per case:

1. the request completes and the engine survives;
2. the realised prompt token count matches the *analytically predicted* count (catches
   processor / patching / token-expansion regressions);
3. the output has not numerically collapsed into repetition (the usual failure mode for
   long context on accelerators, which does not raise);
4. at batch size > 1, the requests in the batch actually ran as distinct sequences —
   same length, different content, different output.

A model's capabilities are declared once, in the registry, and everything else follows:
which modalities it contributes cases for, whether MoE/quantization change its weight
footprint, and how large a single image or video clip may be before its vision
transformer's attention matrix blows the per-core address space (see "Vision attention
ceiling" below).

## Running it

`pytest .` runs every test in one process, which means one process tears an engine
down and builds the next. vLLM's worker processes do not reliably go away when that
happens, and the next engine then contends with the previous one for the devices. Use
the runner instead — it gives each test its own pytest process and reaps whatever that
process leaves behind:

```bash
./run_tests.sh                          # every collected test, one pytest process each
./run_tests.sh --group-by-engine        # one process per engine instead -- much faster
./run_tests.sh -k selftest              # only the device-free self-tests
./run_tests.sh --tier full --tp 8       # full sweep to the ceiling
./run_tests.sh --models Qwen/Qwen3-32B  # just one model
./run_tests.sh --batch 1,4              # add the batch-size axis
./run_tests.sh --modalities text,image_single  # restrict which input shapes run
./run_tests.sh --pl 8192 --gl 64        # run exactly this one (prompt_len, gen_len) point
./run_tests.sh --device-ids 48-63       # run concurrently across 16 devices, queuing as slots free
./run_tests.sh --list                   # show what would run, then exit
./run_tests.sh --help                   # all options
```

Output goes to `ciLogs_vlm_max_context_<timestamp>/`: one `NNN_<test>.log` per test,
plus `run.log` (everything), `summary.txt` (the pass/fail table) and
`vlm_max_context_summary.xlsx` (the suite's own perf tables, collected across
processes, as a "Summary" sheet plus a "Legend" sheet). Exit status is 0 only if every
test passed or skipped.

### Running one specific case end to end

To run exactly one `(model, TP, modality, prompt_len, gen_len)` point rather than a
sweep, combine `--models`/`--tp`/`--modalities` (each already narrows the matrix to
one value) with `--pl`/`--gl`, which must be given together:

```bash
./run_tests.sh --models Qwen/Qwen3-32B --tp 8 --modalities text --pl 8192 --gl 64
```

This forces `sweep.tier` to `"custom"` and `sweep.custom_targets` to that single pair
for the run, so the collected matrix is exactly one case -- no smoke/full auto-ceiling
point gets added alongside it. It still goes through the normal pipeline end to end
(one isolated pytest process, engine build, generation, all four case assertions,
device cleanup) and produces the same `summary.txt` / `run.log` / `vlm_max_context_summary.xlsx`
outputs as any other run, just for the one case.

`--group-by-engine` batches the cases that share an engine — every modality at one
`(model, TP, context, batch)` point, and the three boundary tests per model — so no
single process ever builds a second engine, which is the condition that causes the
crash. Much faster, slightly less isolated.

### Running across a device pool

By default every batch runs strictly one at a time. `--device-ids` hands the whole
run off to `scheduler.py` instead, which runs batches **concurrently** across a pool
of physical device IDs — a range (`48-63`), a comma list (`48,50,55`), or a mix
(`48,50-55,60`):

```bash
./run_tests.sh --device-ids 48-63 --group-by-engine   # 16 devices, batched by engine
```

Each batch's device count is its TP (parsed straight out of its id/key, e.g.
`-tp8-`); a batch with no TP in its id at all (`test_token_math_selftest.py`, which
is device-free) runs immediately as a 0-device job rather than being dropped. This
applies in either `--per-test` or `--group-by-engine` mode. A batch claims a
disjoint slice the moment enough devices are free and releases it — after a
`--settle`-second cooldown, so the runtime has fully torn the previous engine down
before the slice is reused — the instant it finishes; the next queued batch that
fits claims it. Sixteen devices and several TP=8 batches, for example, run two at a
time: two engines build concurrently on `48-55` and `56-63`, and whichever finishes
first is immediately handed the next queued TP=8 batch.

Before actually handing out a slice, `DevicePool` also cross-checks it against a
live `qaic-util -q` query (the same tool `run_tests.sh`'s own `device_status()`
already shells out to) — its own in-memory bookkeeping only knows what *this run*
has allocated, not whether a device is genuinely idle system-wide (a leaked engine
from an earlier run, or another user's job on a shared host, is invisible to it
otherwise). A device that in-memory bookkeeping thinks is free but `qaic-util`
reports busy is skipped in favor of whichever other devices in the pool are
genuinely free, with a short cooldown before it's tried again; the check fails open
(no blocking, same as not having it at all) if `qaic-util` can't be queried at all
(missing, no permission).

Device pinning is `QAIC_VISIBLE_DEVICES`, set by `conftest.py` from each job's
`--device-id` before collection — this suite always runs in eager mode
(`enforce_eager=True`), which is the mode that env var controls (the AOT-oriented
`additional_config={"device_group": ...}` path other suites in this repo use is not
the one that applies here). `run_metrics.append_summary_workbook()` takes a
cross-process lock on a sidecar `.lock` file so concurrent jobs appending to the same
run's `vlm_max_context_summary.xlsx` serialize instead of racing.

`--reap-strays` cannot be combined with `--device-ids`: it sweeps for escaped vLLM
processes by name and age, which cannot safely tell which concurrent job a wandering
process belongs to. `--stop-on-fail` still works — once any job fails, no new job is
dispatched, but jobs already running are left to finish rather than killed
mid-flight.

### How processes get cleaned up

Each pytest runs in its **own process group** (bash job control gives a background job
its own pgid), so after it exits the runner kills that group: TERM, then KILL after
`--grace` seconds. This can only ever hit that pytest and its descendants — it is
incapable of touching anything else you have running on the machine, which is why it is
always on.

A worker that calls `setsid` escapes its group and survives that. `--reap-strays`
sweeps for those, guarded three ways: the process must be owned by you, be younger than
the run, and have `python` as its *executable name* — not merely mention vllm in its
arguments, or a grep, an editor, or the runner's own command line would match. It is
opt-in because it matches by name at all.

Between tests the runner pauses `--settle` seconds (default 10) for the devices to be
released, and logs `qaic-util -q` readiness if that tool is available.

Sample commands for a single test, when you want pytest directly. The tier and batch
axis are `suite_config.json` fields now rather than env vars, so a bare `pytest`
invocation needs the file edited first (`run_tests.sh --tier`/`--batch` do this for
you, see above):

```bash
# Smoke tier -- every modality this model has, at 8K and 32K, TP=8, batch=1.
pytest test_ctx_sweep.py -s -k Qwen3-32B

# The model's ceiling itself, plus over-length rejection, for every selected model.
pytest test_ctx_boundary.py -s

# Full sweep to 128000 for text/images (video stays capped at the video ceiling).
# Set sweep.tier to "full" in suite_config.json, then:
pytest test_ctx_sweep.py -s

# Add batch size 4 to the sweep.
# Set selection.batch_sizes to [1, 4] in suite_config.json, then:
pytest test_ctx_sweep.py -s

# Token-math self-check -- no device, no engine.
pytest test_token_math_selftest.py -s
```

## Layout

| File | Responsibility |
| --- | --- |
| `run_tests.sh` | Runs the tests one at a time (or, with `--device-ids`, concurrently via `scheduler.py`) in isolated processes and reaps leaked vLLM workers. |
| `scheduler.py` | Concurrent dispatch across a device-ID pool for `run_tests.sh --device-ids`: a `DevicePool` hands each batch a disjoint slice sized to its TP, queuing the rest. |
| `suite_config.json` | **Every user-tunable knob.** Models, TP/batch/context selection, engine limits, vision/assertion/device thresholds. The only file most changes need. |
| `model_specs.py` | The `ModelSpec` shape, loaded from `suite_config.json`'s `models` section. Nothing to edit here to add a model. |
| `ctx_config.py` | Loads `suite_config.json` once and exposes it as named constants. No model or device state. |
| `model_geometry.py` | Layers / KV heads / head dim / `max_model_len` / patch geometry / MoE expert counts / quantization method, all derived from the HF config, plus the geometry + tokenizer caches and `resolve()`. |
| `kv_capacity.py` | KV bytes per token (times batch size), the KV budget, weight footprint, the skip-rather-than-OOM precheck, and `required_model_len()` (PL + GL + drift, quantised). |
| `filler_text.py` | Numbered filler prose and `text_of_exact_len()`, with a `variant` offset so batch members differ. |
| `visual_inputs.py` | Synthetic images and video clips with exactly predictable token counts, capped by the ViT attention ceiling. |
| `prompt_builder.py` | `BuiltInput`, the shared `assemble()`, the modality registry, and `build_batch()`. |
| `case_matrix.py` | Which `(model, TP, modality, ctx, batch)` cases exist. Collection never fails. |
| `engine_pool.py` | `LLM(...)` construction (dtype, multimodal kwargs, `max_num_seqs`) and the single-slot engine cache. |
| `ctx_assertions.py` | Token accounting (per-model drift tolerance) + the degeneracy check registry. |
| `run_metrics.py` | Per-case timings (batch-aware), the optional prefill reference, the summary table. |
| `case_runner.py` | `run_case()` — generate a batch once, assert the four properties. |
| `conftest.py` | Platform gate, RNG seed, `--device-id` (sets `QAIC_VISIBLE_DEVICES` for `scheduler.py`), session summary + engine teardown. |
| `test_ctx_sweep.py` | The sweep. |
| `test_ctx_boundary.py` | Landing on each model's ceiling, decoding to it, exceeding it. Always batch 1. |
| `test_token_math_selftest.py` | Device-free validation of the token math, capability declarations, and batch distinctness. |

Helper modules are imported flat (`from ctx_config import ...`), matching
`hw_models/vlm_accuracy/`: pytest inserts this directory into `sys.path` when it
collects a test file, so no `__init__.py` is needed.

## Model coverage

**Text-only: any HF causal LM, no new code.** `model_geometry.py` derives layer
count, KV heads, head dim, `max_model_len`, MoE expert count and quantization method
from `AutoConfig` with several fallback field names per property (see
`ModelGeometry.from_hf()`), and everything downstream of that -- KV sizing, engine
construction, filler text, degeneracy/token-accounting checks, the registry itself --
works off those derived numbers and the tokenizer, never off anything
architecture-specific. Adding a dense, MoE, or quantized text model is one
`suite_config.json` registry entry (model id, `tp_sizes`, an `approx_total_params_b`
hint for the capacity precheck) and nothing else.

**Vision: the Qwen2-VL/2.5-VL/3-VL family specifically, not "any VLM."** The image/
video modalities (`image_single`/`image_many`/`video`/`video_many`/`mixed`) and
everything in `visual_inputs.py`, plus the vision-derivation half of
`model_geometry.py` and the multimodal half of `engine_pool.py`/`prompt_builder.py`,
assume this family's specific processor behaviour:

- one placeholder ("pad") token per image/video that the processor expands 1:1 into
  exactly `rows * cols` visual tokens, computed from `patch_size` /
  `spatial_merge_size` / `temporal_patch_size` in the HF config
  (`ModelGeometry.pixels_per_visual_token`);
- an HF processor that accepts `max_pixels`/`min_pixels`/`fps` kwargs to control
  resolution and frame rate (`engine_pool._multimodal_kwargs`) -- these are literally
  Qwen's processor's kwarg names, not a generic vLLM interface;
- synthetic images/clips sized in *pixels* to hit an exact *token* count
  (`visual_inputs.make_image_of_tokens`/`make_video_of_tokens`), which only makes
  sense under a resolution-dependent token count in the first place.

A VLM that shares this scheme (any Qwen2-VL-generation model) is, like the text-only
case, a registry entry away. A structurally different VLM -- LLaVA/InternVL-style
AnyRes tiling (a fixed grid of tiles plus a thumbnail), a fixed-token-per-image CLIP
encoder (token count independent of resolution), Pixtral's variable-aspect-ratio
patches, or a cross-attention architecture where images never appear as inline
placeholder tokens at all -- is not covered by this math. Pointing this suite at one
needs new builders in `prompt_builder.py`/`visual_inputs.py` (and possibly new
`engine_pool.py` processor kwargs), not just a `suite_config.json` entry with
different placeholder strings.

## Input shapes

Six modalities are registered in `prompt_builder.py`; a text-only model contributes
just `text`, a model with `supports_images`/`supports_video` contributes whichever of
the rest it declares (see `modalities_for()`):

| Modality | What it builds |
| --- | --- |
| `text` | Filler text only, no visual items. |
| `image_single` | One image sized to consume about half the target token budget. |
| `image_many` | `sweep.image_many_count` (default 8) distinct images sharing the budget -- the image-*count* stress point. |
| `video` | One clip. |
| `video_many` | `sweep.video_many_count` (default 2) distinct clips sharing the budget -- the video-*count* stress point, mirroring `image_many`. |
| `mixed` | 2 images + 1 clip (degrades to images-only on a model without video) interleaved with text, to cross mRoPE section boundaries within one prompt. |

`image_many`/`video_many` are not a separate code path from `image_single`/`video` --
`prompt_builder.assemble()` takes `images`/`videos` as lists throughout, so "one item"
and "many items" are just different-length lists through the same function. Every
item within one case is independently clamped to the ViT attention ceiling (see
below) and given distinct synthetic pixel content per `variant`, so a batch of N
requests -- or N images/clips within one request -- can never be served from a single
cached encoding.

## MoE and quantization

`model_specs.ModelSpec` never declares expert counts, quantization, or weight size —
those are read from the HF config at runtime by `model_geometry.py`, the same way
`max_model_len` and patch geometry already were. This keeps the registry a place for
facts that can't be derived (prompt format, which TP sizes to test, approximate param
count) and means a model whose config drifts (a new MoE topology, a new
`quant_method`) is caught by `test_selftest_capabilities_match_config` and
`test_selftest_geometry_and_kv_sizing` instead of silently mis-sizing an engine.

- **MoE** (`geom.is_moe`, `geom.num_experts`, `geom.num_experts_per_tok`): read from
  whichever of `num_local_experts` / `num_experts` / `n_routed_experts` and
  `num_experts_per_tok` / `moe_topk` / `moe_k` the config has. Only printed by the
  self-test today — MoE does not currently change how a case is built, since routing is
  a runtime decision, not a prompt-shape one.
- **Quantization** (`geom.quantization`): the lowercased `quant_method` out of
  `cfg.quantization_config`, or `None`. Feeds `geom.weight_bytes_per_param(spec)` via
  `WEIGHT_BYTES_BY_QUANT` in `model_geometry.py` (1.0 B/param for 4-bit families like
  `mxfp4`/`awq`/`gptq`, 1.1 for 8-bit families like `fp8`/`compressed-tensors`, 2.0
  unquantized) — that number is what `kv_capacity.weights_gb()` uses for the device
  capacity precheck, so a quantized model's much smaller weight footprint is reflected
  instead of assuming full fp16 weights for everyone.
- **Capabilities** (`spec.supports_images`, `spec.supports_video`): declared per model
  and cross-checked against `geom.has_vision` (whether the HF config has a vision
  tower) by `geom.capability_mismatch(spec)`. Declaring images on a text-only model
  fails deep in the processor; forgetting to declare them on a VLM silently never
  exercises the vision path. Both directions are asserted device-free.

## Batch size

`BATCH_SIZES` (`selection.batch_sizes` in `suite_config.json`, default `[1]`) adds a
third axis to the sweep alongside context length and modality. Batch size changes
engine identity, not just the request: with prefix caching off, `batch_size`
concurrent full-length sequences need `batch_size` times the KV of one, so
`kv_capacity.kv_cache_bytes_for(..., num_seqs=
batch_size)` scales linearly and `engine_pool` sets `max_num_seqs=batch_size` on the
engine it builds. The pool's cache key is `(model, tp, engine_len, batch_size)` — two
sweep points that need the same engine length but different batch sizes get different
engines, deliberately, because they need different KV budgets.

Every request in a batch is the same predicted length (so the case still targets one
context point) but distinct content: `prompt_builder.build_batch()` passes each request
its own `variant` index, which offsets the filler-text rotation and the synthetic
image/video pixel content. Without that, N identical prompts in one batch would hash to
one entry in vLLM's multimodal encoder cache, and the batch would exercise the ViT once
instead of N times — `test_selftest_batch_inputs_are_distinct` and
`test_selftest_image_variants_differ`/`test_selftest_video_variants_differ` guard
against that regression directly.
`case_runner.run_case()` asserts the batch's outputs are not all identical, on top of
the per-request checks it already ran at batch 1.

`test_ctx_boundary.py` stays at batch 1 regardless of `selection.batch_sizes`: its
subject is the position limit of one sequence, and a ceiling-length engine times a
batch would need more KV than any single device has.

`QAIC_SDPA_DECODE=1` selects a decode path that only supports batch size 1; sweep cases
with `batch_size > 1` skip under that env var rather than failing (see
`engine_pool._skip_if_batch_unsupported`). Batch sizes above 1 need the paged-attention
decode path — do not set `QAIC_SDPA_DECODE=1` together with `selection.batch_sizes`
values above 1.

## Vision attention ceiling

A single image or video clip is one ViT sequence, and the vision blocks that attend
over all of it materialise a `[heads_on_rank, patches, patches]` fp16 score matrix on
device. Above the per-core virtual-address-space budget that operator cannot be mapped
at all — a hard device failure (`QAIC_ERROR_MMAP_FAILURE`), not a slow path — and it
gets worse at higher TP, since fewer heads per rank does *not* shrink the `patches²`
term. `visual_inputs.max_visual_tokens_per_item(geom, tp_size)` derives the token
ceiling per item from `vision.vit_va_budget_gb` (default 3.9 GiB) and
`geom.vision_num_heads`, and every image/video builder in `prompt_builder.py` clamps its
per-item budget to it before generating content (noted in the case's `detail` string
when it fires). Set `vision.item_autocap` to `false` to disable the clamp and reproduce the
failure directly. `test_selftest_visual_items_fit_the_vit_ceiling` asserts every built
item stays under the ceiling at every TP size the model is tested at.

## Engine sizing

Each sweep case builds its engine at the `max_model_len` **that case needs** — its
predicted prompt length plus its `gen_len`, plus the drift the accounting assertion
tolerates — not at the model's ceiling. Prompt length and generation length are
independent, explicit knobs (`sweep.smoke_targets`/`full_targets` in
`suite_config.json` are `{prompt_len, gen_len}` pairs; see "Configuration" below) --
`ctx_len` (prompt + generation) is a derived, reported total, not the input. On
Qwen2.5-VL-32B at TP=8, batch=1, gen_len=64:

| Context total (PL + GL) | Engine `max_model_len` | KV/worker | Est. device (TP=8 / TP=4) |
| --- | --- | --- | --- |
| 8192 | 9216 | 0.30 GiB | 13.1 GB / 21.7 GB |
| 16384 | 18432 | 0.59 GiB | 13.4 GB / 22.3 GB |
| 32768 | 36864 | 1.18 GiB | 14.0 GB / 23.5 GB |
| 65536 | 72704 | 2.33 GiB | 15.1 GB / 25.8 GB |
| 98304 | 108544 | 3.48 GiB | 16.3 GB / 28.1 GB |
| 128000 | 128000 | 4.10 GiB | 16.9 GB / 29.4 GB |

Ceiling-sized (the previous behaviour) was 4.10 GiB and 16.9 / 29.4 GB for *every*
case, including the 8K ones. A batch of `N` multiplies the KV/worker column by `N`
directly; the device-GB column follows `kv_capacity.estimated_device_gb()`, which now
also derives the weights term from the model's actual quantization instead of assuming
fp16.

The trade is engine builds. The `(prompt_len, gen_len)` pair and batch size are both
part of the engine identity, so the sweep builds one engine per point where a
ceiling-sized, batch-1-only engine served every case. Two things keep that from
becoming one build per case:

- `ENGINE_LEN_BLOCK` (`engine_limits.engine_len_block` in `suite_config.json`, default
  `1024`) rounds engine lengths up, so all modalities at one `(prompt_len, gen_len,
  batch)` point land on the same length and share an engine.
  `test_selftest_engine_sizing` asserts exactly that, and tells you to raise the block
  if it ever stops holding.
- `case_matrix.build_sweep_cases()` emits cases ordered model → TP → `(prompt_len,
  gen_len)` → batch → modality, with modality innermost since it is the one dimension
  that never changes engine identity, so consecutive cases hit the cached engine.

To trade memory back for fewer builds, raise the block — `engine_limits.engine_len_block:
32768` buckets the 8K and 16K points onto one 32768-token engine.

The three boundary tests deliberately opt out and ask for `geom.max_model_len`: each
model's declared ceiling is their subject. They share one engine between them, at
batch 1.

`assert_token_accounting()` and the `prompt + decode` bound in `run_case()` are checked
against the **engine's** length rather than the model ceiling, which is both the
operative limit and a tighter assertion. The drift tolerance for video is per-model
(`spec.video_ratio_slack`, falling back to `assertions.video_token_ratio_slack`) — a
model whose video processor applies a more aggressive pixel budget can declare a wider
slack rather than the whole suite paying for it.

## Extending it

**Add a model** — one entry under `models` in `suite_config.json`. Only prompt-format
and capability facts belong there; every numeric property is derived from the HF
config at runtime.

```jsonc
"Qwen/Qwen3-VL-32B-Instruct": {
  "tp_sizes": [8, 4],
  "approx_total_params_b": 33.5,
  "supports_images": true,
  "supports_video": true,
  "replicated_vision_gb": 1.6,      // per-worker vision-tower weight replication
  "video_ratio_slack": 0.20         // wider drift tolerance than the suite default
}
```

A text-only or dense model just omits the fields it doesn't need — `supports_images`,
`supports_video`, `replicated_vision_gb` and `video_ratio_slack` all default to
"not a VLM" (see `ModelSpec` in `model_specs.py` for the full set of defaults).
`dtype` (default `"float16"`) and `engine_kwargs` (extra `LLM(...)` kwargs, default
`{}`) are there for a model that needs something the others don't. Add the new model
id to `default_models` to include it in every run by default, or leave it out and
select it explicitly via `selection.models`.

**Add an input shape** — one decorated function in `prompt_builder.py`. `MODALITIES`,
the sweep matrix and the self-test all pick it up automatically. `images`/`videos` are
lists throughout `assemble()`, so a fixed-shape builder like this just passes
fixed-length lists -- `image_many`/`video_many` are the same function with a
config-driven count instead.

```python
@modality("two_videos", requires_video=True)
def _build_two_videos(spec, geom, tokenizer, target_tokens, *, tp_size, variant):
    clip_a, tokens_a, frames_a = make_video_of_tokens(geom, budget_a, variant=variant)
    clip_b, tokens_b, frames_b = make_video_of_tokens(geom, budget_b, variant=variant + 1)
    return assemble(
        spec, tokenizer, target_tokens, [], [],
        [clip_a, clip_b], [tokens_a, tokens_b], [frames_a, frames_b],
        "two videos", variant=variant,
    )
```

**Add a degeneracy signature** — one decorated function in `ctx_assertions.py`,
returning a failure message or `None`.

```python
@degeneracy_check
def _check_something(text, token_ids, stats):
    return "why this is degenerate" if bad else None
```

**Add a context point** — add a `{"prompt_len": ..., "gen_len": ...}` entry to
`sweep.full_targets` (or `smoke_targets`) in `suite_config.json`.

## Configuration

`suite_config.json` (next to this file, or wherever `MAX_CTX_CONFIG_FILE` points) is
the one file you edit to change anything the suite runs with. Every other `.py` file
loads named constants from `ctx_config.py`, which loads the JSON once at import; a
missing or malformed config raises immediately, naming the bad key, rather than
falling back to a Python default. `run_tests.sh`'s `--tier`/`--tp`/`--models`/
`--batch` flags patch a copy of the file for that invocation only (see "Running it"
above) — the checked-in file is never touched by them.

| Section | Field | Default | Effect |
| --- | --- | --- | --- |
| (top-level) | `default_models` | the four default models | Used when `selection.models` is `null`. |
| (top-level) | `models` | the registry | One entry per model; see "Extending it". |
| `selection` | `models` | `null` | `null` → use `default_models`. Else a list of model ids from `models`. |
| `selection` | `tp_sizes` | `[8]` | TP sizes, intersected with each model's own `tp_sizes`. |
| `selection` | `batch_sizes` | `[1]` | Each distinct value is a distinct engine (KV scales with it); needs the paged-attention decode path, i.e. do not also set `QAIC_SDPA_DECODE=1`. |
| `selection` | `modalities` | `null` | `null` → every modality the selected model(s) support (see "Input shapes"). Else a list of modality names; a model that doesn't support a listed one simply contributes no cases for it. |
| `sweep` | `tier` | `"smoke"` | `smoke` → 8K, 32K. `full` → 8K/16K/32K/64K/96K/128000 plus each model's ceiling. `custom` → exactly `custom_targets`, nothing added (set by `run_tests.sh --pl`/`--gl`, not normally by hand). |
| `sweep` | `smoke_targets` / `full_targets` / `custom_targets` | see file | `{prompt_len, gen_len}` pairs per tier -- both independent and explicit; `ctx_len` (their sum) is derived, never the input. In `full` tier, each modality also always gets one extra pair at its own ceiling (prompt_len = model max, or `video_ctx_max`, minus the tier's largest configured `gen_len`); `custom` never gets that extra pair. `custom_targets` defaults to `[]` and is normally written by `--pl`/`--gl`, not edited directly. |
| `sweep` | `video_ctx_max` | `32768` | Separate, lower ceiling on `prompt_len` alone for `video`, `video_many` and `mixed` -- generation length doesn't add visual content, so it isn't counted against this cap. Video at the full ceiling needs ~250 frames / ~600 MB of raw uint8. |
| `sweep` | `image_many_count` | `8` | Images in the `image_many` case. |
| `sweep` | `video_many_count` | `2` | Clips in the `video_many` case. Lower default than `image_many_count` -- a clip costs far more per item than an image. |
| `engine_limits` | `max_single_item_tokens` | `16384` | Cap on any single image/video before the ViT ceiling clamp is even considered. Chunked prefill cannot split one multimodal item. |
| `engine_limits` | `engine_len_block` | `1024` | Engine lengths round up to this, so modalities at one `(context, batch)` point share an engine. Raise it to bucket context points together. |
| `engine_limits` | `engine_len_margin_ratio` | `null` | `null` → derive from `assertions.video_token_ratio_slack`. Floored at that value either way — an engine must tolerate at least as much drift as the accounting assertion. |
| `engine_defaults` | `engine_kwargs` | `{}` | Extra `LLM(...)` kwargs applied to every model, layered under a model's own `engine_kwargs` (which wins on key collision). May not set kwargs the suite derives from the case under test (`model`, `dtype`, `seed`, `tensor_parallel_size`, `max_model_len`, `kv_cache_memory_bytes`, `trust_remote_code`, `max_num_seqs`) — `engine_pool.py` raises, naming the offending key, rather than letting it fail with Python's "multiple values for keyword argument". |
| `vision` | `vit_va_budget_gb` | `3.9` | Per-core virtual-address-space budget the ViT attention score matrix must fit, used to derive the per-item token ceiling. |
| `vision` | `item_autocap` | `true` | Clamp image/video items to the ViT attention ceiling. `false` reproduces the device-side failure directly. |
| `assertions` | `token_slack` | `8` | Exact token-accounting slack (text and images). |
| `assertions` | `video_token_ratio_slack` | `0.10` | Default ratio slack for video; a model may override it via its `video_ratio_slack` field. |
| `assertions` | `min_unique_ratio` | `0.15` | Minimum unique-token fraction in the output. |
| `assertions` | `max_consecutive_repeat` | `32` | Longest run of one repeated token id allowed. |
| `assertions` | `max_cycle_fraction` | `0.6` | Largest fraction of the tail allowed to be a short repeating cycle. |
| `assertions` | `perf_tolerance` | `0.08` | Allowed prefill regression against `REF_PERF` in `run_metrics.py`. |
| `sampling` | `temperature` | `0.0` | Decoding temperature. `0.0` reproduces the suite's original greedy-only behavior. |
| `sampling` | `top_p` | `1.0` | |
| `sampling` | `top_k` | `-1` | vLLM's "disabled" sentinel. |
| `sampling` | `repetition_penalty` | `1.0` | |
| `sampling` | `ignore_eos` | `false` | |
| `device` | `device_mem_gb` | `32` | Per-device budget for the capacity precheck. |
| `device` | `activation_headroom_gb` | `3.0` | Allowance for activations / workspace / fragmentation. |
| `device` | `kv_min_gb` | `0` | Optional floor under the computed KV budget, in GiB. Set to `2` to reproduce what the sibling `hw_models` suites hardcode. |
| `reporting` | `summary_file` | `ciLogs_vlm_max_context_summary.xlsx` | Perf summary workbook appended here (`Summary` + `Legend` sheets). |

Overriding `sampling.*` away from its greedy defaults is an explicit opt-in: the
degeneracy checks (`ctx_assertions.py`) and the token-accounting assertion were
written and tuned against deterministic, greedy output, not against sampled
generation. A non-zero temperature may make a case fail for reasons that have
nothing to do with long-context correctness.

Two things stay outside `suite_config.json` on purpose:

- **`MAX_CTX_CONFIG_FILE`** — a file *locator*, not a knob: points `ctx_config.py` at
  an alternate JSON file. This is how `run_tests.sh` applies its flags without
  touching the checked-in config.
- **`QAIC_SDPA_DECODE`** — reflects which decode kernel is actually
  compiled/selected on the box, not a suite parameter, so `engine_pool.py` reads it
  directly from the environment.

`device.device_mem_gb` / `activation_headroom_gb` / `kv_min_gb` were previously the
literal, unprefixed `QAIC_DEVICE_MEM_GB` / `QAIC_ACTIVATION_HEADROOM_GB` /
`QAIC_KV_MIN_GB` env vars shared with sibling `hw_models` suites. They now live only
in this suite's config — a value another suite's script exports for those names no
longer reaches this suite.

Tiering (and every other selection knob) is read from `suite_config.json` at import
time rather than driven by pytest markers, on purpose: `vllm-qaic-os`'s
`pyproject.toml` registers only `qaic_test_config` / `qaic_aot_mode` /
`qaic_disagg_installed`, while `postflight` is declared in `pytorch/pytest.ini`. A
config file read at import works identically from either rootdir; markers would not.

## Known rough edges

- `test_boundary_exact_max_model_len` requires landing within `TOKEN_SLACK` (8 tokens)
  of the ceiling, but `text_of_exact_len()` may legitimately undershoot by more than
  that when BPE re-encoding oscillates. A spurious failure here is a tokenizer artefact,
  not a long-context bug.
- `run_metrics.record_metrics()` reads `first_token_latency` / `scheduled_ts` /
  `first_token_ts` / `last_token_ts` off `RequestOutput.metrics`. On a vLLM build that
  exposes different field names you get an `AttributeError`, not the intended wall-clock
  fallback (which only triggers when `metrics is None`).
- `conftest.py` gates the whole directory on the QAIC platform at import time, so
  `test_token_math_selftest.py` — which needs no device — is not collectable off a QAIC
  host. Move the assert into an autouse fixture in the two device test files to free it.
- `REF_PERF` in `run_metrics.py` is empty and now keyed by `(model, tp, ctx, modality,
  batch_size)`. Missing keys print a notice rather than failing, so populate it from a
  first clean run per model to enable the prefill regression check.
- MoE expert counts are derived and printed but do not currently change case
  construction. If a future case needs to target specific experts or vary
  `num_experts_per_tok` at runtime, that is new work, not something `geom.is_moe`
  already drives.
