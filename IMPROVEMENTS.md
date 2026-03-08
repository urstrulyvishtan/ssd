# SSD improvements (fork)

This document summarizes improvements made over the upstream [tanishqkumar/ssd](https://github.com/tanishqkumar/ssd) implementation.

## 1. Geometric fan-out from empirical data

**Problem:** `fan_out_list` was static and uniform (`[F] * (K+1)`). The paper’s Theorem 12 gives an optimal geometric fan-out that allocates more guesses where acceptance is higher.

**Change:**
- **`ssd/utils/fan_out.py`**: Added `compute_geometric_fan_out_list(K, F_0, alpha, r)` and `estimate_alpha_from_metrics(accepted_suffix_lens_on_hit, K)`, plus `suggest_geometric_fan_out_list(...)` that fits α from `METRICS["accepted_suffix_lens_on_hit"]` and returns a drop-in list (sum preserved).
- **`ssd/config.py`**: Optional `fan_out_alpha` and `fan_out_r`. When `fan_out_list` is `None` and `fan_out_alpha` is set, `fan_out_list` is computed geometrically at init.
- **`ssd/engine/llm_engine.py`**: When printing metrics, if cache-hit data exists, prints a suggested geometric `fan_out_list` for the next run.
- **`bench/bench.py`**: `--fan-out-alpha` and `--fan-out-r` to use geometric fan-out (e.g. after a warmup run).

**Usage:** Run once, read the printed `[metrics] Suggested geometric fan_out_list (next run): [...]`, then pass `--fan-out-alpha <value>` (or set `fan_out_alpha` in config) on the next run; or pass the list explicitly via `--flh`.

---

## 2. Fused NCCL communication (target ↔ draft)

**Problem:** Per speculation step, the target sent 7 separate NCCL ops (cmd, meta, fused int64, then 4 EAGLE sends), and the draft did 2 recvs for the response. Each op adds ~5–10 µs latency.

**Change:**
- **Target (`ssd/engine/speculator_async.py`):**
  - Single **header** send: `[cmd, B, K, F, is_eagle]` (5 int64).
  - Single **int64 payload**: `cache_keys`, `num_tokens`, `block_tables`, `temps`, and when EAGLE: `extend_counts`, `extend_token_ids` (one `send_int64`).
  - When EAGLE: single **float** send: `concat(recovery_activations, extend_eagle_acts)`.
- **Draft (`ssd/engine/draft_runner.py`):**
  - Recv header (5 int64) → parse B, K, F, is_eagle.
  - Recv one fused int64 payload and unpack (including extend_counts / extend_token_ids when is_eagle).
  - When EAGLE: recv one fused float buffer and slice into `target_recovery_activations` and `extend_eagle_acts`.

**Result:** Request path goes from **7 sends → 2 sends** (header + fused int64; +1 float send when EAGLE). Saves on the order of tens of µs per step at high throughput.

---

## 3. Per-position adaptive sampler_x (Saguaro C)

**Problem:** `sampler_x` was a single scalar for all tree positions. The optimal C is position-dependent (entropy: high entropy → more aggressive / lower C).

**Change:**
- **`ssd/utils/async_helpers/async_spec_helpers.py`**:
  - `apply_sampler_x_rescaling(probs, sampler_x, F)` now accepts `sampler_x` as a **float** (unchanged) or a **1D tensor** of per-position C (same length as `probs.shape[0]`).
  - Added `entropy_to_sampler_x(logits)` returning per-position C: `C = 1 - 0.3 * (H / log(V))` clamped to `[0.1, 1]`.
- **`ssd/layers/sampler.py`**: When `sampler_x` is set and `is_tree`, C is computed from logits via `entropy_to_sampler_x` and passed to `apply_sampler_x_rescaling` (per-position adaptive). Verification path still uses the scalar `sampler_x` for consistency with the target’s ratio test.

---

## 4. Setup and platform

- **`pyproject.toml`**: CUDA-only deps (`triton`, `flashinfer-python`, `sgl-kernel`, `nvidia-cutlass-dsl`) are installed only on Linux (`sys_platform == "linux"`), so `uv sync` works on macOS for development.
- **`.env.example`** and **`scripts/check_setup.py`**: Document env vars and a quick setup check.
- **README**: Setup steps clarified (env vars, verify script, platform notes).

---

## Not yet implemented (from analysis)

- **Hybrid per-element fallback (#5):** Serve cache hits from cache and run JIT only for misses (instead of all-or-nothing batch fallback).
- **Mask precomputation (#6):** Cache assembled masks for (K, F, max_context_len) and avoid rebuilding + packbits every step 0.
- **KV cache prefix reuse (#2):** Reuse glue-decode KV prefix across rounds when the accepted prefix matches (harder; touches block manager, scheduler, draft runner).
- **Verify.py sparse probs (#8):** Allocate / compute probs only for batch elements and positions that need temperature (e.g. `accept_until`-scoped).
- **FlashInfer plan() sync (#7):** Batch plan calls at step 0 and avoid per-step CPU sync inside the replay loop.

These are left as future work; the codebase is structured so they can be added incrementally.
