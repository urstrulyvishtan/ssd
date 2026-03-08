<h1 align="center">Speculative Speculative Decoding</h1>

<h3 align="center">
  <a href="https://arxiv.org/pdf/2603.03251">Paper</a>
</h3>

<p align="center">
  <img width="800"
       src="assets/ssd fig1 readme.png" />
</p>

> *"In all fictions, each time a man meets diverse alternatives, he chooses one and eliminates the others; in the work of the almost unfathomable Ts'ui Pên, he chooses — simultaneously — all of them."*
>
> — Jorge Luis Borges, "The Garden of Forking Paths" (1941)

**SSD is a new LLM inference algorithm. It is exact, and it is extremely fast.**

SSD is a new type of speculative decoding (SD). In normal SD, a small and fast model guesses the next few tokens that a larger slower model may generate, and the large model then verifies them in one forward pass: drafting and verification happen one after the other on the same hardware.

In SSD, they happen in parallel, on distinct hardware. The small model anticipates likely verification outcomes in advance, and speculates for all of them at once. If it guessed correctly, the speculation can be returned immediately so drafting overhead is eliminated entirely.

---

### Limitations (upstream) & what this fork changes

**Limitations in the original implementation:**

- **NCCL:** Target↔draft used many round-trips per step (e.g. 7+ sends with EAGLE), adding ~50–90 µs latency per step.
- **Fan-out:** `fan_out_list` was uniform; no geometric allocation from the paper (Theorem 12).
- **Sampler C:** Saguaro `sampler_x` was a single scalar; no per-position adaptive C from entropy.
- **Batch fallback:** On cache miss with `jit_speculate=True`, the whole batch falls back to JIT (no per-element hybrid).
- **Masks:** Step‑0 mask build does heavy CPU work (numpy, packbits) every round; no reuse for (K, F, max_context_len).
- **Tree cache:** Cache is reset each round; no reuse of glue-decode KV prefix when the accepted prefix matches.
- **Verify:** Full-vocab softmax allocated for the whole batch even when only some elements need temperature.
- **FlashInfer:** `plan()` + CPU sync inside the CUDA graph replay loop, once per tree step.

**Improvements in this fork:**

| Area | Change |
|------|--------|
| **NCCL** | Fused request path: one header + one int64 payload (+ one float when EAGLE). 7 sends → 2–3. |
| **Fan-out** | Geometric `fan_out_list` from empirical α (metrics). Config `fan_out_alpha` / `--fan-out-alpha`; metrics suggest a list for the next run. |
| **Sampler C** | Per-position adaptive C from logits entropy in the draft sampler (high entropy → lower C). |
| **Setup** | CUDA-only deps optional on Linux; `.env.example`, `check_setup.py`; README setup clarified. |

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for details. The rest (hybrid per-element fallback, mask precomputation, KV prefix reuse, verify sparse probs, plan sync) are left as future work.

---

This custom inference engine supports:
- A reference implementation of the SSD algorithm
- Optimized SD and autoregressive baselines
- Qwen3 + Llama3 model families
- Tensor Parallelism
- PagedAttention, CUDAgraphs, torch compilation, prefix caching

<div align="center">
  <table><tr><td width="800">
    <video src="https://github.com/user-attachments/assets/588eaa70-d6e5-4522-9e94-e54fc6074aba" />
  </td></tr></table>
</div>

## Setup

**Requirements:** Python 3.11+, and for running inference: **Linux with CUDA ≥ 12.8** (tested on H100s). On macOS, `uv sync` installs core dependencies only (CUDA-only packages are skipped) so you can develop or run scripts that don’t need the GPU stack.

### 1. Install uv (if needed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# if `uv` is not found in this shell:
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/tanishqkumar/ssd && cd ssd
uv sync
# Optional: for download scripts
uv sync --extra scripts
source .venv/bin/activate
```

### 3. Environment variables

Copy the example env file and set your paths:

```bash
cp .env.example .env
# Edit .env: set SSD_HF_CACHE and SSD_DATASET_DIR to your paths
source .env
```

- **`SSD_HF_CACHE`** — HuggingFace **hub** directory (contains `models--org--name/` subdirectories), e.g. `/data/huggingface/hub` (not the parent directory).
- **`SSD_DATASET_DIR`** — Directory with dataset subdirs (`humaneval/`, `alpaca/`, etc.).
- **`SSD_CUDA_ARCH`** — Optional; CUDA arch for kernels: `9.0` = H100/H200, `8.0` = A100, `8.9` = L40/4090.

### 4. Verify setup

```bash
python scripts/check_setup.py           # check env vars
python scripts/check_setup.py --import  # also test `from ssd import LLM` (requires Linux + CUDA)
```

### 5. Download models and datasets

If you already have the models via `huggingface-cli`, you can skip to datasets. Download scripts need the `scripts` extra: `uv sync --extra scripts`.

```bash
# models (uses SSD_HF_CACHE)
python scripts/download_from_hf.py llama

# datasets (writes to $HF_DATASETS_CACHE/processed_datasets)
export HF_DATASETS_CACHE=/path/to  # parent of SSD_DATASET_DIR
python scripts/get_data_from_hf.py --num-samples 10000
```

## Usage

All commands below run from inside the `bench/` directory. Large models (Llama-3 70B, Qwen-3 32B) take a few minutes for load/warmup/compile before generation starts. Always use `python -O` to disable debug overhead.

### Benchmarks

Use `--all` for full eval across four datasets. Since different data distributions are predictable to varying degrees, the speed of SD/SSD depends a lot on the dataset. Averaging over many prompts from many types of datasets 
gives an overall picture. `--numseqs` is per-dataset, so `--numseqs 128 --all` runs 128 × 4 = 512 prompts total.

```bash
cd bench

# AR — Llama 70B, 4 GPUs
python -O bench.py --llama --size 70 --gpus 4 --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# Sync spec decode — 70B target + 1B draft, 4 GPUs, k=6
python -O bench.py --llama --size 70 --gpus 4 --spec --k 6 --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# Async spec decode (SSD) — 70B target (4 GPUs) + 1B draft (1 GPU), k=7, f=3
python -O bench.py --llama --size 70 --gpus 5 --spec --async --k 7 --f 3 --b 1 --temp 0 --numseqs 128 --output_len 512 --all
```

Use `--qwen --size 32` for Qwen models. See `bench/bench.py` for full args. For SGLang/vLLM baselines, see `bench/README.md`.

### Chat

Interactive streaming chat with Llama-3.1 70B only. Supports AR, sync SD, and async SD (SSD). Pass `--metrics` to print token count, speed, and TTFT after each response.

```bash
cd bench

# AR — 4 GPUs
python -O chat.py --ssd --gpus 4

# Sync spec decode — 4 GPUs, k=6
python -O chat.py --ssd --spec --k 6 --gpus 4

# Async spec decode (SSD) — 5 GPUs, k=7, f=3
python -O chat.py --ssd --spec --async --k 7 --f 3 --gpus 5 --metrics
```

SGLang and vLLM chat backends are also supported (launches their servers automatically) for comparison:

```bash
python -O chat.py --sglang        # spec decode
python -O chat.py --sglang --ar   # autoregressive
python -O chat.py --vllm          # spec decode
```

### Roadmap

Features that will be supported in the near future: 
- Draft data parallel (increase speculation cache size) on up to 4 devices to avoid getting compute bound
- OpenAI-compatible inference over HTTP
- New models and MoE support: GPT-OSS and Kimi-K2.5.

Contributions welcome! 

## Citation

Speculative Speculative Decoding will appear at ICLR 2026.

```bibtex
@misc{kumar2026speculativespeculativedecoding,
      title={Speculative Speculative Decoding},
      author={Tanishq Kumar and Tri Dao and Avner May},
      year={2026},
      eprint={2603.03251},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.03251},
}
```

## History

[![Star History Chart](https://api.star-history.com/svg?repos=tanishqkumar/ssd&type=Date)](https://star-history.com/#tanishqkumar/ssd&Date)
