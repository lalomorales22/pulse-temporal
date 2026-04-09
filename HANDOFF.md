# PULSE Temporal Embeddings -- Handoff Document

Created: April 9, 2026
Session: Initial build of pulse-temporal v0.1

---

## What Was Built

### Core Package (`pulse_temporal/`)

A pip-installable Python package that encodes moments as 128-dimensional vectors
capturing experiential time -- not just timestamps. Seven signal layers fuse clock
time with circadian phase, urgency, behavioral context, and temporal surprise.

**Installed locally with:** `pip install -e ".[dev]" --break-system-packages`

**All 53 tests pass:** `python -m pytest tests/ -v`

#### The 7 Layers

| Layer | Dims | File | What it does |
|---|---|---|---|
| log_time | 8D | `layers/log_time.py` | Weber's Law -- logarithmic compression of time. Felt distance between 1 and 2 minutes >> 101 and 102 minutes |
| oscillators | 32D | `layers/oscillators.py` | 16 fixed-frequency sin/cos pairs spanning 1 hour to 4 years. Time2Vec-inspired |
| circadian | 8D | `layers/circadian.py` | 24h + 90min ultradian cycles, plus cognitive/energy curves from circadian research |
| calendar | 24D | `layers/calendar.py` | Day-of-week, month, season, holidays (US), business hours, weekend, all cyclically encoded |
| urgency | 8D | `layers/urgency.py` | Hyperbolic discounting at 3 rates (slow/medium/panic). Supports multiple deadlines |
| temporal_state | 32D | `layers/temporal_state.py` | Multi-scale exponential decay event history. Events, sleep, session detection |
| prediction_error | 16D | `layers/prediction_error.py` | Temporal surprise -- how much does this moment deviate from expected patterns |

#### Layer Weighting

Context-dependent layers are weighted higher so they shift the embedding when context is provided.
Without this, the always-on oscillator/calendar features dominate. Defined in `encoder.py`:

```python
_DEFAULT_LAYER_WEIGHTS = {
    "log_time": 1.0,
    "oscillators": 0.5,       # scaled down -- dense 32D would dominate otherwise
    "circadian": 1.5,
    "calendar": 0.6,
    "urgency": 4.0,           # critical experiential signal
    "temporal_state": 2.0,
    "prediction_error": 3.0,
}
```

#### Key Result

The encoder produces embeddings where experiential similarity > timestamp similarity:

```
Mon 2pm crunch  <->  Wed 10am crunch:  0.978  (same experience, different day/time)
Mon 2pm crunch  <->  Sat 2pm chill:    0.743  (same timestamp structure, different experience)
Mon 2pm crunch  <->  Mon 2am insomnia: 0.303  (same day, totally different experience)
```

### Daemon (`pulse_temporal/daemon/`)

- `state_db.py` -- SQLite store for events, deadlines, behavioral patterns
- `pulse_daemon.py` -- Background heartbeat loop, event logging, deadline tracking, temporal context API

Works. Tested. The daemon's `get_temporal_context()` returns a full context package for LLM injection.

### Training Pipeline (`pulse_temporal/training/`)

- `data_generator.py` -- Generates synthetic temporal reasoning training examples
  - 14 scenario types (crunch, vacation, insomnia, post-lunch dip, etc.)
  - 15 question types (task suitability, urgency, break advice, etc.)
  - Each example has: system prompt with PULSE temporal context + user question + temporally-aware response
  - Run: `python -m pulse_temporal.training.data_generator`
  - Outputs: `data/temporal_train.jsonl` (2000 examples) and `data/temporal_eval.jsonl` (200 examples)

- `temporal_tuner.py` -- LoRA fine-tuning pipeline
  - Supports Qwen 2.5 (0.5B, 1.5B), Gemma 3 1B, or any causal LM
  - Has a manual weight-loading fallback for Python 3.14/PyTorch 2.10 segfault workaround
  - Run: `python -m pulse_temporal.training.temporal_tuner --model Qwen/Qwen2.5-1.5B-Instruct --data data/temporal_train.jsonl`

### Convenience Script

`train.sh` -- One-command training. Sets env vars for MPS compatibility:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

### Colab Notebook

`notebooks/train_on_colab.ipynb` -- Self-contained notebook that:
1. Installs deps
2. Generates training data inline (no pulse_temporal import needed)
3. Fine-tunes Qwen 2.5 1.5B with LoRA on free Colab T4 GPU
4. Uploads trained model to HF Hub

Also uploaded to: https://huggingface.co/lalopenguin/pulse-base-v1/blob/main/train_on_colab.ipynb

---

## Hugging Face -- What Was Created

### 1. `lalopenguin/pulse-base-v1` (Model)
**URL:** https://huggingface.co/lalopenguin/pulse-base-v1

Formula-based encoder model card. Contains:
- `README.md` -- Model card with architecture, usage, comparison table
- `config.json` -- Layer dimensions, weights, oscillator frequencies
- `train_on_colab.ipynb` -- Colab training notebook

This is the PULSE encoder definition, not a neural network with trained weights.

### 2. `lalopenguin/pulse-qwen-1.5b` (Model)
**URL:** https://huggingface.co/lalopenguin/pulse-qwen-1.5b

Trained LoRA adapter. Contains:
- `adapter_config.json` -- LoRA config (r=16, alpha=32)
- `adapter_model.safetensors` -- Trained LoRA weights
- `tokenizer.json` + `tokenizer_config.json`
- `pulse_config.json` -- PULSE metadata (base model, training info)
- `training_args.bin`
- 3 checkpoints: `checkpoint-125/`, `checkpoint-250/`, `checkpoint-375/`

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`

**Status:** Files are present. Training completed (3 checkpoints = 3 epochs).
The model does NOT have a proper README/model card yet -- needs one.

**To use this model:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "lalopenguin/pulse-qwen-1.5b")
tokenizer = AutoTokenizer.from_pretrained("lalopenguin/pulse-qwen-1.5b")
```

**To test locally:**
```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python examples/inference_trained.py \
  --model lalopenguin/pulse-qwen-1.5b \
  --base Qwen/Qwen2.5-1.5B-Instruct
```

### 3. `lalopenguin/pulse-temporal-demo` (Space)
**URL:** https://huggingface.co/spaces/lalopenguin/pulse-temporal-demo

Interactive Gradio demo. Dark theme (black/grey/orange, no purple/pink).
Three tabs: compare two moments, encode a single moment, 6-scenario similarity matrix.

**Status:** RUNNING. Source is also saved locally at `examples/gradio_demo.py`.

### 4. `lalopenguin/pulse-temporal-train` (Space)
**URL:** https://huggingface.co/spaces/lalopenguin/pulse-temporal-train

GPU training Space. Has a Gradio UI to configure and launch training.
Uses `@spaces.GPU(duration=1800)` decorator for ZeroGPU.

**Status:** Uploaded but ZeroGPU requires HF Pro ($9/mo). On CPU-only it would be very slow.
The Colab notebook is the free-GPU alternative.

---

## Known Issues and Workarounds

### Python 3.14 + PyTorch 2.10 Segfault
`AutoModelForCausalLM.from_pretrained()` segfaults when loading Qwen models on Python 3.14.
**Workaround:** Set these env vars:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```
The `temporal_tuner.py` also has a fallback that loads weights manually via safetensors
if `from_pretrained` fails.

### MPS Out of Memory
Local training of the 0.5B model OOM'd at step 154/750 on MPS (9GB VRAM limit).
Training was going well before the crash -- loss dropped from 3.73 to 0.28, accuracy hit 93%.
The OOM happened when the eval step kicked in (eval + train both in memory).
**Fix:** Use Colab (16GB T4) or reduce batch size + disable eval during training.

### HuggingFace 503 Outage
During this session, HF had intermittent 503 errors on upload/download endpoints.
Uploads and model downloads failed repeatedly for ~20 minutes then recovered.
The Space uploads succeeded on retry. Model downloads for Qwen 1.5B Instruct never
completed locally (but the Colab/Space training used HF-hosted infra which worked).

### TRL API Changes
`SFTConfig` in trl 1.0.0 uses `max_length` not `max_seq_length`. Already fixed in
`temporal_tuner.py`. Also `warmup_ratio` is deprecated in favor of `warmup_steps`.

### pyproject.toml Compatibility
Python 3.14 + setuptools required:
- `build-backend = "setuptools.build_meta"` (not the legacy backend)
- No `License :: OSI Approved` classifiers (PEP 639 superseded them)

---

## What Still Needs Doing

### Immediate (v0.1 completion)
- [ ] Add a README/model card to `lalopenguin/pulse-qwen-1.5b` on HF
- [ ] Test the trained model -- run inference, verify temporal reasoning quality
- [ ] Publish to PyPI: `pip install pulse-temporal`
- [ ] First git commit + push to GitHub (repo is initialized but no commits yet)
- [ ] Clean up HF Space `pulse-temporal-train` (needs Pro or remove ZeroGPU dependency)

### Testing the Trained Model
Run inference locally:
```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python examples/inference_trained.py \
  --model lalopenguin/pulse-qwen-1.5b \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --deadline "2026-04-15T17:00:00" \
  --events 5 \
  --sleep 6
```
This downloads the LoRA adapter from HF, merges it with the base model, and starts
an interactive chat with PULSE temporal context injected.

Or test with raw transformers:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "lalopenguin/pulse-qwen-1.5b")

messages = [
    {"role": "system", "content": "You have temporal awareness. Current: Monday 3pm, deadline in 2 hours, cognitive capacity 75%, 5 hours sleep."},
    {"role": "user", "content": "Should I start a complex refactoring task?"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=True)
out = model.generate(inputs, max_new_tokens=200)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

### Retraining (if needed)
If model quality isn't good enough:
1. **More data:** Increase from 2000 to 5000+ examples in `data_generator.py`
2. **Better responses:** The response templates in `data_generator.py` are rule-based. Could use a larger LLM to generate more natural responses.
3. **Larger base model:** Try Qwen 2.5 7B or Gemma 3 4B on Colab A100
4. **More epochs:** 3 epochs may not be enough. Try 5-10.
5. **Use Colab:** Open `notebooks/train_on_colab.ipynb`, change params, run

### Next Milestone (v0.2)
- Calendar/git event source adapters for the daemon
- MCP server so Claude/ChatGPT can query PULSE as a tool
- Temporal context injection middleware for any LLM API

---

## File Inventory

### Source Code
| File | Lines | Purpose |
|---|---|---|
| `pulse_temporal/__init__.py` | 15 | Public API exports |
| `pulse_temporal/encoder.py` | 195 | PulseEncoder main class |
| `pulse_temporal/layers/log_time.py` | 45 | Weber's Law compression |
| `pulse_temporal/layers/oscillators.py` | 50 | Multi-frequency sinusoids |
| `pulse_temporal/layers/circadian.py` | 60 | Circadian + ultradian cycles |
| `pulse_temporal/layers/calendar.py` | 115 | Calendar features |
| `pulse_temporal/layers/urgency.py` | 75 | Hyperbolic deadline urgency |
| `pulse_temporal/layers/temporal_state.py` | 155 | Event history state |
| `pulse_temporal/layers/prediction_error.py` | 135 | Temporal surprise |
| `pulse_temporal/daemon/state_db.py` | 115 | SQLite state store |
| `pulse_temporal/daemon/pulse_daemon.py` | 130 | Background daemon |
| `pulse_temporal/training/data_generator.py` | 325 | Synthetic data gen |
| `pulse_temporal/training/temporal_tuner.py` | 290 | LoRA fine-tuning |
| `pulse_temporal/utils/similarity.py` | 25 | Distance functions |

### Tests
| File | Tests | All Pass |
|---|---|---|
| `tests/test_encoder.py` | 20 | Yes |
| `tests/test_layers.py` | 20 | Yes |
| `tests/test_daemon.py` | 13 | Yes |

### Examples + Notebooks
| File | Purpose |
|---|---|
| `examples/basic_encoding.py` | Encode moments, similarity matrix |
| `examples/daemon_setup.py` | Run daemon, query context |
| `examples/gradio_demo.py` | HF Space app (dark/orange theme) |
| `examples/inference_trained.py` | Chat with PULSE-trained LLM |
| `notebooks/train_on_colab.ipynb` | Self-contained Colab training |

### Config + Data
| File | Purpose |
|---|---|
| `pyproject.toml` | Package config, deps, extras |
| `train.sh` | One-command training script |
| `models/pulse-base-v1/config.json` | Encoder config (dims, weights, frequencies) |
| `models/pulse-base-v1/README.md` | HF model card |
| `data/temporal_train.jsonl` | 2000 training examples (generated) |
| `data/temporal_eval.jsonl` | 200 eval examples (generated) |

---

## Git Status

Git is initialized but **no commits have been made yet**. All source files are staged.
`.gitignore` excludes `__pycache__/`, `*.egg-info/`, `*.db`, `data/*.jsonl`, model checkpoints,
and large binary files (`*.safetensors`, `*.bin`).

To make the first commit:
```bash
git add -A
git commit -m "Initial commit: pulse-temporal v0.1.0 -- experiential time embeddings"
```

To push to GitHub:
```bash
git remote add origin https://github.com/lalomorales22/pulse-temporal.git
git branch -M main
git push -u origin main
```

---

## Environment Notes

- **Python:** 3.14.3
- **PyTorch:** 2.10.0 (MPS available, Apple Silicon)
- **transformers:** 5.3.0
- **peft:** 0.18.1
- **trl:** 1.0.0
- **numpy:** installed (core dep)
- **HF CLI:** `/opt/homebrew/bin/hf` (authenticated as `lalopenguin`)
- **OS:** macOS Darwin 25.3.0

---

*Session 1 built April 9, 2026. Chula Vista, California.*

---
---

# Session 2 — April 9, 2026 (afternoon)

## What Was Done

### 1. Git + GitHub
- Made the first commit (36 files, 4910 lines)
- Created public repo: https://github.com/lalomorales22/pulse-temporal
- All pushed to `master` branch

### 2. Tested the Trained Qwen 1.5B Model
Loaded `lalopenguin/pulse-qwen-1.5b` from HuggingFace and ran 3 inference tests.
The model works and demonstrates temporal reasoning:

```
TEST 1 (2am, 4h sleep, deadline in 30min): "Should I start complex refactoring?"
PULSE: "15% cognitive, 10% energy, Monday 02:00 AM. Sleep deficit (4h) — expect ~20% more errors."

TEST 2 (10:30am peak, well-rested, no deadlines): "What tasks should I tackle?"
PULSE: "Strong. Go for complex debugging, architecture, research."

TEST 3 (1:30pm post-lunch dip, deadline in 4h): "Push through or call it a day?"
PULSE: "Circadian dip, not a wall. 15-min break restores."
```

**Verdict:** Training worked. Model learned temporal reasoning patterns. Responses are
terse/telegraphic (expected for 1.5B params + template training data).

### 3. Gemma 4 E2B Training Attempt (BLOCKED)
Created `notebooks/train_gemma4_colab.ipynb` to fine-tune `google/gemma-4-E2B-it` (~5.1B params)
with QLoRA on free Colab T4 GPU. Improved training data: 3000 examples (up from 2000),
20 scenarios (up from 14), 20 question types (up from 15), richer natural responses.

**The notebook hits multiple issues that were progressively fixed:**

#### Issue 1: transformers too old for gemma4 (FIXED)
```
KeyError: 'gemma4'
ValueError: model type `gemma4` but Transformers does not recognize this architecture
```
**Fix:** Install transformers from source: `pip install git+https://github.com/huggingface/transformers.git`
Plus auto-restart Colab runtime so new version loads.

#### Issue 2: Gemma4ClippableLinear not supported by PEFT (FIXED)
```
ValueError: Target module Gemma4ClippableLinear(...) is not supported.
Currently, only: torch.nn.Linear, torch.nn.Embedding, ...
```
Gemma 4 wraps all Linear layers in a custom `Gemma4ClippableLinear` class.
**Fix:** Unwrap 232 modules before applying LoRA:
```python
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
for name, module in list(model.named_modules()):
    if isinstance(module, Gemma4ClippableLinear):
        parts = name.rsplit('.', 1)
        parent = model.get_submodule(parts[0])
        setattr(parent, parts[1], module.linear)
```

#### Issue 3: Model too large for T4 VRAM (NOT YET FIXED)
```
ValueError: Some modules are dispatched on the CPU or the disk.
Make sure you have enough GPU RAM to fit the quantized model.
```
Even with 4-bit quantization, Gemma 4 E2B's full architecture (5.1B params including
multimodal components) doesn't fully fit in the T4's 16GB VRAM. Some modules get
offloaded to CPU, which then causes:
```
AcceleratorError: CUDA error: an illegal memory access was encountered
```
This happens during training when the model tries to access CPU-offloaded modules
from a CUDA kernel.

**Possible solutions for next session:**
1. **Use Gemma 3 4B instead** — single-modal, well-tested with LoRA, fits on T4
2. **Use Colab A100** (paid, $9.99/mo Colab Pro) — 40GB VRAM, Gemma 4 E2B fits easily
3. **Try aggressive memory optimization** — `max_memory` config, offload vision encoder only
4. **Use Gemma 4 E2B GGUF** — quantized versions exist that might be smaller
5. **Try `unsloth`** — optimized QLoRA library that reduces memory usage by ~60%

### Files Changed/Added
| File | Change |
|---|---|
| `notebooks/train_gemma4_colab.ipynb` | NEW — Gemma 4 QLoRA training notebook |
| `examples/inference_trained.py` | Updated help text for Gemma 4 |
| `README.md` | Added Gemma 4 notebook link, updated roadmap |
| `HANDOFF.md` | Added session 2 notes |

### Git Log (Session 2)
```
86d9b86 Initial commit: pulse-temporal v0.1.0
5e43cb1 Add Gemma 4 E2B training notebook and improved training data
22fcab6 Fix: install transformers from source for gemma4 architecture support
6765de1 Fix Gemma 4 Colab: auto-restart runtime + trust_remote_code fallback
fbecd85 Guard against restart loop in Colab install cell
38a1e55 Fix: unwrap Gemma4ClippableLinear for PEFT/LoRA compatibility
7ff9407 Fix torch.cuda total_mem -> removed, simplify GPU info print
73caaae Fix QLoRA training: prepare_model_for_kbit_training + peft from source
```

---

*Session 2: April 9, 2026 afternoon. Chula Vista, California.*
