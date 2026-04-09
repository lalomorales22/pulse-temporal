"""Fine-tune a small LLM with PULSE temporal awareness.

Takes a base model (Gemma 2B, Qwen 1.5B, etc.) and fine-tunes it
with LoRA on synthetic temporal reasoning data so the model learns
to understand and use PULSE temporal context naturally.

Usage:
    python -m pulse_temporal.training.temporal_tuner \
        --model google/gemma-3-1b-it \
        --data data/temporal_train.jsonl \
        --output models/pulse-gemma-1b \
        --epochs 3

Requirements:
    pip install pulse-temporal[torch] peft trl
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Optional

# Check dependencies upfront
_MISSING = []
try:
    import torch
except ImportError:
    _MISSING.append("torch")
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
except ImportError:
    _MISSING.append("transformers")
try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    _MISSING.append("peft")
try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    _MISSING.append("trl")
try:
    from datasets import Dataset
except ImportError:
    _MISSING.append("datasets")


def check_deps():
    if _MISSING:
        print(f"Missing dependencies: {', '.join(_MISSING)}")
        print(f"Install with: pip install {' '.join(_MISSING)}")
        sys.exit(1)


def detect_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_data(path: str) -> Dataset:
    """Load JSONL chat data into a HuggingFace Dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line.strip())
            if "messages" in ex:
                examples.append(ex)

    return Dataset.from_list(examples)


def format_chat(example, tokenizer):
    """Format messages into the model's chat template."""
    messages = example["messages"]
    # Use the tokenizer's chat template if available
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback: manual formatting
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        text = "\n".join(parts)

    return {"text": text}


def train(
    model_name: str = "google/gemma-3-1b-it",
    train_data: str = "data/temporal_train.jsonl",
    eval_data: Optional[str] = "data/temporal_eval.jsonl",
    output_dir: str = "models/pulse-temporal-gemma",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_seq_length: int = 1024,
    gradient_accumulation: int = 4,
    use_4bit: bool = False,
):
    """Run the fine-tuning pipeline."""
    check_deps()

    device = detect_device()
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Train data: {train_data}")
    print(f"Output: {output_dir}")

    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load model ----
    model_kwargs = {"trust_remote_code": True}

    if use_4bit and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    elif device == "mps":
        model_kwargs["torch_dtype"] = torch.float32  # MPS needs float32 for stability
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        # Workaround: from_pretrained can segfault on some Python/PyTorch combos.
        # Fall back to manual config + safetensors loading.
        print(f"from_pretrained failed ({e}), trying manual load...")
        from safetensors.torch import load_file
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config)
        cache_dir = Path.home() / ".cache/huggingface/hub"
        safe_name = model_name.replace("/", "--")
        patterns = [
            str(cache_dir / f"models--{safe_name}" / "snapshots" / "*" / "model.safetensors"),
            str(cache_dir / f"models--{safe_name}" / "snapshots" / "*" / "model-00001-of-*.safetensors"),
        ]
        for pattern in patterns:
            files = sorted(glob.glob(pattern))
            if files:
                for f in files:
                    state = load_file(f, device="cpu")
                    model.load_state_dict(state, strict=False)
                    del state
                print(f"Loaded weights from {len(files)} safetensors file(s)")
                break

    if device == "mps" and not use_4bit:
        model = model.to(device)

    # ---- LoRA config ----
    # Target the attention and MLP projection layers
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    # ---- Load data ----
    print("Loading data...")
    train_dataset = load_data(train_data)
    train_dataset = train_dataset.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=["messages", "metadata"],
    )

    eval_dataset = None
    if eval_data and Path(eval_data).exists():
        eval_dataset = load_data(eval_data)
        eval_dataset = eval_dataset.map(
            lambda ex: format_chat(ex, tokenizer),
            remove_columns=["messages", "metadata"],
        )

    print(f"Train examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval examples: {len(eval_dataset)}")

    # ---- Training config ----
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        bf16=(device == "cuda"),
        fp16=False,
        max_length=max_seq_length,
        dataset_text_field="text",
        report_to="none",
        seed=42,
    )

    # ---- Train ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    # ---- Save ----
    print(f"\nSaving to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save PULSE metadata
    meta = {
        "base_model": model_name,
        "pulse_version": "0.1.0",
        "training_type": "temporal_awareness_sft",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "description": "Small LLM fine-tuned with PULSE temporal awareness. "
                       "Understands circadian rhythms, urgency, cognitive state, "
                       "and experiential time context.",
    }
    with open(Path(output_dir) / "pulse_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with PULSE temporal awareness")
    parser.add_argument("--model", default="google/gemma-3-1b-it", help="Base model ID from HuggingFace")
    parser.add_argument("--data", default="data/temporal_train.jsonl", help="Training data JSONL")
    parser.add_argument("--eval", default="data/temporal_eval.jsonl", help="Eval data JSONL")
    parser.add_argument("--output", default="models/pulse-temporal-gemma", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization (CUDA only)")
    args = parser.parse_args()

    train(
        model_name=args.model,
        train_data=args.data,
        eval_data=args.eval,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        max_seq_length=args.max_seq_length,
        use_4bit=args.use_4bit,
    )
