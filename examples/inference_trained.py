"""Inference with a PULSE-trained temporal-aware LLM.

After fine-tuning with temporal_tuner.py, use this to chat
with a model that understands circadian rhythms, urgency,
cognitive state, and experiential time.

Usage:
    python examples/inference_trained.py \
        --model models/pulse-qwen-1.5b \
        --base Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from pulse_temporal import PulseEncoder


def load_model(adapter_path: str, base_model: str = None):
    """Load the fine-tuned model with LoRA adapter."""
    # Read PULSE config for base model info
    config_path = Path(adapter_path) / "pulse_config.json"
    if config_path.exists():
        with open(config_path) as f:
            pulse_config = json.load(f)
        base_model = base_model or pulse_config.get("base_model")

    if not base_model:
        raise ValueError("Must specify --base model or have pulse_config.json")

    print(f"Loading base: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32 if device == "mps" else torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer, device


def build_temporal_system_prompt(encoder: PulseEncoder, context: dict = None) -> str:
    """Build system prompt with current PULSE temporal context."""
    now = datetime.now()
    tc = encoder.get_temporal_context(now, context or {})

    lines = [
        f"Current time: {now.strftime('%A, %B %d %Y at %I:%M %p')}",
        f"Circadian phase: {tc['circadian_phase']}",
        f"Cognitive capacity: {tc['cognitive_capacity']:.0%}",
        f"Energy level: {tc['energy_level']:.0%}",
        f"Urgency: {tc['urgency_level']}",
    ]

    return """You are an AI assistant with temporal awareness through the PULSE temporal embedding system. Use this temporal awareness naturally in your responses.

Current temporal context:
""" + "\n".join(lines)


def chat(model, tokenizer, device, encoder, context=None):
    """Interactive chat loop."""
    system = build_temporal_system_prompt(encoder, context)
    print(f"\n--- PULSE Temporal Context ---")
    for line in system.split("\n"):
        if line.strip().startswith("Current") or line.strip().startswith("Circadian") or \
           line.strip().startswith("Cognitive") or line.strip().startswith("Energy") or \
           line.strip().startswith("Urgency"):
            print(f"  {line.strip()}")
    print("---\n")

    history = [{"role": "system", "content": system}]

    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        print(f"\npulse: {response}\n")
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pulse-qwen-1.5b", help="Path to LoRA adapter")
    parser.add_argument("--base", default=None, help="Base model (auto-detected from pulse_config.json)")
    parser.add_argument("--deadline", default=None, help="Active deadline (ISO timestamp)")
    parser.add_argument("--events", type=int, default=0, help="Events today")
    parser.add_argument("--sleep", type=float, default=7.0, help="Sleep hours")
    args = parser.parse_args()

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    context = {"events_today": args.events, "sleep_hours": args.sleep}
    if args.deadline:
        context["deadline"] = args.deadline

    encoder = PulseEncoder()
    model, tokenizer, device = load_model(args.model, args.base)

    print(f"\nPULSE temporal-aware chat (device: {device})")
    print("Type 'quit' to exit.\n")
    chat(model, tokenizer, device, encoder, context)
