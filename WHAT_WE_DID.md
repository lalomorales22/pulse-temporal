# What We Did

## The Idea

AI has no sense of time. It knows what time it *is* but not what that time *means*. Monday 2am after no sleep feels nothing like Saturday 2pm with nothing to do — but to an AI, they're just two timestamps.

We decided to fix that.

## What We Built

### 1. PULSE — a "time feelings" encoder

We wrote a Python library called `pulse-temporal` that takes a moment — a timestamp plus context like "how much did you sleep?" and "when's your deadline?" — and turns it into a 128-number vector that captures what that moment *feels like*.

It combines 7 signals:
- **Clock time** (compressed logarithmically, because the difference between minute 1 and minute 2 feels bigger than minute 101 and 102)
- **Body clock** (circadian rhythm — your brain peaks around 10am-12pm, dips after lunch, rallies around 4pm)
- **Calendar** (weekday vs weekend, holidays, seasons)
- **Oscillators** (repeating patterns at different scales — hourly, weekly, monthly, yearly)
- **Urgency** (how close is the deadline? uses the same math as behavioral economics — urgency isn't linear, it's hyperbolic)
- **Event history** (what's been happening? lots of meetings = context-switching fatigue)
- **Surprise** (was this event expected or not?)

The key result: two "crunch before deadline" moments on different days score 0.97 similarity. A "crunch" moment and a "chill Saturday" at the same clock time score 0.74. The encoder captures *experience*, not just time.

This is `lalopenguin/pulse-base-v1` on HuggingFace. It's not a neural network — it's a formula-based encoder. No trained weights, just math from neuroscience and behavioral economics.

### 2. Training data — teaching an LLM to think about time

We wrote a data generator that creates thousands of synthetic conversations like:

> **System prompt:** "You have temporal awareness. It's Monday 2am. Cognitive capacity: 15%. Energy: 10%. 4 hours sleep. Deadline in 30 minutes."
>
> **User:** "Should I start a complex refactoring task?"
>
> **Assistant:** "Cognition at 15%, energy at 10%. Complex work at this hour creates more bugs than it fixes. Only 4h sleep makes this worse. Save it for tomorrow morning."

The data generator creates realistic scenarios — crunch time, vacation mode, post-lunch dip, all-nighters, back-to-back meetings, etc. — and generates appropriate temporally-aware responses for each one. 20 different scenarios, 20 different question types, 3000 training examples.

### 3. Fine-tuned a small model — pulse-qwen-1.5b

We took `Qwen 2.5 1.5B Instruct` (a small open-source LLM) and fine-tuned it on our temporal training data using LoRA (a technique that only trains a small adapter layer, not the whole model). Trained on a free Google Colab T4 GPU.

**It worked.** The trained model actually reasons about time:
- Asked at 2am with no sleep if it should do complex work → correctly says no, estimates 20% more errors
- Asked at 10:30am well-rested what to tackle → correctly recommends complex debugging, architecture work
- Asked at 1:30pm during the post-lunch dip → correctly identifies it as a circadian dip, not exhaustion, recommends a short break

This is `lalopenguin/pulse-qwen-1.5b` on HuggingFace. It's a LoRA adapter that sits on top of the base Qwen model.

### 4. Tried to train a bigger model (Gemma 4) — blocked by GPU memory

We tried to do the same thing with Google's Gemma 4 E2B (~5B params, much smarter). But it's a multimodal model (text + vision) and even quantized to 4-bit, it doesn't fit on a free T4 GPU. The vision components eat VRAM that we don't need.

So we pivoted to **Gemma 3 4B** — text-only, 4 billion parameters, and using **Unsloth** (a library that cuts memory usage by ~60%). The notebook is ready, we just need GPU time to run it. Free Colab/Kaggle quota was exhausted when we tried.

### 5. Built the v0.2 infrastructure

On top of the core encoder and trained model, we built:

- **MCP server** — so Claude or ChatGPT can query PULSE as a tool (8 tools: get temporal context, encode moments, compare moments, log events, manage deadlines)
- **LLM middleware** — wraps OpenAI/Anthropic API calls to automatically inject temporal context into every conversation
- **Git adapter** — feeds git commit history into the temporal context
- **Calendar adapter** — feeds iCal/Google Calendar events into the temporal context
- **Published to PyPI** — `pip install pulse-temporal` works

## What's on HuggingFace

| Repo | What it is |
|---|---|
| `lalopenguin/pulse-base-v1` | The PULSE encoder definition (formulas, config, not a neural net) + training notebooks |
| `lalopenguin/pulse-qwen-1.5b` | The trained LoRA adapter (actual neural network weights) for Qwen 2.5 1.5B |
| `lalopenguin/pulse-temporal-demo` | Interactive Gradio demo — compare moments, see similarity scores |
| `lalopenguin/pulse-temporal-train` | Training data generator Space — create custom training datasets |

## What's Next

1. **Run the Gemma 3 4B notebook** — waiting for GPU quota to reset, then train a better model
2. **v0.3** — replace the formula-based encoder with a *learned* encoder trained on real human activity data
3. **Long term** — PULSE runs as a background daemon, continuously tracking your temporal state, and any AI you talk to just *knows* what time means for you right now

## The Simple Version

We wrote code that understands "how time feels." We generated training data that teaches AI to reason about time. We trained a small model on that data and it works. Now we're training a bigger model to make it work better.

Everything is open source: [github.com/lalomorales22/pulse-temporal](https://github.com/lalomorales22/pulse-temporal)
