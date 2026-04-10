"""PULSE Temporal Awareness -- Training Data & Config Space

CPU-only Gradio Space that generates PULSE temporal training data,
lets users preview/download it, and provides ready-to-use configs
for Colab training.

No GPU needed. For actual training, use the Colab notebooks:
- Qwen 2.5 1.5B: https://huggingface.co/lalopenguin/pulse-base-v1/blob/main/train_on_colab.ipynb
- Gemma 3 4B: https://huggingface.co/lalopenguin/pulse-base-v1/blob/main/train_gemma3_colab.ipynb
"""

import gradio as gr
import json
import random
import tempfile
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# =====================================================
# PULSE ENCODER (inline, minimal for data generation)
# =====================================================

_COG = np.array([0.20,0.15,0.12,0.10,0.12,0.18,0.30,0.50,0.70,0.85,0.95,0.92,0.85,0.72,0.68,0.75,0.88,0.90,0.82,0.70,0.55,0.40,0.30,0.25], dtype=np.float32)
_NRG = np.array([0.15,0.10,0.08,0.07,0.10,0.20,0.40,0.60,0.75,0.85,0.90,0.88,0.82,0.70,0.65,0.72,0.85,0.88,0.80,0.65,0.50,0.35,0.25,0.18], dtype=np.float32)

def _interp(c, h):
    h0 = int(h) % 24
    return float(c[h0]*(1-(h-int(h))) + c[(h0+1)%24]*(h-int(h)))

def get_phase(h):
    if 6<=h<10: return "morning_ramp"
    if 10<=h<12: return "morning_peak"
    if 12<=h<14: return "post_lunch_dip"
    if 14<=h<17: return "afternoon_peak"
    if 17<=h<20: return "evening_wind_down"
    if 20<=h<23: return "night_transition"
    return "deep_night"


# =====================================================
# DATA GENERATOR
# =====================================================

SYSTEM_TEMPLATE = """You are an AI assistant with temporal awareness through the PULSE temporal embedding system. Before each interaction, you receive a temporal context package describing the current moment — not just the time, but what that time means: circadian phase, cognitive capacity, energy level, urgency, and behavioral context.

Use this temporal awareness naturally in your responses. Don't announce it mechanically — weave it into your reasoning the way a thoughtful colleague would who knows what time it is and what's going on.

Current temporal context:
{temporal_context}"""

CONTEXTS = [
    ("monday_crunch", 2.0, 7, 5), ("critical_deadline", 0.5, 9, 4),
    ("normal_tuesday", None, 3, 7.5), ("focus_day", None, 1, 8),
    ("deadline_tomorrow", 24.0, 5, 7), ("friday_winding", None, 2, 7),
    ("saturday_morning", None, 0, 9), ("sunday_evening", 12.0, 0, 7),
    ("late_night", None, 0, 0), ("early_fresh", None, 0, 8),
    ("post_lunch", None, 4, 7), ("peak_morning", None, 2, 8),
    ("overdue", -2.0, 8, 4), ("vacation", None, 0, 9),
    ("all_nighter", None, 0, 2), ("back_to_back", None, 10, 6),
    ("weekend_project", 48.0, 0, 8), ("interview_prep", 3.0, 2, 6),
    ("launch_day", 1.0, 12, 4), ("recovery_day", None, 0, 10),
]

QUESTIONS = [
    ("Should I start a complex refactoring task right now?", "task_suitability"),
    ("Is this a good time for creative brainstorming?", "task_suitability"),
    ("Should I take a break right now?", "break_advice"),
    ("How much focus can I expect from myself right now?", "cognitive_state"),
    ("What does my current temporal state look like?", "full_state"),
    ("Am I in a good phase for deep work?", "work_phase"),
    ("How urgent is my situation right now?", "urgency_assessment"),
    ("Would I be more productive waiting until tomorrow morning?", "timing_optimization"),
    ("How should I prioritize my remaining tasks today?", "prioritization"),
    ("What kind of tasks should I tackle right now given my state?", "task_matching"),
    ("Should I push through or call it a day?", "endurance_check"),
    ("How does this moment compare to a typical morning?", "relative_state"),
    ("Should I schedule a difficult conversation for this time?", "task_suitability"),
    ("Is my energy level normal for this time of day?", "circadian_comparison"),
    ("Would this be a good time to learn something new?", "task_suitability"),
    ("Can I handle a code review right now?", "task_suitability"),
    ("Should I do the easy tasks first or tackle the hard one?", "prioritization"),
    ("How will I feel in 2 hours compared to now?", "timing_optimization"),
    ("Is this deadline realistic given my current state?", "urgency_assessment"),
    ("What's the best use of the next hour?", "task_matching"),
]


def generate_response(question, q_sub, dt, deadline_str, events, sleep):
    hour = dt.hour
    cog = _interp(_COG, hour + dt.minute/60)
    eng = _interp(_NRG, hour + dt.minute/60)
    has_dl = deadline_str is not None
    hours_left = (datetime.fromisoformat(deadline_str) - dt).total_seconds()/3600 if has_dl else None
    is_overdue = has_dl and hours_left < 0
    is_urgent = has_dl and hours_left is not None and 0 < hours_left < 3
    is_night = hour < 6 or hour >= 22
    is_peak = 10 <= hour < 12 or 14 <= hour < 17
    is_dip = 12 <= hour < 14
    is_low_sleep = sleep < 6

    parts = []
    if q_sub == "task_suitability":
        if any(w in question.lower() for w in ("complex", "refactoring", "deep work", "learn", "difficult", "code review")):
            if is_peak and not is_low_sleep and cog > 0.8:
                parts.append(f"This is an ideal window. Cognitive capacity at {cog:.0%} during {'morning' if hour<12 else 'afternoon'} peak.")
                if is_urgent: parts.append(f"But with the deadline in {hours_left:.1f} hours, prioritize what's due first.")
                elif not has_dl: parts.append("No deadlines pressing. Good time to dive deep.")
            elif is_dip:
                parts.append(f"Post-lunch dip — cognitive capacity around {cog:.0%}. Complex tasks have more errors now. Wait until ~3pm when afternoon focus returns.")
            elif is_night:
                parts.append(f"It's {dt.strftime('%I:%M %p')}, cognition at {cog:.0%}. Complex work at this hour creates more bugs than it fixes.")
                if is_low_sleep: parts.append(f"Only {sleep:.0f}h sleep makes this worse. Save it for tomorrow morning.")
            elif is_low_sleep:
                parts.append(f"With {sleep:.0f}h sleep, cognitive capacity is compromised. Stick to routine tasks today.")
            else:
                parts.append(f"Moderate capacity at {cog:.0%}. You could start, but this isn't your peak window.")
        elif "creative" in question.lower() or "brainstorm" in question.lower():
            if is_dip or is_night:
                parts.append("Reduced executive function can actually help creativity — your inner critic is quieter. Good time for brainstorming.")
            elif is_peak:
                parts.append(f"Peak capacity ({cog:.0%}) is great for structured creative work. For wild brainstorming, the afternoon dip might actually work better.")
        elif "break" in question.lower():
            if cog < 0.5 or eng < 0.4:
                parts.append(f"Yes. Energy at {eng:.0%}, cognition at {cog:.0%}. A 15-20 minute break would help.")
            elif is_dip:
                parts.append("Natural post-lunch dip. A short walk now aligns with your body's rhythm.")
            elif events > 5:
                parts.append(f"{events} events today means serious context-switching fatigue. Break would help.")
            else:
                parts.append(f"Energy ({eng:.0%}) and cognition ({cog:.0%}) are solid. Keep going if you're in flow.")
        elif "difficult conversation" in question.lower():
            if is_peak and not is_low_sleep:
                parts.append(f"Cognitive capacity at {cog:.0%} helps with emotional regulation. Reasonable window for it.")
            else:
                parts.append(f"With cognition at {cog:.0%}, you're more likely to be reactive than reflective. Postpone if possible.")

    elif q_sub in ("full_state", "cognitive_state", "work_phase"):
        parts.append(f"It's {dt.strftime('%A %I:%M %p')}.")
        if is_peak: parts.append(f"Peak cognitive window — {cog:.0%} capacity, {eng:.0%} energy. Prime time for demanding work.")
        elif is_dip: parts.append(f"Post-lunch dip. Cognition {cog:.0%}, energy {eng:.0%}. Passes around 2:30-3pm.")
        elif is_night: parts.append(f"Deep night. Cognition {cog:.0%}, energy {eng:.0%}. Your body wants rest.")
        else: parts.append(f"Cognitive capacity {cog:.0%}, energy {eng:.0%}.")
        if is_low_sleep: parts.append(f"Sleep deficit ({sleep:.0f}h) dragging everything down. Expect ~20% more errors.")
        if is_overdue: parts.append(f"Deadline passed {-hours_left:.1f}h ago. High stress.")
        elif is_urgent: parts.append(f"Deadline in {hours_left:.1f}h. Focused execution mode.")

    elif q_sub == "urgency_assessment":
        if is_overdue: parts.append(f"Critical. Deadline passed {-hours_left:.1f}h ago. Damage control mode.")
        elif is_urgent: parts.append(f"High — {hours_left:.1f}h until deadline. This should be your only focus.")
        elif has_dl and hours_left < 24: parts.append(f"Moderate. Deadline {hours_left:.1f}h away. Start planning.")
        elif has_dl: parts.append(f"Low for now — {hours_left:.1f}h out. Keep it on radar.")
        else: parts.append("No active deadlines. Choose work based on energy and interest.")

    elif q_sub == "timing_optimization":
        if cog < 0.5: parts.append("Yes. Tomorrow 10-12am would give roughly double your current capacity.")
        elif is_peak: parts.append("You're in a good window now. Waiting means losing momentum and context.")
        elif is_urgent: parts.append(f"Deadline in {hours_left:.1f}h. Waiting isn't an option.")
        else: parts.append(f"Current {cog:.0%} vs tomorrow's ~93% peak. Depends on task complexity.")

    elif q_sub == "prioritization":
        if is_urgent: parts.append(f"Deadline work first — {hours_left:.1f}h left. Everything else secondary.")
        elif is_peak: parts.append("Use this peak for your hardest task. Save routine work for the dip.")
        elif is_dip: parts.append("Good for: emails, code review, admin. Save complex work for the 3-4pm peak.")
        else: parts.append(f"At {cog:.0%} capacity, match tasks to state. Routine now, demanding later.")

    elif q_sub == "task_matching":
        if cog > 0.8: parts.append("Strong state. Go for: complex debugging, architecture decisions, learning new concepts.")
        elif cog > 0.5: parts.append("Moderate. Good for: code review, incremental features, documentation, discussions.")
        else: parts.append("Low state. Stick to: email triage, filing issues, light reading, planning tomorrow.")

    elif q_sub == "endurance_check":
        if eng < 0.3: parts.append(f"Call it. Energy {eng:.0%}, cognition {cog:.0%}. Past diminishing returns.")
        elif is_urgent: parts.append(f"Push through — {hours_left:.1f}h to deadline. Take a 5-min reset first.")
        elif is_dip: parts.append("Feels like a wall but it's the circadian dip. 15-min break usually restores enough.")
        elif events > 7: parts.append(f"{events} events today. Context-switching cost has accumulated. You've earned the stop.")
        else: parts.append(f"Energy {eng:.0%}, cognition {cog:.0%}. Some runway left if work is engaging.")

    elif q_sub in ("relative_state", "circadian_comparison"):
        parts.append(f"At {dt.strftime('%I:%M %p')}, typical capacity is ~{cog:.0%}.")
        if is_low_sleep: parts.append(f"Your {sleep:.0f}h sleep puts you below baseline. Well-rested you'd be closer to {min(cog+0.15,0.95):.0%}.")
        if is_peak: parts.append("This is normally productive. " + ("Good shape to use it." if not is_low_sleep else "Sleep deficit eating into your best hours."))
        elif is_dip: parts.append("Post-lunch dip is universal. Not a you problem, it's biology.")

    if not parts:
        parts.append(f"Current: {cog:.0%} cognitive, {eng:.0%} energy, {dt.strftime('%A %I:%M %p')}.")
    return " ".join(parts)


def generate_dataset(n=2000, seed=42):
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        name, dl_off, events, sleep = rng.choice(CONTEXTS)
        base = datetime(2026, rng.randint(1,12), rng.randint(1,28))
        hour = rng.randint(0, 23)
        minute = rng.choice([0,15,30,45])
        dt = base.replace(hour=hour, minute=minute)
        if "late_night" in name or "all_nighter" in name: dt = dt.replace(hour=rng.choice([0,1,2,3,23]))
        elif "early" in name: dt = dt.replace(hour=rng.choice([5,6,7]))
        elif "peak" in name: dt = dt.replace(hour=rng.choice([10,11]))
        elif "post_lunch" in name: dt = dt.replace(hour=rng.choice([13,14]))

        dl_str = (dt + timedelta(hours=dl_off)).isoformat() if dl_off is not None else None
        h = dt.hour + dt.minute/60
        phase = get_phase(dt.hour)
        cog, eng = _interp(_COG, h), _interp(_NRG, h)
        urg_detail = f"deadline in {dl_off:.1f}h" if dl_off and dl_off > 0 else ("OVERDUE" if dl_off and dl_off < 0 else "none")

        tc = f"""Current time: {dt.strftime('%A, %B %d %Y at %I:%M %p')}
Circadian phase: {phase}
Cognitive capacity: {cog:.0%}
Energy level: {eng:.0%}
Urgency: {urg_detail}
Events today: {events}
Sleep last night: {sleep:.1f} hours
Weekend: {'yes' if dt.weekday()>=5 else 'no'}"""

        question, q_sub = rng.choice(QUESTIONS)
        response = generate_response(question, q_sub, dt, dl_str, events, sleep)
        system = SYSTEM_TEMPLATE.format(temporal_context=tc)

        examples.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
        })
    return examples


# =====================================================
# GRADIO UI
# =====================================================

def generate_and_preview(num_examples, seed):
    """Generate data and return preview + download file."""
    data = generate_dataset(int(num_examples), int(seed))

    # Preview first 3 examples
    preview_lines = []
    for i, ex in enumerate(data[:3]):
        msgs = ex["messages"]
        preview_lines.append(f"--- Example {i+1} ---")
        # Extract temporal context snippet
        sys_content = msgs[0]["content"]
        tc_start = sys_content.find("Current time:")
        tc_end = sys_content.find("Weekend:") + 20
        if tc_start >= 0:
            preview_lines.append(sys_content[tc_start:tc_end])
        preview_lines.append(f"Q: {msgs[1]['content']}")
        preview_lines.append(f"A: {msgs[2]['content']}")
        preview_lines.append("")

    preview_lines.append(f"... {len(data)} examples total")
    preview = "\n".join(preview_lines)

    # Write to temp file for download
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, prefix="pulse_train_")
    for ex in data:
        tmp.write(json.dumps(ex) + "\n")
    tmp.close()

    stats = {
        "total_examples": len(data),
        "scenarios": len(CONTEXTS),
        "question_types": len(QUESTIONS),
        "avg_response_length": sum(len(ex["messages"][2]["content"]) for ex in data) / len(data),
    }
    stats_text = (
        f"Generated **{stats['total_examples']}** examples\n"
        f"- {stats['scenarios']} scenario types\n"
        f"- {stats['question_types']} question types\n"
        f"- Avg response: {stats['avg_response_length']:.0f} chars"
    )

    return preview, tmp.name, stats_text


theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fff7ed",c100="#ffedd5",c200="#fed7aa",c300="#fdba74",
        c400="#fb923c",c500="#f97316",c600="#ea580c",c700="#c2410c",
        c800="#9a3412",c900="#7c2d12",c950="#431407",
    ),
    neutral_hue=gr.themes.Color(
        c50="#fafafa",c100="#f4f4f5",c200="#e4e4e7",c300="#d4d4d8",
        c400="#a1a1aa",c500="#71717a",c600="#52525b",c700="#3f3f46",
        c800="#27272a",c900="#18181b",c950="#09090b",
    ),
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0a0a0a", body_background_fill_dark="#0a0a0a",
    body_text_color="#d4d4d4", body_text_color_dark="#d4d4d4",
    background_fill_primary="#111111", background_fill_primary_dark="#111111",
    background_fill_secondary="#0a0a0a", background_fill_secondary_dark="#0a0a0a",
    block_background_fill="#111111", block_background_fill_dark="#111111",
    block_border_color="#1f1f1f", block_border_color_dark="#1f1f1f",
    border_color_primary="#262626", border_color_primary_dark="#262626",
    input_background_fill="#171717", input_background_fill_dark="#171717",
    input_border_color="#262626", input_border_color_dark="#262626",
    button_primary_background_fill="#c2410c", button_primary_background_fill_dark="#c2410c",
    button_primary_background_fill_hover="#ea580c", button_primary_background_fill_hover_dark="#ea580c",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    slider_color="#ea580c", slider_color_dark="#ea580c",
)

css = """
.gradio-container { max-width: 900px !important; }
h1, h2, h3 { color: #e5e5e5 !important; }
.prose strong { color: #fb923c !important; }
.prose code { background: #1a1a1a !important; color: #fb923c !important; }
.prose a { color: #fb923c !important; }
footer { display: none !important; }
"""

with gr.Blocks(theme=theme, css=css, title="PULSE Training Data") as demo:
    gr.Markdown("""
# PULSE Training Data Generator
### generate temporal awareness training data for any LLM

Create training data that teaches LLMs to understand **circadian rhythms**, **cognitive capacity**,
**urgency**, **sleep debt**, and **experiential time**. Download as JSONL and train on Colab.
""")

    with gr.Tab("generate data"):
        with gr.Row():
            with gr.Column(scale=1):
                num_examples = gr.Slider(500, 10000, value=3000, step=500, label="training examples")
                seed = gr.Number(value=42, label="random seed", precision=0)
                gen_btn = gr.Button("generate", variant="primary", size="lg")
                stats_md = gr.Markdown("")
                download = gr.File(label="download JSONL")
            with gr.Column(scale=2):
                preview = gr.Textbox(label="preview (first 3 examples)", lines=25, max_lines=40, interactive=False)

        gen_btn.click(
            generate_and_preview,
            inputs=[num_examples, seed],
            outputs=[preview, download, stats_md],
        )

    with gr.Tab("training configs"):
        gr.Markdown("""
## Ready-to-use training configs

### Option 1: Colab with Qwen 2.5 1.5B (free T4)
Smallest model, fastest training, good baseline results.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lalomorales22/pulse-temporal/blob/master/notebooks/train_on_colab.ipynb)

```
Model: Qwen/Qwen2.5-1.5B-Instruct
Method: LoRA (r=16, alpha=32)
GPU: T4 (free Colab)
Time: ~15 min
VRAM: ~8 GB
```

### Option 2: Colab with Gemma 3 4B + Unsloth (free T4)
Larger model, better quality, still fits on free GPU.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lalomorales22/pulse-temporal/blob/master/notebooks/train_gemma3_colab.ipynb)

```
Model: unsloth/gemma-3-4b-it-unsloth-bnb-4bit
Method: QLoRA via Unsloth (~60% memory savings)
GPU: T4 (free Colab)
Time: ~25 min
VRAM: ~12 GB
```

### Option 3: Local training
```bash
pip install pulse-temporal[dev]
python -m pulse_temporal.training.temporal_tuner \\
    --model Qwen/Qwen2.5-1.5B-Instruct \\
    --data your_data.jsonl
```
""")

    with gr.Tab("about PULSE"):
        gr.Markdown("""
## What is PULSE?

PULSE encodes moments as **128D vectors** that capture not just *when* something happens,
but *what that time means* — urgency, circadian phase, cognitive capacity, and the felt sense of time.

### The 7 Signal Layers

| Layer | Dims | What it captures |
|---|---|---|
| **log_time** | 8D | Weber's Law — felt time is logarithmic |
| **oscillators** | 32D | Multi-frequency sinusoids (1h to 4y cycles) |
| **circadian** | 8D | 24h + 90min biological clock phase |
| **calendar** | 24D | Day/month/season/holiday structure |
| **urgency** | 8D | Hyperbolic deadline proximity |
| **temporal_state** | 32D | Continuous-time event history |
| **prediction_error** | 16D | How surprising is this moment? |

### Key Result

```
Mon 2pm crunch  <->  Wed 10am crunch:  0.978  (same experience)
Mon 2pm crunch  <->  Sat 2pm chill:    0.743  (same time, different feel)
Mon 2pm crunch  <->  Mon 2am insomnia: 0.303  (same day, totally different)
```

### Links

- [GitHub](https://github.com/lalomorales22/pulse-temporal)
- [Encoder model card](https://huggingface.co/lalopenguin/pulse-base-v1)
- [Trained model (Qwen 1.5B)](https://huggingface.co/lalopenguin/pulse-qwen-1.5b)
- [Interactive demo](https://huggingface.co/spaces/lalopenguin/pulse-temporal-demo)
- `pip install pulse-temporal`
""")

demo.launch()
