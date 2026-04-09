"""Synthetic training data generator for temporal-aware LLM fine-tuning.

Generates instruction-following pairs where the model must understand
and reason about PULSE temporal context. The resulting dataset teaches
a small LLM (Gemma 2B, Qwen 1.5B, etc.) to have temporal awareness.

Three data categories:
1. Temporal reasoning -- "is this a good time to start deep work?"
2. Context interpretation -- "describe the current temporal state"
3. Temporal comparison -- "how does now compare to yesterday?"
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

# Import the encoder
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pulse_temporal.encoder import PulseEncoder


# ---- Scenario Templates ----

_CONTEXTS = [
    # (name, deadline_offset_hours, events_today, sleep_hours, description)
    ("monday_crunch", 2.0, 7, 5, "deadline crunch, sleep-deprived, packed schedule"),
    ("monday_crunch_hard", 0.5, 9, 4, "critical deadline in 30 minutes, exhausted"),
    ("tuesday_morning", None, 3, 7.5, "normal Tuesday morning, reasonable sleep"),
    ("wednesday_focus", None, 1, 8, "light day, well-rested, deep focus time"),
    ("thursday_afternoon", 24.0, 5, 7, "deadline tomorrow, moderate schedule"),
    ("friday_winding_down", None, 2, 7, "Friday afternoon, wrapping up the week"),
    ("saturday_morning", None, 0, 9, "Saturday, no obligations, well-rested"),
    ("saturday_night", None, 1, 8, "Saturday evening, relaxed"),
    ("sunday_evening", 12.0, 0, 7, "Sunday night, work deadline Monday morning"),
    ("late_night_coding", None, 0, 0, "2am coding session, been up all night"),
    ("early_morning_fresh", None, 0, 8, "6am, just woke up, quiet before the day starts"),
    ("post_lunch_slump", None, 4, 7, "2pm, post-lunch energy dip"),
    ("peak_morning", None, 2, 8, "10:30am, peak cognitive hours"),
    ("overdue_panic", -2.0, 8, 4, "deadline was 2 hours ago, still not done"),
    ("vacation_mode", None, 0, 9, "vacation day, absolutely nothing to do"),
]

_HOURS = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

_TEMPORAL_QUESTIONS = [
    ("Should I start a complex refactoring task right now?",
     "reasoning", "task_suitability"),
    ("Is this a good time for creative brainstorming?",
     "reasoning", "task_suitability"),
    ("Should I schedule a difficult conversation for this time?",
     "reasoning", "task_suitability"),
    ("Would this be a good time to learn something new?",
     "reasoning", "task_suitability"),
    ("Should I take a break right now?",
     "reasoning", "break_advice"),
    ("How much focus can I expect from myself right now?",
     "interpretation", "cognitive_state"),
    ("What does my current temporal state look like?",
     "interpretation", "full_state"),
    ("Am I in a good phase for deep work?",
     "interpretation", "work_phase"),
    ("How urgent is my situation right now?",
     "interpretation", "urgency_assessment"),
    ("How does this moment compare to a typical morning?",
     "comparison", "relative_state"),
    ("Is my energy level normal for this time of day?",
     "comparison", "circadian_comparison"),
    ("Would I be more productive waiting until tomorrow morning?",
     "reasoning", "timing_optimization"),
    ("How should I prioritize my remaining tasks today?",
     "reasoning", "prioritization"),
    ("What kind of tasks should I tackle right now given my state?",
     "reasoning", "task_matching"),
    ("Should I push through or call it a day?",
     "reasoning", "endurance_check"),
]


def _make_temporal_context_block(dt: datetime, encoder: PulseEncoder,
                                  deadline_str: Optional[str], events: int,
                                  sleep: float) -> str:
    """Build the temporal context string that gets injected into prompts."""
    context = {}
    if deadline_str:
        context["deadline"] = deadline_str
    context["events_today"] = events
    context["sleep_hours"] = sleep

    tc = encoder.get_temporal_context(dt, context)

    # Urgency detail
    if deadline_str:
        dl = datetime.fromisoformat(deadline_str)
        delta = dl - dt
        hours_left = delta.total_seconds() / 3600
        if hours_left > 0:
            urg_detail = f"deadline in {hours_left:.1f} hours"
        else:
            urg_detail = f"deadline was {-hours_left:.1f} hours ago (OVERDUE)"
    else:
        urg_detail = "no active deadlines"

    lines = [
        f"Current time: {dt.strftime('%A, %B %d %Y at %I:%M %p')}",
        f"Circadian phase: {tc['circadian_phase']}",
        f"Cognitive capacity: {tc['cognitive_capacity']:.0%}",
        f"Energy level: {tc['energy_level']:.0%}",
        f"Urgency: {tc['urgency_level']} ({urg_detail})",
        f"Events today: {events}",
        f"Sleep last night: {sleep:.1f} hours",
        f"Weekend: {'yes' if dt.weekday() >= 5 else 'no'}",
        f"Business hours: {'yes' if (dt.weekday() < 5 and 9 <= dt.hour < 17) else 'no'}",
    ]
    return "\n".join(lines)


def _generate_response(question: str, q_type: str, q_sub: str,
                        dt: datetime, deadline_str: Optional[str],
                        events: int, sleep: float,
                        scenario_desc: str) -> str:
    """Generate a training response that demonstrates temporal awareness.

    These are template-based responses that capture the RIGHT reasoning pattern.
    The fine-tuned model will learn to generalize from these patterns.
    """
    hour = dt.hour
    cog = float(np.interp(hour, range(24), [0.20,0.15,0.12,0.10,0.12,0.18,0.30,0.50,0.70,0.85,0.95,0.92,0.85,0.72,0.68,0.75,0.88,0.90,0.82,0.70,0.55,0.40,0.30,0.25]))
    eng = float(np.interp(hour, range(24), [0.15,0.10,0.08,0.07,0.10,0.20,0.40,0.60,0.75,0.85,0.90,0.88,0.82,0.70,0.65,0.72,0.85,0.88,0.80,0.65,0.50,0.35,0.25,0.18]))

    has_deadline = deadline_str is not None
    if has_deadline:
        hours_left = (datetime.fromisoformat(deadline_str) - dt).total_seconds() / 3600
        is_overdue = hours_left < 0
        is_urgent = hours_left < 3 and hours_left > 0
    else:
        hours_left = None
        is_overdue = False
        is_urgent = False

    is_night = hour < 6 or hour >= 22
    is_peak = 10 <= hour < 12 or 14 <= hour < 17
    is_dip = 12 <= hour < 14
    is_low_sleep = sleep < 6
    is_weekend = dt.weekday() >= 5

    # Build response based on question type and temporal state
    parts = []

    if q_sub == "task_suitability":
        if "complex" in question.lower() or "refactoring" in question.lower() or "deep work" in question.lower():
            if is_peak and not is_low_sleep and cog > 0.8:
                parts.append(f"This is actually an ideal window. Your cognitive capacity is at {cog:.0%} during {('morning' if hour < 12 else 'afternoon')} peak hours.")
                if has_deadline and is_urgent:
                    parts.append(f"However, with your deadline in {hours_left:.1f} hours, starting a complex refactor is risky. Focus on what's due first.")
                elif not has_deadline:
                    parts.append("No pressing deadlines either, so you have the mental space for it. Go for it.")
            elif is_dip:
                parts.append(f"You're in the post-lunch dip right now — cognitive capacity is around {cog:.0%}. Complex tasks tend to have more errors during this phase.")
                parts.append("Consider lighter work for the next hour, then tackle this when afternoon focus returns around 2:30-3pm.")
            elif is_night:
                parts.append(f"It's {dt.strftime('%I:%M %p')} and your cognitive capacity is at {cog:.0%}. Complex refactoring at this hour tends to create more bugs than it fixes.")
                if is_low_sleep:
                    parts.append(f"You've only had {sleep:.0f} hours of sleep. Your error rate is significantly elevated. This is not the time for complex changes.")
                parts.append("Save this for tomorrow morning when you're fresh.")
            elif is_low_sleep:
                parts.append(f"With only {sleep:.0f} hours of sleep, your cognitive capacity is compromised even during what would normally be productive hours.")
                parts.append("Stick to routine tasks and save the complex work for a better day.")
            else:
                parts.append(f"Your cognitive capacity is moderate at {cog:.0%}. You could start, but this isn't your peak window.")
                if has_deadline:
                    parts.append(f"With a deadline in {hours_left:.1f} hours, weigh whether this refactor is essential or if it can wait.")

        elif "creative" in question.lower() or "brainstorm" in question.lower():
            if is_dip or (is_night and not is_low_sleep):
                parts.append("Interestingly, slightly reduced executive function can actually help creative thinking — your inner critic is quieter.")
                parts.append("This post-peak window can be good for brainstorming. Let ideas flow without filtering too hard.")
            elif is_peak:
                parts.append(f"You're at peak cognitive capacity ({cog:.0%}), which is great for analytical creativity but can sometimes over-filter loose brainstorming.")
                parts.append("If you want wild ideas, try dropping your guard a bit. If you want structured creative problem-solving, this is the perfect time.")
            else:
                parts.append(f"Energy is at {eng:.0%} right now. Creative work needs a baseline of engagement to be productive.")
                if eng < 0.3:
                    parts.append("You might struggle to generate much at this energy level. Rest first.")

        elif "break" in question.lower():
            if cog < 0.5 or eng < 0.4:
                parts.append(f"Yes. Your energy is at {eng:.0%} and cognitive capacity at {cog:.0%}. A 15-20 minute break would help you reset.")
            elif is_dip:
                parts.append("You're in the natural post-lunch dip. A short walk or brief rest now aligns with your body's rhythm and will pay off in the afternoon.")
            elif events > 5:
                parts.append(f"With {events} events today, context-switching fatigue is real. A break to decompress would help even if your energy feels okay.")
            else:
                parts.append(f"Your energy ({eng:.0%}) and cognition ({cog:.0%}) are solid right now. If you're in flow, keep going. Break when you feel the momentum drop.")

        elif "difficult conversation" in question.lower():
            if is_peak and not is_low_sleep and not is_urgent:
                parts.append(f"Your cognitive capacity is strong at {cog:.0%}, which helps with emotional regulation and careful communication.")
                parts.append("This is a reasonable window for it.")
            elif is_low_sleep or is_night:
                parts.append(f"With {sleep:.0f} hours of sleep and lower cognitive function ({cog:.0%}), you're more likely to be reactive than reflective.")
                parts.append("Postpone if possible — this conversation deserves your best self.")

        else:
            # Generic task suitability
            parts.append(f"Current state: {cog:.0%} cognitive capacity, {eng:.0%} energy, circadian phase is {'favorable' if is_peak else 'suboptimal'}.")
            if has_deadline and is_urgent:
                parts.append(f"Priority should be your deadline in {hours_left:.1f} hours.")

    elif q_sub == "full_state" or q_sub == "cognitive_state" or q_sub == "work_phase":
        parts.append(f"It's {dt.strftime('%A %I:%M %p')}. ", )
        if is_peak:
            parts.append(f"You're in a peak cognitive window — capacity at {cog:.0%}, energy at {eng:.0%}. This is prime time for demanding work.")
        elif is_dip:
            parts.append(f"You're in the post-lunch circadian dip. Cognition is at {cog:.0%}, energy at {eng:.0%}. Normal — it passes around 2:30-3pm.")
        elif is_night:
            parts.append(f"Deep night hours. Cognition at {cog:.0%}, energy at {eng:.0%}. Your body wants rest, even if your mind is still going.")
        else:
            parts.append(f"Cognitive capacity at {cog:.0%}, energy at {eng:.0%}.")

        if is_low_sleep:
            parts.append(f"Sleep deficit ({sleep:.0f}h) is dragging everything down. Expect ~15-20% more errors and slower processing.")
        if has_deadline:
            if is_overdue:
                parts.append(f"Your deadline passed {-hours_left:.1f} hours ago. High stress state.")
            elif is_urgent:
                parts.append(f"Deadline in {hours_left:.1f} hours. Urgency is high — focused execution mode.")
            else:
                parts.append(f"Deadline in {hours_left:.1f} hours — present but not acute.")
        if events > 6:
            parts.append(f"Heavy schedule today ({events} events). Context-switching cost is accumulating.")

    elif q_sub == "urgency_assessment":
        if is_overdue:
            parts.append(f"Critical. The deadline passed {-hours_left:.1f} hours ago. You're in damage control mode.")
        elif has_deadline and is_urgent:
            parts.append(f"High urgency — {hours_left:.1f} hours until deadline. This should be your only focus right now.")
        elif has_deadline and hours_left and hours_left < 24:
            parts.append(f"Moderate urgency. Deadline is {hours_left:.1f} hours away. Start planning your approach if you haven't already.")
        elif has_deadline:
            parts.append(f"Low urgency for now — deadline is {hours_left:.1f} hours out. But keep it on your radar.")
        else:
            parts.append("No active deadlines. You have the luxury of choosing what to work on based on energy and interest rather than pressure.")

    elif q_sub in ("relative_state", "circadian_comparison"):
        typical_peak_cog = 0.93
        parts.append(f"At {dt.strftime('%I:%M %p')}, typical cognitive capacity is around {cog:.0%}.")
        if is_low_sleep:
            parts.append(f"Your {sleep:.0f} hours of sleep puts you below baseline. On a well-rested day at this hour, you'd be closer to {min(cog+0.15, typical_peak_cog):.0%}.")
        if is_peak:
            parts.append("This is normally a productive window. " + ("You're in good shape to use it." if not is_low_sleep else "The sleep deficit is eating into what should be your best hours."))
        elif is_dip:
            parts.append("The post-lunch dip is universal — everyone's numbers drop here. It's not a you problem, it's biology.")

    elif q_sub == "timing_optimization":
        if cog < 0.5:
            parts.append("Yes, almost certainly. Tomorrow morning between 10-12 would give you roughly double your current cognitive capacity.")
        elif is_peak:
            parts.append("You're actually in a good window right now. Waiting until tomorrow means losing this momentum and context.")
        elif has_deadline and hours_left and hours_left < 18:
            parts.append(f"With the deadline in {hours_left:.1f} hours, waiting isn't an option. Work with what you've got.")
        else:
            parts.append(f"Your current capacity is {cog:.0%}. Tomorrow morning peak would give you ~93%. Whether that difference matters depends on the task complexity.")

    elif q_sub == "prioritization":
        if has_deadline and is_urgent:
            parts.append(f"Deadline work first — you have {hours_left:.1f} hours. Everything else is secondary.")
        elif is_peak:
            parts.append("Use this peak window for your hardest cognitive task. Save emails, meetings, and routine work for the dip.")
        elif is_dip:
            parts.append("Good time for: emails, code review, administrative tasks, planning. Save complex problem-solving for the afternoon peak around 3-4pm.")
        else:
            parts.append(f"At {cog:.0%} cognitive capacity, match your tasks to your state. Routine work now, demanding work during your next peak window.")

    elif q_sub == "task_matching":
        if cog > 0.8 and eng > 0.7:
            parts.append("You're in strong shape. This is the window for: complex debugging, architecture decisions, writing important documents, learning new concepts.")
        elif 0.5 < cog < 0.8:
            parts.append("Moderate capacity. Good for: code review, incremental feature work, documentation updates, team discussions.")
        elif cog < 0.5:
            parts.append("Low cognitive state. Stick to: email triage, filing issues, light reading, planning tomorrow, mechanical refactoring with clear patterns.")
        if is_low_sleep:
            parts.append("With the sleep deficit, add an extra margin — anything that feels borderline complex should wait.")

    elif q_sub == "endurance_check":
        if eng < 0.3 and cog < 0.4:
            parts.append(f"Call it. Energy at {eng:.0%}, cognition at {cog:.0%}. You're past the point of diminishing returns — you're in negative returns territory.")
        elif has_deadline and is_urgent:
            parts.append(f"Push through — {hours_left:.1f} hours to deadline. But take a 5-minute reset first. Splash water on your face, stretch, then focus.")
        elif is_dip and eng > 0.5:
            parts.append("This feels like a wall but it's the circadian dip — it'll pass. A 15-minute break or short walk usually restores enough for the afternoon push.")
        elif events > 7:
            parts.append(f"With {events} events today, you've earned the right to stop. Context-switching has a real cognitive cost that accumulates.")
        else:
            parts.append(f"Energy at {eng:.0%}, cognition at {cog:.0%}. You've got some runway left if the work is engaging. Listen to when you start re-reading the same line.")

    if not parts:
        parts.append(f"Current temporal state: {cog:.0%} cognitive capacity, {eng:.0%} energy, {dt.strftime('%A %I:%M %p')}.")

    return " ".join(parts)


SYSTEM_PROMPT = """You are an AI assistant with temporal awareness through the PULSE temporal embedding system. Before each interaction, you receive a temporal context package describing the current moment — not just the time, but what that time means: circadian phase, cognitive capacity, energy level, urgency, and behavioral context.

Use this temporal awareness naturally in your responses. Don't announce it mechanically — weave it into your reasoning the way a thoughtful colleague would who knows what time it is and what's going on.

Current temporal context:
{temporal_context}"""


class TemporalDataGenerator:
    """Generates training data for temporal-aware LLM fine-tuning."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.encoder = PulseEncoder()

    def generate_example(self) -> Dict:
        """Generate a single training example."""
        # Pick a random scenario
        scenario = self.rng.choice(_CONTEXTS)
        name, dl_offset, events, sleep, desc = scenario

        # Pick a random date and hour
        base_date = datetime(2026, self.rng.randint(1, 12), self.rng.randint(1, 28))
        hour = self.rng.choice(_HOURS)
        minute = self.rng.choice([0, 15, 30, 45])
        dt = base_date.replace(hour=hour, minute=minute)

        # Override hour for scenarios that imply specific times
        if "late_night" in name or "insomnia" in name:
            dt = dt.replace(hour=self.rng.choice([0, 1, 2, 3, 23]))
        elif "early_morning" in name:
            dt = dt.replace(hour=self.rng.choice([5, 6, 7]))
        elif "peak_morning" in name:
            dt = dt.replace(hour=self.rng.choice([10, 11]))
        elif "post_lunch" in name:
            dt = dt.replace(hour=self.rng.choice([13, 14]))

        # Deadline
        deadline_str = None
        if dl_offset is not None:
            deadline = dt + timedelta(hours=dl_offset)
            deadline_str = deadline.isoformat()

        # Build temporal context
        temporal_context = _make_temporal_context_block(
            dt, self.encoder, deadline_str, events, sleep
        )

        # Pick a question
        question, q_type, q_sub = self.rng.choice(_TEMPORAL_QUESTIONS)

        # Generate response
        response = _generate_response(
            question, q_type, q_sub, dt, deadline_str, events, sleep, desc
        )

        # Format as chat
        system = SYSTEM_PROMPT.format(temporal_context=temporal_context)

        return {
            "system": system,
            "user": question,
            "assistant": response,
            "metadata": {
                "scenario": name,
                "timestamp": dt.isoformat(),
                "deadline": deadline_str,
                "events_today": events,
                "sleep_hours": sleep,
                "question_type": q_type,
                "question_sub": q_sub,
            }
        }

    def generate_dataset(self, n: int = 1000, output_path: Optional[str] = None) -> List[Dict]:
        """Generate n training examples."""
        examples = [self.generate_example() for _ in range(n)]

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.endswith(".jsonl"):
                with open(path, "w") as f:
                    for ex in examples:
                        f.write(json.dumps(ex) + "\n")
            else:
                with open(path, "w") as f:
                    json.dump(examples, f, indent=2)

            print(f"Generated {n} examples -> {path}")

        return examples

    def generate_chat_format(self, n: int = 1000, output_path: Optional[str] = None) -> List[Dict]:
        """Generate dataset in chat/messages format for transformers."""
        examples = self.generate_dataset(n)

        chat_examples = []
        for ex in examples:
            chat_examples.append({
                "messages": [
                    {"role": "system", "content": ex["system"]},
                    {"role": "user", "content": ex["user"]},
                    {"role": "assistant", "content": ex["assistant"]},
                ],
                "metadata": ex["metadata"],
            })

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                for ex in chat_examples:
                    f.write(json.dumps(ex) + "\n")
            print(f"Generated {n} chat examples -> {path}")

        return chat_examples


if __name__ == "__main__":
    gen = TemporalDataGenerator()

    # Generate a small sample
    print("=== Sample Training Example ===\n")
    ex = gen.generate_example()
    print(f"SYSTEM:\n{ex['system']}\n")
    print(f"USER: {ex['user']}\n")
    print(f"ASSISTANT: {ex['assistant']}\n")
    print(f"META: {json.dumps(ex['metadata'], indent=2)}")

    # Generate full datasets
    gen.generate_chat_format(2000, "data/temporal_train.jsonl")
    gen.generate_chat_format(200, "data/temporal_eval.jsonl")
