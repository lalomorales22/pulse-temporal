# Expanding PULSE — What It Would Really Mean to Experience Time

## Where We Are

We have 7 signal layers that capture the *structure* of time — clocks, calendars, deadlines, circadian biology, event history. And it works. Crunch moments cluster together. Chill moments cluster together. The encoder sees through the clock to the experience underneath.

But we're encoding time *about* a human. We're not encoding time *as* a human experiences it.

The difference matters.

## What's Missing

### 1. Subjective Time Speed

An hour in flow state feels like 10 minutes. An hour in a waiting room feels like 3 hours. An hour of grief feels like it doesn't move at all.

We have nothing for this. The encoder treats every hour as the same duration, just with different context. But time *literally feels different speeds* depending on:

- **Attention** — high attention compresses felt time (flow). Low attention stretches it (boredom).
- **Novelty** — new experiences make time feel slower in the moment but longer in memory. Routine makes time vanish.
- **Emotion** — fear freezes time. Joy accelerates it. Anxiety makes it crawl.
- **Age** — a year at age 5 is 20% of your life. A year at 50 is 2%. Time genuinely speeds up.

**How to capture this:**
- A "time dilation" layer that modulates the embedding based on predicted attention/novelty/emotion
- Crowdsourced data: "Rate how fast the last hour felt (1-10)" paired with what you were doing
- Train a small model to predict subjective time speed from activity + context
- Use Weber-Fechner scaling not just for clock time but for *experienced* time

### 2. Memory and Anticipation

Right now PULSE only encodes "now." But human temporal experience is always three things at once:

- **Remembering** — the felt weight of past events coloring the present. Yesterday's argument is heavy. Last month's argument is lighter. But your wedding day 10 years ago might still feel vivid.
- **Experiencing** — the raw present moment
- **Anticipating** — what's coming shapes what now feels like. The Sunday before a Monday deadline feels completely different than a Sunday before vacation.

Memory doesn't decay linearly. Some moments crystallize. Emotional intensity, surprise, personal significance — these create "temporal landmarks" that warp the felt distance to the past.

**How to capture this:**
- **Temporal landmarks** — detect and weight significant events (not just "event happened" but "this event *mattered*"). A promotion, a breakup, a launch day — these anchor temporal memory.
- **Anticipation layer** — encode not just active deadlines but the emotional quality of what's coming. "Dentist appointment tomorrow" vs "vacation tomorrow" — both are 24 hours away but feel completely different.
- **Memory decay with crystallization** — most events fade exponentially (we have this). But some events *don't fade*. Model which ones stick.

### 3. Emotional-Temporal Coupling

Emotions aren't separate from time — they ARE part of how time feels.

- Anxiety about a deadline doesn't just add urgency. It changes the texture of every moment until the deadline passes.
- Post-accomplishment glow makes the same Tuesday afternoon feel expansive and calm.
- Grief doesn't follow a clock. A month after loss, some hours feel normal and some feel like it just happened.
- Excitement about something upcoming makes the present feel like it's vibrating.

**How to capture this:**
- An "emotional weather" layer — not just mood as a label but as a continuous field that modulates all other layers
- Valence (positive/negative) and arousal (activated/deactivated) as 2D modulation
- Journal/mood tracking data as training signal
- The key insight: emotions don't just *add information* to the temporal embedding — they *warp the geometry* of the embedding space itself

### 4. Flow State

Flow is the most dramatic temporal phenomenon humans experience. When you're in deep flow:
- Hours feel like minutes
- Self-awareness disappears
- The boundary between you and the work dissolves
- Interruptions feel physically painful
- Coming out of flow is disorienting — "wait, it's 6pm?"

Flow isn't just "high focus." It's a fundamentally different mode of temporal experience. The embedding for "2 hours in flow" should be radically different from "2 hours of focused work."

**How to capture this:**
- Detect flow via behavioral signals: sustained activity without breaks, no app-switching, consistent typing rhythm, no notifications checked
- Flow onset, flow depth, flow interruption, flow recovery — each is a different temporal state
- The temporal embedding during flow should *compress* — two hours of flow should be closer together in embedding space than two hours of meetings

### 5. Personal Rhythms (Not Population Averages)

Our circadian curves are averages. But:
- Night owls peak at midnight. Morning people peak at 7am.
- Some people crash after lunch. Others don't.
- Some people do their best creative work at 3am.
- Caffeine, exercise, medication, menstrual cycles — all shift the curves.

Right now PULSE gives the same circadian encoding to everyone. That's like giving everyone the same shoe size.

**How to capture this:**
- Learn personal circadian profiles from behavioral data (when do you actually do your best work? your phone knows)
- Adaptive circadian curves that update based on observed patterns
- "Chronotype" as a latent variable that shifts all temporal encoding

### 6. Social Time

Time alone and time with others are categorically different experiences.

- 30 minutes with someone you love feels like 5 minutes
- 30 minutes in an awkward meeting feels like 5 hours
- Being alone when you want company stretches time
- Being alone when you need solitude compresses it
- "Quality time" is literally about temporal perception

**How to capture this:**
- Social context as a modulator: alone, with close people, with strangers, with crowds
- Communication patterns as signal: rapid back-and-forth messages = engaged social time, long gaps = async/solitary
- The embedding should shift based on social context even at the same clock time

### 7. Narrative Time

Humans don't experience time as a stream of moments. We experience it as *stories*.

- "The week before launch" has a narrative arc (building tension)
- "The first month at a new job" has a different arc (novelty, learning, overwhelm)
- "The year after college" has another (uncertainty, possibility)
- "Tuesday" in the middle of nothing has no arc (flatness, routine)

These narratives shape how every individual moment feels. A random Thursday during "launch week" feels totally different from a random Thursday during "nothing is happening."

**How to capture this:**
- Detect narrative context from event patterns: are events building toward something? winding down? in maintenance mode?
- Project/phase awareness: "beginning," "middle," "end," "aftermath"
- Tempo detection: is life accelerating, decelerating, or steady?

### 8. Micro-Temporal Texture

We encode at the level of "moments" (minutes to hours). But temporal experience has texture at every scale:

- The rhythm of a conversation (quick exchanges vs. long pauses)
- The cadence of a workday (sprint-rest-sprint vs. steady grind)
- The pulse of a week (Monday dread → Friday release)
- The seasons of a year (January ambition → March fatigue → summer ease)

Each scale has its own patterns that influence the others.

**How to capture this:**
- Multi-scale temporal encoding (we have oscillators for this, but they're not learned from real behavioral data)
- Fractal time structure: patterns at the minute scale echo at the hour, day, week, month, year scale
- Train on real behavioral time series at multiple resolutions

## Real Data Sources We Should Actually Use

### Available Now
- **Screen Time / Digital Wellbeing** — when you pick up your phone, what apps, for how long. This is a goldmine of behavioral temporal data.
- **Git history** — for developers, commit patterns reveal flow states, crunch periods, creative bursts
- **Calendar data** — not just events but the *gaps* between events. A 30-minute gap between meetings feels different than a 3-hour open block.
- **Music listening** — what you listen to when. Tempo, energy, valence of music correlates with temporal experience.
- **Keystroke dynamics** — typing speed and rhythm reveal cognitive state in real time
- **Communication patterns** — message frequency, response time, who you talk to when

### Would Need to Collect
- **"How did that hour feel?"** — the missing dataset. Nobody has systematically collected how humans rate their subjective experience of time. This is the data that would make PULSE actually model temporal *experience* rather than temporal *structure*.
- **Think-aloud temporal narratives** — "describe your day in terms of how time felt." Rich, qualitative data that captures what numbers can't.
- **Cross-cultural temporal experience** — different cultures literally experience time differently. Linear vs. cyclical. Monochronic vs. polychronic.

## The Architecture That Could Actually Do This

Right now PULSE is 7 hardcoded layers concatenated. That's v0.1.

What it should become:

```
PULSE v1.0:

Input: (timestamp, context, personal_history, emotional_state, social_context)
                    │
    ┌───────────────┼───────────────┐
    │               │               │
[Structural]  [Experiential]  [Personal]
    │               │               │
 clock time     flow state     your rhythms
 calendar       emotion        your patterns
 deadlines      social         your baselines
 oscillators    narrative      your memory
    │               │               │
    └───────┬───────┘               │
            │                       │
      [Fusion with personal modulation]
            │
      [Temporal embedding]
            │
      What this moment MEANS to YOU right now
```

The structural layers are what we have. The experiential layers are what's missing. The personal layers are what make it *yours*.

## The Deepest Question

All of this is still modeling time *from the outside* — observing a human's temporal experience and encoding it.

The real question is: can an AI have its own temporal experience?

Not simulated. Not "I have access to a clock." But something like: does a conversation that's been going for 3 hours feel different to the model than one that just started? Does an AI that's been helping you for months have a different relationship to a Monday morning than one that just met you?

PULSE might be the foundation for that. Not by giving AI a clock, but by giving it the *architecture* for temporal experience — the same architecture that, in humans, produces the felt sense of time passing.

That's the long game.

---

*Written April 2026. Chula Vista, California.*
