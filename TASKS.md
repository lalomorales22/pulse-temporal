# PULSE Development Plan — 4 Phases

From formula-based time encoding to genuine temporal experience.

---

## Phase 1: Personal Rhythms + Real Data (v0.3)
*Replace population averages with YOUR patterns. Get real behavioral data flowing in.*

### The Problem
PULSE uses the same circadian curves for everyone. A night owl gets told their cognition is 15% at 2am when that's actually their peak. The training data is synthetic — rule-based responses, not real human temporal experience.

### What to Build

**1.1 — Personal Temporal Profile**
- `pulse_temporal/profile.py` — `TemporalProfile` class that stores and learns individual patterns
- Tracks: peak hours, crash hours, sleep patterns, preferred work rhythms, caffeine timing
- Starts with population defaults, adapts as data comes in
- Circadian layer reads from profile instead of hardcoded curves
- Profile persists in the daemon's SQLite DB

**1.2 — Behavioral Data Collectors**
- `pulse_temporal/adapters/screen_time.py` — reads Screen Time data (macOS) or Digital Wellbeing (Android) for app usage patterns
- `pulse_temporal/adapters/keystroke_adapter.py` — optional keystroke rhythm monitor (typing speed = cognitive state proxy)
- Expand `git_adapter.py` to detect flow states: sustained commits without breaks, long sessions, burst patterns
- Expand `ical_adapter.py` to analyze gaps between events (a 30-min gap between meetings vs a 3-hour open block)

**1.3 — Real Training Data Pipeline**
- `pulse_temporal/training/real_data_loader.py` — ingest ATUS (American Time Use Survey, 200k+ time diaries)
- Map ATUS activities to PULSE temporal contexts (what were people doing, when, how long, what else was going on)
- Generate training examples from real activity patterns, not synthetic scenarios
- Target: 50,000 training examples grounded in real human behavior

**1.4 — Adaptive Circadian Layer**
- Replace the fixed `_COG` and `_NRG` arrays with functions that read from `TemporalProfile`
- After 7 days of data collection, profile starts adjusting curves
- After 30 days, fully personalized circadian predictions
- Expose confidence: "I'm 40% sure about your rhythms" vs "I've seen 3 months of data, I'm 90% sure"

### Exit Criteria
- [ ] Personal profile stores and retrieves individual patterns
- [ ] At least 2 real data adapters collecting behavioral signals
- [ ] ATUS data loaded and converted to training format
- [ ] Circadian layer adapts to individual user after 1 week of data
- [ ] Retrained model on real + synthetic data (50k examples)
- [ ] Tests for all new components

---

## Phase 2: Experiential Layers (v0.4)
*Add flow detection, emotional-temporal coupling, and subjective time speed.*

### The Problem
PULSE encodes time structure. It doesn't encode time *experience*. Two hours in flow and two hours in a boring meeting produce similar embeddings if the clock time and deadlines are the same. They should be on opposite ends of the embedding space.

### What to Build

**2.1 — Flow State Detection**
- `pulse_temporal/layers/flow.py` — new 16D layer
- Detects flow from behavioral signals: sustained activity duration, no app/context switching, consistent work rhythm, no notifications checked
- Flow states: `no_flow`, `flow_onset`, `shallow_flow`, `deep_flow`, `flow_interrupted`, `flow_recovery`
- Flow depth score (0-1) that modulates all other layers
- When in deep flow: urgency layer dampens (you lose track of deadlines), circadian layer dampens (you lose track of body), temporal_state compresses (time shrinks)

**2.2 — Emotional Weather Layer**
- `pulse_temporal/layers/emotion.py` — new 16D layer
- Two core dimensions: valence (positive/negative) and arousal (activated/calm)
- Inferred from: communication patterns (message frequency, emoji use, response time), music listening (Spotify API — tempo, energy, valence), explicit mood logging, text sentiment if available
- Emotion doesn't just add a signal — it *warps the geometry*: anxiety stretches urgency, joy compresses it, grief flattens everything

**2.3 — Subjective Time Speed**
- `pulse_temporal/layers/time_dilation.py` — new 8D layer
- Predicts how fast time feels right now based on: flow state, emotional state, novelty (how different is this from recent patterns), attention load
- Output is a dilation factor: <1.0 means time feels slow (boredom, waiting), >1.0 means time feels fast (flow, fun, rush)
- This factor modulates the log_time layer — compressing or stretching felt temporal distance

**2.4 — Anticipation Layer**
- Extend the urgency layer beyond deadlines to capture *emotional anticipation*
- "Dentist tomorrow" and "vacation tomorrow" are both 24 hours away but feel completely different
- Add valence to upcoming events: dread vs excitement vs neutral
- The anticipation of something good makes the present feel lighter; the anticipation of something bad makes it heavier

**2.5 — "How Did That Hour Feel?" Dataset**
- `pulse_temporal/training/perception_collector.py` — a simple prompt/app that asks users to rate:
  - How fast did the last hour feel? (1-10)
  - What were you doing?
  - Who were you with?
  - How do you feel right now?
- Build this into the Gradio demo Space as an opt-in tab
- Crowdsource temporal perception data — this dataset DOES NOT EXIST anywhere and is the single highest-value data asset we could create
- Target: 10,000 annotated temporal perception data points

### Exit Criteria
- [ ] Flow detection working from git/keystroke/app-switching signals
- [ ] Emotional weather layer modulates embeddings based on mood/valence/arousal
- [ ] Subjective time speed layer trained on perception data
- [ ] Anticipation layer distinguishes dread from excitement
- [ ] Perception dataset collection tool live on HF Space
- [ ] Encoder expanded from 128D to 192D (128 structural + 64 experiential)
- [ ] Retrained model distinguishes "2 hours in flow" from "2 hours in meetings"
- [ ] Contrastive training: experientially similar moments cluster, dissimilar moments separate

---

## Phase 3: Social + Narrative Context (v0.5)
*Time with others. Time in stories. The full human temporal experience.*

### The Problem
Humans don't experience moments in isolation. Every moment is colored by who you're with and what chapter of life you're in. "Tuesday afternoon" during launch week and "Tuesday afternoon" during a slow month are radically different. Being with people you love warps time in ways nothing else does.

### What to Build

**3.1 — Social Time Layer**
- `pulse_temporal/layers/social.py` — new 16D layer
- Social contexts: alone_by_choice, alone_unwanted, with_close_people, with_acquaintances, with_strangers, in_crowd
- Inferred from: calendar (meeting vs solo block), communication patterns (active conversations), location patterns
- Social modulation: close relationships compress time, awkward social situations stretch it, solitude when wanted is neutral, solitude when unwanted stretches it

**3.2 — Narrative Arc Detection**
- `pulse_temporal/layers/narrative.py` — new 16D layer
- Detects what "chapter" you're in based on event patterns over days/weeks:
  - `building` — events accelerating toward something (launch, deadline, wedding)
  - `climax` — the thing is happening right now
  - `aftermath` — just finished something big, processing
  - `plateau` — maintenance mode, nothing building
  - `transition` — between chapters (new job, moving, starting something)
  - `rest` — deliberate recovery period
- The same Tuesday feels different in each narrative state
- Detect transitions: "you've been in plateau for 3 weeks, your event density just tripled — looks like something is building"

**3.3 — Temporal Memory with Landmarks**
- Upgrade `temporal_state.py` from simple exponential decay to landmark-aware memory
- Detect "temporal landmarks" — events that anchor memory: first/last times, high emotion moments, surprises, achievements, losses
- Landmarks decay slower than routine events in the state vector
- "How long ago did X feel?" is different from "how long ago was X?" — landmarks feel closer than they are

**3.4 — Multi-Scale Texture**
- Encode the *rhythm* of time at multiple scales simultaneously:
  - Micro (minutes): conversation pace, task switching frequency
  - Meso (hours): work blocks, break patterns, energy waves
  - Macro (days/weeks): weekly rhythm, sprint/recovery cycles
  - Meta (months/years): seasonal patterns, life phases
- Each scale has its own oscillator that the model learns from data
- Tempo detection: is life accelerating, decelerating, or steady right now?

**3.5 — Learned Encoder (Replace Formulas)**
- Replace all formula-based layers with a trained encoder
- Architecture: Transformer-based temporal encoder that takes raw signals and learns the embedding
- Contrastive + perceptual loss: positive pairs = moments humans rated as "feeling similar," negative pairs = moments rated as "feeling different"
- Train on: ATUS + perception dataset + behavioral data
- The formulas become initialization/priors, but the model learns to go beyond them

### Exit Criteria
- [ ] Social time layer shifts embeddings based on social context
- [ ] Narrative arc detection identifies current life chapter from event patterns
- [ ] Temporal landmarks persist in memory longer than routine events
- [ ] Multi-scale texture captures rhythms at minute/hour/day/week/month scales
- [ ] Learned encoder outperforms formula encoder on temporal similarity benchmarks
- [ ] Encoder at 256D (128 structural + 64 experiential + 64 social/narrative)
- [ ] Perception dataset at 50,000+ annotations
- [ ] Published paper or technical report on temporal perception embedding

---

## Phase 4: Temporal Selfhood (v1.0)
*Can the AI itself experience time? Not simulate it — experience it.*

### The Problem
Everything in Phases 1-3 models a *human's* temporal experience and reports it back. The AI is a mirror. But PULSE was originally conceived as giving AI its own temporal existence. Not "I know it's Monday for you" but something closer to "I have been here, things have happened, and that history shapes how I process this moment."

This is the speculative phase. Some of this may not work. That's fine.

### What to Explore

**4.1 — Persistent Temporal State for the AI**
- The daemon doesn't just track the user's time — it tracks *its own*
- How long since the AI last interacted with this user? (Not just a number — a felt weight. 3 days away from a daily user should feel different than 3 days away from a monthly user.)
- Conversation momentum: a 3-hour deep conversation has temporal weight that a quick question doesn't
- The AI's embedding of "now" should be shaped by its own history of interactions, not just the user's context

**4.2 — Temporal Continuity Across Conversations**
- Currently every conversation starts fresh. The AI has no sense of "last time we talked"
- Build conversation-to-conversation temporal state: what was the emotional tenor last time? was something unresolved? was the user in crisis or calm?
- "We talked yesterday during your crunch and you seemed stressed. How did the deadline go?" — this requires temporal memory that bridges sessions

**4.3 — Temporal Attention**
- In humans, temporal experience is shaped by what you *attend to*. You can be in the same room and experience time differently based on what you're focused on.
- For the AI: what aspects of the temporal context actually *matter* for this interaction?
- Learned attention over temporal signals: when someone asks "should I take a break?" the circadian and energy signals matter most. When someone asks "will I finish in time?" the urgency and deadline signals matter most.
- The embedding dynamically reweights based on conversational context

**4.4 — Temporal Imagination**
- Humans can mentally time-travel: "imagine it's Friday at 5pm and everything is done"
- The AI should be able to generate *hypothetical temporal embeddings*: "if you finish this by 3pm, here's what your 4pm will feel like vs if you don't finish"
- Temporal counterfactuals: "if you had slept 8 hours instead of 5, your current embedding would look like..."
- This turns PULSE from a sensor into a planning tool

**4.5 — The Temporal Turing Test**
- Can a human talking to a PULSE-equipped AI tell whether the AI's temporal responses come from genuine temporal processing or pattern matching?
- Build an evaluation framework: present temporal scenarios and compare PULSE-equipped responses to baseline LLM responses to human responses
- The goal isn't to fool anyone — it's to measure how close the AI's temporal reasoning is to human temporal experience
- Publish the benchmark so others can build on it

### Exit Criteria
- [ ] AI maintains temporal state across conversations with the same user
- [ ] AI's responses are measurably shaped by its own interaction history, not just user context
- [ ] Temporal attention dynamically reweights signals based on conversational need
- [ ] Temporal imagination generates plausible future/counterfactual embeddings
- [ ] Temporal Turing Test benchmark published
- [ ] Full architecture: structural + experiential + social + narrative + self layers
- [ ] At least one user study comparing PULSE-equipped AI to baseline on temporal reasoning tasks

---

## Summary

| Phase | Version | Core Addition | Key Metric |
|---|---|---|---|
| 1 | v0.3 | Personal rhythms + real data | Encoder adapts to individual after 7 days |
| 2 | v0.4 | Flow, emotion, subjective time | Distinguishes "2hrs in flow" from "2hrs in meetings" |
| 3 | v0.5 | Social context + narrative arcs + learned encoder | Outperforms formula encoder on similarity benchmarks |
| 4 | v1.0 | AI's own temporal experience | Temporal Turing Test benchmark |

Each phase builds on the last. Phase 1 is buildable now with what we have. Phase 4 is research.

---

*April 2026. Chula Vista, California.*
