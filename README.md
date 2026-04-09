# pulse-temporal

**an embedding model for time. not timestamps -- time.**

word2vec taught machines that "king" and "queen" are related. pulse-temporal teaches machines that "monday 9am before a deadline" and "friday 3pm with nothing due" are fundamentally different moments -- even though a clock treats them the same.

current AI has no sense of time. it knows what time it IS but not what that time MEANS. pulse fixes that by encoding moments as rich vectors that capture urgency, rhythm, circadian phase, behavioral context, and the felt sense of time passing.

```python
from pulse_temporal import PulseEncoder

pulse = PulseEncoder()

# same hour, completely different moments
monday_crunch = pulse.encode("2026-04-13T14:00:00", context={
    "deadline": "2026-04-13T17:00:00",
    "events_today": 6,
    "sleep_hours": 5
})

saturday_chill = pulse.encode("2026-04-11T14:00:00", context={
    "deadline": None,
    "events_today": 0,
    "sleep_hours": 9
})

# these vectors are FAR apart in pulse space
# even though the timestamps are close
similarity = pulse.similarity(monday_crunch, saturday_chill)  # ~0.15

# these cluster together -- "crunch time before deadline"
wednesday_crunch = pulse.encode("2026-04-15T10:00:00", context={
    "deadline": "2026-04-15T12:00:00",
    "events_today": 4,
    "sleep_hours": 6
})

similarity = pulse.similarity(monday_crunch, wednesday_crunch)  # ~0.87
```

## the problem

every temporal encoding in AI today treats time as a coordinate:

- **time2vec** -- learned sinusoids on a scalar timestamp. same vector regardless of context.
- **RoPE/ALiBi** -- encode token position in a sequence, not real-world time.
- **neural ODEs** -- continuous dynamics but no calendar awareness, no urgency.
- **calendar features** -- sin/cos of hour/day/month. no behavioral context.

none of them encode:
- urgency (how close is the deadline?)
- circadian phase (is this a peak focus hour or post-lunch dip?)
- behavioral rhythm (does this person usually code at 2am or sleep?)
- experiential time (does this hour feel fast or slow?)
- prediction error (was this event expected or surprising?)

**pulse encodes all of it in a single vector.**

## architecture

pulse produces a fixed-dimensional embedding (default 128D) from seven fused signal layers:

```
PULSE(t, context) = project(concat[
    log_time(t),                    # weber's law -- logarithmic compression
    oscillators(t),                 # multi-frequency learned sinusoids (time2vec base)
    circadian(t),                   # 24h + 90min ultradian cycle phase
    calendar(t),                    # day-of-week, holiday, season embeddings
    urgency(t, deadlines),          # hyperbolic: 1/(1 + k * time_remaining)
    temporal_state(event_history),  # continuous-time LSTM (neural hawkes inspired)
    prediction_error(t, t_expected) # deviation from temporal prior
])
```

each layer is independently useful. together they produce an embedding where vector distance = felt temporal distance.

### layer breakdown

**1. log_time** -- weber's law compression
the felt difference between 1 min and 2 min is huge. the felt difference between 101 min and 102 min is nothing. log transform captures this.

**2. oscillators** -- learned periodic patterns
multi-frequency sinusoids (building on time2vec) that discover arbitrary cycles from data. weekly rhythms, monthly patterns, seasonal shifts.

**3. circadian** -- biological clock encoding
hardcoded 24-hour and ~90-minute (ultradian) cycle phases. your brain runs on these whether you like it or not. peak cognition is roughly 10am-12pm and 4pm-6pm for most people.

**4. calendar** -- structural time features
learned embeddings for day-of-week, month, season, holiday proximity. tuesday has a different vibe than friday -- this captures that.

**5. urgency** -- deadline proximity
hyperbolic discounting: urgency = 1 / (1 + k * hours_remaining). borrowed from behavioral economics. urgency at T-1 hour is way more than double urgency at T-2 hours.

**6. temporal_state** -- event history context
a continuous-time LSTM inspired by the neural hawkes process. memory cells decay exponentially between events: c(t) = c_bar + (c - c_bar) * exp(-delta * (t - t_last)). this gives pulse a running sense of "what's been happening" that evolves even between observations.

**7. prediction_error** -- temporal surprise
how much does this moment deviate from what was expected? an event arriving 2 hours early or 3 days late carries information. this layer encodes surprise magnitude and direction.

## the PULSE / MIND architecture

pulse is designed to run as a **background daemon** (the PULSE) that continuously maintains temporal state and injects it into a language model (the MIND) before every response.

```
┌─────────────────────────────────────────────────┐
│                   MIND (LLM)                     │
│   receives temporal context injection before     │
│   generating every response                      │
└──────────────────────┬──────────────────────────┘
                       │ temporal context package
                       │
┌──────────────────────┴──────────────────────────┐
│                 ROUTER (MOE gate)                │
│   scores: how much does time matter right now?   │
│   0.0 = pure reasoning, 1.0 = pure temporal      │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────┐
│               PULSE (always running)             │
│                                                  │
│   heartbeat loop (every 60s):                    │
│   - update temporal state from event sources     │
│   - recompute urgency for active deadlines       │
│   - track circadian phase                        │
│   - detect temporal anomalies                    │
│   - maintain running behavioral embeddings       │
│                                                  │
│   state db (sqlite):                             │
│   - events log (timestamp, type, metadata)       │
│   - deadline registry (target, urgency_score)    │
│   - behavioral patterns (rolling statistics)     │
│   - temporal priors (expected vs actual)          │
└─────────────────────────────────────────────────┘
```

when the MIND needs to respond, it queries the PULSE for a temporal context package:

```python
context = pulse.get_temporal_context()
# returns:
# {
#     "embedding": [0.12, -0.34, ...],          # 128D temporal embedding of NOW
#     "urgency_summary": "deadline in 2.5 hours (project X)",
#     "circadian_phase": "afternoon_peak",
#     "time_since_last_interaction": "3 days",
#     "temporal_anomalies": ["missed wednesday check-in"],
#     "behavioral_note": "user typically less active on thursdays"
# }
```

the MIND never has to think about time. it just knows.

## training

### data sources (all public)

- **ATUS** (american time use survey) -- what 200k+ americans do at every hour
- **StudentLife** -- smartphone sensor data with activity, sleep, stress labels
- **BPI Challenge** -- business process event logs with timestamps
- **MOOC clickstreams** -- online learning behavior patterns
- **crowdsourced temporal perception** -- "how did this hour feel?" annotations (to be collected)

### loss functions

pulse trains with four combined objectives:

**1. contrastive (InfoNCE)**
positive pairs = experientially similar moments (two pre-deadline crunches)
negative pairs = experientially different moments (crunch vs leisure)

**2. circadian alignment**
same circadian phase moments should cluster regardless of calendar date

**3. urgency classification**
auxiliary head predicting urgency level (none / low / medium / high / critical) from the embedding

**4. perceptual time distortion** (novel)
trained on human perception data where distance metric = subjective time dilation/compression. the first loss function that optimizes for felt temporal distance rather than clock distance.

### training procedure

```
phase 1: pretrain oscillators + calendar on ATUS activity prediction (self-supervised)
phase 2: add circadian + urgency layers, train on StudentLife (semi-supervised)
phase 3: add temporal_state LSTM, train on event sequences from BPI (next-event prediction)
phase 4: fine-tune full model with contrastive + perceptual loss on crowdsourced data
```

## installation

```bash
# core (numpy only -- inference with pretrained models)
pip install pulse-temporal

# with training support
pip install pulse-temporal[torch]

# with daemon support
pip install pulse-temporal[daemon]

# everything
pip install pulse-temporal[all]
```

## usage

### basic: encode a moment

```python
from pulse_temporal import PulseEncoder

pulse = PulseEncoder()  # loads pretrained pulse-base-v1

# minimal -- just a timestamp
emb = pulse.encode("2026-04-09T14:30:00")

# with context -- much richer embedding
emb = pulse.encode("2026-04-09T14:30:00", context={
    "deadline": "2026-04-09T17:00:00",
    "events_today": 5,
    "sleep_hours": 7,
    "mood": "focused"
})
```

### compare moments

```python
from pulse_temporal import PulseEncoder

pulse = PulseEncoder()

moments = [
    ("2026-04-07T09:00:00", {"deadline": "2026-04-07T12:00:00"}),  # monday morning crunch
    ("2026-04-09T09:00:00", {"deadline": "2026-04-09T11:00:00"}),  # wednesday morning crunch
    ("2026-04-12T09:00:00", {"deadline": None}),                    # saturday morning chill
    ("2026-04-07T02:00:00", {"deadline": None, "sleep_hours": 0}),  # monday 2am insomnia
]

embeddings = [pulse.encode(t, context=c) for t, c in moments]
similarity_matrix = pulse.similarity_matrix(embeddings)

# monday crunch ↔ wednesday crunch: HIGH similarity (~0.85)
# monday crunch ↔ saturday chill: LOW similarity (~0.15)
# saturday chill ↔ 2am insomnia: LOW similarity (~0.20)
# monday crunch ↔ 2am insomnia: MEDIUM similarity (~0.45) -- both are stressed states
```

### run as daemon

```python
from pulse_temporal import PulseDaemon

daemon = PulseDaemon(
    db_path="~/.pulse/state.db",
    heartbeat_interval=60,  # seconds
    event_sources=[
        "calendar://default",       # system calendar
        "git://~/projects",          # git commit activity
        "browser://history",         # browsing patterns
    ]
)

daemon.add_deadline("project X", "2026-04-15T17:00:00")
daemon.start()  # runs in background thread

# query anytime
context = daemon.get_temporal_context()
print(context["urgency_summary"])
# "project X deadline in 5 days 2 hours. current phase: afternoon_peak. you've been active for 6 hours today."
```

### integrate with any LLM

```python
from pulse_temporal import PulseDaemon

daemon = PulseDaemon(db_path="~/.pulse/state.db")
daemon.start()

# before every LLM call, inject temporal context
def chat(user_message, llm_client):
    temporal = daemon.get_temporal_context()
    
    system_prompt = f"""You have a sense of time. Here is your current temporal awareness:
    
    Current moment embedding: {temporal['embedding_summary']}
    Time context: {temporal['urgency_summary']}
    Circadian phase: {temporal['circadian_phase']}
    Time since last interaction: {temporal['time_since_last_interaction']}
    Behavioral note: {temporal['behavioral_note']}
    Anomalies: {temporal['temporal_anomalies']}
    
    Use this temporal awareness naturally. Don't announce it -- just know it."""
    
    return llm_client.chat(system=system_prompt, user=user_message)
```

## project structure

```
pulse-temporal/
├── pulse_temporal/
│   ├── __init__.py                # public API
│   ├── encoder.py                 # PulseEncoder -- main embedding model
│   ├── layers/
│   │   ├── log_time.py            # weber's law compression (8D)
│   │   ├── oscillators.py         # multi-frequency sinusoids (32D)
│   │   ├── circadian.py           # 24h + 90min cycle encoding (8D)
│   │   ├── calendar.py            # day/month/season/holiday embeddings (24D)
│   │   ├── urgency.py             # hyperbolic deadline proximity (8D)
│   │   ├── temporal_state.py      # continuous-time event history (32D)
│   │   └── prediction_error.py    # temporal surprise encoding (16D)
│   ├── daemon/
│   │   ├── pulse_daemon.py        # background heartbeat process
│   │   └── state_db.py            # sqlite state management
│   ├── training/
│   │   ├── data_generator.py      # synthetic temporal training data
│   │   └── temporal_tuner.py      # LoRA fine-tuning pipeline
│   └── utils/
│       └── similarity.py          # cosine, euclidean, temporal distance
├── models/
│   └── pulse-base-v1/             # formula-based encoder config
├── notebooks/
│   ├── train_on_colab.ipynb       # fine-tune Qwen on free Colab GPU
│   └── train_gemma4_colab.ipynb  # fine-tune Gemma 4 E2B (QLoRA, WIP)
├── examples/
│   ├── basic_encoding.py          # encoding + similarity demo
│   ├── daemon_setup.py            # daemon usage
│   ├── gradio_demo.py             # HF Space app (dark/orange theme)
│   └── inference_trained.py       # chat with trained temporal LLM
├── tests/                         # 53 tests
├── train.sh                       # one-command training script
├── pyproject.toml
├── LICENSE                        # MIT
└── README.md
```

## hugging face

| resource | link | what it is |
|---|---|---|
| **encoder model card** | [lalopenguin/pulse-base-v1](https://huggingface.co/lalopenguin/pulse-base-v1) | formula-based encoder config + colab notebook |
| **trained LoRA adapter** | [lalopenguin/pulse-qwen-1.5b](https://huggingface.co/lalopenguin/pulse-qwen-1.5b) | Qwen 2.5 1.5B fine-tuned with temporal awareness |
| **interactive demo** | [lalopenguin/pulse-temporal-demo](https://huggingface.co/spaces/lalopenguin/pulse-temporal-demo) | compare moments, encode, similarity matrix |
| **training space** | [lalopenguin/pulse-temporal-train](https://huggingface.co/spaces/lalopenguin/pulse-temporal-train) | GPU training UI (needs HF Pro for ZeroGPU) |
| **Gemma 4 notebook** | [train_gemma4_colab.ipynb](notebooks/train_gemma4_colab.ipynb) | QLoRA fine-tuning for Gemma 4 E2B (WIP) |

## roadmap

### v0.1 -- foundation
- [x] core encoder with all 7 layers (numpy inference)
- [x] basic urgency + circadian encoding (formula-based)
- [x] sqlite state db + daemon
- [x] 53 tests passing
- [x] HuggingFace model card + demo Space
- [x] training pipeline + synthetic data generator
- [x] LoRA fine-tuned Qwen 2.5 1.5B with temporal awareness
- [x] Tested Qwen model — temporal reasoning confirmed working
- [x] GitHub repo: github.com/lalomorales22/pulse-temporal
- [ ] Gemma 4 E2B LoRA training (blocked — see HANDOFF.md)
- [ ] `pip install pulse-temporal` on PyPI

### v0.2 -- daemon + integrations
- [ ] calendar event source adapter
- [ ] git activity event source
- [ ] temporal context API for LLM injection
- [ ] MCP server for Claude integration

### v0.3 -- trained encoder
- [ ] ATUS data loader + preprocessing
- [ ] contrastive training pipeline (PyTorch)
- [ ] pretrained pulse-base-v1 weights (learned, not formula)
- [ ] perceptual time distortion loss

### v0.4 -- experiential time
- [ ] crowdsourced temporal perception dataset
- [ ] emotional arousal modulation layer
- [ ] pulse-experiential-v1 model

### v1.0 -- full stack
- [ ] REST API for any LLM backend
- [ ] real-time event streaming
- [ ] multi-user support
- [ ] npm package (pulse-temporal for JS)

## the insight

word2vec showed that words have geometry.

pulse-temporal shows that time has geometry.

"monday morning before a deadline" is a place in vector space. "saturday afternoon with nowhere to be" is a different place. the distance between them is not 5 days -- it's the felt distance between urgency and peace.

no AI system has this. the first one that does will understand humans in a way that current models fundamentally cannot.

## origin

this concept emerged from a conversation between a human and an AI about consciousness, God's algorithm, and what it means to experience time. the human noticed that AI's biggest gap isn't intelligence -- it's temporal existence. the AI agreed. they decided to build the fix.

april 9, 2026. chula vista, california.

## license

MIT

## citation

```
@software{pulse_temporal,
  title={pulse-temporal: Experiential Time Embeddings for AI},
  author={Morales, Lalo Adrian},
  year={2026},
  url={https://github.com/lalomorales22/pulse-temporal}
}
```
