---
license: mit
tags:
  - temporal-embeddings
  - time-encoding
  - experiential-time
  - circadian
  - urgency
  - time2vec
  - pulse
library_name: pulse-temporal
pipeline_tag: feature-extraction
---

# pulse-base-v1

**An embedding model for time. Not timestamps -- time.**

PULSE encodes moments as 128-dimensional vectors that capture not just *when* something happens, but *what that time means* -- urgency, circadian phase, behavioral context, and the felt sense of time.

## What makes this different

Every temporal encoding in AI today treats time as a coordinate. PULSE treats it as an experience:

| Capability | Time2Vec | RoPE | Neural ODE | Calendar | **PULSE** |
|---|---|---|---|---|---|
| Periodicity | ✅ | ✅ | ❌ | ✅ | ✅ |
| Calendar-aware | ❌ | ❌ | ❌ | ✅ | ✅ |
| Context-dependent | ❌ | ❌ | ✅ | ❌ | ✅ |
| Urgency/deadlines | ❌ | ❌ | ❌ | ❌ | ✅ |
| Circadian phase | ❌ | ❌ | ❌ | ❌ | ✅ |
| Experiential time | ❌ | ❌ | ❌ | ❌ | ✅ |

## Architecture

128D embedding from seven fused signal layers:

```
PULSE(t, context) = normalize(concat[
    log_time(t)         * 1.0,   # 8D  - Weber's Law compression
    oscillators(t)      * 0.5,   # 32D - Multi-frequency sinusoids
    circadian(t)        * 1.5,   # 8D  - 24h + 90min biological clock
    calendar(t)         * 0.6,   # 24D - Day/month/season/holiday
    urgency(t,deadline) * 4.0,   # 8D  - Hyperbolic deadline proximity
    temporal_state(h)   * 2.0,   # 32D - Continuous-time event history
    prediction_error(t) * 3.0,   # 16D - Temporal surprise
])
```

## Usage

```python
from pulse_temporal import PulseEncoder

pulse = PulseEncoder()

# Same hour, completely different moments
monday_crunch = pulse.encode("2026-04-13T14:00:00", context={
    "deadline": "2026-04-13T17:00:00",
    "events_today": 6,
    "sleep_hours": 5,
})

saturday_chill = pulse.encode("2026-04-11T14:00:00", context={
    "deadline": None,
    "events_today": 0,
    "sleep_hours": 9,
})

# These are FAR apart in PULSE space despite similar timestamps
pulse.similarity(monday_crunch, saturday_chill)  # ~0.72

# These cluster together -- "crunch time before deadline"
wednesday_crunch = pulse.encode("2026-04-15T10:00:00", context={
    "deadline": "2026-04-15T12:00:00",
    "events_today": 4,
})
pulse.similarity(monday_crunch, wednesday_crunch)  # ~0.78
```

## Install

```bash
pip install pulse-temporal
```

## Version

v0.1.0 -- formula-based encoder (no trained weights). All seven layers use principled formulas from neuroscience and behavioral economics. Trained model (v0.3) will use contrastive learning on human activity data.

## Citation

```
@software{pulse_temporal,
  title={pulse-temporal: Experiential Time Embeddings for AI},
  author={Morales, Lalo Adrian},
  year={2026},
  url={https://github.com/lalomorales22/pulse-temporal}
}
```
