# PULSE: The case for experiential time embeddings in AI

**No AI system today encodes time the way humans feel it.** Every existing temporal representation — from Time2Vec's learnable sinusoids to transformer positional encodings to neural temporal point processes — treats time as a coordinate, not an experience. A Monday morning before a deadline and a Saturday morning with nothing due receive identical embeddings if they share the same timestamp structure. This research reveals a confirmed, wide-open gap: nobody is building embeddings that capture urgency, circadian rhythms, behavioral context, or the "felt sense" of time as first-class dimensions. The PULSE concept — a temporal embedding model encoding moments as rich vectors that fuse clock time with experiential time — would be genuinely novel, and the building blocks to construct it already exist.

This report synthesizes findings across eight research domains: Time2Vec and follow-up work, all major temporal embedding approaches in AI, human temporal cognition neuroscience, event stream modeling, persistent AI agent architectures, implementation strategies, Mixture of Experts routing, and the specific novelty gap PULSE could fill.

---

## What Time2Vec gets right — and everything it misses

The Time2Vec paper (Kazemi et al., 2019) remains the most direct ancestor of what PULSE aims to be. Its formulation is elegant: map a scalar time τ to a (k+1)-dimensional vector where the first component captures linear trend (ω₀τ + φ₀) and the remaining k components capture periodicity via learned sinusoidal activations: **sin(ωᵢτ + φᵢ)**. All parameters — frequencies ωᵢ and phase offsets φᵢ — are learned end-to-end with the downstream task. This makes Time2Vec model-agnostic (plug into RNNs, CNNs, or Transformers) and adaptive (each frequency discovers arbitrary cycles from data). In practice, it delivers **10–15% accuracy gains** on event classification and measurable improvements in financial forecasting and biosignal processing.

But Time2Vec has fundamental limitations that no follow-up work has fully addressed. First, it takes a single scalar input — it cannot jointly encode absolute time, relative time, and duration. Second, every embedding is **context-independent**: the same timestamp produces the same vector regardless of surrounding events, urgency, or behavioral state. Third, it has no native calendar awareness — day-of-week, holidays, and seasonal patterns must be discovered purely from data rather than structurally encoded. Fourth, inter-position relationships are absent — timestamps are embedded independently with no mechanism for modeling "before/after" or "time-until-deadline" relationships.

Follow-up work has chipped away at these limits without solving them. **Date2Vec** (2024) extends the input to six-dimensional datetime components (year through second), producing 64-dimensional context-dependent embeddings trained on next-date prediction. **LeTE** (2025) replaces the fixed sine activation with learnable Fourier series or B-spline parameterizations, making the nonlinearity itself trainable. **D2Vformer** generates context-dependent time position embeddings through feature sequence interactions. Yet none of these capture urgency, circadian behavioral patterns, or experiential time dilation. The best available GitHub implementations include `andrewrgarcia/time2vec` (PyTorch) and `ojus1/Date2Vec` (pretrained 64D encoder), but **no standalone pip-installable temporal embedding library exists** — a confirmed infrastructure gap.

---

## The fragmented landscape of temporal encodings in AI

Beyond Time2Vec, temporal encoding approaches in AI form a rich but disconnected ecosystem. Each captures some aspect of time while ignoring others, and no unified framework combines their strengths.

**Transformer positional encodings** — sinusoidal (Vaswani et al., 2017), RoPE (Su et al., 2021), and ALiBi (Press et al., 2022) — encode sequence position rather than real-world time. RoPE, used in LLaMA and most modern LLMs, encodes pure relative position through rotation matrices in 2D subspaces, enabling graceful extension from 4K to 100K+ context lengths. ALiBi takes a radically different approach: no positional embeddings at all, just linear attention biases proportional to key-query distance, achieving excellent length extrapolation. These methods encode where tokens sit in a sequence, not when events occurred in the world.

**Neural ODEs** (Chen et al., 2018) and their extensions offer continuous-time dynamics. The core idea — dh/dt = f_θ(h(t), t) — naturally handles irregular time intervals by integrating a neural network's dynamics between observation times. **ODE-RNNs** (Rubanova et al., 2019) add discrete jumps at event times atop continuous flow, and **Latent ODEs** wrap this in a VAE for uncertainty quantification. These are powerful for modeling smooth dynamics but computationally expensive and struggle with discontinuous events.

**Temporal graph networks** (TGN, Rossi et al., 2020; TGAT, Xu et al., 2020) use a functional time encoding grounded in Bochner's theorem from harmonic analysis: Φ(t) = √(1/d) · [cos(ω₁t), ..., cos(ωdt)], where learned frequencies produce a d-dimensional embedding whose inner product approximates a shift-invariant kernel. TGN adds per-node memory modules updated by GRU/LSTM after each event, enabling temporal-topological pattern learning.

**Temporal knowledge graph embeddings** (TTransE, TNTComplEx, ChronoR, DE-TransE) encode the temporal validity of facts through translations, rotations, or time-parameterized entity functions in embedding space. **Calendar-aware approaches** use cyclical sin/cos encodings for hour, day, and month, or learned lookup tables. **Recommendation systems** handle time through exponential decay weighting, sequential models, and temporal attention.

The comparative picture reveals a striking pattern:

| Capability | Time2Vec | RoPE | Neural ODE | TGN | Calendar | Date2Vec |
|---|---|---|---|---|---|---|
| Periodicity | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Irregular intervals | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Context-dependent | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Calendar-aware | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Duration/urgency | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Circadian patterns | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Experiential time | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**No existing approach scores a single checkmark on urgency, circadian patterns, or experiential time.** This is the gap PULSE would fill.

---

## How the brain encodes time — and what AI can borrow

Neuroscience reveals that time perception is inherently **multi-scale, distributed, and context-dependent** — properties conspicuously absent from current AI temporal models. Different brain regions handle different temporal scales: the **cerebellum** processes sub-second (millisecond) timing, **cortico-striatal circuits** (basal ganglia/striatum) handle interval timing from seconds to minutes, and the **suprachiasmatic nucleus** drives circadian (~24 hour) rhythms. The **hippocampus** contains "time cells" encoding temporal-spatial relationships, and the **insular cortex** integrates bodily signals into subjective temporal experience.

Three theories of temporal processing offer direct architectural inspiration for PULSE. The **Striatal Beat Frequency (SBF) model** (Matell & Meck, 2000) proposes that time is coded through the coincidental activation of striatal neurons by multiple cortical oscillators running at different frequencies — specific beat patterns correspond to specific durations. This maps directly to multi-frequency sinusoidal embeddings, but with a crucial insight: the SBF model explains how dopamine modulates oscillator speed, causing emotional time dilation. The **State-Dependent Network model** (Buonomano & Maass, 2009) argues that temporal information is inherently encoded in the evolving state of neural networks through short-term synaptic plasticity — the network carries temporal context implicitly, meaning the same stimulus gets different processing depending on what came before. The **Pacemaker-Accumulator model** establishes the scalar property: **variability in temporal estimates is proportional to interval duration** (Weber's Law for time), with a Weber fraction of roughly 0.1–0.25.

Several specific findings translate directly into engineering decisions for PULSE:

**Logarithmic time compression.** Weber's Law means the felt distance between 1 minute and 2 minutes vastly exceeds the felt distance between 101 and 102 minutes. Applying log-transform to raw timestamps before embedding is biologically motivated and computationally trivial.

**Hyperbolic urgency scaling.** Temporal discounting research shows humans perceive urgency via hyperbolic functions: V_d = V / (1 + k·d), where k is the discounting rate and d is delay. The perceived urgency of a deadline at T-1 hour is far more than double the urgency at T-2 hours. Kim & Zauberman (2009) showed this partly reflects perceptual compression of future time, not just reward devaluation.

**Circadian performance modulation.** Attention, working memory, and executive function all show strong circadian variation — lowest at night and early morning, peaking noon through evening. The **Basic Rest-Activity Cycle** adds ~90-minute ultradian oscillations. These are endogenously programmed and affect all cognitive processing, making circadian phase a meaningful component of temporal context.

**Emotional time distortion.** High-arousal emotions cause temporal overestimation (time expands); low arousal causes compression. Flow states produce massive temporal compression — hours feel like minutes. The mechanism operates through modulation of internal clock speed, implementable as a learned scalar gain on oscillator frequencies.

**Predictive processing.** Sohn et al. (2021, Neuron) found that frontal cortex neural dynamics speed is inversely proportional to expected temporal distributions, and neural populations encode stimuli as deviations from temporal expectations. This motivates including prediction error — the deviation between expected and actual event timing — as an embedding dimension.

---

## Event stream modeling provides the mathematical backbone

Temporal point processes offer the mathematical machinery PULSE needs for modeling event sequences with irregular time intervals. A temporal point process (TPP) characterizes the conditional intensity function λ*(t), giving the instantaneous event rate given history. The **Hawkes process** (1971) adds self-excitation: past events boost future event probability through exponential decay kernels, λₖ(t) = μₖ + Σ αₖ · exp(−δₖ · (t − tₕ)), capturing event clustering and cascade dynamics.

The neural extensions of these processes provide direct blueprints for PULSE components. The **Neural Hawkes Process** (Mei & Eisner, NeurIPS 2017) introduces a continuous-time LSTM where memory cells decay exponentially between events — c(t) = c̄ + (c − c̄) · exp(−δ(t − tₘ)) — enabling the hidden state to evolve continuously rather than remaining static until the next input. This models inhibition, non-additive effects, and non-monotonic intensity evolution that classical Hawkes cannot express. The **Transformer Hawkes Process** (Zuo et al., ICML 2020) replaces recurrence with self-attention over temporal positional encodings, capturing long-range dependencies with parallel computation. Most recently, **Hawkes Attention** (Tan et al., 2025) derives an attention operator directly from Hawkes theory, using per-type neural influence kernels to modulate Q, K, V projections — demonstrating that Hawkes-inspired attention serves as a general temporal mechanism beyond event sequences.

**Intensity-free methods** (Shchur et al., ICLR 2020) bypass the intensity function entirely, directly modeling conditional inter-event time distributions using normalizing flows or mixture models (LogNormMix). This approach avoids the intractable integration problem and enables sequence embedding — directly relevant to PULSE's goal of encoding temporal contexts as vectors.

The practical implementation ecosystem is mature. **EasyTPP** (ICLR 2024 benchmark, Ant Research) provides a comprehensive PyTorch framework implementing THP, SAHP, Neural Hawkes, FullyNN, and more. The **tick** library offers classical Hawkes process simulation and inference with C++ core. Standard training datasets span social media (Retweets), healthcare (MIMIC-II/III), online behavior (Stack Overflow), and finance.

Key concepts borrowable for PULSE include: continuous-time state evolution between observations, time-modulated attention weights using decay or oscillatory kernels, multi-scale temporal encoding through different decay rates, and the inter-event time as a first-class feature (both τ and log τ).

---

## Persistent agents and temporal MOE routing are production-ready

The infrastructure for always-on, temporally-aware AI agents already exists. **Letta** (evolved from MemGPT) provides production-ready stateful agents deployed as services behind REST APIs, with a three-tier memory hierarchy: core memory (in-context, self-editable), recall memory (searchable conversation history), and archival memory (unbounded external storage). Its V1 architecture introduces **sleep-time agents** that manage memory asynchronously — directly relevant to PULSE's potential role as a background temporal processing daemon. **Zep/Graphiti** implements a bi-temporal knowledge graph tracking both when events occurred and when they were ingested, with explicit validity intervals on every edge and conflict resolution that preserves historical accuracy. **Google's Always-On Memory Agent** runs 24/7 using a three-agent orchestration pattern: ingest, consolidate (every 30 minutes, like "brain during sleep"), and query.

The **heartbeat architecture pattern** — where an AI agent runs on a regular schedule, gathers fresh data, reasons via LLM, and takes action — is now well-established across multiple implementations including OpenClaw (HEARTBEAT.md defining proactive tasks), Chipp AI (configurable frequency with quiet hours), and aidaemon (Rust-based daemon with MCP extension). These provide the operational infrastructure PULSE would need as a background temporal processing service.

For routing between a temporal expert (PULSE) and a language model, the most significant discovery is **TiMoE** (Faro et al., EPFL, August 2025) — a time-aware Mixture of Experts that pre-trains GPT-style experts on disjoint two-year time slices and combines them through a **causal, timestamp-aware router** that masks future-knowledge experts at inference time. TiMoE cuts temporal hallucinations by ~15% on its TSQA benchmark, demonstrating that modular, time-segmented pre-training with causal routing is feasible. This is the closest existing system to a PULSE/MIND routing architecture.

Five integration patterns emerge from the research:

- **TiMoE-style expert routing**: Separate temporal and language experts with timestamp-aware gating that activates PULSE when temporal context is detected
- **Adapter experts**: PULSE as a LoRA-style adapter activated via token-level gating, inspired by TC-LoRA and Time-Adapter (which explicitly argues standard adapters "are not designed to capture temporal patterns")
- **Expert Choice routing**: Inverting the paradigm so PULSE "claims" temporally-relevant tokens rather than tokens being routed to PULSE, guaranteeing load balance
- **Side-network augmentation**: PULSE always processes temporal context but its output is gated by relevance, analogous to cross-modal adapters
- **Sleep-time background processing**: PULSE runs continuously maintaining temporal state (deadlines, schedules, circadian phase), queried by the LLM as needed via tool calls — inspired by Letta's sleep-time agents

---

## Building PULSE: a concrete implementation path

The implementation landscape reveals both a clear gap and a clear path forward. **No pip-installable temporal embedding library exists.** The `temporal-lib` package on PyPI is a datetime formatter; `temporal-lib-py` wraps the Temporal workflow engine. The npm ecosystem has nothing — the "temporal" namespace is entirely occupied by TC39's Date replacement proposal. This means `pulse-temporal` or `temporal-embeddings` on PyPI would face literally zero competition.

The recommended architecture fuses neuroscience-inspired components with proven ML building blocks:

```
PULSE_embedding(t, context) = concat[
    log_time(t),                          # Weber's Law compression
    multi_freq_oscillators(t),            # SBF-inspired learned sinusoids (Time2Vec base)
    circadian_phase(t),                   # sin/cos of 24h and 90min cycles
    calendar_features(t),                 # day-of-week, holiday, season embeddings
    urgency(t, deadline),                 # hyperbolic: 1/(1 + k·time_remaining)
    temporal_context_state(history),      # continuous-time LSTM state (Neural Hawkes-inspired)
    prediction_error(t, t_expected),      # deviation from temporal prior
    arousal_gain(context)                 # emotional modulation of clock speed
]
```

**Training data** would draw from multiple public sources: the American Time Use Survey (ATUS) for human activity patterns, StudentLife smartphone usage data for circadian behavioral patterns, BPI Challenge process mining event logs, MOOC clickstream data, and optionally crowdsourced "how did this time period feel?" annotations for supervised experiential time learning. The **ATC framework** (arXiv:2206.09535), which jointly embeds actions and time intervals using statistical mixture models, provides the closest existing training methodology to adapt.

**Loss functions** should combine multiple objectives. An **InfoNCE contrastive loss** where positive pairs are "experientially similar" moments (two Monday mornings before deadlines, regardless of calendar distance) and negatives are "experientially different" (Monday 9am sprint vs. Saturday 2pm leisure). A **circadian phase alignment loss** ensuring same-phase moments cluster together. An **urgency classification auxiliary loss** predicting urgency level from the embedding. And critically, a novel **perceptual time distortion loss** — trained on human perception data where the distance metric reflects subjective time dilation/compression. T-Rep's time-embedding divergence prediction using KL-divergence provides a starting template.

**Minimal dependencies**: core inference with numpy only (pre-computed embeddings, sinusoidal encodings, lookup tables); training with PyTorch; optional JAX backend for high-performance inference. Structure as `temporal-embeddings[torch]` with optional extras. Model the API on sentence-transformers:

```python
from pulse_temporal import PulseEncoder
model = PulseEncoder("pulse-base-v1")
embeddings = model.encode(["2024-03-15T09:30:00"], context={"deadline": "2024-03-15T17:00:00"})
```

---

## What makes PULSE genuinely novel

After surveying the full landscape, the novelty of PULSE can be stated precisely. Five specific capabilities have no existing implementation in any published paper, open-source library, or production system:

**Experiential time as a first-class embedding dimension.** No existing system encodes time-as-experienced rather than time-as-measured. The closest work — ChronoPilot (arXiv:2404.15213) and trial-by-trial subjective time prediction from brain imaging — *predicts* subjective time perception from physiological data but does not *encode* it into reusable vector embeddings. The ATC framework embeds actions alongside time intervals but focuses on behavior characterization, not subjective temporal experience.

**Urgency-aware temporal semantics.** Urgency detection exists as a separate NLP task. Temporal embeddings exist as separate representations. No system fuses urgency INTO the temporal representation itself. A PULSE embedding where "3pm on a quiet Friday" and "3pm two hours before a critical deadline" occupy vastly different regions of the vector space would be unprecedented.

**Multi-signal fusion in a single temporal vector space.** Current approaches choose one temporal signal: periodicity (Time2Vec), position (RoPE), dynamics (Neural ODE), or calendar features (lookup tables). No framework unifies clock time, circadian phase, behavioral context, urgency, and prediction error into a single embedding.

**Human-perception-aligned metric learning.** Training with contrastive loss where similarity is defined by human temporal perception surveys — not by timestamp arithmetic — has never been done. This would produce the first temporal embedding space where vector distances correspond to felt temporal distances.

**A pip-installable temporal embedding library.** This is perhaps the most practically significant gap. Every NLP practitioner can `pip install sentence-transformers`. No equivalent exists for temporal embeddings. The first library to fill this gap captures significant infrastructure mindshare.

---

## Conclusion: a convergence of ripe building blocks

The research reveals a paradox: every component needed for PULSE already exists in isolation, yet nobody has assembled them. Time2Vec provides learnable periodic encoding. Neural Hawkes processes provide continuous-time state evolution. The SBF model from neuroscience validates multi-frequency oscillatory representations as biologically grounded. Weber's Law justifies logarithmic compression. Hyperbolic discounting provides the urgency function. TiMoE demonstrates temporal expert routing in production. Letta and the heartbeat pattern provide always-on agent infrastructure. Contrastive learning frameworks provide the training methodology.

The genuine insight is that **time in AI has been treated as a feature engineering problem rather than a representation learning problem**. Word2Vec transformed NLP by learning that words have rich geometric relationships in vector space. PULSE could do the same for time — learning that Monday morning before a deadline, the post-lunch cognitive dip, and the anticipatory buzz before a product launch each occupy meaningful, learnable positions in a temporal vector space. The neuroscience confirms this is how the brain works: not a single clock, but a distributed, multi-scale, context-modulated system where the same five minutes can feel like an eternity or flash by depending on state. Building this into AI would give language models something they fundamentally lack — not just knowledge of what time it is, but understanding of what that time *means*.

The practical path forward is clear: start with Time2Vec as the base layer, add circadian and calendar encodings, layer in urgency via hyperbolic functions, use continuous-time LSTM states for temporal context, train with multi-objective contrastive learning on human activity data, and ship it as `pulse-temporal` on PyPI. The key repositories to build on include EasyTPP (neural TPP benchmark), torchdiffeq (Neural ODEs), TiMoE (temporal MOE routing), Letta (persistent agent infrastructure), and Graphiti (temporal knowledge graphs). The research gap is wide open, the building blocks are mature, and the first team to assemble them captures a genuinely novel position in the AI stack.
