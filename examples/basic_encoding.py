"""Basic PULSE encoding example.

Shows how experientially similar moments cluster together
even when their timestamps are far apart.
"""

from pulse_temporal import PulseEncoder

pulse = PulseEncoder()

# --- Same clock time, completely different experiences ---

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

print(f"Monday crunch vs Saturday chill: {pulse.similarity(monday_crunch, saturday_chill):.3f}")

# --- Different clock times, similar experience ---

wednesday_crunch = pulse.encode("2026-04-15T10:00:00", context={
    "deadline": "2026-04-15T12:00:00",
    "events_today": 4,
    "sleep_hours": 6,
})

print(f"Monday crunch vs Wednesday crunch: {pulse.similarity(monday_crunch, wednesday_crunch):.3f}")

# --- Decompose a moment to see what each layer contributes ---

print("\n--- Layer decomposition for Monday 2pm crunch ---")
layers = pulse.decompose("2026-04-13T14:00:00", context={
    "deadline": "2026-04-13T17:00:00",
    "events_today": 6,
    "sleep_hours": 5,
})

for name, vec in layers.items():
    magnitude = float((vec ** 2).sum() ** 0.5)
    print(f"  {name:20s}: dim={len(vec):3d}, magnitude={magnitude:.3f}")

# --- Temporal context package (for LLM injection) ---

print("\n--- Temporal context package ---")
ctx = pulse.get_temporal_context("2026-04-13T14:00:00", context={
    "deadline": "2026-04-13T17:00:00",
    "events_today": 6,
})
for key, value in ctx.items():
    if key != "embedding":
        print(f"  {key}: {value}")

# --- Similarity matrix ---

print("\n--- Similarity matrix ---")
moments = [
    ("2026-04-13T09:00:00", {"deadline": "2026-04-13T12:00:00"}),   # monday AM crunch
    ("2026-04-15T09:00:00", {"deadline": "2026-04-15T11:00:00"}),   # wednesday AM crunch
    ("2026-04-11T09:00:00", {"deadline": None}),                     # saturday AM chill
    ("2026-04-13T02:00:00", {"deadline": None, "sleep_hours": 0}),   # monday 2am insomnia
]
labels = ["Mon crunch", "Wed crunch", "Sat chill", "Mon 2am"]

embeddings = [pulse.encode(t, context=c) for t, c in moments]
sim_matrix = pulse.similarity_matrix(embeddings)

print(f"{'':>12s}", end="")
for label in labels:
    print(f"{label:>12s}", end="")
print()
for i, label in enumerate(labels):
    print(f"{label:>12s}", end="")
    for j in range(len(labels)):
        print(f"{sim_matrix[i, j]:12.3f}", end="")
    print()
