"""PULSE daemon example.

Shows how to run PULSE as a background service that maintains
temporal state and provides context for LLM integration.
"""

import time
from pulse_temporal.daemon import PulseDaemon

# Initialize the daemon
daemon = PulseDaemon(
    db_path="/tmp/pulse_demo.db",
    heartbeat_interval=10,  # fast for demo
)

# Register some deadlines
daemon.add_deadline("project X review", "2026-04-15T17:00:00", priority="high")
daemon.add_deadline("weekly standup", "2026-04-14T09:30:00", priority="medium")

# Log some events
daemon.log_event(event_type="work_session_start")
daemon.log_event(event_type="code_commit", metadata={"repo": "pulse-temporal"})

# Start the background heartbeat
daemon.start()
print(f"Daemon started: {daemon}")

# Query temporal context (this is what the MIND would call)
context = daemon.get_temporal_context()

print("\n--- Current Temporal Context ---")
for key, value in context.items():
    if key != "embedding":
        print(f"  {key}: {value}")

print(f"\n  embedding shape: {context['embedding'].shape}")
print(f"  embedding norm: {float(context['embedding'] @ context['embedding']):.4f}")

# Clean up
daemon.stop()
print(f"\nDaemon stopped: {daemon}")
