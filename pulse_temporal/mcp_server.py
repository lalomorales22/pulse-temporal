"""PULSE Temporal MCP Server.

Exposes PULSE temporal encoding as MCP tools that any MCP-compatible
LLM client (Claude, ChatGPT, etc.) can call.

Run standalone:
    python -m pulse_temporal.mcp_server

Or with the MCP CLI:
    mcp run pulse_temporal.mcp_server
"""

import json
import sys
from datetime import datetime
from typing import Optional

from .encoder import PulseEncoder
from .daemon.pulse_daemon import PulseDaemon


# Global instances, initialized on startup
_encoder: Optional[PulseEncoder] = None
_daemon: Optional[PulseDaemon] = None


def _get_encoder() -> PulseEncoder:
    global _encoder
    if _encoder is None:
        _encoder = PulseEncoder()
    return _encoder


def _get_daemon() -> PulseDaemon:
    global _daemon
    if _daemon is None:
        _daemon = PulseDaemon()
        _daemon.start()
    return _daemon


# --- MCP Protocol Implementation (JSON-RPC over stdio) ---

TOOLS = [
    {
        "name": "get_temporal_context",
        "description": (
            "Get rich temporal context for the current moment. Returns circadian phase, "
            "cognitive capacity, energy level, urgency, and active deadlines. "
            "Inject this into your system prompt to become temporally aware."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "ISO 8601 timestamp. Defaults to now.",
                },
            },
        },
    },
    {
        "name": "encode_moment",
        "description": (
            "Encode a moment as a 128D PULSE temporal embedding vector. "
            "Captures not just when something happens, but what that time feels like."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "ISO 8601 timestamp to encode.",
                },
                "deadline": {
                    "type": "string",
                    "description": "ISO 8601 deadline timestamp (optional).",
                },
                "events_today": {
                    "type": "integer",
                    "description": "Number of events/meetings today.",
                },
                "sleep_hours": {
                    "type": "number",
                    "description": "Hours of sleep last night.",
                },
            },
            "required": ["timestamp"],
        },
    },
    {
        "name": "compare_moments",
        "description": (
            "Compare two moments and return their experiential similarity. "
            "Score near 1.0 means they feel similar; near 0.0 means very different experiences."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "moment_a": {
                    "type": "object",
                    "description": "First moment: {timestamp, deadline?, events_today?, sleep_hours?}",
                    "properties": {
                        "timestamp": {"type": "string"},
                        "deadline": {"type": "string"},
                        "events_today": {"type": "integer"},
                        "sleep_hours": {"type": "number"},
                    },
                    "required": ["timestamp"],
                },
                "moment_b": {
                    "type": "object",
                    "description": "Second moment: {timestamp, deadline?, events_today?, sleep_hours?}",
                    "properties": {
                        "timestamp": {"type": "string"},
                        "deadline": {"type": "string"},
                        "events_today": {"type": "integer"},
                        "sleep_hours": {"type": "number"},
                    },
                    "required": ["timestamp"],
                },
            },
            "required": ["moment_a", "moment_b"],
        },
    },
    {
        "name": "log_event",
        "description": (
            "Log a temporal event (meeting, break, task switch, etc.) "
            "to the PULSE daemon. Events influence future temporal context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "event_type": {
                    "type": "string",
                    "description": "Type of event: work_session_start, work_session_end, break, meeting, task_switch, meal, exercise, sleep_start, sleep_end, etc.",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata (e.g., {\"task\": \"code review\"}).",
                },
            },
            "required": ["event_type"],
        },
    },
    {
        "name": "add_deadline",
        "description": "Register a deadline. Deadlines shift urgency in all future temporal context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "What the deadline is for.",
                },
                "target_time": {
                    "type": "string",
                    "description": "ISO 8601 deadline timestamp.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Priority level. Default: medium.",
                },
            },
            "required": ["name", "target_time"],
        },
    },
    {
        "name": "complete_deadline",
        "description": "Mark a deadline as completed. Removes it from urgency calculations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "deadline_id": {
                    "type": "integer",
                    "description": "ID of the deadline to complete (from list_deadlines).",
                },
            },
            "required": ["deadline_id"],
        },
    },
    {
        "name": "list_deadlines",
        "description": "List all active (non-completed) deadlines with their IDs and time remaining.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "decompose_moment",
        "description": (
            "Break down a moment into its seven temporal signal layers. "
            "Shows exactly what PULSE sees: log_time, oscillators, circadian, "
            "calendar, urgency, temporal_state, prediction_error."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "ISO 8601 timestamp to decompose.",
                },
                "deadline": {
                    "type": "string",
                    "description": "Optional deadline for urgency layer.",
                },
            },
            "required": ["timestamp"],
        },
    },
]


def _build_context(args: dict) -> dict:
    """Build encoder context from tool arguments."""
    ctx = {}
    if args.get("deadline"):
        ctx["deadline"] = args["deadline"]
    if args.get("events_today") is not None:
        ctx["events_today"] = args["events_today"]
    if args.get("sleep_hours") is not None:
        ctx["sleep_hours"] = args["sleep_hours"]
    return ctx


def _handle_tool_call(name: str, arguments: dict) -> dict:
    """Execute a tool and return the result content."""
    encoder = _get_encoder()
    daemon = _get_daemon()

    if name == "get_temporal_context":
        ts = arguments.get("timestamp")
        ctx = daemon.get_temporal_context(ts)
        # Remove the raw embedding (not useful as text)
        ctx.pop("embedding", None)
        return ctx

    elif name == "encode_moment":
        ts = arguments["timestamp"]
        context = _build_context(arguments)
        emb = encoder.encode(ts, context)
        tc = encoder.get_temporal_context(ts, context)
        return {
            "timestamp": ts,
            "embedding_dim": len(emb),
            "embedding_norm": float((emb ** 2).sum() ** 0.5),
            "circadian_phase": tc["circadian_phase"],
            "cognitive_capacity": tc["cognitive_capacity"],
            "energy_level": tc["energy_level"],
            "urgency_level": tc["urgency_level"],
        }

    elif name == "compare_moments":
        a = arguments["moment_a"]
        b = arguments["moment_b"]
        emb_a = encoder.encode(a["timestamp"], _build_context(a))
        emb_b = encoder.encode(b["timestamp"], _build_context(b))
        sim = encoder.similarity(emb_a, emb_b)
        dist = encoder.temporal_distance(emb_a, emb_b)

        # Get context for both
        tc_a = encoder.get_temporal_context(a["timestamp"], _build_context(a))
        tc_b = encoder.get_temporal_context(b["timestamp"], _build_context(b))

        return {
            "similarity": round(sim, 4),
            "temporal_distance": round(dist, 4),
            "interpretation": (
                "nearly identical experience" if sim > 0.95
                else "very similar experience" if sim > 0.85
                else "similar experience" if sim > 0.75
                else "somewhat different" if sim > 0.6
                else "quite different" if sim > 0.4
                else "very different experiences"
            ),
            "moment_a": {
                "phase": tc_a["circadian_phase"],
                "cognitive": tc_a["cognitive_capacity"],
                "urgency": tc_a["urgency_level"],
            },
            "moment_b": {
                "phase": tc_b["circadian_phase"],
                "cognitive": tc_b["cognitive_capacity"],
                "urgency": tc_b["urgency_level"],
            },
        }

    elif name == "log_event":
        event_id = daemon.log_event(
            event_type=arguments["event_type"],
            metadata=arguments.get("metadata"),
        )
        return {"event_id": event_id, "event_type": arguments["event_type"], "logged": True}

    elif name == "add_deadline":
        dl_id = daemon.add_deadline(
            name=arguments["name"],
            target_time=arguments["target_time"],
            priority=arguments.get("priority", "medium"),
        )
        return {"deadline_id": dl_id, "name": arguments["name"], "registered": True}

    elif name == "complete_deadline":
        daemon.complete_deadline(arguments["deadline_id"])
        return {"deadline_id": arguments["deadline_id"], "completed": True}

    elif name == "list_deadlines":
        deadlines = daemon.db.get_active_deadlines()
        now = datetime.now()
        result = []
        for d in deadlines:
            target = datetime.fromisoformat(d["target_time"])
            delta = target - now
            hours = delta.total_seconds() / 3600
            result.append({
                "id": d["id"],
                "name": d["name"],
                "target_time": d["target_time"],
                "priority": d["priority"],
                "time_remaining": f"{hours:.1f} hours" if 0 < hours < 48 else (
                    f"{hours / 24:.1f} days" if hours >= 48 else f"OVERDUE by {-hours:.1f} hours"
                ),
            })
        return {"deadlines": result, "count": len(result)}

    elif name == "decompose_moment":
        ts = arguments["timestamp"]
        context = _build_context(arguments)
        layers = encoder.decompose(ts, context)
        # Summarize each layer
        return {
            "timestamp": ts,
            "layers": {
                name: {
                    "dim": len(vec),
                    "mean": round(float(vec.mean()), 4),
                    "max": round(float(vec.max()), 4),
                    "min": round(float(vec.min()), 4),
                    "nonzero": int((vec != 0).sum()),
                }
                for name, vec in layers.items()
            },
        }

    else:
        raise ValueError(f"Unknown tool: {name}")


def _send(msg: dict):
    """Send a JSON-RPC message to stdout."""
    raw = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(raw)}\r\n\r\n{raw}")
    sys.stdout.flush()


def _read_message() -> Optional[dict]:
    """Read a JSON-RPC message from stdin (Content-Length framing)."""
    headers = {}
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        if ":" in line:
            key, val = line.split(":", 1)
            headers[key.strip()] = val.strip()

    length = int(headers.get("Content-Length", 0))
    if length == 0:
        return None
    body = sys.stdin.read(length)
    return json.loads(body)


def _handle_message(msg: dict) -> Optional[dict]:
    """Handle a single JSON-RPC message and return the response."""
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": "pulse-temporal",
                    "version": "0.2.0",
                },
            },
        }

    elif method == "notifications/initialized":
        return None  # no response for notifications

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOLS},
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            result = _handle_tool_call(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2, default=str),
                        }
                    ],
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "isError": True,
                },
            }

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    else:
        # Unknown method
        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return None


def serve():
    """Run the MCP server on stdio."""
    while True:
        msg = _read_message()
        if msg is None:
            break
        response = _handle_message(msg)
        if response is not None:
            _send(response)


def main():
    """Entry point for `python -m pulse_temporal.mcp_server`."""
    serve()


if __name__ == "__main__":
    main()
