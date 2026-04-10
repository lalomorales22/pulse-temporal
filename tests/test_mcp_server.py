"""Tests for the PULSE MCP server."""

import json
import pytest
from datetime import datetime, timedelta

from pulse_temporal.mcp_server import _handle_tool_call, _handle_message, TOOLS


class TestMCPTools:
    """Test individual tool handlers."""

    def test_tools_defined(self):
        """All expected tools are registered."""
        names = {t["name"] for t in TOOLS}
        assert "get_temporal_context" in names
        assert "encode_moment" in names
        assert "compare_moments" in names
        assert "log_event" in names
        assert "add_deadline" in names
        assert "complete_deadline" in names
        assert "list_deadlines" in names
        assert "decompose_moment" in names

    def test_get_temporal_context(self):
        result = _handle_tool_call("get_temporal_context", {})
        assert "circadian_phase" in result
        assert "cognitive_capacity" in result
        assert "energy_level" in result
        assert "urgency_level" in result
        # No raw embedding in the output
        assert "embedding" not in result

    def test_get_temporal_context_with_timestamp(self):
        result = _handle_tool_call("get_temporal_context", {
            "timestamp": "2026-04-09T14:30:00"
        })
        assert result["circadian_phase"] == "afternoon_peak"

    def test_encode_moment(self):
        result = _handle_tool_call("encode_moment", {
            "timestamp": "2026-04-09T14:30:00",
        })
        assert result["embedding_dim"] == 128
        assert "circadian_phase" in result
        assert "cognitive_capacity" in result

    def test_encode_moment_with_context(self):
        result = _handle_tool_call("encode_moment", {
            "timestamp": "2026-04-09T14:30:00",
            "deadline": "2026-04-09T17:00:00",
            "sleep_hours": 5,
        })
        assert result["urgency_level"] != "none"

    def test_compare_moments(self):
        result = _handle_tool_call("compare_moments", {
            "moment_a": {"timestamp": "2026-04-13T14:00:00", "deadline": "2026-04-13T17:00:00"},
            "moment_b": {"timestamp": "2026-04-11T14:00:00"},
        })
        assert "similarity" in result
        assert "temporal_distance" in result
        assert "interpretation" in result
        assert 0 <= result["similarity"] <= 1
        assert result["temporal_distance"] >= 0

    def test_compare_similar_moments(self):
        """Same type of crunch moment on different days should be similar."""
        result = _handle_tool_call("compare_moments", {
            "moment_a": {
                "timestamp": "2026-04-13T14:00:00",
                "deadline": "2026-04-13T17:00:00",
            },
            "moment_b": {
                "timestamp": "2026-04-15T10:00:00",
                "deadline": "2026-04-15T12:00:00",
            },
        })
        assert result["similarity"] > 0.5

    def test_log_event(self):
        result = _handle_tool_call("log_event", {
            "event_type": "test_event",
            "metadata": {"test": True},
        })
        assert result["logged"] is True
        assert result["event_type"] == "test_event"
        assert "event_id" in result

    def test_add_and_list_deadlines(self):
        target = (datetime.now() + timedelta(hours=5)).isoformat()
        add_result = _handle_tool_call("add_deadline", {
            "name": "test deadline",
            "target_time": target,
            "priority": "high",
        })
        assert add_result["registered"] is True

        list_result = _handle_tool_call("list_deadlines", {})
        assert list_result["count"] > 0
        names = [d["name"] for d in list_result["deadlines"]]
        assert "test deadline" in names

    def test_complete_deadline(self):
        target = (datetime.now() + timedelta(hours=5)).isoformat()
        add_result = _handle_tool_call("add_deadline", {
            "name": "to complete",
            "target_time": target,
        })
        dl_id = add_result["deadline_id"]

        result = _handle_tool_call("complete_deadline", {"deadline_id": dl_id})
        assert result["completed"] is True

    def test_decompose_moment(self):
        result = _handle_tool_call("decompose_moment", {
            "timestamp": "2026-04-09T14:30:00",
        })
        assert "layers" in result
        layers = result["layers"]
        assert "log_time" in layers
        assert "oscillators" in layers
        assert "circadian" in layers
        assert "calendar" in layers
        assert "urgency" in layers
        assert "temporal_state" in layers
        assert "prediction_error" in layers
        for name, info in layers.items():
            assert "dim" in info
            assert "mean" in info

    def test_unknown_tool(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            _handle_tool_call("nonexistent_tool", {})


class TestMCPProtocol:
    """Test JSON-RPC message handling."""

    def test_initialize(self):
        resp = _handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        })
        assert resp["id"] == 1
        assert resp["result"]["serverInfo"]["name"] == "pulse-temporal"
        assert "tools" in resp["result"]["capabilities"]

    def test_tools_list(self):
        resp = _handle_message({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        assert len(resp["result"]["tools"]) == len(TOOLS)

    def test_tools_call(self):
        resp = _handle_message({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_temporal_context",
                "arguments": {"timestamp": "2026-04-09T10:00:00"},
            },
        })
        content = resp["result"]["content"][0]
        assert content["type"] == "text"
        data = json.loads(content["text"])
        assert "circadian_phase" in data

    def test_tools_call_error(self):
        resp = _handle_message({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
        })
        assert resp["result"]["isError"] is True

    def test_ping(self):
        resp = _handle_message({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "ping",
        })
        assert resp["result"] == {}

    def test_notification_no_response(self):
        resp = _handle_message({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        assert resp is None

    def test_unknown_method(self):
        resp = _handle_message({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "unknown/method",
        })
        assert resp["error"]["code"] == -32601
