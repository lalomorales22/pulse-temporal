"""Tests for the PULSE LLM middleware."""

import pytest
from datetime import datetime

from pulse_temporal.middleware import PulseMiddleware


class TestPulseMiddleware:
    """Test temporal context injection middleware."""

    def setup_method(self):
        self.mw = PulseMiddleware()

    def test_init(self):
        assert repr(self.mw) == "PulseMiddleware(preamble=True)"

    def test_init_no_preamble(self):
        mw = PulseMiddleware(include_preamble=False)
        assert mw._include_preamble is False

    def test_get_temporal_context(self):
        ctx = self.mw.get_temporal_context()
        assert "circadian_phase" in ctx
        assert "cognitive_capacity" in ctx
        assert "energy_level" in ctx
        assert "urgency_level" in ctx
        assert "embedding" not in ctx  # should be stripped

    def test_format_temporal_block(self):
        block = self.mw.format_temporal_block()
        assert "[PULSE Temporal Context]" in block
        assert "Circadian phase:" in block
        assert "Cognitive capacity:" in block
        assert "Energy level:" in block
        assert "Urgency:" in block

    def test_format_with_custom_context(self):
        ctx = {
            "circadian_phase": "morning_peak",
            "cognitive_capacity": 0.85,
            "energy_level": 0.9,
            "urgency_level": "high",
            "urgency_score": 0.7,
            "urgency_summary": "Project X in 2.0 hours",
        }
        block = self.mw.format_temporal_block(ctx)
        assert "morning_peak" in block
        assert "Project X in 2.0 hours" in block

    def test_get_temporal_system_prompt(self):
        prompt = self.mw.get_temporal_system_prompt()
        assert "PULSE" in prompt
        assert "temporal awareness" in prompt
        assert "[PULSE Temporal Context]" in prompt

    def test_get_temporal_system_prompt_with_existing(self):
        existing = "You are a helpful coding assistant."
        prompt = self.mw.get_temporal_system_prompt(existing_system=existing)
        assert "helpful coding assistant" in prompt
        assert "[PULSE Temporal Context]" in prompt

    def test_get_temporal_system_prompt_no_preamble(self):
        mw = PulseMiddleware(include_preamble=False)
        prompt = mw.get_temporal_system_prompt()
        assert "You have real-time temporal awareness" not in prompt
        assert "[PULSE Temporal Context]" in prompt

    def test_inject_messages_no_system(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        injected = self.mw.inject_messages(messages)
        assert len(injected) == 2
        assert injected[0]["role"] == "system"
        assert "[PULSE Temporal Context]" in injected[0]["content"]
        assert injected[1]["role"] == "user"

    def test_inject_messages_existing_system(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        injected = self.mw.inject_messages(messages)
        assert len(injected) == 2
        assert "You are helpful." in injected[0]["content"]
        assert "[PULSE Temporal Context]" in injected[0]["content"]

    def test_inject_messages_does_not_mutate_original(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        original_len = len(messages)
        self.mw.inject_messages(messages)
        assert len(messages) == original_len

    def test_inject_messages_preserves_conversation(self):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
        ]
        injected = self.mw.inject_messages(messages)
        assert len(injected) == 4
        assert injected[1]["content"] == "Question 1"
        assert injected[2]["content"] == "Answer 1"
        assert injected[3]["content"] == "Question 2"
