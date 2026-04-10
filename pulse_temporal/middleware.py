"""Temporal context injection middleware for LLM APIs.

Wraps any LLM API client to automatically inject PULSE temporal context
into system prompts. Works with OpenAI, Anthropic, and any chat-completion
compatible API.

Usage with OpenAI:
    from openai import OpenAI
    from pulse_temporal.middleware import PulseMiddleware

    client = OpenAI()
    pulse = PulseMiddleware()

    # Automatic temporal context injection
    response = pulse.chat(client, messages=[
        {"role": "user", "content": "Should I tackle this complex bug now?"}
    ])

Usage with Anthropic:
    from anthropic import Anthropic
    from pulse_temporal.middleware import PulseMiddleware

    client = Anthropic()
    pulse = PulseMiddleware()

    response = pulse.chat_anthropic(client, messages=[
        {"role": "user", "content": "Should I tackle this complex bug now?"}
    ])

Standalone (just get the system prompt):
    pulse = PulseMiddleware()
    system_prompt = pulse.get_temporal_system_prompt()
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from .encoder import PulseEncoder
from .daemon.pulse_daemon import PulseDaemon


class PulseMiddleware:
    """Injects PULSE temporal context into LLM API calls."""

    TEMPORAL_PREAMBLE = (
        "You have real-time temporal awareness via PULSE. "
        "Use this context to give time-appropriate responses — "
        "consider the user's cognitive state, energy, urgency, and circadian phase."
    )

    def __init__(
        self,
        daemon: Optional[PulseDaemon] = None,
        encoder: Optional[PulseEncoder] = None,
        include_preamble: bool = True,
    ):
        self._daemon = daemon
        self._encoder = encoder or PulseEncoder()
        self._include_preamble = include_preamble

    @property
    def daemon(self) -> PulseDaemon:
        if self._daemon is None:
            self._daemon = PulseDaemon()
            self._daemon.start()
        return self._daemon

    def get_temporal_context(self, t: Optional[str] = None) -> Dict:
        """Get current temporal context from the daemon."""
        ctx = self.daemon.get_temporal_context(t)
        ctx.pop("embedding", None)
        return ctx

    def format_temporal_block(self, ctx: Optional[Dict] = None) -> str:
        """Format temporal context as a text block for system prompt injection.

        Returns a concise, structured block that any LLM can parse.
        """
        if ctx is None:
            ctx = self.get_temporal_context()

        now = datetime.now()
        lines = [
            f"[PULSE Temporal Context]",
            f"Time: {now.strftime('%A %I:%M %p')} ({now.strftime('%Y-%m-%d')})",
            f"Circadian phase: {ctx.get('circadian_phase', 'unknown')}",
            f"Cognitive capacity: {ctx.get('cognitive_capacity', '?')}",
            f"Energy level: {ctx.get('energy_level', '?')}",
            f"Urgency: {ctx.get('urgency_level', 'none')} (score: {ctx.get('urgency_score', 0)})",
        ]

        if ctx.get("urgency_summary"):
            lines.append(f"Deadlines: {ctx['urgency_summary']}")

        if ctx.get("time_since_last_interaction"):
            lines.append(f"Last interaction: {ctx['time_since_last_interaction']} ago")

        return "\n".join(lines)

    def get_temporal_system_prompt(
        self,
        existing_system: Optional[str] = None,
        ctx: Optional[Dict] = None,
    ) -> str:
        """Build a system prompt with temporal context injected.

        Args:
            existing_system: User's existing system prompt to augment.
            ctx: Pre-computed temporal context (fetched if None).
        """
        temporal_block = self.format_temporal_block(ctx)
        parts = []

        if self._include_preamble:
            parts.append(self.TEMPORAL_PREAMBLE)
        parts.append(temporal_block)

        if existing_system:
            parts.append(existing_system)

        return "\n\n".join(parts)

    def inject_messages(
        self,
        messages: List[Dict[str, str]],
        ctx: Optional[Dict] = None,
    ) -> List[Dict[str, str]]:
        """Inject temporal context into a message list.

        If a system message exists, augments it. Otherwise, prepends one.
        """
        messages = list(messages)  # don't mutate the original

        existing_system = None
        system_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                existing_system = msg["content"]
                system_idx = i
                break

        new_system = self.get_temporal_system_prompt(existing_system, ctx)

        if system_idx is not None:
            messages[system_idx] = {"role": "system", "content": new_system}
        else:
            messages.insert(0, {"role": "system", "content": new_system})

        return messages

    def chat(
        self,
        client: Any,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        **kwargs,
    ) -> Any:
        """Send a chat completion with temporal context injected.

        Works with any OpenAI-compatible client (OpenAI, Together, Groq, etc.).

        Args:
            client: OpenAI-compatible client with client.chat.completions.create().
            messages: Chat messages.
            model: Model name.
            **kwargs: Additional arguments passed to the API.
        """
        injected = self.inject_messages(messages)
        return client.chat.completions.create(
            model=model,
            messages=injected,
            **kwargs,
        )

    def chat_anthropic(
        self,
        client: Any,
        messages: List[Dict[str, str]],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        **kwargs,
    ) -> Any:
        """Send a message with temporal context via the Anthropic API.

        Anthropic uses a separate `system` parameter instead of a system message.

        Args:
            client: Anthropic client.
            messages: Chat messages (user/assistant only).
            model: Model name.
            max_tokens: Max tokens in response.
            **kwargs: Additional arguments passed to the API.
        """
        ctx = self.get_temporal_context()

        # Extract any existing system from kwargs
        existing_system = kwargs.pop("system", None)
        system_prompt = self.get_temporal_system_prompt(existing_system, ctx)

        # Filter out system messages (Anthropic doesn't accept them in messages)
        filtered = [m for m in messages if m.get("role") != "system"]

        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=filtered,
            **kwargs,
        )

    def wrap_openai(self, client: Any) -> "PulseOpenAIWrapper":
        """Return a drop-in wrapper around an OpenAI client.

        Usage:
            from openai import OpenAI
            raw = OpenAI()
            client = pulse.wrap_openai(raw)
            # Now every call automatically includes temporal context
            client.chat.completions.create(model="gpt-4o", messages=[...])
        """
        return PulseOpenAIWrapper(client, self)

    def __repr__(self) -> str:
        return f"PulseMiddleware(preamble={self._include_preamble})"


class PulseOpenAIWrapper:
    """Drop-in wrapper for OpenAI client that auto-injects temporal context."""

    def __init__(self, client: Any, middleware: PulseMiddleware):
        self._client = client
        self._middleware = middleware
        self.chat = _ChatNamespace(client, middleware)

    def __getattr__(self, name: str):
        return getattr(self._client, name)


class _ChatNamespace:
    def __init__(self, client: Any, middleware: PulseMiddleware):
        self._client = client
        self._middleware = middleware
        self.completions = _CompletionsNamespace(client, middleware)


class _CompletionsNamespace:
    def __init__(self, client: Any, middleware: PulseMiddleware):
        self._client = client
        self._middleware = middleware

    def create(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        injected = self._middleware.inject_messages(messages)
        return self._client.chat.completions.create(messages=injected, **kwargs)
