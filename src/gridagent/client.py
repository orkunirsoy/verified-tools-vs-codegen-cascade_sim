"""Model client: one interface, a real OpenAI backend and an offline stub.

The stub exists so the whole pipeline -- prompt assembly, both arms, the logger, the
grader -- can be built and exercised with no API key and no network. It verifies
everything except the things only a real call can: auth, real token accounting, the
actual response shape, and timeout/rate-limit behaviour. Those are what the one real
smoke call is for, and why it stays a hard gate before any batch.

Deliberately not faked in the stub: token counts default to 0 rather than to invented
numbers. A logger that records plausible-looking zeros is obvious; one that records
plausible-looking fabrications is not.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None
    #: Hidden reasoning tokens, for reasoning models (the GPT-5 family). The API bills
    #: these as OUTPUT and already counts them inside `completion_tokens`, so cost stays
    #: correct without this field -- but "mean tokens" silently stops being comparable
    #: across model families, and that is a headline metric here. Recording the split
    #: keeps a reasoning model's visible output separable from its thinking.
    #: Always 0 for non-reasoning models.
    reasoning_tokens: int = 0

    @property
    def visible_completion_tokens(self) -> int:
        """Output tokens excluding hidden reasoning -- the cross-family comparable figure."""
        return max(0, self.completion_tokens - self.reasoning_tokens)


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0) -> Completion:
        ...


class StubClient:
    """Offline client that replays scripted responses.

    `responses` is consumed in order. A callable receives the messages and returns
    text, which is enough to fake a code-repair loop (first attempt broken, second
    attempt fixed) without touching the network.
    """

    def __init__(self, responses: list[str | Callable[[list[dict]], str]],
                 model: str = "stub"):
        self.model = model
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0) -> Completion:
        self.calls.append(messages)
        if not self._responses:
            return Completion(text="", error="StubClient ran out of scripted responses")
        nxt = self._responses.pop(0)
        text = nxt(messages) if callable(nxt) else nxt
        return Completion(text=text, prompt_tokens=0, completion_tokens=0, latency_s=0.0)


class OpenAIClient:
    """Real backend. Token counts come from the API response, never estimated."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        from openai import OpenAI  # imported lazily so the stub path needs no SDK

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Build and test against StubClient, or "
                "export the key before running a real call."
            )
        self.model = model
        self._client = OpenAI(api_key=key)

    #: Models that reject an explicit `temperature`. The GPT-5 family accepts only the
    #: default (1); passing anything else returns a 400. We run at 1.0, which IS the
    #: default, but the parameter is omitted entirely for these rather than relying on
    #: that coincidence -- and a non-default temperature is refused loudly instead of
    #: being silently ignored, which would make a run look like it honoured a setting it
    #: never received.
    NO_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")

    def _rejects_temperature(self) -> bool:
        return self.model.startswith(self.NO_TEMPERATURE_PREFIXES)

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0) -> Completion:
        t0 = time.time()
        kwargs = {"model": self.model, "messages": messages}
        if self._rejects_temperature():
            if abs(temperature - 1.0) > 1e-9:
                return Completion(
                    text="", latency_s=time.time() - t0,
                    error=(f"ValueError: {self.model} supports only the default "
                           f"temperature (1); got {temperature}. Refusing rather than "
                           f"sending a request whose temperature would be ignored."))
        else:
            kwargs["temperature"] = temperature
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # transport, auth, rate limit -- all recorded, none hidden
            return Completion(text="", latency_s=time.time() - t0, error=f"{type(exc).__name__}: {exc}")
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None)
        return Completion(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
            latency_s=time.time() - t0,
        )
