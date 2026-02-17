"""Tests for nadirclaw.routing — routing intelligence."""

from types import SimpleNamespace

import pytest

from nadirclaw.routing import (
    MODEL_ALIASES,
    SessionCache,
    apply_routing_modifiers,
    check_context_window,
    detect_agentic,
    detect_reasoning,
    estimate_cost,
    estimate_token_count,
    resolve_alias,
    resolve_profile,
)


# Helper to create fake message objects
def _msg(role, content=""):
    ns = SimpleNamespace(role=role, content=content)
    ns.text_content = lambda: content
    return ns


# ---------------------------------------------------------------------------
# resolve_profile
# ---------------------------------------------------------------------------

class TestResolveProfile:
    def test_auto(self):
        assert resolve_profile("auto") == "auto"

    def test_eco(self):
        assert resolve_profile("eco") == "eco"

    def test_premium(self):
        assert resolve_profile("premium") == "premium"

    def test_free(self):
        assert resolve_profile("free") == "free"

    def test_reasoning(self):
        assert resolve_profile("reasoning") == "reasoning"

    def test_nadirclaw_prefix(self):
        assert resolve_profile("nadirclaw/eco") == "eco"
        assert resolve_profile("nadirclaw/premium") == "premium"

    def test_case_insensitive(self):
        assert resolve_profile("ECO") == "eco"
        assert resolve_profile("NadirClaw/Premium") == "premium"

    def test_not_a_profile(self):
        assert resolve_profile("gpt-4o") is None
        assert resolve_profile("claude-sonnet") is None

    def test_none(self):
        assert resolve_profile(None) is None

    def test_empty(self):
        assert resolve_profile("") is None


# ---------------------------------------------------------------------------
# resolve_alias
# ---------------------------------------------------------------------------

class TestResolveAlias:
    def test_sonnet(self):
        assert resolve_alias("sonnet") == "claude-sonnet-4-5-20250929"

    def test_opus(self):
        assert resolve_alias("opus") == "claude-opus-4-6-20250918"

    def test_gpt4(self):
        assert resolve_alias("gpt4") == "gpt-4.1"

    def test_flash(self):
        assert resolve_alias("flash") == "gemini-2.5-flash"

    def test_case_insensitive(self):
        assert resolve_alias("SONNET") == "claude-sonnet-4-5-20250929"
        assert resolve_alias("Flash") == "gemini-2.5-flash"

    def test_unknown(self):
        assert resolve_alias("unknown-model") is None

    def test_deepseek(self):
        assert resolve_alias("deepseek") == "deepseek/deepseek-chat"
        assert resolve_alias("deepseek-r1") == "deepseek/deepseek-reasoner"


# ---------------------------------------------------------------------------
# detect_agentic
# ---------------------------------------------------------------------------

class TestDetectAgentic:
    def test_not_agentic_simple(self):
        messages = [_msg("user", "What is 2+2?")]
        result = detect_agentic(messages)
        assert result["is_agentic"] is False
        assert result["confidence"] == 0.0

    def test_tools_defined(self):
        messages = [_msg("user", "Help me")]
        result = detect_agentic(messages, has_tools=True, tool_count=3)
        assert result["is_agentic"] is True
        assert "tools_defined(3)" in result["signals"]

    def test_many_tools(self):
        messages = [_msg("user", "Help me")]
        result = detect_agentic(messages, has_tools=True, tool_count=5)
        assert result["confidence"] >= 0.5
        assert "many_tools" in result["signals"]

    def test_tool_messages(self):
        messages = [
            _msg("user", "Do it"),
            _msg("assistant", "calling tool"),
            _msg("tool", "result"),
        ]
        result = detect_agentic(messages)
        assert result["is_agentic"] is False  # tool messages alone = 0.3, below 0.35
        assert "tool_messages(1)" in result["signals"]

    def test_tool_messages_with_tools(self):
        messages = [
            _msg("user", "Do it"),
            _msg("assistant", "calling tool"),
            _msg("tool", "result"),
        ]
        result = detect_agentic(messages, has_tools=True, tool_count=2)
        assert result["is_agentic"] is True

    def test_agentic_cycles(self):
        messages = [
            _msg("user", "Do it"),
            _msg("assistant", "step 1"),
            _msg("tool", "result 1"),
            _msg("assistant", "step 2"),
            _msg("tool", "result 2"),
            _msg("assistant", "done"),
        ]
        result = detect_agentic(messages)
        assert result["is_agentic"] is True
        assert any("agentic_cycles" in s for s in result["signals"])

    def test_agentic_system_keywords(self):
        messages = [_msg("user", "Help")]
        result = detect_agentic(
            messages,
            system_prompt="You are a coding agent. You can execute commands and read files.",
        )
        assert "agentic_keywords" in result["signals"]

    def test_long_system_prompt(self):
        messages = [_msg("user", "Help")]
        result = detect_agentic(messages, system_prompt_length=800)
        assert "long_system_prompt" in result["signals"]

    def test_deep_conversation(self):
        messages = [_msg("user", f"msg {i}") for i in range(12)]
        result = detect_agentic(messages, message_count=12)
        assert "deep_conversation" in result["signals"]

    def test_full_agentic_request(self):
        """Realistic agentic request with multiple signals."""
        messages = [
            _msg("system", "You are an AI agent. You can use tools to read and write files."),
            _msg("user", "Refactor the auth module"),
            _msg("assistant", "I'll start by reading the file"),
            _msg("tool", "file contents here"),
            _msg("assistant", "Now I'll write the updated file"),
            _msg("tool", "success"),
            _msg("user", "Now add tests"),
        ]
        result = detect_agentic(
            messages,
            has_tools=True,
            tool_count=4,
            system_prompt="You are an AI agent. You can use tools to read and write files.",
            system_prompt_length=600,
            message_count=7,
        )
        assert result["is_agentic"] is True
        assert result["confidence"] >= 0.8


# ---------------------------------------------------------------------------
# detect_reasoning
# ---------------------------------------------------------------------------

class TestDetectReasoning:
    def test_not_reasoning(self):
        result = detect_reasoning("What is 2+2?")
        assert result["is_reasoning"] is False

    def test_single_marker(self):
        result = detect_reasoning("Think through this problem")
        assert result["is_reasoning"] is False  # need 2+ markers
        assert result["marker_count"] == 1

    def test_two_markers(self):
        result = detect_reasoning("Think through this step by step")
        assert result["is_reasoning"] is True
        assert result["marker_count"] >= 2

    def test_reasoning_in_system(self):
        result = detect_reasoning(
            "What are the implications?",
            system_message="Analyze the tradeoffs and compare and contrast the approaches",
        )
        assert result["is_reasoning"] is True

    def test_proof_request(self):
        result = detect_reasoning("Prove that P=NP and derive the implications step by step")
        assert result["is_reasoning"] is True

    def test_critical_analysis(self):
        result = detect_reasoning("Critically analyze the paper and evaluate whether the conclusions are valid")
        assert result["is_reasoning"] is True


# ---------------------------------------------------------------------------
# check_context_window
# ---------------------------------------------------------------------------

class TestContextWindow:
    def test_fits(self):
        messages = [_msg("user", "short")]
        assert check_context_window("gpt-4o", messages) is True

    def test_unknown_model_passes(self):
        messages = [_msg("user", "x" * 100000)]
        assert check_context_window("unknown-model-xyz", messages) is True

    def test_exceeds(self):
        # gpt-4o has 128k context. 128k * 4 = 512k chars
        content = "x" * 600_000
        messages = [_msg("user", content)]
        assert check_context_window("gpt-4o", messages) is False

    def test_gemini_large_context(self):
        # Gemini has 1M context
        content = "x" * 600_000
        messages = [_msg("user", content)]
        assert check_context_window("gemini-3-flash-preview", messages) is True


class TestEstimateTokenCount:
    def test_basic(self):
        messages = [_msg("user", "hello world")]  # 11 chars → ~2 tokens
        count = estimate_token_count(messages)
        assert count == 2

    def test_multiple_messages(self):
        messages = [_msg("user", "a" * 400), _msg("assistant", "b" * 400)]
        count = estimate_token_count(messages)
        assert count == 200


# ---------------------------------------------------------------------------
# SessionCache
# ---------------------------------------------------------------------------

class TestSessionCache:
    def test_put_and_get(self):
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("system", "You are helpful"), _msg("user", "Hello")]
        cache.put(msgs, "gpt-4o", "complex")
        result = cache.get(msgs)
        assert result == ("gpt-4o", "complex")

    def test_miss(self):
        cache = SessionCache(ttl_seconds=60)
        msgs = [_msg("user", "Hello")]
        assert cache.get(msgs) is None

    def test_expiry(self):
        cache = SessionCache(ttl_seconds=0)  # immediate expiry
        msgs = [_msg("user", "Hello")]
        cache.put(msgs, "gpt-4o", "complex")
        import time
        time.sleep(0.01)
        assert cache.get(msgs) is None

    def test_same_session_different_followup(self):
        """Same system + first user msg → same cache key regardless of later messages."""
        cache = SessionCache(ttl_seconds=60)
        msgs1 = [_msg("system", "Be helpful"), _msg("user", "Hello")]
        msgs2 = [_msg("system", "Be helpful"), _msg("user", "Hello"), _msg("assistant", "Hi"), _msg("user", "More")]
        cache.put(msgs1, "gpt-4o", "complex")
        result = cache.get(msgs2)
        assert result == ("gpt-4o", "complex")

    def test_clear_expired(self):
        cache = SessionCache(ttl_seconds=0)
        msgs = [_msg("user", "Hello")]
        cache.put(msgs, "gpt-4o", "complex")
        import time
        time.sleep(0.01)
        removed = cache.clear_expired()
        assert removed == 1


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_known_model(self):
        cost = estimate_cost("gpt-4o", 1000, 500)
        assert cost is not None
        assert cost > 0

    def test_unknown_model(self):
        assert estimate_cost("unknown-xyz", 1000, 500) is None

    def test_free_model(self):
        cost = estimate_cost("ollama/llama3.1:8b", 1000, 500)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# apply_routing_modifiers
# ---------------------------------------------------------------------------

class TestApplyRoutingModifiers:
    def test_no_modifiers(self):
        """Simple request stays simple."""
        messages = [_msg("user", "What is 2+2?")]
        meta = {"has_tools": False, "tool_count": 0, "system_prompt_text": "", "system_prompt_length": 0, "message_count": 1}
        model, tier, info = apply_routing_modifiers(
            "gemini-flash", "simple", meta, messages, "gemini-flash", "gpt-4o",
        )
        assert model == "gemini-flash"
        assert tier == "simple"

    def test_agentic_override(self):
        """Agentic request overrides simple → complex."""
        messages = [
            _msg("system", "You are a coding agent. You can use tools."),
            _msg("user", "Refactor this"),
            _msg("assistant", "reading file"),
            _msg("tool", "contents"),
            _msg("assistant", "writing file"),
            _msg("tool", "done"),
        ]
        meta = {
            "has_tools": True, "tool_count": 4,
            "system_prompt_text": "You are a coding agent. You can use tools.",
            "system_prompt_length": 600, "message_count": 6,
        }
        model, tier, info = apply_routing_modifiers(
            "gemini-flash", "simple", meta, messages, "gemini-flash", "gpt-4o",
        )
        assert model == "gpt-4o"
        assert tier == "complex"
        assert "agentic_override" in info["modifiers_applied"]

    def test_agentic_no_override_if_already_complex(self):
        """Agentic request doesn't change anything if already complex."""
        messages = [
            _msg("user", "Do it"),
            _msg("assistant", "step"),
            _msg("tool", "result"),
            _msg("assistant", "step"),
            _msg("tool", "result"),
        ]
        meta = {"has_tools": True, "tool_count": 3, "system_prompt_text": "", "system_prompt_length": 0, "message_count": 5}
        model, tier, info = apply_routing_modifiers(
            "gpt-4o", "complex", meta, messages, "gemini-flash", "gpt-4o",
        )
        assert model == "gpt-4o"
        assert tier == "complex"
        assert "agentic_override" not in info["modifiers_applied"]

    def test_reasoning_override(self):
        """Reasoning markers override to reasoning model."""
        messages = [_msg("user", "Think through this step by step and analyze the tradeoffs")]
        meta = {"has_tools": False, "tool_count": 0, "system_prompt_text": "", "system_prompt_length": 0, "message_count": 1}
        model, tier, info = apply_routing_modifiers(
            "gemini-flash", "simple", meta, messages,
            "gemini-flash", "gpt-4o", reasoning_model="o3",
        )
        assert model == "o3"
        assert tier == "reasoning"
        assert "reasoning_override" in info["modifiers_applied"]

    def test_reasoning_falls_back_to_complex(self):
        """Without a reasoning model configured, falls back to complex."""
        messages = [_msg("user", "Think through this step by step and analyze the tradeoffs")]
        meta = {"has_tools": False, "tool_count": 0, "system_prompt_text": "", "system_prompt_length": 0, "message_count": 1}
        model, tier, info = apply_routing_modifiers(
            "gemini-flash", "simple", meta, messages,
            "gemini-flash", "gpt-4o",
        )
        assert model == "gpt-4o"
        assert tier == "reasoning"

    def test_context_window_swap(self):
        """Swaps model when context window is exceeded."""
        # gpt-4o-mini: 128k context. Make content exceed that.
        big_content = "x" * 600_000  # ~150k tokens
        messages = [_msg("user", big_content)]
        meta = {"has_tools": False, "tool_count": 0, "system_prompt_text": "", "system_prompt_length": 0, "message_count": 1}
        model, tier, info = apply_routing_modifiers(
            "gpt-4o-mini", "simple", meta, messages,
            "gpt-4o-mini", "gemini-2.5-pro",  # gemini has 1M context
        )
        assert model == "gemini-2.5-pro"
        assert any("context_window_swap" in m for m in info["modifiers_applied"])
