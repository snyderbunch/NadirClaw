"""Tests for nadirclaw.setup — setup wizard logic."""

import json
import os
import platform
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from nadirclaw.setup import (
    ENV_FILE,
    classify_model_tier,
    detect_existing_config,
    fetch_provider_models,
    get_available_models_for_providers,
    is_first_run,
    select_default_model,
    write_env_file,
)


@pytest.fixture(autouse=True)
def tmp_nadirclaw_dir(tmp_path, monkeypatch):
    """Redirect ~/.nadirclaw to a temp directory for each test."""
    fake_config = tmp_path / ".nadirclaw"
    fake_config.mkdir()
    fake_env = fake_config / ".env"

    monkeypatch.setattr("nadirclaw.setup.CONFIG_DIR", fake_config)
    monkeypatch.setattr("nadirclaw.setup.ENV_FILE", fake_env)

    # Also redirect credentials to avoid touching real ones
    creds_file = fake_config / "credentials.json"
    monkeypatch.setattr(
        "nadirclaw.credentials._credentials_path", lambda: creds_file
    )
    # Clear env vars
    for var in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
        "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    return fake_config, fake_env


# ---------------------------------------------------------------------------
# is_first_run
# ---------------------------------------------------------------------------

class TestIsFirstRun:
    def test_no_env_file(self, tmp_nadirclaw_dir):
        """No .env file means first run."""
        assert is_first_run() is True

    def test_env_file_exists(self, tmp_nadirclaw_dir):
        """Existing .env means not first run."""
        _, fake_env = tmp_nadirclaw_dir
        fake_env.write_text("NADIRCLAW_SIMPLE_MODEL=test\n")
        assert is_first_run() is False


# ---------------------------------------------------------------------------
# classify_model_tier
# ---------------------------------------------------------------------------

class TestClassifyModelTier:
    def test_mini_is_simple(self):
        assert classify_model_tier("gpt-4.1-mini") == "simple"
        assert classify_model_tier("gpt-5-mini") == "simple"

    def test_nano_is_simple(self):
        assert classify_model_tier("gpt-4.1-nano") == "simple"

    def test_flash_is_simple(self):
        assert classify_model_tier("gemini-2.5-flash") == "simple"

    def test_haiku_is_simple(self):
        assert classify_model_tier("claude-haiku-4-5-20251001") == "simple"

    def test_o3_is_reasoning(self):
        assert classify_model_tier("o3") == "reasoning"
        assert classify_model_tier("o3-mini") == "reasoning"

    def test_o4_is_reasoning(self):
        assert classify_model_tier("o4-mini") == "reasoning"

    def test_reasoner_is_reasoning(self):
        assert classify_model_tier("deepseek/deepseek-reasoner") == "reasoning"

    def test_ollama_is_free(self):
        assert classify_model_tier("ollama/llama3.1:8b") == "free"
        assert classify_model_tier("ollama/qwen3:32b") == "free"

    def test_sonnet_is_complex(self):
        assert classify_model_tier("claude-sonnet-4-5-20250929") == "complex"

    def test_opus_is_complex(self):
        assert classify_model_tier("claude-opus-4-6-20250918") == "complex"

    def test_gpt5_is_complex(self):
        assert classify_model_tier("gpt-5") == "complex"
        assert classify_model_tier("gpt-5.2") == "complex"

    def test_gemini_pro_is_complex(self):
        assert classify_model_tier("gemini-2.5-pro") == "complex"


# ---------------------------------------------------------------------------
# get_available_models_for_providers (with fetched models)
# ---------------------------------------------------------------------------

class TestGetAvailableModels:
    def test_fetched_models_used(self):
        """API-fetched models should be used as primary source."""
        fetched = {"openai": ["gpt-4.1", "gpt-4.1-mini", "o3"]}
        tiers = get_available_models_for_providers(["openai"], fetched_models=fetched)
        all_names = [m["model"] for tier in tiers.values() for m in tier]
        assert "gpt-4.1" in all_names
        assert "gpt-4.1-mini" in all_names
        assert "o3" in all_names

    def test_fetched_models_classified_correctly(self):
        """Fetched models should be classified into correct tiers."""
        fetched = {"openai": ["gpt-4.1", "gpt-4.1-mini", "o3"]}
        tiers = get_available_models_for_providers(["openai"], fetched_models=fetched)
        simple_names = [m["model"] for m in tiers["simple"]]
        complex_names = [m["model"] for m in tiers["complex"]]
        reasoning_names = [m["model"] for m in tiers["reasoning"]]
        assert "gpt-4.1-mini" in simple_names
        assert "gpt-4.1" in complex_names
        assert "o3" in reasoning_names

    def test_fallback_to_registry(self):
        """Providers without fetched models should fall back to static registry."""
        tiers = get_available_models_for_providers(["google"], fetched_models={})
        all_names = [m["model"] for tier in tiers.values() for m in tier]
        assert any("gemini" in m for m in all_names)

    def test_empty_providers(self):
        """No providers means no models."""
        tiers = get_available_models_for_providers([])
        for tier_models in tiers.values():
            assert len(tier_models) == 0

    def test_ollama_fetched(self):
        """Ollama fetched models should go to free tier."""
        fetched = {"ollama": ["ollama/llama3.1:8b", "ollama/mistral:7b"]}
        tiers = get_available_models_for_providers(["ollama"], fetched_models=fetched)
        free_names = [m["model"] for m in tiers["free"]]
        assert "ollama/llama3.1:8b" in free_names
        assert "ollama/mistral:7b" in free_names

    def test_mixed_fetched_and_fallback(self):
        """Fetched for one provider, fallback for another."""
        fetched = {"openai": ["gpt-5.2", "gpt-5-mini"]}
        tiers = get_available_models_for_providers(["openai", "google"], fetched_models=fetched)
        all_names = [m["model"] for tier in tiers.values() for m in tier]
        # OpenAI from fetch
        assert "gpt-5.2" in all_names
        # Google from registry fallback
        assert any("gemini" in m for m in all_names)


# ---------------------------------------------------------------------------
# select_default_model
# ---------------------------------------------------------------------------

class TestSelectDefaultModel:
    def test_google_simple(self):
        result = select_default_model("simple", ["google"])
        assert result == "gemini-2.5-flash"

    def test_anthropic_complex(self):
        result = select_default_model("complex", ["anthropic"])
        assert result == "claude-sonnet-4-5-20250929"

    def test_openai_reasoning(self):
        result = select_default_model("reasoning", ["openai"])
        assert result == "o3"

    def test_ollama_free(self):
        result = select_default_model("free", ["ollama"])
        assert result == "ollama/llama3.1:8b"

    def test_no_matching_provider(self):
        result = select_default_model("simple", ["nonexistent"])
        assert result is None

    def test_respects_available_list(self):
        """Should only return a default that appears in the available list."""
        available = [{"model": "gpt-4.1-mini"}, {"model": "gpt-5-mini"}]
        result = select_default_model("simple", ["openai"], available=available)
        assert result == "gpt-4.1-mini"

    def test_skips_unavailable_default(self):
        """If preferred default isn't in available list, try next provider."""
        available = [{"model": "gemini-2.5-flash"}]
        result = select_default_model("simple", ["openai", "google"], available=available)
        assert result == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# fetch_provider_models (mocked)
# ---------------------------------------------------------------------------

class TestFetchProviderModels:
    def test_openai_fetch(self, monkeypatch):
        """Should parse OpenAI /v1/models response."""
        mock_response = json.dumps({
            "data": [
                {"id": "gpt-4.1"},
                {"id": "gpt-4.1-mini"},
                {"id": "gpt-5-mini"},
                {"id": "dall-e-3"},  # should be filtered
                {"id": "text-embedding-3-large"},  # should be filtered
                {"id": "o3"},
                {"id": "tts-1"},  # should be filtered
            ]
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: mock_resp)

        models = fetch_provider_models("openai", "sk-test")
        assert "gpt-4.1" in models
        assert "gpt-4.1-mini" in models
        assert "o3" in models
        assert "dall-e-3" not in models
        assert "tts-1" not in models

    def test_anthropic_fetch(self, monkeypatch):
        """Should parse Anthropic /v1/models response."""
        mock_response = json.dumps({
            "data": [
                {"id": "claude-opus-4-6-20250918"},
                {"id": "claude-sonnet-4-5-20250929"},
                {"id": "claude-haiku-4-5-20251001"},
            ]
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: mock_resp)

        models = fetch_provider_models("anthropic", "sk-ant-test")
        assert "claude-opus-4-6-20250918" in models
        assert "claude-sonnet-4-5-20250929" in models

    def test_google_fetch(self, monkeypatch):
        """Should parse Google GenAI models response."""
        mock_response = json.dumps({
            "models": [
                {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            ]
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: mock_resp)

        models = fetch_provider_models("google", "AIza-test")
        assert "gemini-2.5-flash" in models
        assert "gemini-2.5-pro" in models
        assert "text-embedding-004" not in models

    def test_fetch_failure_returns_empty(self, monkeypatch):
        """API failure should return empty list, not raise."""
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(Exception("fail")))
        models = fetch_provider_models("openai", "bad-key")
        assert models == []

    def test_ollama_fetch(self, monkeypatch):
        """Should parse Ollama /api/tags response."""
        mock_response = json.dumps({
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "qwen3:32b"},
            ]
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: mock_resp)

        models = fetch_provider_models("ollama", "")
        assert "ollama/llama3.1:8b" in models
        assert "ollama/qwen3:32b" in models


# ---------------------------------------------------------------------------
# write_env_file
# ---------------------------------------------------------------------------

class TestWriteEnvFile:
    def test_creates_file(self, tmp_nadirclaw_dir):
        _, fake_env = tmp_nadirclaw_dir
        path = write_env_file(
            simple="gemini-2.5-flash",
            complex_model="gpt-4.1",
        )
        assert path == fake_env
        assert fake_env.exists()

        content = fake_env.read_text()
        assert "NADIRCLAW_SIMPLE_MODEL=gemini-2.5-flash" in content
        assert "NADIRCLAW_COMPLEX_MODEL=gpt-4.1" in content

    def test_includes_api_keys(self, tmp_nadirclaw_dir):
        _, fake_env = tmp_nadirclaw_dir
        write_env_file(
            simple="flash",
            complex_model="gpt-4.1",
            api_keys={"OPENAI_API_KEY": "sk-test-123"},
        )
        content = fake_env.read_text()
        assert "OPENAI_API_KEY=sk-test-123" in content

    def test_includes_optional_tiers(self, tmp_nadirclaw_dir):
        _, fake_env = tmp_nadirclaw_dir
        write_env_file(
            simple="flash",
            complex_model="gpt-4.1",
            reasoning="o3",
            free="ollama/llama3.1:8b",
        )
        content = fake_env.read_text()
        assert "NADIRCLAW_REASONING_MODEL=o3" in content
        assert "NADIRCLAW_FREE_MODEL=ollama/llama3.1:8b" in content

    def test_creates_backup(self, tmp_nadirclaw_dir):
        fake_config, fake_env = tmp_nadirclaw_dir
        fake_env.write_text("OLD_CONFIG=true\n")

        write_env_file(simple="flash", complex_model="gpt-4.1")

        backups = list(fake_config.glob(".env.backup-*"))
        assert len(backups) == 1
        assert "OLD_CONFIG=true" in backups[0].read_text()

    def test_file_permissions(self, tmp_nadirclaw_dir):
        if platform.system() == "Windows":
            pytest.skip("Permission check not applicable on Windows")

        _, fake_env = tmp_nadirclaw_dir
        write_env_file(simple="flash", complex_model="gpt-4.1")
        mode = fake_env.stat().st_mode & 0o777
        assert mode == 0o600

    def test_omits_reasoning_when_none(self, tmp_nadirclaw_dir):
        _, fake_env = tmp_nadirclaw_dir
        write_env_file(simple="flash", complex_model="gpt-4.1")
        content = fake_env.read_text()
        assert "NADIRCLAW_REASONING_MODEL" not in content


# ---------------------------------------------------------------------------
# detect_existing_config
# ---------------------------------------------------------------------------

class TestDetectExistingConfig:
    def test_no_file(self, tmp_nadirclaw_dir):
        assert detect_existing_config() == {}

    def test_reads_config(self, tmp_nadirclaw_dir):
        _, fake_env = tmp_nadirclaw_dir
        fake_env.write_text("NADIRCLAW_SIMPLE_MODEL=flash\nNADIRCLAW_PORT=9000\n")
        config = detect_existing_config()
        assert config["NADIRCLAW_SIMPLE_MODEL"] == "flash"
        assert config["NADIRCLAW_PORT"] == "9000"

    def test_ignores_comments(self, tmp_nadirclaw_dir):
        _, fake_env = tmp_nadirclaw_dir
        fake_env.write_text("# comment\n\nKEY=value\n")
        config = detect_existing_config()
        assert len(config) == 1
        assert config["KEY"] == "value"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestSetupCLI:
    def test_setup_help(self):
        from nadirclaw.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["setup", "--help"])
        assert result.exit_code == 0
        assert "setup wizard" in result.output.lower() or "configure" in result.output.lower()

    def test_setup_already_configured(self, tmp_nadirclaw_dir):
        _, fake_env = tmp_nadirclaw_dir
        fake_env.write_text("NADIRCLAW_SIMPLE_MODEL=test\n")

        from nadirclaw.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["setup"], input="n\n")
        assert result.exit_code == 0
