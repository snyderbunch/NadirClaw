"""Tests for nadirclaw.setup — setup wizard logic."""

import os
import platform
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from nadirclaw.setup import (
    ENV_FILE,
    detect_existing_config,
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
# get_available_models_for_providers
# ---------------------------------------------------------------------------

class TestGetAvailableModels:
    def test_google_only(self):
        """Google-only should show gemini models."""
        tiers = get_available_models_for_providers(["google"])
        model_names = [m["model"] for tier in tiers.values() for m in tier]
        assert any("gemini" in m for m in model_names)
        # Should not include OpenAI or Anthropic models
        assert not any("gpt" in m for m in model_names)
        assert not any("claude" in m for m in model_names)

    def test_ollama_only(self):
        """Ollama-only should show local models."""
        tiers = get_available_models_for_providers(["ollama"])
        model_names = [m["model"] for tier in tiers.values() for m in tier]
        assert any("ollama" in m for m in model_names)
        assert not any("gemini" in m for m in model_names)

    def test_empty_providers(self):
        """No providers means no models."""
        tiers = get_available_models_for_providers([])
        for tier_models in tiers.values():
            assert len(tier_models) == 0

    def test_multiple_providers(self):
        """Multiple providers should union models."""
        tiers = get_available_models_for_providers(["google", "openai"])
        model_names = [m["model"] for tier in tiers.values() for m in tier]
        assert any("gemini" in m for m in model_names)
        assert any("gpt" in m for m in model_names)

    def test_free_tier_has_zero_cost(self):
        """Free tier should only contain zero-cost models."""
        tiers = get_available_models_for_providers(["ollama"])
        for m in tiers["free"]:
            assert m["cost_in"] == 0
            assert m["cost_out"] == 0

    def test_simple_tier_is_cheap(self):
        """Simple tier should only contain models with cost <= 0.20."""
        tiers = get_available_models_for_providers(["google", "openai", "deepseek", "ollama"])
        for m in tiers["simple"]:
            assert m["cost_in"] <= 0.20


# ---------------------------------------------------------------------------
# select_default_model
# ---------------------------------------------------------------------------

class TestSelectDefaultModel:
    def test_google_simple(self):
        """Google should default to gemini-3-flash-preview for simple."""
        result = select_default_model("simple", ["google"])
        assert result == "gemini-3-flash-preview"

    def test_anthropic_complex(self):
        """Anthropic should default to claude-sonnet for complex."""
        result = select_default_model("complex", ["anthropic"])
        assert result == "claude-sonnet-4-20250514"

    def test_openai_reasoning(self):
        """OpenAI should default to o3 for reasoning."""
        result = select_default_model("reasoning", ["openai"])
        assert result == "o3"

    def test_ollama_free(self):
        """Ollama should default to llama for free tier."""
        result = select_default_model("free", ["ollama"])
        assert result == "ollama/llama3.1:8b"

    def test_no_matching_provider(self):
        """Unknown provider returns None."""
        result = select_default_model("simple", ["nonexistent"])
        assert result is None

    def test_priority_order(self):
        """First matching provider in preference order wins."""
        # For complex, anthropic comes before openai in _TIER_DEFAULTS
        result = select_default_model("complex", ["openai", "anthropic"])
        assert result == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# write_env_file
# ---------------------------------------------------------------------------

class TestWriteEnvFile:
    def test_creates_file(self, tmp_nadirclaw_dir):
        """Should create .env with correct variables."""
        _, fake_env = tmp_nadirclaw_dir
        path = write_env_file(
            simple="gemini-3-flash-preview",
            complex_model="gpt-4o",
        )
        assert path == fake_env
        assert fake_env.exists()

        content = fake_env.read_text()
        assert "NADIRCLAW_SIMPLE_MODEL=gemini-3-flash-preview" in content
        assert "NADIRCLAW_COMPLEX_MODEL=gpt-4o" in content

    def test_includes_api_keys(self, tmp_nadirclaw_dir):
        """Should include API keys when provided."""
        _, fake_env = tmp_nadirclaw_dir
        write_env_file(
            simple="flash",
            complex_model="gpt-4o",
            api_keys={"OPENAI_API_KEY": "sk-test-123"},
        )
        content = fake_env.read_text()
        assert "OPENAI_API_KEY=sk-test-123" in content

    def test_includes_optional_tiers(self, tmp_nadirclaw_dir):
        """Should include reasoning and free models when provided."""
        _, fake_env = tmp_nadirclaw_dir
        write_env_file(
            simple="flash",
            complex_model="gpt-4o",
            reasoning="o3",
            free="ollama/llama3.1:8b",
        )
        content = fake_env.read_text()
        assert "NADIRCLAW_REASONING_MODEL=o3" in content
        assert "NADIRCLAW_FREE_MODEL=ollama/llama3.1:8b" in content

    def test_creates_backup(self, tmp_nadirclaw_dir):
        """Should backup existing .env before overwriting."""
        fake_config, fake_env = tmp_nadirclaw_dir
        fake_env.write_text("OLD_CONFIG=true\n")

        write_env_file(simple="flash", complex_model="gpt-4o")

        # Check a backup file was created
        backups = list(fake_config.glob(".env.backup-*"))
        assert len(backups) == 1
        assert "OLD_CONFIG=true" in backups[0].read_text()

    def test_file_permissions(self, tmp_nadirclaw_dir):
        """Should set 0o600 permissions on Unix."""
        if platform.system() == "Windows":
            pytest.skip("Permission check not applicable on Windows")

        _, fake_env = tmp_nadirclaw_dir
        write_env_file(simple="flash", complex_model="gpt-4o")
        mode = fake_env.stat().st_mode & 0o777
        assert mode == 0o600

    def test_omits_reasoning_when_none(self, tmp_nadirclaw_dir):
        """Should not include REASONING_MODEL when None."""
        _, fake_env = tmp_nadirclaw_dir
        write_env_file(simple="flash", complex_model="gpt-4o")
        content = fake_env.read_text()
        assert "NADIRCLAW_REASONING_MODEL" not in content


# ---------------------------------------------------------------------------
# detect_existing_config
# ---------------------------------------------------------------------------

class TestDetectExistingConfig:
    def test_no_file(self, tmp_nadirclaw_dir):
        """No .env returns empty dict."""
        assert detect_existing_config() == {}

    def test_reads_config(self, tmp_nadirclaw_dir):
        """Reads key=value pairs from .env."""
        _, fake_env = tmp_nadirclaw_dir
        fake_env.write_text("NADIRCLAW_SIMPLE_MODEL=flash\nNADIRCLAW_PORT=9000\n")
        config = detect_existing_config()
        assert config["NADIRCLAW_SIMPLE_MODEL"] == "flash"
        assert config["NADIRCLAW_PORT"] == "9000"

    def test_ignores_comments(self, tmp_nadirclaw_dir):
        """Skips comment lines and empty lines."""
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
        """nadirclaw setup --help should work."""
        from nadirclaw.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["setup", "--help"])
        assert result.exit_code == 0
        assert "setup wizard" in result.output.lower() or "configure" in result.output.lower()

    def test_setup_already_configured(self, tmp_nadirclaw_dir):
        """nadirclaw setup when already configured should ask to re-run."""
        _, fake_env = tmp_nadirclaw_dir
        fake_env.write_text("NADIRCLAW_SIMPLE_MODEL=test\n")

        from nadirclaw.cli import main
        runner = CliRunner()
        # Answer 'n' to "Already configured. Re-run setup?"
        result = runner.invoke(main, ["setup"], input="n\n")
        assert result.exit_code == 0
