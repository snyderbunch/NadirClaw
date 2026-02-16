"""Interactive setup wizard for NadirClaw.

Guides users through provider selection, credential entry, and model
configuration on first run or via `nadirclaw setup`.
"""

import os
import shutil
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click

from nadirclaw.routing import MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------

PROVIDER_INFO: Dict[str, Dict] = {
    "openai": {
        "display": "OpenAI",
        "description": "GPT-4o, o3, Codex",
        "env_var": "OPENAI_API_KEY",
        "key_prefix": "sk-",
        "oauth": True,
        "credential_key": "openai",
    },
    "anthropic": {
        "display": "Anthropic",
        "description": "Claude Sonnet, Opus, Haiku",
        "env_var": "ANTHROPIC_API_KEY",
        "key_prefix": "sk-ant-",
        "oauth": True,
        "credential_key": "anthropic",
    },
    "google": {
        "display": "Google / Gemini",
        "description": "Gemini Flash, Pro",
        "env_var": "GEMINI_API_KEY",
        "key_prefix": "AIza",
        "oauth": True,
        "credential_key": "google",
    },
    "deepseek": {
        "display": "DeepSeek",
        "description": "DeepSeek Chat, Reasoner",
        "env_var": "DEEPSEEK_API_KEY",
        "key_prefix": "sk-",
        "oauth": False,
        "credential_key": "deepseek",
    },
    "ollama": {
        "display": "Ollama (local)",
        "description": "Llama, Qwen — no API key",
        "env_var": None,
        "key_prefix": None,
        "oauth": False,
        "credential_key": "ollama",
    },
}

PROVIDER_ORDER = ["openai", "anthropic", "google", "deepseek", "ollama"]

# Model → provider mapping (for filtering available models)
_MODEL_PROVIDER_MAP: Dict[str, str] = {
    "gemini-3-flash-preview": "google",
    "gemini-2.5-pro": "google",
    "gemini-2.5-flash": "google",
    "gemini/gemini-3-flash-preview": "google",
    "gemini/gemini-2.5-pro": "google",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "o3": "openai",
    "o3-mini": "openai",
    "openai-codex/gpt-5.3-codex": "openai",
    "claude-opus-4-20250514": "anthropic",
    "claude-sonnet-4-20250514": "anthropic",
    "claude-haiku-4-20250514": "anthropic",
    "deepseek/deepseek-chat": "deepseek",
    "deepseek/deepseek-reasoner": "deepseek",
    "ollama/llama3.1:8b": "ollama",
    "ollama/qwen3:32b": "ollama",
}

# Tier defaults — ordered preference per provider
_TIER_DEFAULTS = {
    "simple": {
        "google": "gemini-3-flash-preview",
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek/deepseek-chat",
        "ollama": "ollama/llama3.1:8b",
        "anthropic": "claude-haiku-4-20250514",
    },
    "complex": {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "google": "gemini-2.5-pro",
        "deepseek": "deepseek/deepseek-reasoner",
        "ollama": "ollama/qwen3:32b",
    },
    "reasoning": {
        "openai": "o3",
        "deepseek": "deepseek/deepseek-reasoner",
        "anthropic": "claude-sonnet-4-20250514",
        "google": "gemini-2.5-pro",
    },
    "free": {
        "ollama": "ollama/llama3.1:8b",
    },
}

# Config directory
CONFIG_DIR = Path.home() / ".nadirclaw"
ENV_FILE = CONFIG_DIR / ".env"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_first_run() -> bool:
    """Check if NadirClaw has been configured (i.e. .env exists)."""
    return not ENV_FILE.exists()


def detect_existing_config() -> Dict[str, str]:
    """Read existing .env file and return key-value pairs."""
    config: Dict[str, str] = {}
    if not ENV_FILE.exists():
        return config
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    return config


def detect_existing_credentials() -> List[str]:
    """Return list of providers that already have credentials configured."""
    from nadirclaw.credentials import get_credential

    found = []
    for provider_key, info in PROVIDER_INFO.items():
        if provider_key == "ollama":
            continue
        cred_key = info["credential_key"]
        if get_credential(cred_key):
            found.append(provider_key)
        elif info["env_var"] and os.getenv(info["env_var"], ""):
            found.append(provider_key)
    return found


def _format_cost(cost: float) -> str:
    """Format a cost value for display."""
    if cost == 0:
        return "FREE"
    return f"${cost:.2f}"


def _format_context(ctx: int) -> str:
    """Format context window for display."""
    if ctx >= 1_000_000:
        return f"{ctx // 1_000_000}M ctx"
    return f"{ctx // 1_000}K ctx"


# ---------------------------------------------------------------------------
# Step 1: Welcome
# ---------------------------------------------------------------------------

def print_welcome():
    """Print welcome banner."""
    click.echo()
    click.echo("=" * 56)
    click.echo("  NadirClaw Setup Wizard")
    click.echo("=" * 56)
    click.echo()
    click.echo("This wizard will help you configure NadirClaw by:")
    click.echo("  1. Selecting your LLM providers")
    click.echo("  2. Entering API keys or logging in via OAuth")
    click.echo("  3. Choosing models for each routing tier")
    click.echo()
    click.echo("Your configuration will be saved to ~/.nadirclaw/.env")
    click.echo()


# ---------------------------------------------------------------------------
# Step 2: Provider selection
# ---------------------------------------------------------------------------

def prompt_provider_selection(existing: Optional[List[str]] = None) -> List[str]:
    """Multi-select providers via numbered menu."""
    click.echo("Which LLM providers do you want to use?")
    click.echo()

    for i, key in enumerate(PROVIDER_ORDER, 1):
        info = PROVIDER_INFO[key]
        marker = " *" if existing and key in existing else ""
        click.echo(f"  {i}. {info['display']:20s} ({info['description']}){marker}")

    if existing:
        click.echo()
        click.echo("  * = credential already configured")

    click.echo()
    raw = click.prompt(
        "Select [comma-separated, e.g. 1,3,5]",
        default="1,3" if not existing else None,
    )

    selected = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(PROVIDER_ORDER):
                selected.append(PROVIDER_ORDER[idx])
    if not selected:
        click.echo("No valid selections. Defaulting to Google/Gemini.")
        selected = ["google"]

    names = ", ".join(PROVIDER_INFO[p]["display"] for p in selected)
    click.echo(f"\nSelected: {names}\n")
    return selected


# ---------------------------------------------------------------------------
# Step 3: Credential collection
# ---------------------------------------------------------------------------

def _check_ollama_connectivity() -> bool:
    """Check if Ollama is running at localhost:11434."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def prompt_credential_for_provider(provider: str, reconfigure: bool = False) -> Optional[str]:
    """Prompt user for credentials for a single provider.

    Returns the credential string, or None if skipped.
    """
    from nadirclaw.credentials import get_credential, save_credential

    info = PROVIDER_INFO[provider]

    # Ollama needs no key
    if provider == "ollama":
        click.echo(f"  {info['display']}: Checking connectivity...")
        if _check_ollama_connectivity():
            click.echo("    Ollama is running at localhost:11434")
        else:
            click.echo("    Ollama not detected at localhost:11434")
            click.echo("    Make sure Ollama is running before using local models.")
        click.echo()
        return "local"

    # Check existing credential
    cred_key = info["credential_key"]
    existing = get_credential(cred_key)
    if existing and not reconfigure:
        masked = existing[:8] + "..." + existing[-4:] if len(existing) > 12 else existing[:4] + "***"
        click.echo(f"  {info['display']}: credential already configured ({masked})")
        if not click.confirm("    Update this credential?", default=False):
            click.echo()
            return existing

    click.echo(f"\n  {info['display']}:")

    if info["oauth"]:
        click.echo("    1. Enter API key")
        click.echo("    2. Login via OAuth (opens browser)")
        choice = click.prompt("    Choose", type=click.Choice(["1", "2"]), default="1")
    else:
        choice = "1"

    if choice == "1":
        key = click.prompt(f"    {info['display']} API key", hide_input=True)
        key = key.strip()
        if not key:
            click.echo("    Skipped (empty key).")
            click.echo()
            return None
        save_credential(cred_key, key, source="manual")
        click.echo(f"    Saved!")
        click.echo()
        return key
    else:
        # OAuth flow
        return _run_oauth_for_provider(provider)


def _run_oauth_for_provider(provider: str) -> Optional[str]:
    """Run the OAuth flow for a provider. Returns access token or None."""
    from nadirclaw.credentials import save_oauth_credential

    try:
        if provider == "openai":
            from nadirclaw.oauth import login_openai
            token_data = login_openai(timeout=300)
            if token_data and token_data.get("access_token"):
                import time
                expires_in = max(int(token_data.get("expires_at", 0) - time.time()), 3600)
                save_oauth_credential(
                    "openai-codex",
                    token_data["access_token"],
                    token_data.get("refresh_token", ""),
                    expires_in,
                )
                click.echo("    OpenAI OAuth login successful!")
                click.echo()
                return token_data["access_token"]
        elif provider == "anthropic":
            from nadirclaw.oauth import validate_anthropic_setup_token
            click.echo("    Paste your Anthropic setup token (from `claude setup-token`):")
            token = click.prompt("    Token", hide_input=True).strip()
            error = validate_anthropic_setup_token(token)
            if error:
                click.echo(f"    Invalid token: {error}")
                click.echo()
                return None
            from nadirclaw.credentials import save_credential
            save_credential("anthropic", token, source="setup-token")
            click.echo("    Anthropic token saved!")
            click.echo()
            return token
        elif provider == "google":
            from nadirclaw.oauth import login_gemini
            token_data = login_gemini(timeout=300)
            if token_data and token_data.get("access_token"):
                import time
                expires_in = max(int(token_data.get("expires_at", 0) - time.time()), 3600)
                save_oauth_credential(
                    "gemini",
                    token_data["access_token"],
                    token_data.get("refresh_token", ""),
                    expires_in,
                    metadata={
                        "project_id": token_data.get("project_id", ""),
                        "email": token_data.get("email", ""),
                    },
                )
                click.echo("    Gemini OAuth login successful!")
                click.echo()
                return token_data["access_token"]
    except Exception as e:
        click.echo(f"    OAuth failed: {e}")
        click.echo("    You can try again later with `nadirclaw auth <provider> login`.")
        click.echo()
        return None

    click.echo()
    return None


# ---------------------------------------------------------------------------
# Step 4: Model selection
# ---------------------------------------------------------------------------

def get_available_models_for_providers(providers: List[str]) -> Dict[str, List[dict]]:
    """Filter MODEL_REGISTRY by configured providers, grouped by tier.

    Returns dict with keys: simple, complex, reasoning, free.
    Each value is a list of dicts: {model, cost_in, cost_out, context, provider}.
    """
    # Deduplicate models — skip gemini/ prefixed duplicates
    skip_prefixed = set()
    for model in MODEL_REGISTRY:
        if model.startswith("gemini/"):
            skip_prefixed.add(model)

    available = []
    for model, info in MODEL_REGISTRY.items():
        if model in skip_prefixed:
            continue
        model_provider = _MODEL_PROVIDER_MAP.get(model)
        if model_provider and model_provider in providers:
            available.append({
                "model": model,
                "cost_in": info["cost_per_m_input"],
                "cost_out": info["cost_per_m_output"],
                "context": info["context_window"],
                "provider": model_provider,
            })

    # Classify into tiers
    tiers: Dict[str, List[dict]] = {
        "simple": [],
        "complex": [],
        "reasoning": [],
        "free": [],
    }

    for m in available:
        # Free tier
        if m["cost_in"] == 0 and m["cost_out"] == 0:
            tiers["free"].append(m)

        # Simple tier: cheap models
        if m["cost_in"] <= 0.20:
            tiers["simple"].append(m)

        # Complex tier: expensive models
        if m["cost_in"] > 0.20:
            tiers["complex"].append(m)

        # Reasoning tier: specific models
        if m["model"] in ("o3", "o3-mini", "deepseek/deepseek-reasoner"):
            tiers["reasoning"].append(m)

    # Sort each tier by cost
    for tier in tiers.values():
        tier.sort(key=lambda x: (x["cost_in"], x["model"]))

    return tiers


def format_model_table(models: List[dict], tier: str) -> str:
    """Format a model selection table for display."""
    tier_labels = {
        "simple": "Simple model (cheap/fast)",
        "complex": "Complex model (premium)",
        "reasoning": "Reasoning model (chain-of-thought)",
        "free": "Free model (zero cost)",
    }
    lines = [f"\n{tier_labels.get(tier, tier)}:"]

    for i, m in enumerate(models, 1):
        cost_str = (
            "FREE"
            if m["cost_in"] == 0
            else f"${m['cost_in']:.2f}/${m['cost_out']:.2f} per 1M"
        )
        ctx_str = _format_context(m["context"])
        # Mark recommended (first model in sorted list)
        rec = "  [RECOMMENDED]" if i == 1 else ""
        lines.append(f"  {i}. {m['model']:36s} {cost_str:24s} {ctx_str}{rec}")

    return "\n".join(lines)


def select_default_model(tier: str, providers: List[str]) -> Optional[str]:
    """Pick the best default model for a tier based on configured providers."""
    tier_prefs = _TIER_DEFAULTS.get(tier, {})
    for provider in tier_prefs:
        if provider in providers:
            model = tier_prefs[provider]
            # Verify model is in registry
            if model in MODEL_REGISTRY:
                return model
    return None


def prompt_model_selection(tier: str, models: List[dict], providers: List[str]) -> Optional[str]:
    """Show model table and prompt for selection. Returns model name or None."""
    if not models:
        return None

    table = format_model_table(models, tier)
    click.echo(table)

    default_model = select_default_model(tier, providers)
    default_idx = "1"
    for i, m in enumerate(models, 1):
        if m["model"] == default_model:
            default_idx = str(i)
            break

    is_optional = tier in ("reasoning", "free")
    prompt_text = f"Select [1-{len(models)}]"
    if is_optional:
        prompt_text += " or 's' to skip"

    raw = click.prompt(prompt_text, default=default_idx)
    raw = raw.strip().lower()

    if is_optional and raw in ("s", "skip", ""):
        click.echo(f"  Skipped {tier} tier.\n")
        return None

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(models):
            chosen = models[idx]["model"]
            click.echo(f"  Selected: {chosen}\n")
            return chosen

    # Fallback to first
    chosen = models[0]["model"]
    click.echo(f"  Selected: {chosen}\n")
    return chosen


# ---------------------------------------------------------------------------
# Step 5: Write config + summary
# ---------------------------------------------------------------------------

def write_env_file(
    simple: str,
    complex_model: str,
    reasoning: Optional[str] = None,
    free: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None,
) -> Path:
    """Write ~/.nadirclaw/.env with model configuration.

    Creates backup of existing .env if present. Sets 0o600 permissions.
    Returns path to written file.
    """
    import platform

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Backup existing .env
    if ENV_FILE.exists():
        backup_name = f".env.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        backup_path = CONFIG_DIR / backup_name
        shutil.copy2(ENV_FILE, backup_path)
        click.echo(f"  Backed up existing config to {backup_path}")

    lines = [
        "# NadirClaw configuration",
        f"# Generated by `nadirclaw setup` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # API keys
    if api_keys:
        lines.append("# API Keys")
        for env_var, value in sorted(api_keys.items()):
            lines.append(f"{env_var}={value}")
        lines.append("")

    # Model routing
    lines.append("# Model Routing")
    lines.append(f"NADIRCLAW_SIMPLE_MODEL={simple}")
    lines.append(f"NADIRCLAW_COMPLEX_MODEL={complex_model}")
    if reasoning:
        lines.append(f"NADIRCLAW_REASONING_MODEL={reasoning}")
    if free:
        lines.append(f"NADIRCLAW_FREE_MODEL={free}")
    lines.append("")

    # Server defaults
    lines.append("# Server")
    lines.append("NADIRCLAW_PORT=8856")
    lines.append("")

    ENV_FILE.write_text("\n".join(lines) + "\n")

    # Restrict permissions
    if platform.system() != "Windows":
        ENV_FILE.chmod(0o600)

    return ENV_FILE


def print_summary(
    providers: List[str],
    simple: str,
    complex_model: str,
    reasoning: Optional[str],
    free: Optional[str],
):
    """Print configuration summary and next steps."""
    click.echo()
    click.echo("=" * 56)
    click.echo("  Setup Complete!")
    click.echo("=" * 56)
    click.echo()
    click.echo("  Configuration:")
    click.echo(f"    Providers:     {', '.join(PROVIDER_INFO[p]['display'] for p in providers)}")
    click.echo(f"    Simple model:  {simple}")
    click.echo(f"    Complex model: {complex_model}")
    if reasoning:
        click.echo(f"    Reasoning:     {reasoning}")
    if free:
        click.echo(f"    Free model:    {free}")
    click.echo(f"    Config file:   {ENV_FILE}")
    click.echo()
    click.echo("  Next steps:")
    click.echo("    nadirclaw serve           # Start the router")
    click.echo("    nadirclaw status          # Check configuration")
    click.echo("    nadirclaw codex onboard   # Configure Codex integration")
    click.echo("    nadirclaw setup --reconfigure  # Re-run this wizard")
    click.echo()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_setup_wizard(reconfigure: bool = False):
    """Run the full interactive setup wizard."""
    # Step 1: Welcome
    print_welcome()

    # Detect existing state
    existing_creds = detect_existing_credentials() if reconfigure else []

    # Step 2: Provider selection
    providers = prompt_provider_selection(existing=existing_creds or None)

    # Step 3: Credential collection
    click.echo("-" * 56)
    click.echo("  Credentials")
    click.echo("-" * 56)
    click.echo()

    api_keys: Dict[str, str] = {}
    for provider in providers:
        cred = prompt_credential_for_provider(provider, reconfigure=reconfigure)
        # Collect API keys for .env (only plain keys, not OAuth tokens)
        if cred and provider != "ollama":
            info = PROVIDER_INFO[provider]
            if info["env_var"]:
                # Only write to .env if it looks like an API key (not an OAuth token)
                if not cred.startswith("eyJ"):  # JWT tokens start with eyJ
                    api_keys[info["env_var"]] = cred

    # Step 4: Model selection
    click.echo("-" * 56)
    click.echo("  Model Selection")
    click.echo("-" * 56)

    tiers = get_available_models_for_providers(providers)

    # Simple (required)
    simple_model = None
    if tiers["simple"]:
        simple_model = prompt_model_selection("simple", tiers["simple"], providers)
    if not simple_model:
        simple_model = select_default_model("simple", providers) or "gemini-3-flash-preview"
        click.echo(f"  Using default simple model: {simple_model}\n")

    # Complex (required)
    complex_model = None
    if tiers["complex"]:
        complex_model = prompt_model_selection("complex", tiers["complex"], providers)
    if not complex_model:
        complex_model = select_default_model("complex", providers) or "gpt-4o"
        click.echo(f"  Using default complex model: {complex_model}\n")

    # Reasoning (optional)
    reasoning_model = None
    if tiers["reasoning"]:
        reasoning_model = prompt_model_selection("reasoning", tiers["reasoning"], providers)

    # Free (optional)
    free_model = None
    if tiers["free"]:
        free_model = prompt_model_selection("free", tiers["free"], providers)

    # Step 5: Write config + summary
    click.echo("-" * 56)
    click.echo("  Writing Configuration")
    click.echo("-" * 56)
    click.echo()

    env_path = write_env_file(
        simple=simple_model,
        complex_model=complex_model,
        reasoning=reasoning_model,
        free=free_model,
        api_keys=api_keys,
    )
    click.echo(f"  Wrote {env_path}")

    print_summary(providers, simple_model, complex_model, reasoning_model, free_model)
