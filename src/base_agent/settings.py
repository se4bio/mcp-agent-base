"""Configuration defaults for the MCP agent service."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a local .env file if present.
load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("DEFAULT_MAX_OUTPUT_TOKENS", "1024"))
INCLUDE_TOOL_LOGS = os.getenv("INCLUDE_TOOL_LOGS", "true").lower() in {"1", "true", "yes", "on"}
SIMULATE = os.getenv("SIMULATE", "false").lower() in {"1", "true", "yes", "on"}
MOCK_EMPTY_MCP = os.getenv("MOCK_EMPTY_MCP", "false").lower() in {"1", "true", "yes", "on"}

# Extended thinking controls.
THINKING_ENABLED = os.getenv("THINKING_ENABLED", "false").lower() in {"1", "true", "yes"}
DEFAULT_THINKING_BUDGET_TOKENS = int(os.getenv("THINKING_BUDGET_TOKENS", "1024"))

# AWS Bedrock configuration
USE_BEDROCK = os.getenv("USE_BEDROCK", "false").lower() in {"1", "true", "yes"}
AWS_PROFILE = os.getenv("AWS_PROFILE")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Default Bedrock model (Claude Sonnet 4.5)
DEFAULT_BEDROCK_MODEL = os.getenv("DEFAULT_BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")


def load_system_prompt() -> str:
    """Load the system prompt from the bundled prompts/system_prompt.txt file."""
    candidate_paths = []

    # Common locations: working directory and project root when installed.
    candidate_paths.append(Path.cwd() / "prompts" / "system_prompt.txt")
    candidate_paths.append(Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt")

    for prompt_path in candidate_paths:
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()

    return ""


DEFAULT_SYSTEM_PROMPT = load_system_prompt()

# Keep the user-agent identifier simple so MCP servers can see who is calling.
CLIENT_NAME = os.getenv("CLIENT_NAME", "mcp-agent-base")
