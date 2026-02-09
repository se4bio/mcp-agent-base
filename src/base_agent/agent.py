"""Claude agent that loops through MCP tool calls until a final answer is ready."""

from __future__ import annotations

import asyncio
import logging
import os
import json
from typing import AsyncIterable, List, Optional, Any, Dict

from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from anthropic.types import MessageParam, ToolResultBlockParam

from .mcp_client import MCPClient, EmptyMCPClient, MCPConnectionError
from .settings import (
    MCP_SERVER_URL,
    DEFAULT_MODEL,
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    THINKING_ENABLED,
    INCLUDE_TOOL_LOGS,
    SIMULATE,
    MOCK_EMPTY_MCP,
    USE_BEDROCK,
    AWS_PROFILE,
    AWS_REGION,
)

# Simple module logger; defaults to INFO if not configured by the host app.
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
# Quiet noisy third-party loggers; keep our own logs at INFO.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
logging.getLogger("mcp.client").setLevel(logging.WARNING)

try:  # Optional imports for more specific error handling.
    from anthropic import APIConnectionError, APITimeoutError, APIStatusError  # type: ignore
except Exception:  # pragma: no cover - defensive import for older anthropic versions
    APIConnectionError = None
    APITimeoutError = None
    APIStatusError = None

try:  # Optional dependency; used only for error matching.
    import httpcore  # type: ignore
    import httpx  # type: ignore
except Exception:  # pragma: no cover - fallback if httpx isn't available
    httpcore = None
    httpx = None


def _iter_exceptions(exc: BaseException):
    if isinstance(exc, BaseExceptionGroup):
        for nested in exc.exceptions:
            yield from _iter_exceptions(nested)
    else:
        yield exc


class ClaudeMCPAgent:
    """Runs a Claude tool-use loop backed by the MCP server."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        mcp_server_url: str = MCP_SERVER_URL,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        temperature: float = DEFAULT_TEMPERATURE,
        thinking_enabled: bool = THINKING_ENABLED,
        thinking_budget_tokens: Optional[int] = DEFAULT_THINKING_BUDGET_TOKENS,
        simulate: bool = SIMULATE,
        mock_empty_mcp: Optional[bool] = None,
        llm_max_retries: int = 3,
        llm_retry_backoff_seconds: float = 0.5,
        use_bedrock: bool = USE_BEDROCK,
        aws_profile: Optional[str] = AWS_PROFILE,
        aws_region: str = AWS_REGION,
    ):
        self.use_bedrock = use_bedrock
        self.max_output_tokens = max_output_tokens
        self.system_prompt = system_prompt or ""
        self.temperature = temperature
        self.thinking_enabled = thinking_enabled
        self.thinking_budget_tokens = thinking_budget_tokens
        self.simulate = simulate

        # Select appropriate model based on provider
        if use_bedrock:
            self.model = self._normalize_bedrock_model(model)
        else:
            self.model = self._normalize_model(model)

        logger.info(
            "Using system prompt",
            extra={"system_prompt_preview": (self.system_prompt[:120] if self.system_prompt else "<empty>")},
        )

        if self.simulate:
            self.client = None
            self.api_key = None
        elif use_bedrock:
            # Use AWS Bedrock
            logger.info(f"Using AWS Bedrock with model {self.model} in region {aws_region}")
            if aws_profile:
                import boto3
                session = boto3.Session(profile_name=aws_profile)
                credentials = session.get_credentials()
                self.client = AsyncAnthropicBedrock(
                    aws_access_key=credentials.access_key,
                    aws_secret_key=credentials.secret_key,
                    aws_session_token=credentials.token,
                    aws_region=aws_region,
                )
            else:
                # Use default AWS credential chain
                self.client = AsyncAnthropicBedrock(aws_region=aws_region)
            self.api_key = None
        else:
            # Use Anthropic API directly
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY is required (or set USE_BEDROCK=true).")
            self.client = AsyncAnthropic(api_key=self.api_key)

        mock_empty = MOCK_EMPTY_MCP if mock_empty_mcp is None else mock_empty_mcp
        if self.simulate:
            self.mcp_client = None
        elif mock_empty:
            logger.info("MOCK_EMPTY_MCP enabled; skipping tool discovery.")
            self.mcp_client = EmptyMCPClient()
        else:
            self.mcp_client = MCPClient(mcp_server_url)
        self._llm_max_retries = max(1, llm_max_retries)
        self._llm_retry_backoff_seconds = max(0.0, llm_retry_backoff_seconds)
        self._last_model_info: Optional[Dict[str, Any]] = None

    def _is_llm_connection_error(self, exc: BaseException) -> bool:
        for err in _iter_exceptions(exc):
            if APIConnectionError is not None and isinstance(err, APIConnectionError):
                return True
            if APITimeoutError is not None and isinstance(err, APITimeoutError):
                return True
            if APIStatusError is not None and isinstance(err, APIStatusError):
                status = getattr(err, "status_code", None) or getattr(
                    getattr(err, "response", None), "status_code", None
                )
                if status is not None and int(status) >= 500:
                    return True
            if isinstance(err, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
                return True
            if httpx is not None and isinstance(
                err, (httpx.ConnectError, httpx.ReadTimeout, httpx.TimeoutException)
            ):
                return True
            if httpcore is not None and isinstance(
                err, (httpcore.ConnectError, httpcore.ReadTimeout, httpcore.TimeoutException)
            ):
                return True
        return False

    async def ask_stream(
        self,
        messages: List[MessageParam],
        *,
        stream_mode: bool = True,
        include_tool_logs: bool = INCLUDE_TOOL_LOGS,
        include_model_info: bool = False,
    ) -> AsyncIterable[Any]:
        """Yield intermediate and final responses while invoking MCP tools.

        stream_mode=True streams live pieces; stream_mode=False emits only final pieces in order.
        """
        self._last_model_info = None
        if self.simulate:
            simulated = "Simulated response."
            if stream_mode:
                yield {"delta": {"content": simulated}}
                yield {"delta": {}, "finish_reason": "stop"}
            else:
                yield simulated
            return

        logger.info(
            "Starting ask",
            extra={
                "model": self.model,
                "mcp_server": self.mcp_client.server_url if self.mcp_client else "<simulation>",
            },
        )
        try:
            tools = [tool.to_anthropic() for tool in await self.mcp_client.list_tools()]
        except MCPConnectionError as exc:
            logger.warning("MCP connection failed during tool discovery: %s", exc)
            message = (
                f"Unable to connect to the MCP server at {self.mcp_client.server_url}. "
                "Please check that it is running and try again."
            )
            if stream_mode:
                yield {"delta": {"content": message}}
                yield {"delta": {}, "finish_reason": "stop"}
            else:
                yield message
            return

        conversation: List[MessageParam] = list(messages)
        answer_parts: List[str] = []
        thinking_parts: List[str] = []
        usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        saw_usage = False
        saw_input_tokens = False
        saw_output_tokens = False
        saw_cache_creation_tokens = False
        saw_cache_read_tokens = False

        while True:
            block_types: Dict[int, str] = {}
            system_content = [{"type": "text", "text": str(self.system_prompt)}] if self.system_prompt else None
            request_kwargs = dict(
                model=self.model,
                max_tokens=self.max_output_tokens,
                tools=tools,
                temperature=self.temperature,
                messages=conversation,
            )
            if system_content is not None:
                request_kwargs["system"] = system_content
            if self.thinking_enabled and self.thinking_budget_tokens:
                logger.info("Enabling extended thinking with budget of %d tokens", self.thinking_budget_tokens)
                request_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget_tokens,
                }
            payload_preview = {
                "model": self.model,
                "system": system_content,
                "messages": conversation,
                "tools": tools,
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "thinking": request_kwargs.get("thinking"),
            }
            logger.info(
                "Sending request to Anthropic (counts toward input tokens; preview truncated): %s",
                json.dumps(payload_preview, default=str)[:5000],
            )

            tool_requests: List[Any] = []
            final_message = None
            for attempt in range(1, self._llm_max_retries + 1):
                tool_requests = []
                block_types = {}
                had_events = False
                try:
                    async with self.client.messages.stream(**request_kwargs) as stream:
                        async for event in stream:
                            had_events = True
                            if event.type == "content_block_start":
                                block_types[event.index] = getattr(event.content_block, "type", None)
                                if block_types.get(event.index) == "thinking" and stream_mode:
                                    yield {"delta": {"content": "**Thoughts:**\n"}}
                            if event.type == "content_block_delta":
                                # Anthropic thinking deltas may not always use type="text_delta"; look for a text attribute.
                                delta_text = getattr(event.delta, "text", None) or ""
                                if not delta_text and hasattr(event.delta, "thinking"):
                                    delta_text = getattr(event.delta, "thinking") or ""
                                if delta_text:
                                    block_type = block_types.get(event.index)
                                    is_thinking = (
                                        block_type == "thinking"
                                        or hasattr(event.delta, "thinking")
                                        or getattr(event.delta, "type", None) == "thinking_delta"
                                    )
                                    if is_thinking:
                                        thinking_parts.append(delta_text)
                                        if stream_mode:
                                            yield {"delta": {"thinking": delta_text, "content": delta_text}}
                                    else:
                                        if stream_mode:
                                            yield {"delta": {"content": delta_text}}
                            if event.type == "content_block_stop" and stream_mode:
                                if block_types.get(event.index) == "thinking":
                                    yield {"delta": {"content": "\n\n---\n\n"}}
                                yield {"delta": {"content": "\n"}}
                                
                            if event.type == "content_block_stop" and getattr(event.content_block, "type", None) == "tool_use":
                                tool_requests.append(event.content_block)
                                if stream_mode:
                                    tool_call = {
                                        "id": event.content_block.id,
                                        "type": "function",
                                        "function": {
                                            "name": event.content_block.name,
                                            "arguments": json.dumps(event.content_block.input or {}),
                                        },
                                    }
                                    yield {"delta": {"tool_calls": [tool_call]}}

                        final_message = await stream.get_final_message()
                        break
                except Exception as exc:
                    if not self._is_llm_connection_error(exc):
                        raise
                    logger.warning(
                        "LLM connection failed (attempt %d/%d): %s",
                        attempt,
                        self._llm_max_retries,
                        exc,
                    )
                    if attempt < self._llm_max_retries and not had_events:
                        if self._llm_retry_backoff_seconds:
                            await asyncio.sleep(self._llm_retry_backoff_seconds * attempt)
                        continue
                    message = (
                        "Unable to connect to the LLM provider. "
                        "Please check your network connectivity and try again."
                    )
                    if stream_mode:
                        yield {"delta": {"content": message}}
                        yield {"delta": {}, "finish_reason": "stop"}
                    else:
                        yield message
                    return

            if final_message is None:
                message = (
                    "Unable to connect to the LLM provider. "
                    "Please check your network connectivity and try again."
                )
                if stream_mode:
                    yield {"delta": {"content": message}}
                    yield {"delta": {}, "finish_reason": "stop"}
                else:
                    yield message
                return

            response_preview = {"role": "assistant", "content": final_message.content}
            try:
                response_dump = json.dumps(response_preview, default=str)
            except TypeError:
                response_dump = str(response_preview)
            logger.info(
                "Received response from Anthropic (counts toward output tokens; preview truncated): %s",
                response_dump[:5000],
            )
            usage = getattr(final_message, "usage", None)
            if usage:
                input_tokens = getattr(usage, "input_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
                saw_usage = True
                if input_tokens is not None:
                    usage_totals["input_tokens"] += input_tokens
                    saw_input_tokens = True
                if output_tokens is not None:
                    usage_totals["output_tokens"] += output_tokens
                    saw_output_tokens = True
                cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", None)
                if cache_creation_tokens is not None:
                    usage_totals["cache_creation_input_tokens"] += cache_creation_tokens
                    saw_cache_creation_tokens = True
                cache_read_tokens = getattr(usage, "cache_read_input_tokens", None)
                if cache_read_tokens is not None:
                    usage_totals["cache_read_input_tokens"] += cache_read_tokens
                    saw_cache_read_tokens = True
                total_tokens = (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None
                    else None
                )
                logger.info(
                    "Token usage: input=%s output=%s total=%s cache_creation_input=%s cache_read_input=%s",
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cache_creation_tokens,
                    cache_read_tokens,
                )
            conversation.append({"role": "assistant", "content": final_message.content})

            text_blocks = [
                block.text
                for block in final_message.content
                if getattr(block, "text", None) and getattr(block, "type", None) != "thinking"
            ]
            for text in text_blocks:
                answer_parts.append(text)
            thinking_blocks = [
                block.text
                for block in final_message.content
                if getattr(block, "text", None) and getattr(block, "type", None) == "thinking"
            ]
            thinking_parts.extend(thinking_blocks)

            if not tool_requests:
                break

            tool_results: List[ToolResultBlockParam] = []
            for tool_use in tool_requests:
                logger.info(
                    "Calling tool %s with args: %s",
                    tool_use.name,
                    json.dumps(tool_use.input or {}),
                )

                if stream_mode and include_tool_logs:
                    yield f"\n\nCalling `{tool_use.name}` with args:\n```json\n{json.dumps(tool_use.input or {})}\n```"

                if include_tool_logs:
                    answer_parts.append(f"\n\nCalling `{tool_use.name}` with args:\n```json\n{json.dumps(tool_use.input or {}, indent=2)}\n```")
                
                try:
                    tool_output = await self.mcp_client.call_tool(
                        tool_name=tool_use.name, arguments=tool_use.input or {}
                    )
                except MCPConnectionError as exc:
                    logger.warning(
                        "MCP connection failed while calling tool %s: %s",
                        tool_use.name,
                        exc,
                    )
                    message = (
                        f"Unable to connect to the MCP server at {self.mcp_client.server_url} "
                        f"while calling `{tool_use.name}`. Please check that it is running "
                        "and try again."
                    )
                    if stream_mode:
                        yield {"delta": {"content": message}}
                        yield {"delta": {}, "finish_reason": "stop"}
                    else:
                        yield message
                    return

                try:
                    parsed_tool_output = json.loads(tool_output)
                except (TypeError, json.JSONDecodeError):
                    parsed_tool_output = tool_output

                logger.info("Result from %s:\n%s", tool_use.name, parsed_tool_output)
                if stream_mode and tool_output and include_tool_logs:
                    rendered_output = (
                        json.dumps(parsed_tool_output)
                        if not isinstance(parsed_tool_output, str)
                        else parsed_tool_output
                    )
                    yield {
                        "delta": {
                            "role": "tool",
                            "tool_call_id": tool_use.id,
                            "content": f"\n\nResult from `{tool_use.name}`:\n```json\n{rendered_output}\n```\n",
                        }
                    }
                if include_tool_logs:
                    answer_parts.append(f"\n\nResult from `{tool_use.name}`:\n```json\n{tool_output}\n```\n")

                tool_results.append(
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=tool_use.id,
                        content=tool_output or "No content returned by tool.",
                    )
                )

            # Feed tool results back to Claude for another reasoning step.
            conversation.append({"role": "user", "content": tool_results})

        # Append collected pieces.
        if saw_usage:
            usage_summary: Dict[str, Optional[int]] = {
                "input_tokens": usage_totals["input_tokens"] if saw_input_tokens else None,
                "output_tokens": usage_totals["output_tokens"] if saw_output_tokens else None,
                "total_tokens": (
                    usage_totals["input_tokens"] + usage_totals["output_tokens"]
                    if saw_input_tokens and saw_output_tokens
                    else None
                ),
                "cache_creation_input_tokens": (
                    usage_totals["cache_creation_input_tokens"]
                    if saw_cache_creation_tokens
                    else None
                ),
                "cache_read_input_tokens": (
                    usage_totals["cache_read_input_tokens"] if saw_cache_read_tokens else None
                ),
            }
        else:
            usage_summary = None
        model_info = None
        if include_model_info:
            model_info = {
                "model": self.model,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "thinking_enabled": self.thinking_enabled,
                "thinking_budget_tokens": self.thinking_budget_tokens,
                "usage": usage_summary,
            }
        self._last_model_info = model_info

        final_sections = []
        if thinking_parts:
            final_sections.append(f"**Thoughts:**\n\n{''.join(thinking_parts)}\n\n---\n\n**Answer:**\n\n")

        final_sections.extend([part for part in answer_parts if part])
        final_answer = "\n\n".join(final_sections).strip()
        if stream_mode:
            chunk = {"delta": {}, "finish_reason": "stop"}
            if include_model_info:
                chunk["model_info"] = model_info
            yield chunk
        else:
            yield final_answer

    async def ask(
        self,
        messages: List[MessageParam],
    ) -> str:
        """Collect all streamed chunks into a single answer string."""
        parts: List[str] = []
        async for chunk in self.ask_stream(
            messages,
            stream_mode=False,
        ):
            parts.append(str(chunk))
        return "".join(parts).strip()

    async def ask_with_model_info(
        self,
        messages: List[MessageParam],
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Return the final answer plus aggregated model info from the full exchange."""
        parts: List[str] = []
        async for chunk in self.ask_stream(
            messages,
            stream_mode=False,
            include_model_info=True,
        ):
            parts.append(str(chunk))
        answer = "".join(parts).strip()
        return answer, self._last_model_info

    @staticmethod
    def _normalize_model(model: str) -> str:
        """Drop provider prefixes like 'anthropic:' so the client accepts the model name."""
        normalized = (model or DEFAULT_MODEL).strip()
        if normalized.startswith("anthropic:"):
            return normalized.split(":", 1)[1]
        return normalized

    @staticmethod
    def _normalize_bedrock_model(model: str) -> str:
        """Convert model name to Bedrock format if needed."""
        normalized = (model or DEFAULT_BEDROCK_MODEL).strip()
        # If already a Bedrock model ID, use as-is
        if "anthropic." in normalized:
            return normalized
        # Drop provider prefixes
        if normalized.startswith("anthropic:"):
            normalized = normalized.split(":", 1)[1]
        if normalized.startswith("bedrock:"):
            return normalized.split(":", 1)[1]
        # Use default Bedrock model
        return DEFAULT_BEDROCK_MODEL
