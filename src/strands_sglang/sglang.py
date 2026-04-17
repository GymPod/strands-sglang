# Copyright 2025-2026 Horizon RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGLang model provider with token-in/token-out support."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Any,
    TypedDict,
    TypeVar,
    cast,
)

from pydantic import BaseModel
from strands.models import Model
from strands.types.content import ContentBlock, Messages, SystemContentBlock
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)
from strands.types.streaming import StopReason, StreamEvent
from strands.types.tools import ToolChoice, ToolResultContent, ToolSpec
from transformers import PreTrainedTokenizerBase
from typing_extensions import Unpack, override

from .client import SGLangClient
from .exceptions import SGLangContextLengthError, SGLangThrottledError
from .token import TokenManager
from .tool_parsers import HermesToolParser, ToolParser

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class _PreparedPrompt:
    """Prompt prepared for one `/generate` call."""

    input_ids: list[int]
    full_input_ids: list[int]
    new_input_ids: list[int]
    image_data: list[str]
    extends_active_context: bool


class SGLangModel(Model):
    """SGLang native `/generate` API provider with token-in/token-out support.

    Example:
        >>> from transformers import AutoTokenizer
        >>> from strands_sglang import SGLangClient, SGLangModel
        >>> client = SGLangClient(base_url="http://localhost:30000")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        >>> model = SGLangModel(client=client, tokenizer=tokenizer)
        >>> # After generation:
        >>> model.token_manager.token_ids    # Full token trajectory
        >>> model.token_manager.loss_mask    # Boolean mask for loss computation
        >>> model.token_manager.logprobs     # Log probabilities
    """

    class SGLangConfig(TypedDict, total=False):
        """Configuration options for SGLang generation."""

        sampling_params: dict[str, Any] | None  # Passed to /generate endpoint
        return_logprob: bool | None  # Return logprobs for all tokens (default: True)
        enable_thinking: bool | None  # Enable thinking mode for Qwen3 hybrid models
        return_routed_experts: bool | None  # Record MoE routing decisions for routing replay

    def __init__(
        self,
        *,
        client: SGLangClient,
        tokenizer: PreTrainedTokenizerBase,
        tool_parser: ToolParser | None = None,
        **config: Unpack[SGLangConfig],
    ) -> None:
        """Initialize SGLang model provider.

        Args:
            client: `SGLangClient` for HTTP communication with the SGLang server.
            tokenizer: HuggingFace tokenizer for chat template and tokenization.
            tool_parser: `ToolParser` for tool calls (default: `HermesToolParser`).
            **config: Additional SGLang generation configuration (see `SGLangConfig`).
        """
        self.client = client
        self.tokenizer = tokenizer
        self.tool_parser = tool_parser or HermesToolParser()
        self.config = dict(config)
        self._chat_template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "enable_thinking": self.config.get("enable_thinking", True),
        }

        # State tracking (this makes SGLangModel stateful)
        self.token_manager = TokenManager()
        self._active_input_ids: list[int] = []
        self._active_image_data: list[str] = []
        self.message_count: int = 0
        self.tool_parse_errors: dict[str, int] = {}  # per-tool parse error count
        self.image_data: list[str] = []  # current prompt image data URLs (VLM only)

        logger.debug("initialized with config: %s", self.config)

    def reset(self) -> None:
        """Reset all state for a new episode."""
        self.token_manager.reset()
        self._active_input_ids = []
        self._active_image_data = []
        self.message_count = 0
        self.tool_parse_errors = {}
        self.image_data = []

    # -------------------------------------------------------------------------
    # Model interface implementation
    # -------------------------------------------------------------------------

    @override
    def update_config(self, **model_config: Unpack[SGLangConfig]) -> None:  # type: ignore[override]
        """Update the model configuration."""
        self.config.update(model_config)

    @override
    def get_config(self) -> SGLangConfig:
        """Get the model configuration."""
        return cast(SGLangModel.SGLangConfig, self.config)

    # -------------------------------------------------------------------------
    # Chat template and message formatting
    # -------------------------------------------------------------------------

    @cached_property
    def message_separator(self) -> str:
        """Auto-detect text bridging the previous response's stop token and the next message.

        Probes the chat template with a terminal assistant message. The text after the
        marker is `stop_token + separator`. Strip `stop_token` to get the separator if it exists.
        """
        probe = str(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "U"}, {"role": "assistant", "content": "__M__"}],
                tokenize=False,
                add_generation_prompt=False,
            )
        )
        sep = self.tokenizer.encode(probe.split("__M__", 1)[1], add_special_tokens=False)[1:]
        return self.tokenizer.decode(sep) if sep else ""

    @classmethod
    def format_content_block(
        cls, content_block: ContentBlock | ToolResultContent, is_multimodal: bool = False
    ) -> dict[str, Any] | str:
        """Convert a single Strands `ContentBlock` or `ToolResultContent` to HF chat template format."""
        # ContentBlock / ToolResultContent is a TypedDict with exactly one key set at runtime
        ((key, value),) = content_block.items()
        result: dict[str, Any] = {}
        match key, value:
            case "text", str() as text:
                result = {"type": "text", "text": text}
            case "image", dict() as image:
                mime = f"image/{image['format']}"
                encoded = base64.b64encode(image["source"]["bytes"]).decode()
                result = {"type": "image", "image": f"data:{mime};base64,{encoded}"}
            case "json", data:
                result = {"type": "text", "text": json.dumps(data)}
            # TODO: add support for other content types
            case _:
                raise TypeError(f"content_type=<{key}> | unsupported type")
        # flatten to text if not multimodal
        if not is_multimodal:
            return str(result["text"])
        return result

    @classmethod
    def format_messages(
        cls, messages: Messages, system_prompt: str | None = None, is_multimodal: bool = False
    ) -> list[dict[str, Any]]:
        """Convert Strands Messages to HF chat template format.

        When `is_multimodal=False` (default), content is flattened to a plain string.
        When `is_multimodal=True`, content is kept as a list of dicts.
        """
        result: list[dict[str, Any]] = []

        if system_prompt:
            content: Any = [{"type": "text", "text": system_prompt}] if is_multimodal else system_prompt
            result.append({"role": "system", "content": content})

        # Each Strands message is {"role": str, "content": [ContentBlock, ...]}
        # One Strands message maps to one HF message, except toolResult blocks
        # which each become a separate HF message with role="tool".
        for msg in messages:
            if "toolResult" in msg["content"][0]:
                # Each toolResult → its own HF message (different tool_call_id)
                for cb in msg["content"]:
                    assert "toolResult" in cb
                    tr = cb["toolResult"]
                    content = [cls.format_content_block(c, is_multimodal) for c in tr["content"]]
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr["toolUseId"],
                            "content": content if is_multimodal else content[0],
                        }
                    )
            else:
                # Non-tool content → one HF message (text, image, etc.; toolUse skipped)
                content = [cls.format_content_block(c, is_multimodal) for c in msg["content"] if "toolUse" not in c]
                result.append({"role": msg["role"], "content": content if is_multimodal else content[0]})

        return result

    def format_tool_specs(self, tool_specs: list[ToolSpec]) -> list[dict]:
        """Format strands ToolSpecs to HF chat template format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["inputSchema"]["json"],
                },
            }
            for spec in tool_specs
        ]

    @staticmethod
    def sort_tool_results(messages: Messages) -> Messages:
        """Sort tool results by ID to match original call order.

        Notes:
            In strands' format, parallel tool results are batched into a single message.
        """
        return [
            {**msg, "content": sorted(msg["content"], key=lambda c: c["toolResult"]["toolUseId"])}
            if "toolResult" in msg["content"][0]
            else msg
            for msg in messages
        ]

    def tokenize_prompt_messages(
        self,
        messages: Messages,
        system_prompt: str | None,
        tool_specs: list[ToolSpec] | None = None,
        is_multimodal: bool = False,
    ) -> list[int]:
        """Tokenize prompt messages for the next generation call.

        Returns the prompt tokens newly introduced relative to the active model
        context. When conversation management resets or compresses history, the
        active context no longer prefixes the current prompt, so the full prompt
        is returned as a new prompt segment.
        """
        return self._prepare_prompt_messages(
            messages=messages,
            system_prompt=system_prompt,
            tool_specs=tool_specs,
            is_multimodal=is_multimodal,
        ).new_input_ids

    @staticmethod
    def _messages_have_images(messages: Messages) -> bool:
        """Return whether any message content includes image data."""

        def _content_has_image(content_block: ContentBlock | ToolResultContent) -> bool:
            if "image" in content_block:
                return True
            tool_result = content_block.get("toolResult")
            if not isinstance(tool_result, dict):
                return False
            nested_content = tool_result.get("content")
            if not isinstance(nested_content, list):
                return False
            return any(isinstance(item, dict) and _content_has_image(item) for item in nested_content)

        return any(_content_has_image(content_block) for message in messages for content_block in message["content"])

    @staticmethod
    def _extract_image_data(hf_messages: list[dict[str, Any]]) -> list[str]:
        """Collect image data URLs from HF-formatted multimodal messages."""
        image_data: list[str] = []
        for msg in hf_messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    image = part.get("image")
                    if not isinstance(image, str):
                        raise TypeError(f"Expected image data URL to be str, got {type(image).__name__}")
                    image_data.append(image)
        return image_data

    def _render_prompt_messages(
        self,
        messages: Messages,
        system_prompt: str | None,
        tool_specs: list[ToolSpec] | None,
        is_multimodal: bool,
    ) -> tuple[str, list[str]]:
        """Render messages through the chat template and collect current image data."""
        multimodal = is_multimodal or self._messages_have_images(messages)
        hf_messages = self.format_messages(messages, system_prompt, is_multimodal=multimodal)
        image_data = self._extract_image_data(hf_messages) if multimodal else []
        tools = self.format_tool_specs(tool_specs) if tool_specs else None
        prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                hf_messages,
                tools=cast(list, tools),
                add_generation_prompt=True,
                **self._chat_template_kwargs,
            ),
        )
        return prompt, self._drop_unexpanded_images(image_data, prompt)

    def _prepare_prompt_messages(
        self,
        messages: Messages,
        system_prompt: str | None,
        tool_specs: list[ToolSpec] | None,
        is_multimodal: bool,
    ) -> _PreparedPrompt:
        """Prepare the current prompt and incremental training segment.

        For additive turns, keep the exact previously generated token stream and
        only tokenize the newly introduced messages. Assistant outputs and tool
        calls are not guaranteed to round-trip through the chat template byte
        for byte, so rebuilding the entire prompt on every turn can shift the
        live model context even when conversation history was not rewritten.
        """
        prompt, image_data = self._render_prompt_messages(messages, system_prompt, tool_specs, is_multimodal)
        full_input_ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        active_prefix_len = len(self._active_input_ids)

        if active_prefix_len == 0:
            return _PreparedPrompt(
                input_ids=full_input_ids,
                full_input_ids=full_input_ids,
                new_input_ids=full_input_ids,
                image_data=image_data,
                extends_active_context=True,
            )

        if full_input_ids == self._active_input_ids and image_data == self._active_image_data:
            raise RuntimeError(
                f"No new messages to tokenize (active_input_len={active_prefix_len}, got {len(messages)} messages)"
            )

        multimodal = is_multimodal or self._messages_have_images(messages)
        additive_messages = len(messages) > self.message_count
        preserves_image_prefix = image_data[: len(self._active_image_data)] == self._active_image_data
        if additive_messages and preserves_image_prefix:
            new_hf_messages = self.format_messages(
                self.sort_tool_results(messages[self.message_count :]),
                is_multimodal=multimodal,
            )
            fake_messages: Messages = [
                {"role": "system", "content": [{"text": "FAKE SYSTEM PROMPT"}]},
                {"role": "user", "content": [{"text": "FAKE USER MESSAGE"}]},
            ]
            fake_hf_messages = self.format_messages(fake_messages, is_multimodal=multimodal)
            delta_prompt_with_prefix = cast(
                str,
                self.tokenizer.apply_chat_template(
                    fake_hf_messages + new_hf_messages,
                    add_generation_prompt=True,
                    **self._chat_template_kwargs,
                ),
            )
            fake_prefix_prompt = cast(
                str,
                self.tokenizer.apply_chat_template(
                    fake_hf_messages,
                    add_generation_prompt=False,
                    **self._chat_template_kwargs,
                ),
            )
            if not delta_prompt_with_prefix.startswith(fake_prefix_prompt):
                raise AssertionError("incremental prompt must start with the fake prefix prompt")
            delta_prompt = self.message_separator + delta_prompt_with_prefix[len(fake_prefix_prompt) :]
            new_input_ids = list(self.tokenizer.encode(delta_prompt, add_special_tokens=False))
            input_ids = self._active_input_ids + new_input_ids
            return _PreparedPrompt(
                input_ids=input_ids,
                full_input_ids=full_input_ids,
                new_input_ids=new_input_ids,
                image_data=image_data,
                extends_active_context=True,
            )

        extends_active_context = (
            full_input_ids[:active_prefix_len] == self._active_input_ids
            and image_data[: len(self._active_image_data)] == self._active_image_data
        )
        if extends_active_context:
            new_input_ids = full_input_ids[active_prefix_len:]
        else:
            new_input_ids = full_input_ids

        return _PreparedPrompt(
            input_ids=full_input_ids,
            full_input_ids=full_input_ids,
            new_input_ids=new_input_ids,
            image_data=image_data,
            extends_active_context=extends_active_context,
        )

    def _drop_unexpanded_images(self, image_data: list[str], template_output: str) -> list[str]:
        """Remove images whose data URLs survive in the chat template output.

        When a chat template expands an image, it *consumes* the data URL and
        replaces it with model-specific vision tokens (e.g. ``<|vision_start|>``).
        If a data URL appears verbatim in the rendered string, the template did
        not generate placeholder tokens for it and SGLang will warn about more
        image data items than corresponding tokens.  Drop those entries so the
        positional mapping stays correct.
        """
        if not image_data:
            return []
        kept = [img for img in image_data if img not in template_output]
        n_dropped = len(image_data) - len(kept)
        if n_dropped:
            logger.warning(
                "Dropped %d/%d images whose data URLs were not expanded by the chat template "
                "(likely images in message roles the template does not support)",
                n_dropped,
                len(image_data),
            )
        return kept

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Non-streaming chat completion via SGLang's `/generate` endpoint."""
        # Prepare request
        config = self.get_config()
        sampling_params: dict[str, Any] = dict(config.get("sampling_params") or {})
        sampling_params.setdefault("skip_special_tokens", False)
        return_logprob = config.get("return_logprob", True)
        prepared_prompt = self._prepare_prompt_messages(
            messages=messages,
            system_prompt=system_prompt,
            tool_specs=tool_specs,
            is_multimodal=await self.client.is_multimodal(),
        )
        input_ids = prepared_prompt.input_ids
        self.image_data = prepared_prompt.image_data

        # Assistant message start
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}

        # Call SGLang's `/generate` endpoint
        return_routed_experts = config.get("return_routed_experts", False)
        try:
            response = await self.client.generate(
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_logprob=return_logprob,
                logprob_start_len=(
                    max(0, len(self._active_input_ids) - 1)
                    if return_logprob and prepared_prompt.extends_active_context and self._active_input_ids
                    else 0
                    if return_logprob
                    else None
                ),
                image_data=self.image_data or None,
                return_routed_experts=return_routed_experts,
            )

            # Extract response data
            text = response["text"]
            output_ids = response["output_ids"]
            meta_info = response["meta_info"]
            input_token_logprobs = meta_info.get("input_token_logprobs")
            output_token_logprobs = meta_info.get("output_token_logprobs")

            # Assistant message content delta (single delta for non-streaming)
            yield {"contentBlockDelta": {"delta": {"text": text}}}

        except SGLangContextLengthError as e:
            raise ContextWindowOverflowException(f"Context length exceeded: {e.body}") from e
        except SGLangThrottledError as e:
            raise ModelThrottledException(f"Service throttled (status={e.status}): {e.body}") from e

        # Update token trajectory
        if not prepared_prompt.extends_active_context and len(self.token_manager) > 0:
            self.token_manager.start_new_trajectory()
        self.token_manager.add_prompt(
            token_ids=prepared_prompt.new_input_ids,
            logprobs=(
                [entry[0] for entry in input_token_logprobs[-len(prepared_prompt.new_input_ids) :]]
                if input_token_logprobs and prepared_prompt.new_input_ids
                else None
            ),
        )
        self.token_manager.add_response(
            token_ids=output_ids,
            logprobs=[e[0] for e in output_token_logprobs] if output_token_logprobs else None,
        )
        self._active_input_ids = prepared_prompt.input_ids + output_ids
        self._active_image_data = list(prepared_prompt.image_data)
        self.message_count = len(messages) + 1

        # Store routed experts for routing replay (overwrite semantics —
        # SGLang returns routing for ALL tokens each turn, not incremental)
        if return_routed_experts:
            routed_experts_data = meta_info.get("routed_experts")
            if routed_experts_data:
                self.token_manager.add_routed_experts(routed_experts_data)

        # Assistant message content stop
        yield {"contentBlockStop": {}}

        # Assistant message tool use content - start, delta, stop
        parsed_tool_calls = self.tool_parser.parse(text)
        for tool_call in parsed_tool_calls:
            if tool_call.is_error:
                logger.warning("Tool parse error for '%s': %s", tool_call.name, (tool_call.raw or "")[:100])
                self.tool_parse_errors[tool_call.name] = self.tool_parse_errors.get(tool_call.name, 0) + 1

            yield {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": tool_call.id,
                            "name": tool_call.name,
                        }
                    }
                }
            }
            yield {
                "contentBlockDelta": {
                    "delta": {
                        "toolUse": {
                            "input": tool_call.payload,
                        }
                    }
                }
            }
            yield {"contentBlockStop": {}}

        # Assistant message stop
        stop_reason: str = "tool_use" if parsed_tool_calls else "end_turn"
        if meta_info["finish_reason"]["type"] == "length":
            stop_reason = "max_tokens"
        yield {"messageStop": {"stopReason": cast(StopReason, stop_reason)}}

        # Assistant message usage metadata
        yield {
            "metadata": {
                "usage": {
                    "inputTokens": meta_info["prompt_tokens"],
                    "outputTokens": meta_info["completion_tokens"],
                    "totalTokens": meta_info["prompt_tokens"] + meta_info["completion_tokens"],
                    "cacheReadInputTokens": meta_info["cached_tokens"],
                },
                "metrics": {"latencyMs": int(meta_info["e2e_latency"] * 1000)},
            }
        }

    @override
    async def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Structured output via SGLang's `json_schema` constrained decoding.

        Notes:
            Does not update `token_manager` (no token trajectory tracking).
        """
        # Convert Pydantic model to JSON schema string
        json_schema = json.dumps(output_model.model_json_schema())

        # Format and tokenize prompt (no tools for structured output)
        prepared_prompt = self._prepare_prompt_messages(
            messages=prompt,
            system_prompt=system_prompt,
            tool_specs=None,
            is_multimodal=await self.client.is_multimodal(),
        )

        # Build sampling params with json_schema constraint
        config = self.get_config()
        sampling_params: dict[str, Any] = dict(config.get("sampling_params") or {})
        sampling_params["json_schema"] = json_schema

        # Call SGLang /generate endpoint
        try:
            response = await self.client.generate(
                input_ids=prepared_prompt.input_ids,
                sampling_params=sampling_params,
                return_logprob=False,  # No need for logprobs in structured output
                image_data=prepared_prompt.image_data or None,
            )
        except SGLangContextLengthError as e:
            raise ContextWindowOverflowException(f"Context length exceeded: {e.body}") from e
        except SGLangThrottledError as e:
            raise ModelThrottledException(f"Service throttled (status={e.status}): {e.body}") from e

        # Parse and validate response
        text = response["text"]
        parsed = output_model.model_validate_json(text)

        yield {"output": parsed}
