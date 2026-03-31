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

"""Unit tests for SGLangModel helper methods (no API calls needed)."""

import base64
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands_sglang import SGLangModel
from strands_sglang.client import SGLangClient


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.name_or_path = "/nonexistent"
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    return tokenizer


@pytest.fixture
def model(mock_tokenizer):
    """Create an SGLangModel with mock tokenizer."""
    client = SGLangClient(base_url="http://localhost:30000")
    client._is_multimodal = False
    model = SGLangModel(client=client, tokenizer=mock_tokenizer)
    model.__dict__["message_separator"] = ""  # override cached_property (mock has no real template)
    return model


class TestFormatTools:
    """Tests for format_tool_specs method."""

    def test_format_single_tool(self, model):
        """Format a single tool spec into HF function-calling format."""
        tool_specs = [
            {
                "name": "calculator",
                "description": "Perform calculations",
                "inputSchema": {"json": {"type": "object", "properties": {"expr": {"type": "string"}}}},
            }
        ]
        result = model.format_tool_specs(tool_specs)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "calculator"
        assert result[0]["function"]["description"] == "Perform calculations"
        assert "properties" in result[0]["function"]["parameters"]

    def test_format_multiple_tools(self, model):
        """Format multiple tool specs preserving order."""
        tool_specs = [
            {"name": "tool1", "description": "First tool", "inputSchema": {"json": {}}},
            {"name": "tool2", "description": "Second tool", "inputSchema": {"json": {}}},
            {"name": "tool3", "description": "Third tool", "inputSchema": {"json": {}}},
        ]
        result = model.format_tool_specs(tool_specs)

        assert len(result) == 3
        assert [t["function"]["name"] for t in result] == ["tool1", "tool2", "tool3"]

    def test_format_tool_missing_fields_raises(self, model):
        """Missing inputSchema raises KeyError."""
        with pytest.raises(KeyError):
            model.format_tool_specs([{"name": "minimal"}])


class TestFormatMessages:
    """Tests for format_messages — especially parallel tool results."""

    def test_parallel_tool_results_split_into_separate_messages(self):
        """All toolResult blocks in one Strands message must produce separate HF messages."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call_0", "status": "success", "content": [{"text": "result 0"}]}},
                    {"toolResult": {"toolUseId": "call_1", "status": "success", "content": [{"text": "result 1"}]}},
                    {"toolResult": {"toolUseId": "call_2", "status": "success", "content": [{"text": "result 2"}]}},
                ],
            }
        ]
        result = SGLangModel.format_messages(messages)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(tool_msgs) == 3
        assert {m["tool_call_id"] for m in tool_msgs} == {"call_0", "call_1", "call_2"}

    def test_single_tool_result(self):
        """Single toolResult produces one HF tool message with flattened content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call_0", "status": "success", "content": [{"text": "ok"}]}},
                ],
            }
        ]
        result = SGLangModel.format_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "ok"

    def test_tooluse_skipped(self):
        """toolUse blocks are skipped — tool calls live in raw text."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"text": "<tool_call>...</tool_call>"},
                    {"toolUse": {"toolUseId": "call_0", "name": "fn", "input": {}}},
                ],
            }
        ]
        result = SGLangModel.format_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "<tool_call>...</tool_call>"


class TestTokenizePromptMessages:
    """Tests for tokenize_prompt_messages error handling."""

    def test_no_new_messages_raises(self, model):
        """Raises RuntimeError when message_count matches input length."""
        model.token_manager.add_prompt([1, 2, 3])
        model.message_count = 2

        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
        ]

        with pytest.raises(RuntimeError, match="No new messages to tokenize"):
            model.tokenize_prompt_messages(messages, system_prompt=None)


class TestSortToolResults:
    """Tests for sort_tool_results method."""

    def test_sort_by_sequential_id(self, model):
        """Tool results are sorted by sequential ID."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call_0002", "content": [{"text": "third"}]}},
                    {"toolResult": {"toolUseId": "call_0000", "content": [{"text": "first"}]}},
                    {"toolResult": {"toolUseId": "call_0001", "content": [{"text": "second"}]}},
                ],
            },
        ]

        sorted_msgs = model.sort_tool_results(messages)

        results = sorted_msgs[0]["content"]
        assert results[0]["toolResult"]["toolUseId"] == "call_0000"
        assert results[1]["toolResult"]["toolUseId"] == "call_0001"
        assert results[2]["toolResult"]["toolUseId"] == "call_0002"

    def test_preserves_non_tool_messages(self, model):
        """Non-tool messages pass through unchanged."""
        messages = [
            {"role": "assistant", "content": [{"text": "Hello"}]},
            {"role": "user", "content": [{"text": "Hi"}]},
        ]

        assert model.sort_tool_results(messages) == messages

    def test_mixed_message_types(self, model):
        """Mixed assistant + user messages: only user tool results are sorted."""
        messages = [
            {"role": "assistant", "content": [{"text": "I'll call some tools"}]},
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call_0001", "content": [{"text": "b"}]}},
                    {"toolResult": {"toolUseId": "call_0000", "content": [{"text": "a"}]}},
                ],
            },
        ]

        sorted_msgs = model.sort_tool_results(messages)

        # Assistant message unchanged
        assert sorted_msgs[0] == messages[0]
        # User tool results sorted
        assert sorted_msgs[1]["content"][0]["toolResult"]["toolUseId"] == "call_0000"
        assert sorted_msgs[1]["content"][1]["toolResult"]["toolUseId"] == "call_0001"


class TestStreamDefaults:
    """Tests for stream() default behavior."""

    async def test_skip_special_tokens_defaults_to_false(self, mock_tokenizer):
        """stream() passes skip_special_tokens=False to client.generate by default."""
        from unittest.mock import AsyncMock

        client = SGLangClient(base_url="http://localhost:30000")
        client._is_multimodal = False
        client.generate = AsyncMock(
            return_value={
                "text": "hello",
                "output_ids": [1, 2],
                "meta_info": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop"},
                    "e2e_latency": 0.1,
                },
            }
        )
        model = SGLangModel(client=client, tokenizer=mock_tokenizer)

        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        async for _ in model.stream(messages):
            pass

        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs["sampling_params"]["skip_special_tokens"] is False


# ---------------------------------------------------------------------------
# Helpers for routing replay end-to-end tests
# ---------------------------------------------------------------------------

NUM_LAYERS = 2
TOP_K = 2
EXPERTS_PER_TOKEN = NUM_LAYERS * TOP_K  # int32 values per token


def _make_routing_b64(expert_ids: list[int]) -> str:
    """Encode int32 expert IDs to base64 (matching SGLang format)."""
    return base64.b64encode(struct.pack(f"<{len(expert_ids)}i", *expert_ids)).decode("ascii")


def _decode_routing(data: bytes) -> list[int]:
    """Decode raw bytes routing data to int32 expert IDs."""
    return list(struct.unpack(f"<{len(data) // 4}i", data))


def _make_generate_response(
    text: str,
    output_ids: list[int],
    num_input_tokens: int,
    *,
    routing_start: int = 0,
    include_routing: bool = False,
) -> dict:
    """Build a mock SGLang /generate response.

    When include_routing is True, generates deterministic per-token routing:
    token at position i gets experts [i*10, i*10+1, i*10+2, i*10+3]
    (for NUM_LAYERS=2, TOP_K=2).
    """
    num_output = len(output_ids)
    total = num_input_tokens + num_output

    meta_info = {
        "finish_reason": {"type": "stop"},
        "prompt_tokens": num_input_tokens,
        "completion_tokens": num_output,
        "cached_tokens": 0,
        "e2e_latency": 0.1,
    }

    if include_routing:
        # Generate routing for tokens from routing_start to total-1
        expert_ids = []
        for pos in range(routing_start, total):
            expert_ids.extend([pos * 10 + k for k in range(EXPERTS_PER_TOKEN)])
        meta_info["routed_experts"] = _make_routing_b64(expert_ids)

    input_logprobs = [[-0.1, tid] for tid in range(num_input_tokens)]
    output_logprobs = [[-0.2, tid] for tid in output_ids]

    return {
        "text": text,
        "output_ids": output_ids,
        "meta_info": {**meta_info, "input_token_logprobs": input_logprobs, "output_token_logprobs": output_logprobs},
    }


async def _collect_stream(stream):
    """Collect all events from an async iterable."""
    return [event async for event in stream]


class TestRoutedExpertsE2E:
    """End-to-end tests for routing replay through SGLangModel.stream()."""

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        tokenizer.encode.return_value = [10, 20, 30]
        return tokenizer

    @pytest.fixture
    def model(self, mock_tokenizer):
        client = SGLangClient(base_url="http://localhost:30000")
        client._is_multimodal = False
        m = SGLangModel(client=client, tokenizer=mock_tokenizer, return_routed_experts=True)
        m.__dict__["message_separator"] = ""
        return m

    async def test_single_turn_routing(self, model):
        """Single turn: routing covers prompt + response tokens."""
        prompt_tokens = [10, 20, 30]
        output_ids = [40, 50]

        response = _make_generate_response(
            text="Hello!",
            output_ids=output_ids,
            num_input_tokens=len(prompt_tokens),
            routing_start=0,
            include_routing=True,
        )

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response):
            messages = [{"role": "user", "content": [{"text": "Hi"}]}]
            await _collect_stream(model.stream(messages))

        assert model.token_manager.token_ids == prompt_tokens + output_ids
        assert len(model.token_manager) == 5

        routing = model.token_manager.routed_experts
        assert routing is not None
        decoded = _decode_routing(routing)
        assert len(decoded) == 5 * EXPERTS_PER_TOKEN

        for pos in range(5):
            chunk = decoded[pos * EXPERTS_PER_TOKEN : (pos + 1) * EXPERTS_PER_TOKEN]
            assert chunk == [pos * 10 + k for k in range(EXPERTS_PER_TOKEN)]

    async def test_multi_turn_with_tool_call(self, model, mock_tokenizer):
        """Multi-turn with overwrite semantics: turn 2 returns routing for ALL tokens.

        SGLang does not support routed_experts_start_len, so each response
        includes routing for every token in the sequence. The latest response
        overwrites (not accumulates) the previous routing data.
        """
        # --- Turn 1: user prompt -> model generates tool call ---
        prompt_tokens_t1 = [10, 20, 30]
        output_ids_t1 = [40, 50]
        mock_tokenizer.encode.return_value = prompt_tokens_t1

        response_t1 = _make_generate_response(
            text='<tool_call>{"name": "calc", "arguments": {"expr": "1+1"}}</tool_call>',
            output_ids=output_ids_t1,
            num_input_tokens=len(prompt_tokens_t1),
            routing_start=0,
            include_routing=True,
        )

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response_t1):
            messages_t1 = [{"role": "user", "content": [{"text": "What is 1+1?"}]}]
            await _collect_stream(model.stream(messages_t1, tool_specs=[{"name": "calc", "description": "calc", "inputSchema": {"json": {}}}]))

        assert model.token_manager.token_ids == prompt_tokens_t1 + output_ids_t1
        total_after_t1 = len(model.token_manager)  # 5

        # --- Turn 2: tool result -> model generates final answer ---
        # SGLang returns routing for ALL tokens (routing_start=0), not just new ones
        tool_result_tokens = [60, 70]
        output_ids_t2 = [80, 90, 100]
        mock_tokenizer.encode.return_value = tool_result_tokens

        total_input_t2 = total_after_t1 + len(tool_result_tokens)  # 7
        response_t2 = _make_generate_response(
            text="The answer is 2.",
            output_ids=output_ids_t2,
            num_input_tokens=total_input_t2,
            routing_start=0,  # SGLang returns routing for ALL tokens
            include_routing=True,
        )

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response_t2) as mock_gen:
            messages_t2 = messages_t1 + [
                {
                    "role": "assistant",
                    "content": [
                        {"text": '<tool_call>{"name": "calc", "arguments": {"expr": "1+1"}}</tool_call>'},
                        {"toolUse": {"toolUseId": "call_0000", "name": "calc", "input": {"expr": "1+1"}}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"toolResult": {"toolUseId": "call_0000", "content": [{"text": "2"}]}},
                    ],
                },
            ]
            await _collect_stream(model.stream(messages_t2, tool_specs=[{"name": "calc", "description": "calc", "inputSchema": {"json": {}}}]))

            # Verify return_routed_experts is passed to generate
            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs["return_routed_experts"] is True

        # Full token trajectory
        expected_ids = prompt_tokens_t1 + output_ids_t1 + tool_result_tokens + output_ids_t2
        assert model.token_manager.token_ids == expected_ids
        total_tokens = len(expected_ids)  # 10

        # Routing is from turn 2 only (overwrite, not accumulate).
        # Turn 2 returned routing for ALL 10 tokens (routing_start=0).
        routing = model.token_manager.routed_experts
        assert routing is not None
        decoded = _decode_routing(routing)
        assert len(decoded) == total_tokens * EXPERTS_PER_TOKEN

        # Verify per-token expert IDs from turn 2's response
        for pos in range(total_tokens):
            chunk = decoded[pos * EXPERTS_PER_TOKEN : (pos + 1) * EXPERTS_PER_TOKEN]
            assert chunk == [pos * 10 + k for k in range(EXPERTS_PER_TOKEN)]

    async def test_routing_aligns_with_loss_mask(self, model, mock_tokenizer):
        """Routing entries align 1:1 with token_ids and loss_mask."""
        prompt_tokens = [10, 20, 30]
        output_ids = [40, 50]
        mock_tokenizer.encode.return_value = prompt_tokens

        response = _make_generate_response(
            text="answer",
            output_ids=output_ids,
            num_input_tokens=len(prompt_tokens),
            routing_start=0,
            include_routing=True,
        )

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response):
            messages = [{"role": "user", "content": [{"text": "Hi"}]}]
            await _collect_stream(model.stream(messages))

        n_tokens = len(model.token_manager.token_ids)
        routing_entries = len(_decode_routing(model.token_manager.routed_experts)) // EXPERTS_PER_TOKEN

        assert routing_entries == n_tokens
        assert len(model.token_manager.loss_mask) == n_tokens
        assert len(model.token_manager.logprobs) == n_tokens

    async def test_routing_disabled_by_default(self, mock_tokenizer):
        """When return_routed_experts is not set, no routing data is recorded."""
        client = SGLangClient(base_url="http://localhost:30000")
        client._is_multimodal = False
        model = SGLangModel(client=client, tokenizer=mock_tokenizer)
        model.__dict__["message_separator"] = ""

        mock_tokenizer.encode.return_value = [10, 20]
        response = _make_generate_response(
            text="hi",
            output_ids=[30],
            num_input_tokens=2,
            include_routing=False,
        )

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response) as mock_gen:
            messages = [{"role": "user", "content": [{"text": "Hi"}]}]
            await _collect_stream(model.stream(messages))

            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs["return_routed_experts"] is False

        assert model.token_manager.routed_experts is None

    async def test_routing_absent_from_response(self, model, mock_tokenizer):
        """If server doesn't return routing data, routed_experts stays None."""
        mock_tokenizer.encode.return_value = [10, 20]
        response = _make_generate_response(
            text="hi",
            output_ids=[30],
            num_input_tokens=2,
            include_routing=False,
        )

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response):
            messages = [{"role": "user", "content": [{"text": "Hi"}]}]
            await _collect_stream(model.stream(messages))

        assert model.token_manager.routed_experts is None

    async def test_reset_clears_routing(self, model, mock_tokenizer):
        """model.reset() clears accumulated routing data."""
        mock_tokenizer.encode.return_value = [10, 20]
        response = _make_generate_response(
            text="hi",
            output_ids=[30],
            num_input_tokens=2,
            routing_start=0,
            include_routing=True,
        )

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response):
            messages = [{"role": "user", "content": [{"text": "Hi"}]}]
            await _collect_stream(model.stream(messages))

        assert model.token_manager.routed_experts is not None

        model.reset()

        assert model.token_manager.routed_experts is None
