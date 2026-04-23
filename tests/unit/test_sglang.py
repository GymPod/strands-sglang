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
    model.__dict__["assistant_stop_token_ids"] = []
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
    """Tests for tokenize_prompt_messages prompt-state behavior."""

    def test_no_new_messages_raises(self, model, mock_tokenizer):
        """Raises RuntimeError when the prompt is unchanged."""
        model._active_input_ids = [1, 2, 3]
        mock_tokenizer.encode.return_value = [1, 2, 3]

        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
        ]

        with pytest.raises(RuntimeError, match="No new messages to tokenize"):
            model.tokenize_prompt_messages(messages, system_prompt=None)

    def test_additive_prompt_returns_incremental_suffix(self, model, mock_tokenizer):
        """Continuation prompts only return the suffix beyond the active context."""
        model._active_input_ids = [10, 11]
        model.message_count = 2
        mock_tokenizer.encode.return_value = [10, 11, 12, 13]

        token_ids = model.tokenize_prompt_messages(
            [
                {"role": "user", "content": [{"text": "Hello"}]},
                {"role": "assistant", "content": [{"text": "Continue"}]},
            ],
            system_prompt=None,
        )

        assert token_ids == [12, 13]

    def test_context_reset_returns_full_prompt(self, model, mock_tokenizer):
        """Compressed/reset prompts are re-tokenized from full current context."""
        model._active_input_ids = [10, 11, 12]
        mock_tokenizer.encode.return_value = [90, 91]

        token_ids = model.tokenize_prompt_messages(
            [{"role": "user", "content": [{"text": "Summary"}]}],
            system_prompt=None,
        )

        assert token_ids == [90, 91]


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

    async def test_stream_uses_full_prompt_after_context_reset(self, mock_tokenizer):
        """A context reset sends the new full prompt, not the old trajectory prefix."""
        client = SGLangClient(base_url="http://localhost:30000")
        client._is_multimodal = False
        client.generate = AsyncMock(
            return_value={
                "text": "reset",
                "output_ids": [7],
                "meta_info": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop"},
                    "e2e_latency": 0.1,
                    "input_token_logprobs": [[-0.1], [-0.2]],
                    "output_token_logprobs": [[-0.3]],
                },
            }
        )
        mock_tokenizer.encode.return_value = [41, 42]
        model = SGLangModel(client=client, tokenizer=mock_tokenizer)
        model._active_input_ids = [10, 11, 12]
        model.message_count = 1

        messages = [{"role": "user", "content": [{"text": "compressed summary"}]}]
        async for _ in model.stream(messages):
            pass

        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs["input_ids"] == [41, 42]
        assert call_kwargs.kwargs["logprob_start_len"] == 0
        assert model.token_manager.turn_trajectory_ids == [0]
        assert len(model.token_manager.trajectories) == 1

    async def test_stream_additive_turn_keeps_exact_prior_token_stream(self, mock_tokenizer):
        """Additive turns must reuse raw previously generated tokens, not reserialized history."""
        client = SGLangClient(base_url="http://localhost:30000")
        client._is_multimodal = False
        client.generate = AsyncMock(
            side_effect=[
                _make_generate_response(text="first", output_ids=[13], num_input_tokens=2),
                _make_generate_response(text="second", output_ids=[23], num_input_tokens=5),
            ]
        )
        mock_tokenizer.apply_chat_template.side_effect = [
            "turn-1 prompt",
            "retokenized full prompt for turn 2",
            "FAKE PREFIXfollow-up",
            "FAKE PREFIX",
        ]
        mock_tokenizer.encode.side_effect = [
            [11, 12],
            [81, 82, 83, 84],
            [21, 22],
        ]

        model = SGLangModel(client=client, tokenizer=mock_tokenizer)
        model.__dict__["message_separator"] = ""
        model.__dict__["assistant_stop_token_ids"] = []

        async for _ in model.stream([{"role": "user", "content": [{"text": "turn 1"}]}]):
            pass
        async for _ in model.stream(
            [
                {"role": "user", "content": [{"text": "turn 1"}]},
                {"role": "assistant", "content": [{"text": "assistant reformatted"}]},
                {"role": "user", "content": [{"text": "follow-up"}]},
            ]
        ):
            pass

        second_call_kwargs = client.generate.call_args_list[1].kwargs
        assert second_call_kwargs["input_ids"] == [11, 12, 13, 21, 22]
        assert second_call_kwargs["logprob_start_len"] == 2
        assert model.token_manager.turn_trajectory_ids == [0, 0]
        assert [trajectory.token_ids for trajectory in model.token_manager.trajectories] == [[11, 12, 13, 21, 22, 23]]

    async def test_stream_tool_result_turn_inserts_missing_assistant_stop_token(self, mock_tokenizer):
        """Tool-result continuations must restore the assistant stop token if SGLang omits it."""
        client = SGLangClient(base_url="http://localhost:30000")
        client._is_multimodal = False
        client.generate = AsyncMock(
            side_effect=[
                _make_generate_response(
                    text='<tool_call>{"name": "calc", "arguments": {"expr": "1+1"}}</tool_call>',
                    output_ids=[13],
                    num_input_tokens=2,
                ),
                _make_generate_response(text="The answer is 2.", output_ids=[23], num_input_tokens=6),
            ]
        )
        mock_tokenizer.apply_chat_template.side_effect = [
            "turn-1 prompt",
            "retokenized full prompt for turn 2",
            "FAKE PREFIXfollow-up tool result",
            "FAKE PREFIX",
        ]
        mock_tokenizer.encode.side_effect = [
            [11, 12],
            [11, 12, 13, 99, 21, 22],
            [21, 22],
        ]

        model = SGLangModel(client=client, tokenizer=mock_tokenizer)
        model.__dict__["message_separator"] = ""
        model.__dict__["assistant_stop_token_ids"] = [99]

        turn_1_messages = [{"role": "user", "content": [{"text": "What is 1+1?"}]}]
        async for _ in model.stream(
            turn_1_messages,
            tool_specs=[{"name": "calc", "description": "calc", "inputSchema": {"json": {}}}],
        ):
            pass

        assert model._active_input_ids == [11, 12, 13, 99]

        turn_2_messages = turn_1_messages + [
            {
                "role": "assistant",
                "content": [
                    {"text": '<tool_call>{"name": "calc", "arguments": {"expr": "1+1"}}</tool_call>'},
                    {"toolUse": {"toolUseId": "call_0000", "name": "calc", "input": {"expr": "1+1"}}},
                ],
            },
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "call_0000", "content": [{"text": "2"}]}}],
            },
        ]
        async for _ in model.stream(
            turn_2_messages,
            tool_specs=[{"name": "calc", "description": "calc", "inputSchema": {"json": {}}}],
        ):
            pass

        second_call_kwargs = client.generate.call_args_list[1].kwargs
        assert second_call_kwargs["input_ids"] == [11, 12, 13, 99, 21, 22]
        assert second_call_kwargs["logprob_start_len"] == 3
        assert model.token_manager.turn_trajectory_ids == [0, 0]
        assert [trajectory.token_ids for trajectory in model.token_manager.trajectories] == [[11, 12, 13, 21, 22, 23]]

    async def test_stream_marks_new_trajectory_after_context_reset(self, mock_tokenizer):
        """A non-prefix prompt reset should start a new grouped trajectory."""
        client = SGLangClient(base_url="http://localhost:30000")
        client._is_multimodal = False
        client.generate = AsyncMock(
            side_effect=[
                {
                    "text": "first",
                    "output_ids": [13],
                    "meta_info": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                        "finish_reason": {"type": "stop"},
                        "e2e_latency": 0.1,
                        "input_token_logprobs": [[-0.1], [-0.2]],
                        "output_token_logprobs": [[-0.3]],
                    },
                },
                {
                    "text": "second",
                    "output_ids": [91],
                    "meta_info": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                        "finish_reason": {"type": "stop"},
                        "e2e_latency": 0.1,
                        "input_token_logprobs": [[-0.4], [-0.5]],
                        "output_token_logprobs": [[-0.6]],
                    },
                },
            ]
        )
        mock_tokenizer.encode.side_effect = [[11, 12], [41, 42]]

        model = SGLangModel(client=client, tokenizer=mock_tokenizer)
        model.__dict__["message_separator"] = ""
        model.__dict__["assistant_stop_token_ids"] = []

        async for _ in model.stream([{"role": "user", "content": [{"text": "first"}]}]):
            pass
        async for _ in model.stream([{"role": "user", "content": [{"text": "summary"}]}]):
            pass

        assert model.token_manager.turn_trajectory_ids == [0, 1]
        assert [trajectory.token_ids for trajectory in model.token_manager.trajectories] == [[11, 12, 13], [41, 42, 91]]

    async def test_summary_style_resets_preserve_exact_token_partition(self, mock_tokenizer):
        """Summary-like rewrites should partition the global token stream without loss or duplication."""
        client = SGLangClient(base_url="http://localhost:30000")
        client._is_multimodal = False
        client.generate = AsyncMock(
            side_effect=[
                _make_generate_response(text="first", output_ids=[13], num_input_tokens=2),
                _make_generate_response(text="summary", output_ids=[23], num_input_tokens=2),
                _make_generate_response(text="post-summary", output_ids=[33], num_input_tokens=2),
            ]
        )
        mock_tokenizer.encode.side_effect = [[11, 12], [21, 22], [31, 32]]

        model = SGLangModel(client=client, tokenizer=mock_tokenizer)
        model.__dict__["message_separator"] = ""
        model.__dict__["assistant_stop_token_ids"] = []

        async for _ in model.stream([{"role": "user", "content": [{"text": "original"}]}]):
            pass
        async for _ in model.stream([{"role": "user", "content": [{"text": "Please summarize this conversation."}]}]):
            pass
        async for _ in model.stream([{"role": "user", "content": [{"text": "continue from summary"}]}]):
            pass

        assert model.token_manager.turn_trajectory_ids == [0, 1, 2]
        assert [trajectory.token_ids for trajectory in model.token_manager.trajectories] == [
            [11, 12, 13],
            [21, 22, 23],
            [31, 32, 33],
        ]
        assert [trajectory.loss_mask for trajectory in model.token_manager.trajectories] == [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
        _assert_trajectory_partition(model)


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

    When include_routing is True, generates deterministic per-transition routing:
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
        # Generate routing for transitions from routing_start to total-2.
        expert_ids = []
        for pos in range(routing_start, max(total - 1, routing_start)):
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


def _assert_trajectory_partition(model: SGLangModel) -> None:
    """Assert grouped trajectories exactly partition the flat token stream."""
    trajectories = model.token_manager.trajectories
    assert trajectories

    flat_token_ids = model.token_manager.token_ids
    flat_logprobs = model.token_manager.logprobs
    flat_loss_mask = model.token_manager.loss_mask

    assert [token_id for trajectory in trajectories for token_id in trajectory.token_ids] == flat_token_ids
    assert [logprob for trajectory in trajectories for logprob in trajectory.logprobs] == flat_logprobs
    assert [mask for trajectory in trajectories for mask in trajectory.loss_mask] == flat_loss_mask

    expected_offset = 0
    for trajectory in trajectories:
        assert trajectory.token_offset == expected_offset
        if trajectory.routed_experts is not None:
            routing_entries = len(_decode_routing(trajectory.routed_experts)) // EXPERTS_PER_TOKEN
            assert routing_entries == max(len(trajectory.token_ids) - 1, 0)
        expected_offset += len(trajectory.token_ids)


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
        m.__dict__["assistant_stop_token_ids"] = []
        return m

    async def test_single_turn_routing(self, model):
        """Single turn: routing covers local trajectory transitions."""
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
        assert len(decoded) == 4 * EXPERTS_PER_TOKEN

        for pos in range(4):
            chunk = decoded[pos * EXPERTS_PER_TOKEN : (pos + 1) * EXPERTS_PER_TOKEN]
            assert chunk == [pos * 10 + k for k in range(EXPERTS_PER_TOKEN)]
        assert model.token_manager.trajectories[0].routed_experts == routing

    async def test_multi_turn_with_tool_call(self, model, mock_tokenizer):
        """Multi-turn with overwrite semantics within the active trajectory.

        SGLang does not support routed_experts_start_len, so each response
        includes routing for every local transition in the active trajectory.
        The latest response overwrites (not accumulates) the previous routing
        data for that trajectory.
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
            await _collect_stream(
                model.stream(
                    messages_t1, tool_specs=[{"name": "calc", "description": "calc", "inputSchema": {"json": {}}}]
                )
            )

        assert model.token_manager.token_ids == prompt_tokens_t1 + output_ids_t1
        total_after_t1 = len(model.token_manager)  # 5

        # --- Turn 2: tool result -> model generates final answer ---
        # SGLang returns routing for ALL active transitions (routing_start=0), not just new ones
        tool_result_tokens = [60, 70]
        output_ids_t2 = [80, 90, 100]
        mock_tokenizer.encode.return_value = tool_result_tokens

        total_input_t2 = total_after_t1 + len(tool_result_tokens)  # 7
        response_t2 = _make_generate_response(
            text="The answer is 2.",
            output_ids=output_ids_t2,
            num_input_tokens=total_input_t2,
            routing_start=0,  # SGLang returns routing for all active transitions
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
            await _collect_stream(
                model.stream(
                    messages_t2, tool_specs=[{"name": "calc", "description": "calc", "inputSchema": {"json": {}}}]
                )
            )

            # Verify return_routed_experts is passed to generate
            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs["return_routed_experts"] is True

        # Full token trajectory
        expected_ids = prompt_tokens_t1 + output_ids_t1 + tool_result_tokens + output_ids_t2
        assert model.token_manager.token_ids == expected_ids
        total_tokens = len(expected_ids)  # 10

        # Routing is from turn 2 only (overwrite, not accumulate).
        # Turn 2 returned routing for ALL 9 transitions in the 10-token trajectory.
        routing = model.token_manager.routed_experts
        assert routing is not None
        decoded = _decode_routing(routing)
        assert len(decoded) == (total_tokens - 1) * EXPERTS_PER_TOKEN

        # Verify per-token expert IDs from turn 2's response
        for pos in range(total_tokens - 1):
            chunk = decoded[pos * EXPERTS_PER_TOKEN : (pos + 1) * EXPERTS_PER_TOKEN]
            assert chunk == [pos * 10 + k for k in range(EXPERTS_PER_TOKEN)]

        trajectories = model.token_manager.trajectories
        assert len(trajectories) == 1
        assert trajectories[0].trajectory_id == 0
        assert trajectories[0].token_offset == 0
        assert trajectories[0].token_ids == expected_ids
        assert trajectories[0].routed_experts == routing

    async def test_routing_aligns_with_loss_mask(self, model, mock_tokenizer):
        """Routing entries align 1:1 with token transitions."""
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

        assert routing_entries == n_tokens - 1
        assert len(model.token_manager.loss_mask) == n_tokens
        assert len(model.token_manager.logprobs) == n_tokens

    async def test_context_reset_keeps_routing_local_to_each_trajectory(self, model, mock_tokenizer):
        """Managed context resets must snapshot routing per split trajectory."""
        model.client.generate = AsyncMock(
            side_effect=[
                _make_generate_response(text="first", output_ids=[13], num_input_tokens=2, include_routing=True),
                _make_generate_response(text="summary", output_ids=[23], num_input_tokens=2, include_routing=True),
            ]
        )
        mock_tokenizer.encode.side_effect = [[11, 12], [21, 22]]

        async for _ in model.stream([{"role": "user", "content": [{"text": "first"}]}]):
            pass
        async for _ in model.stream([{"role": "user", "content": [{"text": "summary"}]}]):
            pass

        trajectories = model.token_manager.trajectories
        assert len(trajectories) == 2
        assert [trajectory.token_ids for trajectory in trajectories] == [[11, 12, 13], [21, 22, 23]]
        assert [trajectory.token_offset for trajectory in trajectories] == [0, 3]

        first_routing = trajectories[0].routed_experts
        second_routing = trajectories[1].routed_experts
        assert first_routing is not None
        assert second_routing is not None
        assert _decode_routing(first_routing) == [0, 1, 2, 3, 10, 11, 12, 13]
        assert _decode_routing(second_routing) == [0, 1, 2, 3, 10, 11, 12, 13]
        assert model.token_manager.routed_experts == second_routing

    async def test_routing_disabled_by_default(self, mock_tokenizer):
        """When return_routed_experts is not set, no routing data is recorded."""
        client = SGLangClient(base_url="http://localhost:30000")
        client._is_multimodal = False
        model = SGLangModel(client=client, tokenizer=mock_tokenizer)
        model.__dict__["message_separator"] = ""
        model.__dict__["assistant_stop_token_ids"] = []

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
