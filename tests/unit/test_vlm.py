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

"""Unit tests for VLM (Vision Language Model) support."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands_sglang import SGLangModel
from strands_sglang.client import SGLangClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 1x1 red PNG (smallest valid PNG)
_RED_PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
_RED_PIXEL_B64 = base64.b64encode(_RED_PIXEL_PNG).decode()
_RED_PIXEL_DATA_URL = f"data:image/png;base64,{_RED_PIXEL_B64}"


def _image_block() -> dict:
    """Strands ImageContent block."""
    return {"image": {"format": "png", "source": {"bytes": _RED_PIXEL_PNG}}}


def _image_block_from_bytes(image_bytes: bytes) -> dict:
    """Strands ImageContent block with caller-provided bytes."""
    return {"image": {"format": "png", "source": {"bytes": image_bytes}}}


def _image_data_url(image_bytes: bytes) -> str:
    """Build the expected data URL for an image block."""
    return f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.name_or_path = "/nonexistent"
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    return tokenizer


@pytest.fixture
def client():
    return SGLangClient(base_url="http://localhost:30000")


@pytest.fixture
def vlm_model(client, mock_tokenizer):
    """SGLangModel in VLM mode."""
    client._is_multimodal = True
    model = SGLangModel(client=client, tokenizer=mock_tokenizer)
    model.__dict__["message_separator"] = ""  # override cached_property (mock has no real template)
    return model


@pytest.fixture
def text_model(client, mock_tokenizer):
    """SGLangModel in text-only mode."""
    client._is_multimodal = False
    model = SGLangModel(client=client, tokenizer=mock_tokenizer)
    model.__dict__["message_separator"] = ""  # override cached_property (mock has no real template)
    return model


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


class TestVLMAutoDetect:
    @pytest.mark.asyncio
    async def test_has_image_understanding_true(self, client):
        """Detects multimodal from server's has_image_understanding field."""
        client.model_info = AsyncMock(return_value={"has_image_understanding": True})
        assert await client.is_multimodal() is True

    @pytest.mark.asyncio
    async def test_has_image_understanding_false(self, client):
        """Text-only models have has_image_understanding=False."""
        client.model_info = AsyncMock(return_value={"has_image_understanding": False})
        assert await client.is_multimodal() is False

    @pytest.mark.asyncio
    async def test_missing_field_defaults_false(self, client):
        """Older SGLang servers without has_image_understanding default to False."""
        client.model_info = AsyncMock(return_value={"model_path": "some/model"})
        assert await client.is_multimodal() is False

    @pytest.mark.asyncio
    async def test_server_unreachable_defaults_false(self, client):
        """When server is unreachable, defaults to False."""
        client.model_info = AsyncMock(return_value=None)
        assert await client.is_multimodal() is False

    @pytest.mark.asyncio
    async def test_result_is_cached(self, client):
        """Second call returns cached result without querying server again."""
        client.model_info = AsyncMock(return_value={"has_image_understanding": True})
        await client.is_multimodal()
        await client.is_multimodal()
        client.model_info.assert_awaited_once()


# ---------------------------------------------------------------------------
# format_content_block
# ---------------------------------------------------------------------------


class TestFormatContentBlockMultimodal:
    """format_content_block with is_multimodal=True returns dicts, not strings."""

    def test_text_block_returns_dict(self):
        result = SGLangModel.format_content_block({"text": "hello"}, is_multimodal=True)
        assert result == {"type": "text", "text": "hello"}

    def test_text_block_returns_string_when_not_multimodal(self):
        result = SGLangModel.format_content_block({"text": "hello"}, is_multimodal=False)
        assert result == "hello"

    def test_image_block_returns_data_url(self):
        result = SGLangModel.format_content_block(_image_block(), is_multimodal=True)
        assert result == {"type": "image", "image": _RED_PIXEL_DATA_URL}

    def test_image_block_raises_when_not_multimodal(self):
        """Image blocks require is_multimodal=True (no 'text' key to flatten to)."""
        with pytest.raises(KeyError):
            SGLangModel.format_content_block(_image_block(), is_multimodal=False)

    def test_json_block_returns_dict(self):
        result = SGLangModel.format_content_block({"json": {"key": "val"}}, is_multimodal=True)
        assert result == {"type": "text", "text": '{"key": "val"}'}


# ---------------------------------------------------------------------------
# format_messages with is_multimodal
# ---------------------------------------------------------------------------


class TestFormatMessagesMultimodal:
    def test_mixed_text_and_image(self):
        """Text + image in same message are grouped into list content."""
        messages = [{"role": "user", "content": [{"text": "what is this?"}, _image_block()]}]
        result = SGLangModel.format_messages(messages, is_multimodal=True)
        assert len(result) == 1
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0] == {"type": "text", "text": "what is this?"}
        assert result[0]["content"][1]["type"] == "image"

    def test_tool_result_with_text_and_image(self):
        """Tool result with text + image grouped into list content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_001",
                            "status": "success",
                            "content": [{"text": "Image loaded"}, _image_block()],
                        }
                    }
                ],
            }
        ]
        result = SGLangModel.format_messages(messages, is_multimodal=True)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0] == {"type": "text", "text": "Image loaded"}
        assert result[0]["content"][1]["type"] == "image"


# ---------------------------------------------------------------------------
# Image accumulation across turns
# ---------------------------------------------------------------------------


class TestImageAccumulation:
    def test_text_only_server_still_formats_image_prompts_multimodally(self, text_model, mock_tokenizer):
        """Prompt images override a stale text-only server probe."""
        messages = [{"role": "user", "content": [{"text": "describe"}, _image_block()]}]

        token_ids = text_model.tokenize_prompt_messages(messages, system_prompt=None, is_multimodal=False)

        assert token_ids == [1, 2, 3]
        rendered_messages = mock_tokenizer.apply_chat_template.call_args.args[0]
        assert rendered_messages == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "describe"}, {"type": "image", "image": _RED_PIXEL_DATA_URL}],
            }
        ]

    def test_prepare_prompt_messages_tracks_current_prompt_images(self, vlm_model, mock_tokenizer):
        """Prepared prompt image data should reflect the current visible prompt, not historical accumulation."""
        messages = [
            {"role": "user", "content": [{"text": "describe"}, _image_block()]},
            {
                "role": "assistant",
                "content": [
                    {"text": "I'll use the tool."},
                    {"toolUse": {"toolUseId": "call_001", "name": "screenshot", "input": {}}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_001",
                            "status": "success",
                            "content": [_image_block()],
                        }
                    }
                ],
            },
        ]

        prepared = vlm_model._prepare_prompt_messages(messages, system_prompt=None, tool_specs=None, is_multimodal=True)

        assert prepared.image_data == [_RED_PIXEL_DATA_URL, _RED_PIXEL_DATA_URL]
        assert vlm_model.image_data == []

    def test_reset_clears_current_image_state(self, vlm_model, mock_tokenizer):
        vlm_model.image_data = [_RED_PIXEL_DATA_URL]

        vlm_model.reset()
        assert vlm_model.image_data == []

    def test_unexpanded_images_dropped(self, vlm_model, mock_tokenizer):
        """Images whose data URLs survive in the template output are dropped."""
        mock_tokenizer.apply_chat_template.return_value = f"tool result: {_RED_PIXEL_DATA_URL}"
        messages = [{"role": "user", "content": [{"text": "describe"}, _image_block()]}]

        prepared = vlm_model._prepare_prompt_messages(messages, system_prompt=None, tool_specs=None, is_multimodal=True)

        assert prepared.image_data == []

    def test_expanded_images_kept(self, vlm_model, mock_tokenizer):
        """Images expanded by the template (data URL consumed) are kept."""
        mock_tokenizer.apply_chat_template.return_value = "<|vision_start|><|image_pad|><|vision_end|>describe"
        messages = [{"role": "user", "content": [{"text": "describe"}, _image_block()]}]

        prepared = vlm_model._prepare_prompt_messages(messages, system_prompt=None, tool_specs=None, is_multimodal=True)

        assert prepared.image_data == [_RED_PIXEL_DATA_URL]

    @pytest.mark.asyncio
    async def test_image_drop_reset_keeps_only_current_prompt_images(self, vlm_model, mock_tokenizer):
        """Image-window resets should use only current images and start a fresh token trajectory."""
        image_1 = b"img-1"
        image_2 = b"img-2"
        image_1_url = _image_data_url(image_1)
        image_2_url = _image_data_url(image_2)

        mock_tokenizer.encode.side_effect = [[11, 12], [21, 22]]
        mock_tokenizer.apply_chat_template.side_effect = [
            "<|vision_start|><|image_pad|><|vision_end|>turn-1",
            "<|vision_start|><|image_pad|><|vision_end|>turn-2",
        ]
        vlm_model.client.generate = AsyncMock(
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
                    "output_ids": [23],
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

        messages_turn_1 = [{"role": "user", "content": [{"text": "old image"}, _image_block_from_bytes(image_1)]}]
        messages_turn_2 = [{"role": "user", "content": [{"text": "new image only"}, _image_block_from_bytes(image_2)]}]

        async for _ in vlm_model.stream(messages_turn_1):
            pass
        async for _ in vlm_model.stream(messages_turn_2):
            pass

        first_call = vlm_model.client.generate.call_args_list[0].kwargs
        second_call = vlm_model.client.generate.call_args_list[1].kwargs

        assert first_call["image_data"] == [image_1_url]
        assert second_call["image_data"] == [image_2_url]
        assert vlm_model.token_manager.turn_trajectory_ids == [0, 1]
        assert [trajectory.token_ids for trajectory in vlm_model.token_manager.trajectories] == [
            [11, 12, 13],
            [21, 22, 23],
        ]
        assert [trajectory.loss_mask for trajectory in vlm_model.token_manager.trajectories] == [
            [0, 0, 1],
            [0, 0, 1],
        ]
        assert [trajectory.token_offset for trajectory in vlm_model.token_manager.trajectories] == [0, 3]
        assert [
            token_id for trajectory in vlm_model.token_manager.trajectories for token_id in trajectory.token_ids
        ] == [
            11,
            12,
            13,
            21,
            22,
            23,
        ]

    @pytest.mark.asyncio
    async def test_image_rewrite_with_same_prompt_tokens_starts_new_trajectory(self, vlm_model, mock_tokenizer):
        """Changing retained image content should reset even if chat-template token IDs stay prefix-compatible."""
        image_1 = b"img-1"
        image_2 = b"img-2"
        image_1_url = _image_data_url(image_1)
        image_2_url = _image_data_url(image_2)

        mock_tokenizer.encode.side_effect = [[11, 12], [11, 12, 13, 14]]
        mock_tokenizer.apply_chat_template.side_effect = [
            "<|vision_start|><|image_pad|><|vision_end|>turn-1",
            "<|vision_start|><|image_pad|><|vision_end|>turn-1 plus follow-up",
        ]
        vlm_model.client.generate = AsyncMock(
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
                    "output_ids": [15],
                    "meta_info": {
                        "prompt_tokens": 4,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                        "finish_reason": {"type": "stop"},
                        "e2e_latency": 0.1,
                        "input_token_logprobs": [[-0.4], [-0.5], [-0.6], [-0.7]],
                        "output_token_logprobs": [[-0.8]],
                    },
                },
            ]
        )

        messages_turn_1 = [{"role": "user", "content": [{"text": "image one"}, _image_block_from_bytes(image_1)]}]
        messages_turn_2 = [
            {"role": "user", "content": [{"text": "image two now"}, _image_block_from_bytes(image_2)]},
            {"role": "assistant", "content": [{"text": "first"}]},
            {"role": "user", "content": [{"text": "follow up"}]},
        ]

        async for _ in vlm_model.stream(messages_turn_1):
            pass
        async for _ in vlm_model.stream(messages_turn_2):
            pass

        first_call = vlm_model.client.generate.call_args_list[0].kwargs
        second_call = vlm_model.client.generate.call_args_list[1].kwargs

        assert first_call["image_data"] == [image_1_url]
        assert second_call["image_data"] == [image_2_url]
        assert vlm_model.token_manager.turn_trajectory_ids == [0, 1]
        assert [trajectory.token_ids for trajectory in vlm_model.token_manager.trajectories] == [
            [11, 12, 13],
            [11, 12, 13, 14, 15],
        ]


# ---------------------------------------------------------------------------
# stream() — image_data forwarding
# ---------------------------------------------------------------------------


class TestStreamImageData:
    @pytest.mark.asyncio
    async def test_image_data_passed_to_client(self, vlm_model, mock_tokenizer):
        """When message contains an image, image_data is forwarded to client.generate."""
        with patch.object(vlm_model.client, "generate", new_callable=_async_mock_generate) as mock_gen:
            vlm_model.client.is_multimodal = AsyncMock(return_value=True)
            async for _ in vlm_model.stream(
                messages=[{"role": "user", "content": [{"text": "describe"}, _image_block()]}],
            ):
                pass
            assert mock_gen.call_args.kwargs["image_data"] == [_RED_PIXEL_DATA_URL]

    @pytest.mark.asyncio
    async def test_no_image_data_passes_none(self, text_model, mock_tokenizer):
        """When image_data is empty, image_data=None is passed."""
        with patch.object(text_model.client, "generate", new_callable=_async_mock_generate) as mock_gen:
            text_model.client.is_multimodal = AsyncMock(return_value=False)
            async for _ in text_model.stream(
                messages=[{"role": "user", "content": [{"text": "hello"}]}],
            ):
                pass
            assert mock_gen.call_args.kwargs["image_data"] is None


def _async_mock_generate():
    """Factory for async mock of client.generate."""
    mock = MagicMock()

    async def _generate(**kwargs):
        return {
            "text": "response",
            "output_ids": [100, 101],
            "meta_info": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop"},
                "e2e_latency": 0.1,
            },
        }

    mock.side_effect = _generate
    return mock
