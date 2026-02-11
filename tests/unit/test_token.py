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

"""Unit tests for token module."""

import base64
import struct

import pytest

from strands_sglang import Token, TokenManager


class TestToken:
    """Tests for Token dataclass."""

    def test_token_defaults(self):
        """Token has correct default values."""
        token = Token(token_id=42)
        assert token.token_id == 42
        assert token.logprob is None
        assert token.loss_mask is True


class TestTokenManagerBasic:
    """Basic TokenManager tests."""

    def test_init(self):
        """TokenManager starts empty."""
        manager = TokenManager()
        assert len(manager) == 0

    def test_empty_manager(self):
        """Empty manager returns empty lists."""
        manager = TokenManager()
        assert manager.tokens == []
        assert manager.token_ids == []
        assert manager.loss_mask == []
        assert manager.logprobs == []
        assert manager.segments == []
        assert manager.segment_info == []


class TestTokenManagerAddPrompt:
    """Tests for add_prompt method."""

    def test_add_prompt_basic(self):
        """add_prompt adds tokens with loss_mask=False."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])

        assert manager.token_ids == [1, 2, 3]
        assert manager.loss_mask == [False, False, False]
        assert manager.logprobs == [None, None, None]

    def test_add_prompt_with_logprobs(self):
        """add_prompt accepts logprobs."""
        manager = TokenManager()
        manager.add_prompt([10, 20], logprobs=[-0.1, -0.2])

        assert manager.token_ids == [10, 20]
        assert manager.logprobs == [-0.1, -0.2]

    def test_add_prompt_empty(self):
        """add_prompt with empty list does nothing."""
        manager = TokenManager()
        manager.add_prompt([])
        assert len(manager) == 0
        assert manager.segments == []

    def test_add_prompt_mismatched_logprobs_raises(self):
        """add_prompt raises ValueError if logprobs length doesn't match token_ids."""
        manager = TokenManager()
        with pytest.raises(ValueError, match="logprobs length"):
            manager.add_prompt([1, 2, 3], logprobs=[-0.1])


class TestTokenManagerAddResponse:
    """Tests for add_response method."""

    def test_add_response_basic(self):
        """add_response adds tokens with loss_mask=True."""
        manager = TokenManager()
        manager.add_prompt([1])
        manager.add_response([4, 5, 6])

        assert manager.token_ids == [1, 4, 5, 6]
        assert manager.loss_mask == [False, True, True, True]

    def test_add_response_with_logprobs(self):
        """add_response accepts logprobs."""
        manager = TokenManager()
        manager.add_prompt([1])
        manager.add_response([100, 200], logprobs=[-0.5, -0.6])

        assert manager.logprobs[1:] == [-0.5, -0.6]

    def test_add_response_empty(self):
        """add_response with empty list does nothing."""
        manager = TokenManager()
        manager.add_response([])
        assert len(manager) == 0

    def test_add_response_without_prompt_raises(self):
        """add_response raises RuntimeError if no prompt segment exists."""
        manager = TokenManager()
        with pytest.raises(RuntimeError, match="First segment must be a prompt"):
            manager.add_response([4, 5, 6])

    def test_add_response_mismatched_logprobs_raises(self):
        """add_response raises ValueError if logprobs length doesn't match token_ids."""
        manager = TokenManager()
        manager.add_prompt([0])
        with pytest.raises(ValueError, match="logprobs length"):
            manager.add_response([1, 2, 3], logprobs=[-0.1, -0.2])


class TestTokenManagerMultipleSegments:
    """Tests for multiple segment operations."""

    def test_prompt_response_sequence(self):
        """Typical prompt-response sequence works correctly."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5], logprobs=[-0.1, -0.2])

        assert manager.token_ids == [1, 2, 3, 4, 5]
        assert manager.loss_mask == [False, False, False, True, True]
        assert manager.logprobs == [None, None, None, -0.1, -0.2]

    def test_multi_turn_conversation(self):
        """Multi-turn conversation with tool calls."""
        manager = TokenManager()

        # Initial prompt
        manager.add_prompt([1, 2])
        # Model response (tool call)
        manager.add_response([3, 4], logprobs=[-0.1, -0.2])
        # Tool result (treated as prompt)
        manager.add_prompt([5, 6])
        # Final model response
        manager.add_response([7, 8], logprobs=[-0.3, -0.4])

        assert manager.token_ids == [1, 2, 3, 4, 5, 6, 7, 8]
        assert manager.loss_mask == [False, False, True, True, False, False, True, True]
        assert manager.logprobs == [None, None, -0.1, -0.2, None, None, -0.3, -0.4]
        assert len(manager) == 8

    def test_segments_property(self):
        """segments property returns segment data."""
        manager = TokenManager()
        manager.add_prompt([1, 2])
        manager.add_response([3, 4])

        segments = manager.segments
        assert len(segments) == 2
        assert all(not t.loss_mask for t in segments[0])
        assert all(t.loss_mask for t in segments[1])

    def test_initial_prompt(self):
        """initial_prompt returns the first segment."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5])

        prompt = manager.initial_prompt
        assert [t.token_id for t in prompt] == [1, 2, 3]
        assert all(not t.loss_mask for t in prompt)

    def test_initial_prompt_empty(self):
        """initial_prompt returns empty list when no segments exist."""
        manager = TokenManager()
        assert manager.initial_prompt == []

    def test_segment_info(self):
        """segment_info returns correct metadata."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5])
        manager.add_prompt([6])

        assert manager.segment_info == [(False, 3), (True, 2), (False, 1)]


class TestTokenManagerReset:
    """Tests for reset functionality."""

    def test_reset_clears_all(self):
        """reset clears all accumulated tokens."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5])

        manager.reset()

        assert len(manager) == 0
        assert manager.tokens == []
        assert manager.segments == []

    def test_reset_clears_routed_experts(self):
        """reset clears accumulated routed experts."""
        manager = TokenManager()
        chunk = base64.b64encode(b"\x01\x00\x00\x00").decode("ascii")
        manager.add_routed_experts(chunk)

        manager.reset()

        assert manager.routed_experts is None

    def test_reset_allows_reuse(self):
        """Manager can be reused after reset."""
        manager = TokenManager()
        manager.add_prompt([1, 2])
        manager.add_response([3, 4])

        manager.reset()

        manager.add_prompt([10, 20])
        manager.add_response([30])

        assert manager.token_ids == [10, 20, 30]
        assert len(manager) == 3


class TestTokenManagerTokensProperty:
    """Tests for tokens property and iteration."""

    def test_tokens_returns_flat_list(self):
        """tokens property returns all tokens in order."""
        manager = TokenManager()
        manager.add_prompt([1, 2])
        manager.add_response([3, 4])

        tokens = manager.tokens
        assert len(tokens) == 4
        assert [t.token_id for t in tokens] == [1, 2, 3, 4]

    def test_len_counts_all_tokens(self):
        """__len__ returns total token count across segments."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4])
        manager.add_prompt([5, 6])

        assert len(manager) == 6


class TestEdgeCases:
    """Edge case tests."""

    def test_single_token_segments(self):
        """Single-token segments work correctly."""
        manager = TokenManager()
        manager.add_prompt([1])
        manager.add_response([2])
        manager.add_prompt([3])

        assert manager.token_ids == [1, 2, 3]
        assert manager.loss_mask == [False, True, False]

    def test_logprobs_longer_than_tokens(self):
        """Extra logprobs are ignored (only uses indices up to token count)."""
        manager = TokenManager()
        manager.add_prompt([1, 2], logprobs=[-0.1, -0.2, -0.3, -0.4])

        # Only first 2 logprobs used
        assert manager.logprobs == [-0.1, -0.2]

    def test_zero_logprob(self):
        """Zero logprob is valid and preserved."""
        manager = TokenManager()
        manager.add_prompt([0])
        manager.add_response([1], logprobs=[0.0])
        assert manager.logprobs[1:] == [0.0]

    def test_negative_token_ids(self):
        """Negative token IDs are accepted (some tokenizers use them)."""
        manager = TokenManager()
        manager.add_prompt([-1, -100])
        assert manager.token_ids == [-1, -100]

    def test_large_token_ids(self):
        """Large token IDs are handled."""
        manager = TokenManager()
        manager.add_prompt([100000, 999999])
        assert manager.token_ids == [100000, 999999]


def _make_routed_experts_b64(expert_ids: list[int]) -> str:
    """Helper: encode a list of int32 expert IDs to base64 (matching SGLang format)."""
    raw = struct.pack(f"<{len(expert_ids)}i", *expert_ids)
    return base64.b64encode(raw).decode("ascii")


class TestRoutedExperts:
    """Tests for routed experts accumulation."""

    def test_empty_manager_returns_none(self):
        """No routing data returns None."""
        manager = TokenManager()
        assert manager.routed_experts is None

    def test_single_chunk(self):
        """Single chunk round-trips through base64 correctly."""
        manager = TokenManager()
        experts = [0, 1, 2, 3]
        chunk = _make_routed_experts_b64(experts)

        manager.add_routed_experts(chunk)

        result = manager.routed_experts
        assert result is not None
        decoded = struct.unpack(f"<{len(experts)}i", base64.b64decode(result))
        assert list(decoded) == experts

    def test_multi_turn_accumulation(self):
        """Multiple chunks are concatenated in order."""
        manager = TokenManager()

        # Turn 1: prompt + response routing (3 tokens, 1 layer, top_k=2)
        turn1_experts = [10, 20, 30, 40, 50, 60]  # 3 tokens * 1 layer * 2 experts
        manager.add_routed_experts(_make_routed_experts_b64(turn1_experts))

        # Turn 2: new tokens routing (2 tokens, 1 layer, top_k=2)
        turn2_experts = [70, 80, 90, 100]  # 2 tokens * 1 layer * 2 experts
        manager.add_routed_experts(_make_routed_experts_b64(turn2_experts))

        result = manager.routed_experts
        assert result is not None
        all_experts = turn1_experts + turn2_experts
        decoded = struct.unpack(f"<{len(all_experts)}i", base64.b64decode(result))
        assert list(decoded) == all_experts

    def test_reset_then_reuse(self):
        """Routing data can be accumulated after reset."""
        manager = TokenManager()
        manager.add_routed_experts(_make_routed_experts_b64([1, 2]))
        manager.reset()

        new_experts = [5, 6, 7]
        manager.add_routed_experts(_make_routed_experts_b64(new_experts))

        result = manager.routed_experts
        assert result is not None
        decoded = struct.unpack(f"<{len(new_experts)}i", base64.b64decode(result))
        assert list(decoded) == new_experts
