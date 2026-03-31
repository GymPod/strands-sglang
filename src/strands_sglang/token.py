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

"""Token management for TITO (Token-In/Token-Out) training.

This module provides:
- Token: A single token with ID, logprob, and loss mask
- TokenManager: Manages segment-based token accumulation

For RL training, you typically want:
- token_ids: Flat list of all tokens for the trajectory
- loss_mask: Integer mask for loss computation (1 = model output, 0 = prompt/tool)
- logprobs: Log probabilities for policy gradient
- routed_experts: Raw bytes of MoE routing decisions for routing replay
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Token:
    """A single token with its ID, logprob, and loss mask."""

    token_id: int
    logprob: float | None = None
    loss_mask: bool = True


class TokenManager:
    """Manages token accumulation with segment-based prompt/response tracking.

    Notes:
        - Tokens are organized into `segments`, where each segment is either:
            - `PROMPT`: System messages, user input, tool results (loss_mask=False)
            - `RESPONSE`: Model outputs (loss_mask=True)
        - During an agent loop with the `SGLangModel` backend, segments are added in this order:
            - `segments[0]`: `PROMPT`   — initial prompt (system + tools + user message / conversation history)
            - `segments[1]`: `RESPONSE` — first model output (may include tool calls)
            - `segments[2]`: `PROMPT`   — tool results (if tool use occurred)
            - `segments[3]`: `RESPONSE` — next model output
            - ...                   — alternating `PROMPT`/`RESPONSE` until the loop ends
        - `segments[0]` always contains the full initial prompt from the first generation call. Everything after it is the rollout.

    Example:
        >>> manager = TokenManager()
        >>> manager.add_prompt([1, 2, 3])
        >>> manager.add_response([4, 5], [0.1, 0.2])
        >>> manager.token_ids      # [1, 2, 3, 4, 5]
        >>> manager.loss_mask      # [0, 0, 0, 1, 1]
        >>> manager.logprobs       # [None, None, None, 0.1, 0.2]
    """

    def __init__(self) -> None:
        """Create a TokenManager."""
        self._segments: list[list[Token]] = []
        self._routed_experts_bytes: bytes | None = None

    def reset(self) -> None:
        """Reset token accumulation for a new episode."""
        self._segments = []
        self._routed_experts_bytes = None

    def add_prompt(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        """Add a prompt segment (system messages, user input, tool results)."""
        if not token_ids:
            return
        if logprobs is not None and len(logprobs) != len(token_ids):
            raise ValueError(f"logprobs length ({len(logprobs)}) must match token_ids length ({len(token_ids)})")

        tokens = [
            Token(
                token_id=tid,
                logprob=logprobs[i] if logprobs is not None else None,
                loss_mask=False,
            )
            for i, tid in enumerate(token_ids)
        ]
        self._segments.append(tokens)

    def add_response(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        """Add a response segment (model output)."""
        if not token_ids:
            return
        if not self._segments:
            raise RuntimeError("First segment must be a prompt. Call add_prompt() before add_response().")
        if logprobs is not None and len(logprobs) != len(token_ids):
            raise ValueError(f"logprobs length ({len(logprobs)}) must match token_ids length ({len(token_ids)})")

        tokens = [
            Token(
                token_id=tid,
                logprob=logprobs[i] if logprobs is not None else None,
                loss_mask=True,
            )
            for i, tid in enumerate(token_ids)
        ]
        self._segments.append(tokens)

    def add_routed_experts(self, data: str) -> None:
        """Store routed experts from an SGLang response.

        Decodes the base64 wire format and stores raw bytes. SGLang returns
        routed experts as a base64-encoded string of flattened int32 values
        with shape ``[num_tokens, num_layers, top_k]``.  Each response covers
        ALL tokens in the sequence (not just new tokens), because SGLang does
        not support ``routed_experts_start_len``. Therefore we overwrite rather
        than accumulate — the latest call always has the complete routing.

        **Design tradeoffs (overwrite vs accumulate)**:

        Current approach is *overwrite*: each call replaces all stored routing.
        For single-turn rollouts, this is trivially correct. For multi-turn,
        SGLang re-computes routing for the full sequence each turn. Within a
        single episode (no weight update between turns), the MoE router is
        deterministic — same weights + same tokens = same routing — so
        overwriting produces identical results to accumulating.

        If model weights update between turns (async RL), the re-computed
        prefix routing reflects the new weights rather than the weights that
        originally generated those tokens. An *accumulate* approach (keep
        prior turns' routing, append only new tokens' portion) would preserve
        per-token routing-logprob correspondence. This can be done client-side
        by slicing at ``len(existing_bytes)`` without SGLang changes.
        Overwrite is simpler and sufficient for the common case; accumulate
        is a future optimization for long multi-turn async rollouts.

        Args:
            data: Base64-encoded routed experts string from
                ``meta_info["routed_experts"]``.
        """
        import base64

        self._routed_experts_bytes = base64.b64decode(data)

    @property
    def routed_experts(self) -> bytes | None:
        """Get routed experts as raw bytes.

        Returns the decoded routing data from the most recent SGLang response,
        which covers all ``seqlen - 1`` tokens in the trajectory. The bytes
        contain flattened int32 expert IDs with logical shape
        ``[total_tokens - 1, num_layers, top_k]``.

        Returns:
            Raw bytes of int32 expert IDs, or ``None`` if no routing data
            has been recorded.
        """
        return self._routed_experts_bytes

    @property
    def tokens(self) -> list[Token]:
        """Get all tokens as a flat list."""
        return [token for segment in self._segments for token in segment]

    @property
    def token_ids(self) -> list[int]:
        """Get all token IDs as a flat list."""
        return [token.token_id for token in self.tokens]

    @property
    def loss_mask(self) -> list[int]:
        """Get loss mask for all tokens (1 = model output, 0 = prompt/tool).

        Notes:
            Only compute loss on tokens where mask is 1 (model outputs).
        """
        return [int(token.loss_mask) for token in self.tokens]

    @property
    def logprobs(self) -> list[float | None]:
        """Get log probabilities for all tokens."""
        return [token.logprob for token in self.tokens]

    @property
    def initial_prompt(self) -> list[Token]:
        """Get the initial prompt tokens (`segments[0]`).

        Notes:
            Contains the full input context from the first generation call:
            system prompt + tool definitions + user message (or conversation history).
        """
        return self._segments[0] if self._segments else []

    @property
    def segments(self) -> list[list[Token]]:
        """Get tokens organized by segment."""
        return self._segments

    @property
    def segment_info(self) -> list[tuple[bool, int]]:
        """Get segment metadata as `(is_output, length)` tuples."""
        return [(seg[0].loss_mask if seg else False, len(seg)) for seg in self._segments]

    def __len__(self) -> int:
        """Return total number of tokens."""
        return sum(len(seg) for seg in self._segments)

    def __repr__(self) -> str:
        """Return string representation."""
        n_segments = len(self._segments)
        n_tokens = len(self)
        n_output = sum(1 for token in self.tokens if token.loss_mask)
        return f"TokenManager(segments={n_segments}, tokens={n_tokens}, output_tokens={n_output})"
