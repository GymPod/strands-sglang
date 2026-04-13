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
- TokenTrajectory: A grouped token trajectory after any context rewrites
- TokenManager: Manages segment-based prompt/response tracking and trajectory boundaries

For RL training, you typically want:
- token_ids: Flat list of all tokens for the rollout
- loss_mask: Integer mask for loss computation (1 = model output, 0 = prompt/tool)
- logprobs: Log probabilities for policy gradient
- routed_experts: Raw bytes of MoE routing decisions for routing replay
- trajectories: Grouped token trajectories split whenever managed context stops
  extending the previous active prompt+response
"""

from __future__ import annotations

import base64
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Token:
    """A single token with its ID, logprob, and loss mask."""

    token_id: int
    logprob: float | None = None
    loss_mask: bool = True


@dataclass(frozen=True, slots=True)
class TokenTrajectory:
    """A grouped token trajectory tracked inside ``TokenManager``.

    Attributes:
        trajectory_id: Zero-based trajectory index within the rollout.
        token_ids: Flattened token IDs for this trajectory.
        logprobs: Flattened per-token log probabilities (prompt tokens may be ``None``).
        loss_mask: Integer mask aligned to ``token_ids``.
        segment_info: Segment metadata as ``(is_output, length)`` tuples.
        token_offset: Global starting token offset within the full rollout token stream.
    """

    trajectory_id: int
    token_ids: list[int]
    logprobs: list[float | None]
    loss_mask: list[int]
    segment_info: list[tuple[bool, int]]
    token_offset: int


class TokenManager:
    """Manages token accumulation with segment-based prompt/response tracking.

    Notes:
        - Tokens are organized into ``segments``, where each segment is either:
            - ``PROMPT``: System messages, user input, tool results (loss_mask=False)
            - ``RESPONSE``: Model outputs (loss_mask=True)
        - During an agent loop with the ``SGLangModel`` backend, segments are added in this order:
            - ``segments[0]``: ``PROMPT``   — initial prompt
            - ``segments[1]``: ``RESPONSE`` — first model output
            - ``segments[2]``: ``PROMPT``   — tool results or managed prompt delta
            - ``segments[3]``: ``RESPONSE`` — next model output
            - ... alternating ``PROMPT``/``RESPONSE`` until the loop ends
        - ``start_new_trajectory()`` marks that the next segment begins a new
          training trajectory because prompt context was rewritten rather than
          extended additively.
    """

    def __init__(self) -> None:
        """Create a TokenManager."""
        self._segments: list[list[Token]] = []
        self._routed_experts_bytes: bytes | None = None
        self._trajectory_start_segment_indices: list[int] = []
        self._turn_trajectory_ids: list[int] = []

    def reset(self) -> None:
        """Reset token accumulation for a new episode."""
        self._segments = []
        self._routed_experts_bytes = None
        self._trajectory_start_segment_indices = []
        self._turn_trajectory_ids = []

    def start_new_trajectory(self) -> None:
        """Mark the next segment as the start of a new training trajectory.

        This should be called when the visible prompt sent to the model no
        longer prefix-extends the previous active prompt+response context.
        """
        if not self._segments:
            return
        next_start = len(self._segments)
        if self._trajectory_start_segment_indices and self._trajectory_start_segment_indices[-1] == next_start:
            return
        self._trajectory_start_segment_indices.append(next_start)

    @property
    def current_trajectory_id(self) -> int:
        """Get the trajectory ID that new segments belong to."""
        return len(self._trajectory_start_segment_indices)

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
        self._turn_trajectory_ids.append(self.current_trajectory_id)

    def add_routed_experts(self, data: str) -> None:
        """Store routed experts from an SGLang response.

        Decodes the base64 wire format and stores raw bytes. SGLang returns
        routed experts as a base64-encoded string of flattened int32 values
        with shape ``[num_tokens, num_layers, top_k]``. Each response covers
        ALL tokens in the sequence, so the latest call always overwrites the
        stored routing view.

        Args:
            data: Base64-encoded routed experts string from
                ``meta_info["routed_experts"]``.
        """
        self._routed_experts_bytes = base64.b64decode(data)

    @property
    def routed_experts(self) -> bytes | None:
        """Get routed experts as raw bytes."""
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
        """Get loss mask for all tokens (1 = model output, 0 = prompt/tool)."""
        return [int(token.loss_mask) for token in self.tokens]

    @property
    def logprobs(self) -> list[float | None]:
        """Get log probabilities for all tokens."""
        return [token.logprob for token in self.tokens]

    @property
    def initial_prompt(self) -> list[Token]:
        """Get the initial prompt tokens (``segments[0]``)."""
        return self._segments[0] if self._segments else []

    @property
    def segments(self) -> list[list[Token]]:
        """Get tokens organized by segment."""
        return self._segments

    @property
    def segment_info(self) -> list[tuple[bool, int]]:
        """Get segment metadata as ``(is_output, length)`` tuples."""
        return [(seg[0].loss_mask if seg else False, len(seg)) for seg in self._segments]

    @property
    def trajectory_start_segment_indices(self) -> list[int]:
        """Get segment indices where each grouped trajectory begins."""
        if not self._segments:
            return []
        return [0, *self._trajectory_start_segment_indices]

    @property
    def turn_trajectory_ids(self) -> list[int]:
        """Get the grouped trajectory ID for each response/model turn."""
        return list(self._turn_trajectory_ids)

    @property
    def trajectories(self) -> list[TokenTrajectory]:
        """Get grouped token trajectories split by managed-context rewrites."""
        if not self._segments:
            return []

        starts = self.trajectory_start_segment_indices
        ends = starts[1:] + [len(self._segments)]
        trajectories: list[TokenTrajectory] = []
        token_offset = 0

        for trajectory_id, (segment_start, segment_end) in enumerate(zip(starts, ends, strict=True)):
            segment_slice = self._segments[segment_start:segment_end]
            tokens = [token for segment in segment_slice for token in segment]
            trajectories.append(
                TokenTrajectory(
                    trajectory_id=trajectory_id,
                    token_ids=[token.token_id for token in tokens],
                    logprobs=[token.logprob for token in tokens],
                    loss_mask=[int(token.loss_mask) for token in tokens],
                    segment_info=[(segment[0].loss_mask if segment else False, len(segment)) for segment in segment_slice],
                    token_offset=token_offset,
                )
            )
            token_offset += len(tokens)
        return trajectories

    def __len__(self) -> int:
        """Return total number of tokens."""
        return sum(len(seg) for seg in self._segments)

    def __repr__(self) -> str:
        """Return string representation."""
        n_segments = len(self._segments)
        n_tokens = len(self)
        n_output = sum(1 for token in self.tokens if token.loss_mask)
        return f"TokenManager(segments={n_segments}, tokens={n_tokens}, output_tokens={n_output})"
