"""Iteration-level scheduler with admission by free blocks and preemption by recompute (D-004).

One call to :meth:`Scheduler.step` is one scheduling decision and one pass over the GPU:

1. make room for every running sequence's next token, preempting the most recently admitted
   sequences if the pool cannot hold them all;
2. decode one token for every running sequence in a single batched forward pass;
3. admit waiting requests that fit (oldest first), prefilling each and producing its first token.

Preemption by recompute frees all of a sequence's blocks and puts the request back at the front
of the waiting queue. Its generated tokens are kept, and when it is admitted again the prompt
plus those tokens are prefilled in one pass, so no swap path or CPU memory pool is needed.

Greedy decoding only. A request that could never fit in the pool alone is refused up front,
which guarantees the oldest running sequence can always make progress (no livelock).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor

from engine.config import LlamaConfig
from engine.kv_cache import KVCache
from engine.paged_attention import DecodeAttention
from engine.paged_cache import KVPool, PagedBatchCache, PagedSequenceCache
from engine.prefix_cache import match_prefix


class LanguageModel(Protocol):
    """What the scheduler needs from a model: the Llama forward pass or the timing-only mock."""

    @property
    def cfg(self) -> LlamaConfig: ...

    @property
    def device(self) -> torch.device: ...

    def __call__(
        self,
        input_ids: Tensor,
        positions: Tensor | None = None,
        cache: KVCache | None = None,
        last_only: bool = False,
    ) -> Tensor: ...


@dataclass
class Request:
    id: int
    prompt: list[int]
    max_new_tokens: int
    eos_ids: frozenset[int] = frozenset()
    generated: list[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: str | None = None  # "stop" (EOS), "length" or "abort"
    preemptions: int = 0

    @property
    def remaining(self) -> int:
        return self.max_new_tokens - len(self.generated)

    def tokens(self) -> list[int]:
        """Everything the model has to have seen to continue: the prompt and what it produced."""
        return self.prompt + self.generated


@dataclass
class StepResult:
    """What one step did. Token lists are in the order the tokens were produced."""

    new_tokens: dict[int, list[int]] = field(default_factory=dict)
    finished: list[int] = field(default_factory=list)
    finish_reasons: dict[int, str] = field(default_factory=dict)  # for the ids in ``finished``
    preempted: list[int] = field(default_factory=list)
    prefilled: list[int] = field(default_factory=list)
    decoded: int = 0  # sequences in the batched decode


class Scheduler:
    def __init__(
        self,
        model: LanguageModel,
        pool: KVPool,
        max_batch_size: int = 8,
        enable_prefix_caching: bool = True,
        decode_attention: DecodeAttention | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.model = model
        self.pool = pool
        # None: the decode step gathers blocks into a padded tensor (the default, D-003).
        # A function: it reads the blocks in place instead (D-025).
        self.decode_attention = decode_attention
        self.max_batch_size = max_batch_size
        self.prefix_caching = enable_prefix_caching
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # in admission order, oldest first
        self._caches: dict[int, PagedSequenceCache] = {}
        self._known: set[int] = set()
        self.total_preemptions = 0
        self.aborted = 0
        self.prefix_hit_tokens = 0  # prompt tokens served from cached blocks, not recomputed
        self.prefix_prompt_tokens = 0  # prompt tokens of every admission (hits plus computed)
        self.steps = 0
        self.decode_steps = 0
        self.decoded_sequences = 0

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def validate(self, prompt_len: int, max_new_tokens: int) -> None:
        """Raise ``ValueError`` if a request of this size can never be served.

        Only reads constants, so it is safe to call from another thread before submitting.
        """
        if prompt_len <= 0:
            raise ValueError("prompt must contain at least one token")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        total = prompt_len + max_new_tokens
        if total > self.model.cfg.max_position_embeddings:
            raise ValueError(
                f"{total} tokens exceed the model's maximum of "
                f"{self.model.cfg.max_position_embeddings}"
            )
        allocator = self.pool.allocator
        if allocator.blocks_for(total) > allocator.num_blocks:
            raise ValueError(
                f"a request of {total} tokens needs {allocator.blocks_for(total)} blocks but the "
                f"pool only has {allocator.num_blocks}"
            )

    def add_request(
        self,
        request_id: int,
        prompt: list[int],
        max_new_tokens: int,
        eos_token_id: int | Collection[int] | None = None,
    ) -> Request:
        self.validate(len(prompt), max_new_tokens)
        if request_id in self._known:
            raise ValueError(f"request id {request_id} is already in use")
        if eos_token_id is None:
            eos_ids: frozenset[int] = frozenset()
        elif isinstance(eos_token_id, int):
            eos_ids = frozenset({eos_token_id})
        else:
            eos_ids = frozenset(eos_token_id)
        request = Request(request_id, list(prompt), max_new_tokens, eos_ids)
        self._known.add(request_id)
        self.waiting.append(request)
        return request

    @torch.inference_mode()
    def step(self) -> StepResult:
        result = StepResult()
        self.steps += 1
        self._make_room_for_decode(result)
        self._decode(result)
        self._admit(result)
        return result

    def abort(self, request_id: int) -> bool:
        """Drop a request (for example when its client disconnected) and free its blocks.

        Returns False if it is unknown or already finished.
        """
        for request in self.running:
            if request.id == request_id:
                self.running.remove(request)
                self._caches.pop(request_id).release()
                self._finish(request, "abort")
                self.aborted += 1
                return True
        for request in self.waiting:
            if request.id == request_id:
                self.waiting.remove(request)
                self._finish(request, "abort")
                self.aborted += 1
                return True
        return False

    def stats(self) -> dict[str, int | float]:
        allocator = self.pool.allocator
        return {
            "steps": self.steps,
            "running": len(self.running),
            "waiting": len(self.waiting),
            "preemptions": self.total_preemptions,
            "aborted": self.aborted,
            "prefix_caching": int(self.prefix_caching),
            "prefix_hit_tokens": self.prefix_hit_tokens,
            "prefix_prompt_tokens": self.prefix_prompt_tokens,
            "prefix_hit_rate": self.prefix_hit_tokens / self.prefix_prompt_tokens
            if self.prefix_prompt_tokens
            else 0.0,
            "blocks_cached": allocator.num_cached,
            "block_evictions": allocator.evictions,
            "decode_steps": self.decode_steps,
            "mean_decode_batch": self.decoded_sequences / self.decode_steps
            if self.decode_steps
            else 0.0,
            "blocks_total": allocator.num_blocks,
            "blocks_used": allocator.num_used,
        }

    def run_to_completion(self) -> dict[int, list[int]]:
        """Step until nothing is left; returns generated tokens by request id (for tests)."""
        outputs: dict[int, list[int]] = {}
        requests: dict[int, Request] = {r.id: r for r in [*self.waiting, *self.running]}
        while self.has_work:
            self.step()
        for request_id, request in requests.items():
            outputs[request_id] = request.generated
        return outputs

    # -- internals -------------------------------------------------------------------------

    def _make_room_for_decode(self, result: StepResult) -> None:
        allocator = self.pool.allocator
        while self.running:
            needed = sum(
                self._caches[r.id].blocks_missing(self._caches[r.id].length + 1)
                for r in self.running
            )
            if allocator.can_allocate(needed):
                return
            # Newest first: it has produced the least, so recomputing it wastes the least, and
            # the oldest request keeps its place and is guaranteed to be able to finish.
            self._preempt(self.running[-1], result)

    def _preempt(self, request: Request, result: StepResult) -> None:
        self.running.remove(request)
        self._caches.pop(request.id).release()
        request.preemptions += 1
        self.total_preemptions += 1
        self.waiting.appendleft(request)
        result.preempted.append(request.id)

    def _decode(self, result: StepResult) -> None:
        if not self.running:
            return
        batch = self.running
        caches = [self._caches[r.id] for r in batch]
        step_cache = PagedBatchCache(caches, attention=self.decode_attention)
        last_tokens = torch.tensor(
            [[r.tokens()[-1]] for r in batch], dtype=torch.long, device=self.model.device
        )
        logits = self.model(last_tokens, positions=step_cache.positions(), cache=step_cache)
        next_tokens = logits[:, -1].argmax(dim=-1).tolist()
        result.decoded = len(batch)
        self.decode_steps += 1
        self.decoded_sequences += len(batch)
        for request, token in zip(list(batch), next_tokens, strict=True):
            self._accept(request, int(token), result)

    def _admit(self, result: StepResult) -> None:
        if result.preempted:
            return  # memory was just short; admitting now would only evict again next step
        allocator = self.pool.allocator
        while self.waiting and len(self.running) < self.max_batch_size:
            request = self.waiting[0]
            tokens = request.tokens()
            cache = PagedSequenceCache(self.pool)
            if self.prefix_caching:
                # Leave at least one token to compute: the model needs a pass to produce the next.
                blocks, hashes = match_prefix(allocator, tokens, max_tokens=len(tokens) - 1)
                cache.adopt(blocks, hashes)
            # Room for the rest of the prompt (anything recomputed included) and the slot of the
            # first new token. Blocks taken from the cache are already counted as held.
            if not allocator.can_allocate(cache.blocks_missing(len(tokens) + 1)):
                cache.release()
                return
            self.waiting.popleft()
            self._caches[request.id] = cache
            self.prefix_hit_tokens += cache.length
            self.prefix_prompt_tokens += len(tokens)
            ids = torch.tensor([tokens[cache.length :]], dtype=torch.long, device=self.model.device)
            logits = self.model(ids, cache=cache, last_only=True)
            self.running.append(request)
            result.prefilled.append(request.id)
            self._accept(request, int(logits[0, -1].argmax()), result)

    def _accept(self, request: Request, token: int, result: StepResult) -> None:
        request.generated.append(token)
        result.new_tokens.setdefault(request.id, []).append(token)
        if self.prefix_caching:  # before a finished request lets go of its blocks
            self._caches[request.id].register_full_blocks(request.tokens())
        if token in request.eos_ids or request.remaining == 0:
            self.running.remove(request)
            self._caches.pop(request.id).release()
            self._finish(request, "stop" if token in request.eos_ids else "length")
            result.finished.append(request.id)
            result.finish_reasons[request.id] = str(request.finish_reason)

    @staticmethod
    def _finish(request: Request, reason: str) -> None:
        request.finished = True
        request.finish_reason = reason
