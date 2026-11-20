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
from dataclasses import dataclass, field

import torch

from engine.model import Llama
from engine.paged_cache import KVPool, PagedBatchCache, PagedSequenceCache


@dataclass
class Request:
    id: int
    prompt: list[int]
    max_new_tokens: int
    eos_token_id: int | None = None
    generated: list[int] = field(default_factory=list)
    finished: bool = False
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
    preempted: list[int] = field(default_factory=list)
    prefilled: list[int] = field(default_factory=list)
    decoded: int = 0  # sequences in the batched decode


class Scheduler:
    def __init__(self, model: Llama, pool: KVPool, max_batch_size: int = 8) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.model = model
        self.pool = pool
        self.max_batch_size = max_batch_size
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # in admission order, oldest first
        self._caches: dict[int, PagedSequenceCache] = {}
        self._known: set[int] = set()
        self.total_preemptions = 0

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def add_request(
        self,
        request_id: int,
        prompt: list[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> Request:
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if request_id in self._known:
            raise ValueError(f"request id {request_id} is already in use")
        total = len(prompt) + max_new_tokens
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
        request = Request(request_id, list(prompt), max_new_tokens, eos_token_id)
        self._known.add(request_id)
        self.waiting.append(request)
        return request

    @torch.inference_mode()
    def step(self) -> StepResult:
        result = StepResult()
        self._make_room_for_decode(result)
        self._decode(result)
        self._admit(result)
        return result

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
        step_cache = PagedBatchCache(caches)
        last_tokens = torch.tensor(
            [[r.tokens()[-1]] for r in batch], dtype=torch.long, device=self.model.device
        )
        logits = self.model(last_tokens, positions=step_cache.positions(), cache=step_cache)
        next_tokens = logits[:, -1].argmax(dim=-1).tolist()
        result.decoded = len(batch)
        for request, token in zip(list(batch), next_tokens, strict=True):
            self._accept(request, int(token), result)

    def _admit(self, result: StepResult) -> None:
        if result.preempted:
            return  # memory was just short; admitting now would only evict again next step
        allocator = self.pool.allocator
        while self.waiting and len(self.running) < self.max_batch_size:
            request = self.waiting[0]
            tokens = request.tokens()
            # Room for the prompt (plus anything recomputed) and the first new token's slot.
            if not allocator.can_allocate(allocator.blocks_for(len(tokens) + 1)):
                return
            self.waiting.popleft()
            cache = PagedSequenceCache(self.pool)
            self._caches[request.id] = cache
            ids = torch.tensor([tokens], dtype=torch.long, device=self.model.device)
            logits = self.model(ids, cache=cache, last_only=True)
            self.running.append(request)
            result.prefilled.append(request.id)
            self._accept(request, int(logits[0, -1].argmax()), result)

    def _accept(self, request: Request, token: int, result: StepResult) -> None:
        request.generated.append(token)
        result.new_tokens.setdefault(request.id, []).append(token)
        if token == request.eos_token_id or request.remaining == 0:
            request.finished = True
            self.running.remove(request)
            self._caches.pop(request.id).release()
            result.finished.append(request.id)
