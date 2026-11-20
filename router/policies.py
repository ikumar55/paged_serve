"""Routing policies for a fleet of engine replicas (D-007).

Four policies, from least to most informed:

* :class:`RoundRobin`: take the replicas in turn.
* :class:`LeastOutstanding`: send to the replica with the fewest requests in flight.
* :class:`ApproximatePrefix`: remember which replica each chunk of recent prompts went to and send a
  new prompt to the replica that last saw the longest matching start of it. This is a guess: the
  replica may have evicted those blocks since, or restarted, and the router cannot know.
* :class:`PrecisePrefix`: score each replica by how many leading blocks of the prompt it reports
  holding (the replicas' ``GET /cache``), which is what its cache contains now.

Both prefix policies fall back to the least-loaded replica when the preferred one already has
``load_cap`` requests in flight: a cache hit is worth little if the request waits behind a long
queue, and sending every request that shares a popular prefix to one replica would leave the others
idle.

A policy only chooses; the router owns the in-flight counts and health, which it passes in as
:class:`ReplicaState`. Prompts are lists of token ids, so a "chunk" is a run of ``chunk_tokens``
token ids, and the chain hash of :mod:`engine.prefix_cache` is reused, which is what makes the
precise policy's scores comparable to a replica's real block hashes.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from engine.prefix_cache import block_hashes


@dataclass
class ReplicaState:
    """What the router knows about one replica."""

    url: str
    outstanding: int = 0  # requests sent and not yet finished
    served: int = 0  # requests ever sent
    healthy: bool = True
    cached: frozenset[int] = frozenset()  # block hashes the replica last reported (precise policy)
    block_size: int | None = None  # as last reported by the replica


DEFAULT_MAX_ENTRIES = 200_000  # prompt chunks the approximate policy remembers


class Policy(Protocol):
    name: str

    def route(
        self, prompt: Sequence[int], replicas: Sequence[ReplicaState], candidates: Sequence[int]
    ) -> int:
        """Pick one of ``candidates`` (indexes into ``replicas``, all healthy and non-empty)."""
        ...

    def stats(self) -> dict[str, int]: ...


def least_loaded(replicas: Sequence[ReplicaState], candidates: Sequence[int]) -> int:
    """Fewest requests in flight; ties go to the replica that has served the fewest, then lowest."""
    return min(candidates, key=lambda i: (replicas[i].outstanding, replicas[i].served, i))


class RoundRobin:
    name = "round-robin"

    def __init__(self) -> None:
        self._next = 0
        self._decisions = 0

    def route(
        self, prompt: Sequence[int], replicas: Sequence[ReplicaState], candidates: Sequence[int]
    ) -> int:
        # Take the first candidate at or after the pointer, so an unhealthy replica is skipped
        # without making its neighbour take double.
        ordered = sorted(candidates, key=lambda i: (i - self._next) % len(replicas))
        choice = ordered[0]
        self._next = (choice + 1) % len(replicas)
        self._decisions += 1
        return choice

    def stats(self) -> dict[str, int]:
        return {"decisions": self._decisions}


class LeastOutstanding:
    name = "least-outstanding"

    def __init__(self) -> None:
        self._decisions = 0

    def route(
        self, prompt: Sequence[int], replicas: Sequence[ReplicaState], candidates: Sequence[int]
    ) -> int:
        self._decisions += 1
        return least_loaded(replicas, candidates)

    def stats(self) -> dict[str, int]:
        return {"decisions": self._decisions}


@dataclass
class _PrefixStats:
    decisions: int = 0
    matched: int = 0  # went to the replica the policy believed held the prefix
    no_match: int = 0  # no replica was believed to hold any of it
    load_fallbacks: int = 0  # a replica was believed to hold it but was over the cap
    matched_chunks: int = 0  # chunks of prompt believed cached, summed over matched requests
    extra: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int]:
        return {
            "decisions": self.decisions,
            "matched": self.matched,
            "no_match": self.no_match,
            "load_fallbacks": self.load_fallbacks,
            "matched_chunks": self.matched_chunks,
            **self.extra,
        }


class ApproximatePrefix:
    """Remembers where each prompt chunk went; no tokenizer and no talking to the replicas."""

    name = "approx-prefix"

    def __init__(
        self,
        chunk_tokens: int = 16,
        load_cap: int = 16,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        if chunk_tokens <= 0 or load_cap <= 0 or max_entries <= 0:
            raise ValueError("chunk_tokens, load_cap and max_entries must be positive")
        self.chunk_tokens = chunk_tokens
        self.load_cap = load_cap
        self.max_entries = max_entries
        self._owner: OrderedDict[int, int] = OrderedDict()  # chunk hash -> replica, oldest first
        self._stats = _PrefixStats()

    def route(
        self, prompt: Sequence[int], replicas: Sequence[ReplicaState], candidates: Sequence[int]
    ) -> int:
        chain = block_hashes(prompt, self.chunk_tokens)
        preferred: int | None = None
        matched = 0
        for content_hash in chain:
            owner = self._owner.get(content_hash)
            if owner is None or owner not in candidates:
                break  # the chain is a prefix: one miss ends the match
            preferred, matched = owner, matched + 1

        stats = self._stats
        stats.decisions += 1
        if preferred is None:
            stats.no_match += 1
            choice = least_loaded(replicas, candidates)
        elif replicas[preferred].outstanding >= self.load_cap:
            stats.load_fallbacks += 1
            choice = least_loaded(replicas, candidates)
        else:
            stats.matched += 1
            stats.matched_chunks += matched
            choice = preferred
        self._remember(chain, choice)
        return choice

    def _remember(self, chain: Sequence[int], replica: int) -> None:
        for content_hash in chain:
            self._owner[content_hash] = replica
            self._owner.move_to_end(content_hash)
        while len(self._owner) > self.max_entries:
            self._owner.popitem(last=False)

    def stats(self) -> dict[str, int]:
        return {**self._stats.as_dict(), "entries": len(self._owner)}


class PrecisePrefix:
    """Scores replicas by the blocks they report holding; needs the replicas' ``/cache``."""

    name = "precise-prefix"

    def __init__(self, load_cap: int = 16) -> None:
        if load_cap <= 0:
            raise ValueError("load_cap must be positive")
        self.load_cap = load_cap
        self._stats = _PrefixStats()

    def route(
        self, prompt: Sequence[int], replicas: Sequence[ReplicaState], candidates: Sequence[int]
    ) -> int:
        chains: dict[int, list[int]] = {}  # one chain per block size in use
        scores: dict[int, int] = {}
        for index in candidates:
            size = replicas[index].block_size
            if size is None or not replicas[index].cached:
                scores[index] = 0
                continue
            chain = chains.setdefault(size, block_hashes(prompt, size))
            held = replicas[index].cached
            score = 0
            for content_hash in chain:
                if content_hash not in held:
                    break
                score += 1
            scores[index] = score

        stats = self._stats
        stats.decisions += 1
        best = max(scores.values(), default=0)
        if best == 0:
            stats.no_match += 1
            return least_loaded(replicas, candidates)
        holders = [i for i in candidates if scores[i] == best]
        under_cap = [i for i in holders if replicas[i].outstanding < self.load_cap]
        if not under_cap:
            stats.load_fallbacks += 1
            return least_loaded(replicas, candidates)
        stats.matched += 1
        stats.matched_chunks += best
        return least_loaded(replicas, under_cap)

    def stats(self) -> dict[str, int]:
        return self._stats.as_dict()


POLICIES = ("round-robin", "least-outstanding", "approx-prefix", "precise-prefix")


def make_policy(
    name: str,
    *,
    chunk_tokens: int = 16,
    load_cap: int = 16,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> Policy:
    """``max_entries`` is the approximate policy's memory (remembered chunks); others ignore it."""
    if name == "round-robin":
        return RoundRobin()
    if name == "least-outstanding":
        return LeastOutstanding()
    if name == "approx-prefix":
        return ApproximatePrefix(
            chunk_tokens=chunk_tokens, load_cap=load_cap, max_entries=max_entries
        )
    if name == "precise-prefix":
        return PrecisePrefix(load_cap=load_cap)
    raise ValueError(f"unknown policy {name!r}; choose from {', '.join(POLICIES)}")
