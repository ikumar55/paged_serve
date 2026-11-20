"""Routing policies (D-007), as pure logic: no servers, no network."""

import pytest

from engine.prefix_cache import block_hashes
from router.policies import (
    DEFAULT_MAX_ENTRIES,
    POLICIES,
    ApproximatePrefix,
    LeastOutstanding,
    PrecisePrefix,
    ReplicaState,
    RoundRobin,
    make_policy,
)

CHUNK = 4


def fleet(count: int = 3, **overrides: object) -> list[ReplicaState]:
    return [ReplicaState(url=f"http://r{i}", **overrides) for i in range(count)]  # type: ignore[arg-type]


def prompt(*chunks: int, tail: int = 2) -> list[int]:
    """A prompt made of whole chunks; chunk ``n`` is the four tokens ``n n n n``."""
    return [token for chunk in chunks for token in [chunk] * CHUNK] + [900] * tail


def route(policy: object, tokens: list[int], replicas: list[ReplicaState]) -> int:
    candidates = [i for i, r in enumerate(replicas) if r.healthy]
    return policy.route(tokens, replicas, candidates)  # type: ignore[attr-defined, no-any-return]


# -- round robin and least outstanding ---------------------------------------------------------


def test_round_robin_takes_each_replica_in_turn() -> None:
    policy, replicas = RoundRobin(), fleet(3)
    assert [route(policy, [1], replicas) for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]


def test_round_robin_skips_an_unhealthy_replica_without_doubling_its_neighbour() -> None:
    policy, replicas = RoundRobin(), fleet(3)
    replicas[1].healthy = False
    assert [route(policy, [1], replicas) for _ in range(6)] == [0, 2, 0, 2, 0, 2]
    replicas[1].healthy = True
    assert [route(policy, [1], replicas) for _ in range(3)] == [0, 1, 2]  # back in turn


def test_least_outstanding_picks_the_idlest_and_spreads_ties_by_requests_served() -> None:
    policy, replicas = LeastOutstanding(), fleet(3)
    replicas[0].outstanding, replicas[1].outstanding, replicas[2].outstanding = 4, 1, 3
    assert route(policy, [1], replicas) == 1
    for replica in replicas:
        replica.outstanding = 0
    replicas[0].served, replicas[1].served, replicas[2].served = 5, 2, 2
    assert route(policy, [1], replicas) == 1  # tied at zero in flight: fewest served, then lowest


def test_only_healthy_candidates_are_considered() -> None:
    policy, replicas = LeastOutstanding(), fleet(2)
    replicas[0].healthy = False
    replicas[1].outstanding = 50
    assert route(policy, [1], replicas) == 1


# -- approximate prefix ------------------------------------------------------------------------


def test_a_repeated_prefix_goes_back_to_the_replica_that_saw_it() -> None:
    policy, replicas = ApproximatePrefix(chunk_tokens=CHUNK), fleet(3)
    first = route(policy, prompt(1, 2, 3), replicas)
    replicas[first].outstanding += 1  # busy, but under the cap
    again = route(policy, prompt(1, 2, 3, tail=3), replicas)
    assert again == first
    stats = policy.stats()
    assert stats["decisions"] == 2 and stats["matched"] == 1 and stats["no_match"] == 1
    assert stats["matched_chunks"] == 3


def test_a_miss_ends_the_match_so_a_repeated_chunk_elsewhere_does_not_count() -> None:
    policy, replicas = ApproximatePrefix(chunk_tokens=CHUNK), fleet(2)
    replicas[0].outstanding = 0
    replicas[1].outstanding = 5
    owner = route(policy, prompt(1, 2), replicas)  # remembered: [1] and [1, 2]
    assert owner == 0
    replicas[0].outstanding = 9  # the owner is now busier than the other replica
    # Starts with chunk 1 (owned) then differs: partial match, still prefers the owner.
    assert route(policy, prompt(1, 7), replicas) == 0
    # Starts with a chunk nobody owns: nothing matches even though chunk 2 is known further on.
    before = policy.stats()["no_match"]
    assert route(policy, prompt(7, 2), replicas) == 1  # least loaded
    assert policy.stats()["no_match"] == before + 1


def test_over_the_load_cap_it_falls_back_to_the_least_loaded_replica() -> None:
    policy, replicas = ApproximatePrefix(chunk_tokens=CHUNK, load_cap=3), fleet(2)
    owner = route(policy, prompt(1, 2), replicas)
    other = 1 - owner
    replicas[owner].outstanding = 3  # at the cap
    assert route(policy, prompt(1, 2), replicas) == other
    stats = policy.stats()
    assert stats["load_fallbacks"] == 1 and stats["matched"] == 0
    # The fallback replica will now compute and cache that prefix, so it is remembered as owner.
    replicas[other].outstanding = 0
    replicas[owner].outstanding = 0
    assert route(policy, prompt(1, 2), replicas) == other


def test_an_unhealthy_owner_is_not_chosen() -> None:
    policy, replicas = ApproximatePrefix(chunk_tokens=CHUNK), fleet(2)
    owner = route(policy, prompt(1, 2), replicas)
    replicas[owner].healthy = False
    assert route(policy, prompt(1, 2), replicas) == 1 - owner


def test_memory_of_past_routing_is_bounded_and_forgets_the_oldest_first() -> None:
    policy, replicas = ApproximatePrefix(chunk_tokens=CHUNK, max_entries=3), fleet(2)
    route(policy, prompt(1, 2), replicas)  # two entries
    route(policy, prompt(3, 4), replicas)  # four would be needed: the oldest go
    assert policy.stats()["entries"] == 3
    # Chunk 1 (the oldest) is gone, so the first prompt no longer matches.
    before = policy.stats()["matched"]
    route(policy, prompt(1, 2), replicas)
    assert policy.stats()["matched"] == before


def test_prompts_shorter_than_a_chunk_have_nothing_to_match() -> None:
    policy, replicas = ApproximatePrefix(chunk_tokens=CHUNK), fleet(2)
    for _ in range(4):
        route(policy, [5, 6], replicas)
    assert policy.stats()["matched"] == 0 and policy.stats()["entries"] == 0


# -- precise prefix ----------------------------------------------------------------------------


def holding(replica: ReplicaState, *prompts: list[int]) -> None:
    held = set(replica.cached)
    for tokens in prompts:
        held.update(block_hashes(tokens, CHUNK))
    replica.cached, replica.block_size = frozenset(held), CHUNK


def test_precise_routes_to_the_replica_that_reports_the_longest_prefix() -> None:
    policy, replicas = PrecisePrefix(), fleet(3)
    holding(replicas[0], prompt(1, 2))
    holding(replicas[2], prompt(1, 2, 3, 4))
    assert route(policy, prompt(1, 2, 3, 4, 5), replicas) == 2  # four chunks beats two
    assert route(policy, prompt(1, 2, 9), replicas) in (0, 2)  # both hold two chunks
    assert policy.stats()["matched"] == 2


def test_precise_matches_leading_blocks_only() -> None:
    policy, replicas = PrecisePrefix(), fleet(2)
    wanted = prompt(1, 2, 3)
    # Replica 1 holds the second and third blocks of this very prompt but has evicted the first,
    # so none of them can be reused: a block is only valid after the blocks before it.
    replicas[1].cached = frozenset(block_hashes(wanted, CHUNK)[1:])
    replicas[1].block_size = CHUNK
    replicas[0].outstanding = 3
    assert route(policy, wanted, replicas) == 1  # no match, so least loaded (replica 1 is idle)
    assert policy.stats()["no_match"] == 1 and policy.stats()["matched"] == 0


def test_precise_ties_go_to_the_less_loaded_holder() -> None:
    policy, replicas = PrecisePrefix(), fleet(3)
    for index in (0, 1):
        holding(replicas[index], prompt(1, 2))
    replicas[0].outstanding = 5
    assert route(policy, prompt(1, 2), replicas) == 1


def test_precise_over_the_cap_falls_back_even_to_a_replica_without_the_prefix() -> None:
    policy, replicas = PrecisePrefix(load_cap=4), fleet(2)
    holding(replicas[0], prompt(1, 2))
    replicas[0].outstanding = 4
    assert route(policy, prompt(1, 2), replicas) == 1
    assert policy.stats()["load_fallbacks"] == 1
    replicas[0].outstanding = 3
    assert route(policy, prompt(1, 2), replicas) == 0


def test_precise_knows_nothing_until_replicas_have_reported() -> None:
    policy, replicas = PrecisePrefix(), fleet(2)
    replicas[1].outstanding = 2
    assert route(policy, prompt(1, 2), replicas) == 0  # least loaded
    assert policy.stats()["no_match"] == 1


def test_precise_skips_an_unhealthy_holder() -> None:
    policy, replicas = PrecisePrefix(), fleet(2)
    holding(replicas[0], prompt(1, 2))
    replicas[0].healthy = False
    assert route(policy, prompt(1, 2), replicas) == 1


# -- construction ------------------------------------------------------------------------------


def test_make_policy_builds_every_named_policy_and_rejects_others() -> None:
    assert [make_policy(name).name for name in POLICIES] == list(POLICIES)
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("random")
    with pytest.raises(ValueError):
        ApproximatePrefix(chunk_tokens=0)
    with pytest.raises(ValueError):
        PrecisePrefix(load_cap=0)


def test_make_policy_passes_the_approximate_policys_memory_limit_through() -> None:
    policy = make_policy("approx-prefix", chunk_tokens=CHUNK, max_entries=5)
    assert isinstance(policy, ApproximatePrefix) and policy.max_entries == 5
    replicas = fleet(2)
    for group in range(10):  # ten prompts of three chunks each would be 30 entries unbounded
        route(policy, prompt(group, group + 100, group + 200), replicas)
    assert policy.stats()["entries"] == 5
    default = make_policy("approx-prefix")
    assert isinstance(default, ApproximatePrefix) and default.max_entries == DEFAULT_MAX_ENTRIES
    make_policy("round-robin", max_entries=5)  # other policies have no such memory and ignore it
