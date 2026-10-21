import random
from dataclasses import replace
from itertools import chain

import pytest

ta = pytest.importorskip("textarena")

import autoenvscaling.envs  # noqa: F401  (registers the TextArena ids)
from autoenvscaling.envs.unstable_unicorns import (
    Game,
    GameOver,
    UnstableUnicornsEnv,
    parse_choice,
)
from autoenvscaling.envs.unstable_unicorns.cards import BABY, DECK, NURSERY_SIZE

CARDS = {c.name: c for _, c in DECK} | {BABY.name: BABY}
TOTAL = NURSERY_SIZE + sum(count for count, _ in DECK)


def cards(*names):
    return [replace(CARDS[name]) for name in names]


def start(hands, stables=None, **kwargs):
    """A game with the given hands and Stables over a deck of Basic Unicorns; its first decision."""
    g = Game(len(hands), random.Random(0), **kwargs)
    g.hands = [cards(*h) for h in hands]
    g.stables = [cards(*s) for s in stables] if stables else g.stables
    g.deck = cards(*["Basic Unicorn"] * 20)
    run = g.play()
    return g, run, next(run)


def pick(run, decision, label):
    return run.send(decision.options.index(label))


def names(pile):
    return sorted(c.name for c in pile)


@pytest.mark.parametrize("neighs", [3, 2])
def test_neigh_chain_alternates(neighs):
    g, run, d = start([["Unicorn Poison", "Neigh"], ["Neigh", "Neigh"]],
                      [["Baby Unicorn"], ["Baby Unicorn", "Basic Unicorn"]])
    d = pick(run, d, "play Unicorn Poison")
    for i in range(neighs):
        assert d.player == (i + 1) % 2 and d.options == ("play Neigh", "pass")
        d = pick(run, d, "play Neigh")
    stopped = neighs % 2 == 1
    if not stopped:
        d = pick(run, pick(run, d, "pass"), "Basic Unicorn (Player 1)")
    assert d.player == 1 and d.prompt.startswith("Action phase")
    assert g.count(1) == 1 + stopped
    assert names(g.discard_pile) == sorted(["Neigh"] * neighs + ["Unicorn Poison"]
                                           + ["Basic Unicorn"] * (not stopped))


def test_super_neigh_and_yay_cannot_be_neighed():
    g, run, d = start([["Basic Unicorn", "Neigh"], ["Super Neigh"]])
    d = pick(run, pick(run, d, "play Basic Unicorn"), "play Super Neigh")
    assert d.player == 1 and d.prompt.startswith("Action phase")  # Player 0's Neigh is not asked
    g, run, d = start([["Basic Unicorn"], ["Neigh"]], [["Yay"], ["Baby Unicorn"]])
    d = pick(run, d, "play Basic Unicorn")
    assert d.player == 1 and g.count(0) == 1


def test_destroy_respects_rainbow_aura_and_kittencorn():
    g, run, d = start([["Unicorn Poison"], []], [["Baby Unicorn"], ["Rainbow Aura", "Basic Unicorn"]])
    assert pick(run, d, "play Unicorn Poison").player == 1 and g.count(1) == 1
    g, run, d = start([["Unicorn Poison"], []],
                      [["Rhinocorn"], ["Magical Kittencorn", "Basic Unicorn"]])
    assert d.options == ("Magical Kittencorn (Player 1)", "Basic Unicorn (Player 1)", "skip")
    d = pick(run, pick(run, d, "skip"), "play Unicorn Poison")  # only the Basic Unicorn is hit
    assert d.player == 1 and names(g.stables[1]) == ["Magical Kittencorn"]


def test_leaving_a_stable():
    g, run, d = start([["Two-For-One"], []],
                      [["Baby Unicorn", "Stabby the Unicorn"], ["Baby Unicorn", "Greedy Flying Unicorn"]])
    d = pick(run, pick(run, d, "play Two-For-One"), "Stabby the Unicorn (Player 0)")
    # Stabby destroys on leaving; then Two-For-One's first destroy has one target left, taken
    # without asking, and its second has none.
    d = pick(run, d, "Greedy Flying Unicorn (Player 1)")
    assert d.player == 1 and d.prompt.startswith("Action phase")
    assert names(g.hands[1]) == ["Basic Unicorn", "Greedy Flying Unicorn"]
    assert len(g.nursery) == NURSERY_SIZE - 1 and g.stables[1] == []


def test_reaching_the_goal_wins():
    g, run, d = start([["Ginormous Unicorn"], []], goal=3)
    with pytest.raises(GameOver) as over:
        pick(run, d, "play Ginormous Unicorn")
    assert over.value.winners == [0] and g.count(0) == 3


def test_parse_choice():
    options = ("draw a card", "play Neigh", "pass")
    assert parse_choice("I will take [1]", options) == 1
    assert parse_choice("[0] or [2]? [2]", options) == 2
    assert parse_choice("[PASS]", options) == 2
    assert parse_choice("[3]", options) is None and parse_choice("pass", options) is None


def _zones(g):
    return [*g.deck, *g.discard_pile, *g.nursery, *chain(*g.hands, *g.stables)]


@pytest.mark.parametrize("num_players", range(2, 9))
def test_random_self_play_terminates_and_conserves_cards(num_players):
    for seed in range(15):
        env = ta.make("UnstableUnicorns-v0-train")
        env.reset(num_players=num_players, seed=seed)
        rng, done = random.Random(seed), False
        while not done:
            g, d = env.game, env.decision
            player, observation = env.get_observation()
            assert player == d.player and observation.endswith(f"[{len(d.options) - 1}] "
                                                               f"{d.options[-1]}")
            if d.prompt.startswith("Action phase"):
                zones = _zones(g)
                assert len(zones) == len({id(c) for c in zones}) == TOTAL
            done, _ = env.step(f"[{rng.randrange(len(d.options))}]")
        rewards, _ = env.close()
        winners = [p for p, r in rewards.items() if r == 1]
        assert sorted(rewards) == list(range(num_players)) and set(rewards.values()) <= {-1, 0, 1}
        assert winners == env.game.leaders() or set(rewards.values()) == {0}


def test_seeded_games_replay_identically():
    def transcript(seed):
        env = UnstableUnicornsEnv()
        env.reset(num_players=3, seed=seed)
        done = False
        while not done:
            done, _ = env.step(f"[{len(env.decision.options) - 1}]")
        return env.state.logs, env.close()[0]

    assert transcript(7) == transcript(7) != transcript(8)


def test_invalid_moves_end_the_game_for_the_offender():
    env = ta.make("UnstableUnicorns-v0", error_allowance=1)
    env.reset(num_players=3, seed=0)
    offender = env.decision.player
    assert env.step("I draw.") == (False, {})
    player, observation = env.get_observation()
    assert player == offender and "Reply with the number" in observation
    done, _ = env.step("[99]")
    assert done and env.close()[0] == {p: -1.0 if p == offender else 0.0 for p in range(3)}
