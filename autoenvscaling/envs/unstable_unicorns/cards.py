"""The cards, with each card's effect written in the verbs of the game.

An ``Effect`` is a generator ``effect(game, me, card)`` run for player ``me``: it yields a
``Decision`` whenever someone must choose, is resumed with the index of the chosen option, and
returns whether it did anything, so ``then(x, y)`` reads like the card text "you may X. If you do,
Y." Every card text below states exactly what its effect implements.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoenvscaling.envs.unstable_unicorns.game import Decision, Game

Effect = Callable[["Game", int, "Card"], Generator["Decision", int, bool]]


class Kind(StrEnum):
    BABY = "Baby Unicorn"
    BASIC = "Basic Unicorn"
    MAGICAL = "Magical Unicorn"
    MAGIC = "Magic"
    UPGRADE = "Upgrade"
    DOWNGRADE = "Downgrade"
    INSTANT = "Instant"


UNICORN = frozenset({Kind.BABY, Kind.BASIC, Kind.MAGICAL})
STABLE = UNICORN | {Kind.UPGRADE, Kind.DOWNGRADE}


@dataclass(frozen=True, eq=False)
class Card:
    """One physical card. Equality is identity: two Basic Unicorns are two cards."""

    name: str
    kind: Kind
    text: str
    enter: Effect | None = None  # when it enters your Stable
    turn: Effect | None = None  # at the beginning of your turn, while it is in your Stable
    leave: Effect | None = None  # when it is sacrificed or destroyed
    play: Effect | None = None  # a Magic card's effect
    flies: bool = False  # returns to your hand instead of the discard pile

    @property
    def unicorn(self) -> bool:
        return self.kind in UNICORN


def seq(*effects: Effect) -> Effect:
    """Each effect in turn: "X, then Y"."""
    def effect(g, me, src):
        for e in effects:
            yield from e(g, me, src)
        return True
    return effect


def then(first: Effect, *rest: Effect) -> Effect:
    """``rest`` only if ``first`` did anything: "you may X. If you do, Y"."""
    def effect(g, me, src):
        return (yield from first(g, me, src)) and (yield from seq(*rest)(g, me, src))
    return effect


def each(e: Effect) -> Effect:
    """Every player does ``e``, starting with you."""
    def effect(g, me, src):
        for p in g.seats(me):
            yield from e(g, p, src)
        return True
    return effect


def _immediate(action: Callable[[Game, int], object]) -> Effect:
    def effect(g, me, src):
        action(g, me)
        return True
        yield  # an effect is a generator, even one that asks nobody anything
    return effect


def draw(n: int = 1) -> Effect:
    return _immediate(lambda g, me: g.draw(me, n))


def discard(n: int = 1, *, may: bool = False) -> Effect:
    return lambda g, me, src: g.discard(me, n, f"{src.name}: DISCARD a card.", may=may)


end_turn = _immediate(lambda g, me: setattr(g, "ended", True))
extra_turn = _immediate(lambda g, me: setattr(g, "extra", True))
reshuffle = _immediate(lambda g, me: g.reshuffle())


def _stable_verb(verb: str, how: str, *, own: bool = False) -> Callable[..., Effect]:
    """A verb on one card in a Stable: your own if ``own``, else another player's."""
    def factory(kinds: frozenset[Kind] = UNICORN, *, may: bool = False) -> Effect:
        def effect(g, me, src):
            options = {f"{c.name} (Player {p})": (p, c) for p in ([me] if own else g.others(me))
                       for c in g.stables[p]
                       if c.kind in kinds and not (how == "destroyed" and g.immune(p, c, src))}
            hit = yield from g.choose(me, f"{src.name}: {verb} a card.", options, may=may)
            if hit is None:
                return False
            yield from g.remove(*hit, how)
            if how == "stolen":
                yield from g.enter(me, hit[1])
            return True
        return effect
    return factory


destroy = _stable_verb("DESTROY", "destroyed")
sacrifice = _stable_verb("SACRIFICE", "sacrificed", own=True)
steal = _stable_verb("STEAL", "stolen")
bounce = _stable_verb("RETURN", "returned")


def sacrifice_this(g, me, src):
    """You may SACRIFICE this card."""
    if src not in g.stables[me]:
        return False
    if not (yield from g.choose(me, f"{src.name}: SACRIFICE it?", {"sacrifice": True}, may=True)):
        return False
    yield from g.remove(me, src, "sacrificed")
    return True


def _rescue(into_stable: bool) -> Callable[..., Effect]:
    """Take a card out of the discard pile into your Stable or your hand."""
    def factory(kinds: frozenset[Kind], *, may: bool = False) -> Effect:
        def effect(g, me, src):
            options = {c.name: c for c in g.discard_pile if c.kind in kinds}
            card = yield from g.choose(me, f"{src.name}: take a card from the discard pile.",
                                       options, may=may)
            if card is None:
                return False
            g.discard_pile.remove(card)
            g.tell(f"Player {me} takes {card.name} from the discard pile.")
            if into_stable:
                yield from g.enter(me, card)
            else:
                g.hands[me].append(card)
            return True
        return effect
    return factory


revive = _rescue(into_stable=True)
salvage = _rescue(into_stable=False)


def adopt(g, me, src):
    if not g.nursery:
        return False
    yield from g.enter(me, g.nursery.pop())
    return True


def pull(*, look: bool, may: bool = False) -> Effect:
    """Take a card from another player's hand: one you choose if you ``look``, else at random."""
    def effect(g, me, src):
        victims = {f"Player {p}": p for p in g.others(me) if g.hands[p]}
        p = yield from g.choose(me, f"{src.name}: whose hand?", victims, may=may)
        if p is None:
            return False
        hand = g.hands[p]
        card = ((yield from g.choose(me, f"{src.name}: take a card.", {c.name: c for c in hand}))
                if look else g.rng.choice(hand))
        hand.remove(card)
        g.hands[me].append(card)
        g.tell(f"Player {me} takes a card from Player {p}'s hand.")
        g.tell(f"You take {card.name}.", to=me)
        g.tell(f"Player {me} took your {card.name}.", to=p)
        return True
    return effect


def trade(g, me, src):
    p = yield from g.choose(me, f"{src.name}: trade hands with whom?",
                            {f"Player {p}": p for p in g.others(me)})
    g.hands[me], g.hands[p] = g.hands[p], g.hands[me]
    g.tell(f"Player {me} and Player {p} trade hands.")
    return True


BABY = Card("Baby Unicorn", Kind.BABY, "Each player starts with one in their Stable. If it would "
            "leave a Stable for anywhere but another Stable, it returns to the Nursery.")

DECK: tuple[tuple[int, Card], ...] = (
    (22, Card("Basic Unicorn", Kind.BASIC, "No effect.")),
    (1, Card("Rhinocorn", Kind.MAGICAL, "At the beginning of your turn, you may DESTROY a "
             "Unicorn. If you do, your turn ends immediately.",
             turn=then(destroy(may=True), end_turn))),
    (2, Card("Extremely Fertile Unicorn", Kind.MAGICAL, "At the beginning of your turn, you may "
             "DISCARD a card. If you do, bring a Baby Unicorn from the Nursery into your Stable.",
             turn=then(discard(may=True), adopt))),
    (1, Card("Angel Unicorn", Kind.MAGICAL, "At the beginning of your turn, you may SACRIFICE "
             "this card. If you do, bring a Unicorn from the discard pile into your Stable.",
             turn=then(sacrifice_this, revive(UNICORN)))),
    (2, Card("Unicorn on the Cob", Kind.MAGICAL, "When this card enters your Stable, DRAW 2 "
             "cards, then DISCARD a card.", enter=seq(draw(2), discard()))),
    (2, Card("Shark With a Horn", Kind.MAGICAL, "When this card enters your Stable, you may "
             "SACRIFICE it. If you do, DESTROY a Unicorn.", enter=then(sacrifice_this, destroy()))),
    (2, Card("Seductive Unicorn", Kind.MAGICAL, "When this card enters your Stable, you may "
             "DISCARD a card. If you do, STEAL a Unicorn.",
             enter=then(discard(may=True), steal()))),
    (1, Card("Llamacorn", Kind.MAGICAL, "When this card enters your Stable, each player DISCARDS "
             "a card.", enter=each(discard()))),
    (2, Card("Americorn", Kind.MAGICAL, "When this card enters your Stable, you may take a "
             "random card from another player's hand.", enter=pull(look=False, may=True))),
    (2, Card("Mermaid Unicorn", Kind.MAGICAL, "When this card enters your Stable, RETURN a card "
             "in another player's Stable to their hand.", enter=bounce(STABLE))),
    (2, Card("Greedy Flying Unicorn", Kind.MAGICAL, "When this card enters your Stable, DRAW a "
             "card. If it is sacrificed or destroyed, it returns to your hand.",
             enter=draw(), flies=True)),
    (1, Card("Majestic Flying Unicorn", Kind.MAGICAL, "When this card enters your Stable, you may "
             "take a Unicorn from the discard pile into your hand. If it is sacrificed or "
             "destroyed, it returns to your hand.", enter=salvage(UNICORN, may=True), flies=True)),
    (1, Card("Swift Flying Unicorn", Kind.MAGICAL, "When this card enters your Stable, you may "
             "take an Instant from the discard pile into your hand. If it is sacrificed or "
             "destroyed, it returns to your hand.",
             enter=salvage(frozenset({Kind.INSTANT}), may=True), flies=True)),
    (1, Card("Stabby the Unicorn", Kind.MAGICAL, "If this card is sacrificed or destroyed, you "
             "may DESTROY a Unicorn.", leave=destroy(may=True))),
    (1, Card("Magical Kittencorn", Kind.MAGICAL, "Cannot be destroyed by Magic cards.")),
    (1, Card("Ginormous Unicorn", Kind.MAGICAL, "Counts as 2 Unicorns. You cannot play Neighs.")),
    (4, Card("Unicorn Poison", Kind.MAGIC, "DESTROY a Unicorn.", play=destroy())),
    (2, Card("Good Deal", Kind.MAGIC, "DRAW 3 cards, then DISCARD a card.",
             play=seq(draw(3), discard()))),
    (2, Card("Change of Luck", Kind.MAGIC, "DRAW 2 cards, then DISCARD 3 cards. Take another "
             "turn after this one.", play=seq(draw(2), discard(3), extra_turn))),
    (2, Card("Kiss of Life", Kind.MAGIC, "Bring a Unicorn from the discard pile into your Stable.",
             play=revive(UNICORN))),
    (2, Card("Two-For-One", Kind.MAGIC, "SACRIFICE a card. If you do, DESTROY 2 cards.",
             play=then(sacrifice(STABLE), destroy(STABLE), destroy(STABLE)))),
    (2, Card("Unfair Bargain", Kind.MAGIC, "Trade hands with another player.", play=trade)),
    (2, Card("Blatant Thievery", Kind.MAGIC, "Look at another player's hand and take a card from "
             "it.", play=pull(look=True))),
    (1, Card("Mystical Vortex", Kind.MAGIC, "Each player DISCARDS a card, then the discard pile "
             "is shuffled into the deck.", play=seq(each(discard()), reshuffle))),
    (2, Card("Rainbow Aura", Kind.UPGRADE, "Your Unicorns cannot be destroyed.")),
    (2, Card("Yay", Kind.UPGRADE, "Cards you play cannot be Neighed.")),
    (1, Card("Double Dutch", Kind.UPGRADE, "Your Action phase has two actions.")),
    (2, Card("Glitter Bomb", Kind.UPGRADE, "At the beginning of your turn, you may SACRIFICE a "
             "card. If you do, DESTROY a card.",
             turn=then(sacrifice(STABLE, may=True), destroy(STABLE)))),
    (2, Card("Slowdown", Kind.DOWNGRADE, "You cannot play Neighs.")),
    (1, Card("Broken Stable", Kind.DOWNGRADE, "You cannot play Upgrades.")),
    (2, Card("Barbed Wire", Kind.DOWNGRADE, "Each time a Unicorn enters or leaves your Stable, "
             "DISCARD a card.")),
    (1, Card("Tiny Stable", Kind.DOWNGRADE, "Each time a Unicorn enters your Stable while you "
             "have more than 5, SACRIFICE a Unicorn.")),
    (1, Card("Nanny Cam", Kind.DOWNGRADE, "Your hand is visible to all players.")),
    (1, Card("Sadistic Ritual", Kind.DOWNGRADE, "At the beginning of your turn, SACRIFICE a "
             "Unicorn. If you do, DRAW a card.", turn=then(sacrifice(), draw()))),
    (14, Card("Neigh", Kind.INSTANT, "When another player plays a card, play this to send that "
              "card to the discard pile without effect.")),
    (1, Card("Super Neigh", Kind.INSTANT, "As Neigh, but it cannot be Neighed.")),
)
NURSERY_SIZE = 13


def deck() -> list[Card]:
    return [replace(card) for count, card in DECK for _ in range(count)]


def nursery() -> list[Card]:
    return [replace(BABY) for _ in range(NURSERY_SIZE)]
