"""The rules of Unstable Unicorns, independent of TextArena.

``Game.play()`` runs a whole game as one generator: it yields a ``Decision`` whenever a player
must choose and is resumed with the index of the chosen option, so turns, Neigh chains and card
effects read as straight-line code. The game ends by raising ``GameOver``.

Each player starts with a Baby Unicorn in their Stable and five cards in hand. A turn has four
phases: beginning-of-turn effects, draw a card, one action (play a card or draw another), and
discard down to seven cards. The first player with ``goal`` Unicorns in their Stable wins; if the
deck runs out or the turn limit is reached, whoever has the most Unicorns wins.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass

from autoenvscaling.envs.unstable_unicorns.cards import (
    UNICORN,
    Card,
    Kind,
    deck,
    nursery,
    sacrifice,
)

HAND_SIZE = 5
HAND_LIMIT = 7
MAX_PLAYERS = 8


@dataclass(frozen=True)
class Decision:
    player: int
    prompt: str
    options: tuple[str, ...]


class GameOver(Exception):
    """Raised out of ``Game.play()`` when the game ends; the ``winners`` share the win."""

    def __init__(self, winners: list[int], reason: str):
        super().__init__(reason)
        self.winners = winners


class Game:
    """The zones of one game (deck, discard pile, Nursery, hands, Stables) and the rules that
    move cards between them. ``tell(message, to=None)`` narrates to one player or to everyone."""

    def __init__(self, num_players: int, rng: random.Random | None = None, *,
                 goal: int | None = None, max_turns: int | None = None,
                 tell: Callable[..., None] | None = None):
        if not 2 <= num_players <= MAX_PLAYERS:
            raise ValueError(f"Unstable Unicorns is for 2-{MAX_PLAYERS} players, not {num_players}")
        self.n = num_players
        self.rng = rng or random.Random()
        self.goal = goal or (7 if num_players <= 5 else 6)
        self.max_turns = max_turns
        self.tell = tell or (lambda message, to=None: None)
        self.deck = deck()
        self.rng.shuffle(self.deck)
        self.nursery = nursery()
        self.discard_pile: list[Card] = []
        self.stables = [[self.nursery.pop()] for _ in range(num_players)]
        self.hands = [[self.deck.pop() for _ in range(HAND_SIZE)] for _ in range(num_players)]
        self.turns = 0
        self.ended = self.extra = False

    def seats(self, p: int) -> list[int]:
        """Every player in turn order, starting with ``p``."""
        return [(p + i) % self.n for i in range(self.n)]

    def others(self, p: int) -> list[int]:
        return self.seats(p)[1:]

    def has(self, p: int, name: str) -> bool:
        return any(c.name == name for c in self.stables[p])

    def count(self, p: int) -> int:
        """The Unicorns in Player ``p``'s Stable; a Ginormous Unicorn counts twice."""
        return sum(1 + (c.name == "Ginormous Unicorn") for c in self.stables[p] if c.unicorn)

    def leaders(self) -> list[int]:
        best = max(map(self.count, range(self.n)))
        return [p for p in range(self.n) if self.count(p) == best]

    def immune(self, p: int, card: Card, src: Card) -> bool:
        """Whether ``src`` cannot destroy ``card`` in Player ``p``'s Stable."""
        return ((card.unicorn and self.has(p, "Rainbow Aura"))
                or (card.name == "Magical Kittencorn" and src.kind is Kind.MAGIC))

    def choose[T](self, player: int, prompt: str, options: Mapping[str, T], *, may: bool = False,
                  skip: str = "skip") -> Generator[Decision, int, T | None]:
        """Ask ``player`` to pick one of ``options`` (label to value) and return its value.

        ``may`` adds an opt-out labelled ``skip`` that returns None, as do empty ``options``. A
        forced choice of a single option is made without asking.
        """
        labels = [*options, skip] if may else [*options]
        if not options:
            return None
        if len(labels) == 1:
            return options[labels[0]]
        index = yield Decision(player, prompt, tuple(labels))
        return options.get(labels[index])

    def play(self) -> Generator[Decision, int, None]:
        for me in itertools.cycle(range(self.n)):
            yield from self.turn(me)
            while self.extra:
                self.extra = False
                yield from self.turn(me)

    def turn(self, me: int) -> Generator[Decision, int, None]:
        if self.turns == self.max_turns:
            raise GameOver(self.leaders(), f"the {self.max_turns}-turn limit was reached")
        self.turns += 1
        self.ended = False
        self.tell(f"Turn {self.turns}: Player {me}.")
        for card in list(self.stables[me]):
            if card.turn and card in self.stables[me] and not self.ended:
                yield from card.turn(self, me, card)
        if not self.ended:
            self.draw(me)
        for _ in range(1 + self.has(me, "Double Dutch")):
            if not self.ended:
                yield from self.action(me)
        if (excess := len(self.hands[me]) - HAND_LIMIT) > 0:
            yield from self.discard(me, excess, f"End of turn: DISCARD down to {HAND_LIMIT} cards.")

    def action(self, me: int) -> Generator[Decision, int, None]:
        playable = {f"play {c.name}": c for c in self.hands[me] if c.kind is not Kind.INSTANT
                    and not (c.kind is Kind.UPGRADE and self.has(me, "Broken Stable"))}
        card = yield from self.choose(me, "Action phase: play a card, or draw a card instead.",
                                      {"draw a card": None, **playable})
        if card is None:
            self.draw(me)
            return
        self.hands[me].remove(card)
        owner = me
        if card.kind in (Kind.UPGRADE, Kind.DOWNGRADE):
            owner = yield from self.choose(me, f"Whose Stable does {card.name} go into?",
                                           {f"Player {p}": p for p in self.seats(me)})
        self.tell(f"Player {me} plays {card.name}"
                  + ("." if owner == me else f" into Player {owner}'s Stable."))
        if (yield from self.neighed(me, card)):
            self.discard_pile.append(card)
        elif card.kind is Kind.MAGIC:
            self.discard_pile.append(card)
            yield from card.play(self, me, card)
        else:
            yield from self.enter(owner, card)

    def neighed(self, me: int, card: Card) -> Generator[Decision, int, bool]:
        """Run the Neigh chain over ``card``, played by ``me``; True if it was stopped."""
        last, top, stopped = me, card, False
        while top.name != "Super Neigh" and not self.has(last, "Yay"):
            for p in self.others(last):
                neighs = {f"play {c.name}": c for c in self.hands[p] if c.kind is Kind.INSTANT
                          and not (self.has(p, "Slowdown") or self.has(p, "Ginormous Unicorn"))}
                neigh = yield from self.choose(p, f"Player {last} is playing {top.name}.", neighs,
                                               may=True, skip="pass")
                if neigh is not None:
                    break
            else:
                return stopped
            self.hands[p].remove(neigh)
            self.discard_pile.append(neigh)
            self.tell(f"Player {p} plays {neigh.name} on Player {last}'s {top.name}!")
            last, top, stopped = p, neigh, not stopped
        return stopped

    def draw(self, p: int, n: int = 1) -> None:
        for _ in range(n):
            if not self.deck:
                raise GameOver(self.leaders(), "the deck ran out")
            card = self.deck.pop()
            self.hands[p].append(card)
            self.tell(f"You draw {card.name}.", to=p)
        self.tell(f"Player {p} draws {n} card{'s' * (n > 1)}.")

    def discard(self, p: int, n: int = 1, prompt: str = "DISCARD a card.", *,
                may: bool = False) -> Generator[Decision, int, bool]:
        """Player ``p`` discards ``n`` cards; False if they skipped or ran out of cards."""
        for _ in range(n):
            card = yield from self.choose(p, prompt, {c.name: c for c in self.hands[p]}, may=may)
            if card is None:
                return False
            self.hands[p].remove(card)
            self.discard_pile.append(card)
            self.tell(f"Player {p} discards {card.name}.")
        return True

    def reshuffle(self) -> None:
        self.deck += self.discard_pile
        self.discard_pile.clear()
        self.rng.shuffle(self.deck)
        self.tell("The discard pile is shuffled into the deck.")

    def enter(self, p: int, card: Card) -> Generator[Decision, int, None]:
        self.stables[p].append(card)
        self.tell(f"{card.name} enters Player {p}'s Stable.")
        if self.count(p) >= self.goal:
            raise GameOver([p], f"Player {p} has {self.count(p)} Unicorns")
        if card.enter:
            yield from card.enter(self, p, card)
        if card.unicorn:
            yield from self._unicorn_moved(p, entered=True)

    def remove(self, p: int, card: Card, how: str) -> Generator[Decision, int, None]:
        """Take ``card`` out of Player ``p``'s Stable: destroyed, sacrificed, stolen or returned.

        A stolen card is left for the thief to bring into their Stable.
        """
        self.stables[p].remove(card)
        self.tell(f"Player {p}'s {card.name} is {how}.")
        if how != "stolen":
            pile = (self.nursery if card.kind is Kind.BABY
                    else self.hands[p] if how == "returned" or card.flies else self.discard_pile)
            pile.append(card)
        if card.leave and how in ("destroyed", "sacrificed"):
            yield from card.leave(self, p, card)
        if card.unicorn:
            yield from self._unicorn_moved(p, entered=False)

    def _unicorn_moved(self, p: int, entered: bool) -> Generator[Decision, int, None]:
        """The Downgrades that watch Unicorns entering and leaving Player ``p``'s Stable."""
        if self.has(p, "Barbed Wire"):
            yield from self.discard(p, prompt="Barbed Wire: DISCARD a card.")
        tiny = next((c for c in self.stables[p] if c.name == "Tiny Stable"), None)
        if entered and tiny and sum(c.unicorn for c in self.stables[p]) > 5:
            yield from sacrifice(UNICORN)(self, p, tiny)
