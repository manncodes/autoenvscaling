"""Unstable Unicorns as a TextArena environment for 2-8 players.

Every step answers one player's pending ``Decision``: that player sees the board and a numbered
list of options, and replies with an index in brackets, e.g. ``[0]`` (or an option's label, e.g.
``[pass]``). Only the acting player sees their own replies; the game narrates everything public.

Rewards: +1 for the winners and -1 for everyone else, or 0 for all if every player ties. A player
who runs out of invalid-move allowance gets -1 and ends the game, with 0 for the others.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

import textarena as ta

from autoenvscaling.envs.unstable_unicorns.cards import BABY, DECK, NURSERY_SIZE, Card
from autoenvscaling.envs.unstable_unicorns.game import HAND_LIMIT, HAND_SIZE, Game, GameOver

RULES = """\
You are Player {player} in a {n}-player game of Unstable Unicorns. The first player with {goal} \
Unicorns in their Stable wins. If the deck runs out, or after {max_turns} turns, the player with \
the most Unicorns wins.

Everyone starts with a Baby Unicorn in their Stable and {hand} cards in hand. On your turn: \
(1) beginning-of-turn effects of the cards in your Stable happen, (2) you draw a card, (3) you \
either play a card from your hand or draw another, (4) you discard down to {limit} cards.
Unicorns enter your own Stable; Upgrades and Downgrades enter any player's Stable; Magic cards \
take effect, then go to the discard pile. Whenever a player plays a card, every other player \
may play a Neigh to stop it, and a Neigh can itself be Neighed.
DESTROY: move a card in another player's Stable to the discard pile. SACRIFICE: the same, for a \
card in your own Stable. STEAL: move a card from another player's Stable into yours. DISCARD: \
move a card from your hand to the discard pile.

Whenever you must decide, you see your options as a numbered list. Reply with the number of \
your choice in brackets, e.g. [0].

The cards (name, type, copies):
{cards}"""


def parse_choice(action: str, options: Sequence[str]) -> int | None:
    """The option named by the last bracketed index or label in ``action``, if any."""
    labels = [o.lower() for o in options]
    for token in reversed(re.findall(r"\[([^\[\]]*)\]", action)):
        token = token.strip().lower()
        if token.isdigit() and int(token) < len(options):
            return int(token)
        if token in labels:
            return labels.index(token)
    return None


def _names(cards: list[Card]) -> str:
    return ", ".join(c.name for c in cards) or "(empty)"


class UnstableUnicornsEnv(ta.Env):
    def __init__(self, goal: int | None = None, max_rounds: int = 30, error_allowance: int = 3):
        """``goal`` defaults to 7 Unicorns (6 with six or more players); the game also ends after
        ``max_rounds`` turns per player."""
        self.goal = goal
        self.max_rounds = max_rounds
        self.error_allowance = error_allowance

    def reset(self, num_players: int, seed: int | None = None):
        self.game = Game(num_players, random.Random(seed), goal=self.goal,
                         max_turns=self.max_rounds * num_players, tell=self._tell)
        self.state = ta.FFAMultiPlayerState(num_players=num_players, seed=seed,
                                            error_allowance=self.error_allowance)
        self.state.reset(player_prompt_function=self._prompt)
        self.engine = self.game.play()
        self._resume(None)
        self._ask()

    def step(self, action: str) -> tuple[bool, ta.Info]:
        player, options = self.decision.player, self.decision.options
        self.state.add_observation(from_id=player, to_id=player, message=action,
                                   observation_type=ta.ObservationType.PLAYER_ACTION)
        choice = parse_choice(action, options)
        if choice is not None:
            self._resume(choice)
        elif self.state.set_invalid_move(f"Reply with the number of one of the {len(options)} "
                                         "options in brackets, e.g. [0]."):
            self.state.set_game_outcome(
                {p: -1.0 if p == player else 0.0 for p in range(self.game.n)},
                f"Player {player} ran out of invalid moves.")
        done, info = self.state.step(rotate_player=False)
        if not done and choice is not None:
            self._ask()
        return done, info

    def get_board_str(self) -> str:
        return self.render()

    def render(self, viewer: int | None = None) -> str:
        """The board as ``viewer`` sees it: everyone's Stable, and their own hand."""
        g = self.game
        header = (f"Turn {g.turns}. Deck: {len(g.deck)}, discard pile: {len(g.discard_pile)}, "
                  f"Nursery: {len(g.nursery)}. First to {g.goal} Unicorns wins.")
        lines = [header]
        for p in range(g.n):
            lines.append(f"Player {p}{' (you)' * (p == viewer)}: {g.count(p)} Unicorns, "
                         f"{len(g.hands[p])} cards in hand.\n  Stable: {_names(g.stables[p])}")
            if p == viewer or g.has(p, "Nanny Cam"):
                lines.append(f"  Hand: {_names(g.hands[p])}")
        return "\n".join(lines)

    def _resume(self, choice: int | None) -> None:
        """Run the game up to its next decision, or to its end."""
        try:
            self.decision = self.engine.send(choice)
        except GameOver as over:
            counts = ", ".join(f"Player {p}: {self.game.count(p)}" for p in range(self.game.n))
            reason = f"Game over: {over}. Unicorns: {counts}."
            if len(over.winners) == self.game.n:
                self.state.set_draw(reason)
            else:
                self.state.set_winners(over.winners, reason)

    def _ask(self) -> None:
        d = self.decision
        self.state.manually_set_current_player_id(d.player, force=True)
        options = "\n".join(f"[{i}] {label}" for i, label in enumerate(d.options))
        board = f"{self.render(d.player)}\n\n{d.prompt}\n{options}"
        self.state.add_observation(to_id=d.player, message=board,
                                   observation_type=ta.ObservationType.GAME_BOARD)

    def _tell(self, message: str, to: int | None = None) -> None:
        self.state.add_observation(message=message, to_id=-1 if to is None else to,
                                   observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

    def _prompt(self, player_id: int, game_state: object) -> str:
        cards = "\n".join(f"- {c.name} ({c.kind}, x{count}): {c.text}"
                          for count, c in ((NURSERY_SIZE, BABY), *DECK))
        return RULES.format(player=player_id, n=self.game.n, goal=self.game.goal,
                            max_turns=self.game.max_turns, hand=HAND_SIZE, limit=HAND_LIMIT,
                            cards=cards)
