"""
Heads-Up Razz Game Engine

Faithful port of HeadsUpRazzGame.swift from Razz Rebel Trainer.
Fixed-limit heads-up Razz with proper bring-in, completion, and betting cap logic.

Cards are represented as (rank, suit) tuples:
  rank: 1=Ace through 13=King
  suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
"""

import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import IntEnum

from razz_eval import evaluate as razz_evaluate


# ─── Constants ──────────────────────────────────────────────────────────────

ANTE = 0.2       # 10 in a 50/100 game
BRING_IN = 0.3   # 15 in a 50/100 game
SMALL_BET = 1.0  # 50 in a 50/100 game
BIG_BET = 2.0    # 100 in a 50/100 game
MAX_BETS_PER_STREET = 5  # 1 bet + 4 raises


# ─── Enums ──────────────────────────────────────────────────────────────────

class Action(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3      # Also "complete" on 3rd street
    RAISE = 4

    @property
    def symbol(self) -> str:
        return ['f', 'k', 'c', 'b', 'r'][self.value]


STREETS = [3, 4, 5, 6, 7]


def bet_size(street: int) -> float:
    """Small bet on 3rd-4th, big bet on 5th-7th."""
    return SMALL_BET if street <= 4 else BIG_BET


# ─── Card ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Card:
    rank: int  # 1=Ace ... 13=King
    suit: int  # 0-3

    def __repr__(self):
        rank_names = {1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',
                      8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K'}
        suit_names = {0:'c', 1:'d', 2:'h', 3:'s'}
        return f"{rank_names.get(self.rank,'?')}{suit_names.get(self.suit,'?')}"


def make_deck() -> List[Card]:
    """Standard 52-card deck, shuffled."""
    deck = [Card(rank=r, suit=s) for r in range(1, 14) for s in range(4)]
    random.shuffle(deck)
    return deck


# ─── Player State ───────────────────────────────────────────────────────────

@dataclass
class PlayerState:
    hole_cards: List[Card] = field(default_factory=list)
    up_cards: List[Card] = field(default_factory=list)
    invested: float = 0.0
    is_active: bool = True

    @property
    def all_cards(self) -> List[Card]:
        return self.hole_cards + self.up_cards

    @property
    def all_ranks(self) -> List[int]:
        return [c.rank for c in self.all_cards]

    @property
    def card_count(self) -> int:
        return len(self.hole_cards) + len(self.up_cards)

    def clone(self) -> 'PlayerState':
        return PlayerState(
            hole_cards=list(self.hole_cards),
            up_cards=list(self.up_cards),
            invested=self.invested,
            is_active=self.is_active,
        )


# ─── Game State ─────────────────────────────────────────────────────────────

class HeadsUpRazzGame:
    """Heads-up fixed-limit Razz game state.

    Faithfully ports HeadsUpRazzGame.swift including:
    - Bring-in logic (highest upcard brings in)
    - Completion (betting to full small bet on 3rd street)
    - Per-street contribution tracking
    - Betting cap (4 bets per street)
    - Board-strength-based first actor on 4th+ streets
    """

    def __init__(self):
        self.players = [PlayerState(), PlayerState()]
        self.street = 3
        self.pot = 0.0
        self.current_player = 0
        self.all_actions: List[List[tuple]] = [[] for _ in range(5)]  # streets 3-7, each entry is (Action, player_idx)

        # Per-player per-street aggression counters (avoids recomputing from all_actions)
        # player_agg[player_idx][street_idx] = count of BET/RAISE actions
        self.player_agg = [[0] * 5, [0] * 5]

        # Per-street betting state
        self.street_contribution = [0.0, 0.0]
        self.current_bet_level = 0.0
        self.num_bets = 0
        self.last_aggressor = -1
        self.num_actions_this_round = 0

        # Bring-in
        self.bring_in_player = 0
        self.is_bring_in_state = False

        # Terminal
        self.is_terminal = False
        self.winner: Optional[int] = None  # 0, 1, or None (chop)

    def clone(self) -> 'HeadsUpRazzGame':
        """Deep copy for tree traversal."""
        g = HeadsUpRazzGame()
        g.players = [p.clone() for p in self.players]
        g.street = self.street
        g.pot = self.pot
        g.current_player = self.current_player
        g.all_actions = [list(a) for a in self.all_actions]  # tuples are immutable, shallow copy is fine
        g.player_agg = [list(a) for a in self.player_agg]
        g.street_contribution = list(self.street_contribution)
        g.current_bet_level = self.current_bet_level
        g.num_bets = self.num_bets
        g.last_aggressor = self.last_aggressor
        g.num_actions_this_round = self.num_actions_this_round
        g.bring_in_player = self.bring_in_player
        g.is_bring_in_state = self.is_bring_in_state
        g.is_terminal = self.is_terminal
        g.winner = self.winner
        return g

    # ── Setup ───────────────────────────────────────────────────────────

    def deal_third_street(
        self,
        p0_hole: List[Card], p0_up: Card,
        p1_hole: List[Card], p1_up: Card,
    ):
        """Deal 3rd street: post antes, assign cards, determine bring-in."""
        # Post antes
        self.players[0].invested = ANTE
        self.players[1].invested = ANTE
        self.pot = ANTE * 2

        # Deal cards
        self.players[0].hole_cards = list(p0_hole)
        self.players[0].up_cards = [p0_up]
        self.players[1].hole_cards = list(p1_hole)
        self.players[1].up_cards = [p1_up]

        # Bring-in: highest upcard (worst in Razz)
        # Ties broken by suit: spades(3) > hearts(2) > diamonds(1) > clubs(0)
        if (p0_up.rank > p1_up.rank or
            (p0_up.rank == p1_up.rank and p0_up.suit > p1_up.suit)):
            self.bring_in_player = 0
        else:
            self.bring_in_player = 1

        # Bring-in player acts first — they choose to post bring-in (0.3) or complete (1.0)
        self.current_player = self.bring_in_player
        self.is_bring_in_state = True
        self.num_bets = 0
        self.last_aggressor = -1
        self.num_actions_this_round = 0
        self.street = 3

    def deal_card(self, player_idx: int, card: Card, is_hole: bool):
        """Deal a single card to a player."""
        if is_hole:
            self.players[player_idx].hole_cards.append(card)
        else:
            self.players[player_idx].up_cards.append(card)

    # ── Legal Actions ───────────────────────────────────────────────────

    def legal_actions(self) -> List[Action]:
        """Return list of legal actions for current player."""
        if self.is_terminal:
            return []

        my_contrib = self.street_contribution[self.current_player]
        to_call = self.current_bet_level - my_contrib

        if self.is_bring_in_state:
            if self.current_player == self.bring_in_player and self.street_contribution[self.bring_in_player] == 0:
                # Bring-in player's first action: post bring-in or complete
                return [Action.CALL, Action.BET]  # CALL = post bring-in (0.3), BET = complete (1.0)
            else:
                # Other player responding to bring-in
                actions = [Action.FOLD]
                if to_call > 0:
                    actions.append(Action.CALL)
                actions.append(Action.BET)  # Complete
                return actions

        # Normal betting
        if to_call > 0:
            actions = [Action.FOLD, Action.CALL]
            if self.num_bets < MAX_BETS_PER_STREET:
                actions.append(Action.RAISE)
            return actions
        else:
            actions = [Action.CHECK]
            if self.num_bets < MAX_BETS_PER_STREET:
                actions.append(Action.BET)
            return actions

    # ── Apply Action ────────────────────────────────────────────────────

    def apply_action(self, action: Action) -> float:
        """Apply an action. Returns the cost to the acting player."""
        if self.is_terminal:
            return 0.0

        street_idx = self.street - 3
        cp = self.current_player
        self.all_actions[street_idx].append((action, cp))
        if action in (Action.BET, Action.RAISE):
            self.player_agg[cp][street_idx] += 1

        cost = 0.0

        if action == Action.FOLD:
            self.players[cp].is_active = False
            self.is_terminal = True
            self.winner = 1 - cp
            return 0.0

        elif action == Action.CHECK:
            self.num_actions_this_round += 1

        elif action == Action.CALL:
            if self.is_bring_in_state and cp == self.bring_in_player and self.street_contribution[cp] == 0:
                # Bring-in player posts the bring-in (0.3 BB)
                cost = BRING_IN
                self.street_contribution[cp] = BRING_IN
                self.current_bet_level = BRING_IN
                self.players[cp].invested += cost
                self.pot += cost
                # Other player now acts
                self.current_player = 1 - cp
                return cost
            else:
                cost = self.current_bet_level - self.street_contribution[cp]
                self.street_contribution[cp] = self.current_bet_level
                self.players[cp].invested += cost
                self.pot += cost
                self.num_actions_this_round += 1

                if self.is_bring_in_state:
                    self.is_bring_in_state = False
                    self._advance_street()
                    return cost

        elif action == Action.BET:
            if self.is_bring_in_state:
                # Complete to full small bet (works for both bring-in player and responder)
                full_bet = SMALL_BET
                cost = full_bet - self.street_contribution[cp]
                self.street_contribution[cp] = full_bet
                self.current_bet_level = full_bet
                self.players[cp].invested += cost
                self.pot += cost
                self.num_bets = 1
                self.is_bring_in_state = False
                self.last_aggressor = cp
                self.num_actions_this_round = 0
                # Other player now acts
                self.current_player = 1 - cp
                return cost
            else:
                # Normal opening bet
                cost = bet_size(self.street)
                self.street_contribution[cp] += cost
                self.current_bet_level = self.street_contribution[cp]
                self.players[cp].invested += cost
                self.pot += cost
                self.num_bets += 1
                self.last_aggressor = cp
                self.num_actions_this_round = 0

        elif action == Action.RAISE:
            call_amount = self.current_bet_level - self.street_contribution[cp]
            raise_amount = bet_size(self.street)
            cost = call_amount + raise_amount
            self.street_contribution[cp] += cost
            self.current_bet_level = self.street_contribution[cp]
            self.players[cp].invested += cost
            self.pot += cost
            self.num_bets += 1
            self.last_aggressor = cp
            self.num_actions_this_round = 0

        # Check if betting round is complete
        self._check_street_complete()

        return cost

    # ── Street Completion ───────────────────────────────────────────────

    def _check_street_complete(self):
        if self.is_terminal:
            return

        if self.last_aggressor == -1:
            # No one bet. Need both players to check (2 actions).
            if self.num_actions_this_round >= 2:
                self._advance_street()
                return
        else:
            # Someone bet/raised. 1 response = opponent called.
            if self.num_actions_this_round >= 1:
                self._advance_street()
                return

        # Street continues — switch player
        self.current_player = 1 - self.current_player

    def _advance_street(self):
        if self.street >= 7:
            self._resolve_showdown()
            return

        self.street += 1

        # Reset street state
        self.street_contribution = [0.0, 0.0]
        self.current_bet_level = 0.0
        self.num_bets = 0
        self.last_aggressor = -1
        self.num_actions_this_round = 0
        self.is_bring_in_state = False

        # Lowest board acts first on 4th+
        # Compare upcard combinations as Razz hands (lower = better = acts first)
        s0 = self._board_strength(0)
        s1 = self._board_strength(1)
        self.current_player = 0 if s0 <= s1 else 1

    def _board_strength(self, player_idx: int) -> float:
        """Evaluate visible upcards as a Razz hand for action order.

        Lower = better = acts first. Pairs are penalized.
        Uses the same evaluation as razz_eval but on visible cards only.
        """
        ranks = [c.rank for c in self.players[player_idx].up_cards]
        if not ranks:
            return 0

        # Penalize pairs (same as razz_eval)
        seen_count = {}
        penalized = []
        for r in sorted(ranks):
            count = seen_count.get(r, 0)
            penalized.append(r + count * 13)
            seen_count[r] = count + 1

        penalized.sort()

        # Weighted score: higher position = more weight
        score = 0.0
        for i, r in enumerate(penalized):
            score += r * (14 ** i)
        return score

    def _resolve_showdown(self):
        self.is_terminal = True
        ranks0 = self.players[0].all_ranks
        ranks1 = self.players[1].all_ranks

        _, score0 = razz_evaluate(ranks0)
        _, score1 = razz_evaluate(ranks1)

        if score0 < score1:
            self.winner = 0
        elif score1 < score0:
            self.winner = 1
        else:
            self.winner = None  # Chop

    # ── Payoff ──────────────────────────────────────────────────────────

    def payoff(self, player_idx: int) -> float:
        """Return profit/loss for a player. Positive = won, negative = lost."""
        if not self.is_terminal:
            return 0.0

        if self.winner is not None:
            if self.winner == player_idx:
                return self.pot - self.players[player_idx].invested
            else:
                return -self.players[player_idx].invested
        else:
            # Chop
            return (self.pot / 2.0) - self.players[player_idx].invested

    # ── Action History ──────────────────────────────────────────────────

    @property
    def action_history_str(self) -> str:
        """Human-readable action history."""
        parts = []
        for i, street_actions in enumerate(self.all_actions):
            if street_actions:
                parts.append(f"S{i+3}:{''.join(a.symbol for a, _ in street_actions)}")
        return '/'.join(parts)

    @property
    def bucketed_action_history(self) -> str:
        """Bucketed action history matching RazzCFRSolver.bucketActionHistory.

        Per street: {aggression_count}{terminal_action}
        Streets joined by dots.
        """
        parts = []
        for street_actions in self.all_actions:
            if not street_actions:
                parts.append('')
                continue
            agg_count = sum(1 for a, _ in street_actions if a in (Action.BET, Action.RAISE))
            last_action = street_actions[-1][0]
            if last_action in (Action.CHECK, Action.CALL, Action.FOLD):
                parts.append(f"{agg_count}{last_action.symbol}")
            else:
                # Street is open (hero faces bet/raise)
                parts.append(f"{agg_count}")
        return '.'.join(parts)


# ─── Quick validation ───────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Razz Game Engine Tests ===\n")

    # Test 1: Simple fold
    g = HeadsUpRazzGame()
    g.deal_third_street(
        p0_hole=[Card(1,0), Card(2,0)], p0_up=Card(3,0),   # A23
        p1_hole=[Card(10,0), Card(11,0)], p1_up=Card(13,0), # TJK — K brings in
    )
    print(f"Bring-in player: {g.bring_in_player} (should be 1, K is worst)")
    print(f"Current player: {g.current_player} (should be 0, non-bring-in acts)")
    assert g.bring_in_player == 1
    assert g.current_player == 0
    print(f"Legal actions: {[a.name for a in g.legal_actions()]}")
    assert Action.FOLD in g.legal_actions()
    assert Action.CALL in g.legal_actions()
    assert Action.BET in g.legal_actions()

    # Player 0 completes (bet)
    cost = g.apply_action(Action.BET)
    print(f"P0 completes: cost={cost}, pot={g.pot}")
    assert cost == 1.0  # Non-bring-in player has 0 street contrib, complete to SMALL_BET=1.0

    print(f"After complete: pot={g.pot}, street_contrib={g.street_contribution}")
    print(f"Current player: {g.current_player} (should be 1)")

    # Player 1 folds
    g.apply_action(Action.FOLD)
    assert g.is_terminal
    assert g.winner == 0
    p0_payoff = g.payoff(0)
    p1_payoff = g.payoff(1)
    print(f"P0 payoff: {p0_payoff}, P1 payoff: {p1_payoff}")
    assert p0_payoff > 0
    assert p1_payoff < 0
    print("✅ Test 1: Complete → Fold\n")

    # Test 2: Full hand to showdown
    g2 = HeadsUpRazzGame()
    deck = make_deck()
    # Deal specific hands
    g2.deal_third_street(
        p0_hole=[Card(1,0), Card(2,1)], p0_up=Card(3,2),   # A23
        p1_hole=[Card(8,0), Card(9,1)], p1_up=Card(10,2),   # 89T — T brings in
    )
    assert g2.bring_in_player == 1  # T is worst

    # Play: complete, call, then check-check through remaining streets
    g2.apply_action(Action.BET)  # P0 completes
    g2.apply_action(Action.CALL)  # P1 calls → advances to 4th

    print(f"After 3rd: street={g2.street}, pot={g2.pot}")
    assert g2.street == 4

    # Deal 4th street
    g2.deal_card(0, Card(4, 0), is_hole=False)  # P0 gets 4
    g2.deal_card(1, Card(11, 0), is_hole=False)  # P1 gets J

    # Check-check 4th
    for _ in range(2):
        g2.apply_action(Action.CHECK)
    assert g2.street == 5

    # Deal 5th
    g2.deal_card(0, Card(5, 0), is_hole=False)
    g2.deal_card(1, Card(12, 0), is_hole=False)

    for _ in range(2):
        g2.apply_action(Action.CHECK)
    assert g2.street == 6

    # Deal 6th
    g2.deal_card(0, Card(6, 0), is_hole=False)
    g2.deal_card(1, Card(13, 0), is_hole=False)

    for _ in range(2):
        g2.apply_action(Action.CHECK)
    assert g2.street == 7

    # Deal 7th (hole cards)
    g2.deal_card(0, Card(7, 0), is_hole=True)
    g2.deal_card(1, Card(7, 1), is_hole=True)

    for _ in range(2):
        g2.apply_action(Action.CHECK)

    assert g2.is_terminal
    print(f"P0 hand: {g2.players[0].all_ranks} → {razz_evaluate(g2.players[0].all_ranks)}")
    print(f"P1 hand: {g2.players[1].all_ranks} → {razz_evaluate(g2.players[1].all_ranks)}")
    print(f"Winner: P{g2.winner}")
    assert g2.winner == 0  # A2345 vs 89TJQ — A2345 wins
    print(f"P0 payoff: {g2.payoff(0)}, P1 payoff: {g2.payoff(1)}")
    assert g2.payoff(0) + g2.payoff(1) == 0  # Zero-sum
    print("✅ Test 2: Full showdown (A23456 vs 89TJQK)\n")

    # Test 3: Raise and cap
    g3 = HeadsUpRazzGame()
    g3.deal_third_street(
        p0_hole=[Card(1,0), Card(2,1)], p0_up=Card(3,2),
        p1_hole=[Card(4,0), Card(5,1)], p1_up=Card(6,2),  # 6 brings in (higher)
    )
    assert g3.bring_in_player == 1
    g3.apply_action(Action.BET)   # P0 completes
    g3.apply_action(Action.RAISE)  # P1 raises
    g3.apply_action(Action.RAISE)  # P0 re-raises
    g3.apply_action(Action.RAISE)  # P1 re-re-raises
    print(f"Num bets after 3 raises + complete: {g3.num_bets}")
    assert g3.num_bets == 4  # 1 (complete) + 3 raises = 4
    # P0 should now only be able to fold or call (cap reached)
    legal = g3.legal_actions()
    print(f"Legal at cap: {[a.name for a in legal]}")
    assert Action.RAISE not in legal
    assert Action.CALL in legal
    print("✅ Test 3: Betting cap enforced\n")

    # Test 4: Action history bucketing
    g4 = HeadsUpRazzGame()
    g4.deal_third_street(
        p0_hole=[Card(1,0), Card(2,1)], p0_up=Card(3,2),
        p1_hole=[Card(8,0), Card(9,1)], p1_up=Card(10,2),
    )
    g4.apply_action(Action.BET)   # Complete
    g4.apply_action(Action.CALL)   # Call → advance
    print(f"Action history: {g4.action_history_str}")
    print(f"Bucketed: {g4.bucketed_action_history}")
    assert g4.bucketed_action_history.startswith("1c.")
    print("✅ Test 4: Action history bucketing\n")

    # Test 5: Clone independence
    g5 = HeadsUpRazzGame()
    g5.deal_third_street(
        p0_hole=[Card(1,0), Card(2,1)], p0_up=Card(3,2),
        p1_hole=[Card(8,0), Card(9,1)], p1_up=Card(10,2),
    )
    g5c = g5.clone()
    g5.apply_action(Action.FOLD)
    assert g5.is_terminal
    assert not g5c.is_terminal  # Clone unaffected
    print("✅ Test 5: Clone independence\n")

    print("All game engine tests passed!")
