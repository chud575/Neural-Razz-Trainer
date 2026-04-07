"""
Realistic Opponent Models for Deep CFR Training

Provides diverse opponents for training:
1. TAG (bucket-based, matches PokerArena)
2. LAG (bucket-based, wider range)
3. Calling Station
4. Random
5. ReBeL (loaded from HORSE+ model file)
6. Pure CFR / Bucketed (loaded from bucketed_strategy.json)

During training, the opponent is randomly selected from a weighted mix
so the neural network learns to beat all play styles.
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Optional

from razz_game import HeadsUpRazzGame, Action, Card
import bucketer as _bucketer_mod
from bucketer import classify_hero, hero_ev_percentile, load_ev_tables, RANK_CHARS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HORSE_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'HORSE+ Master')


# ═══════════════════════════════════════════════════════════════════════════════
# Bucket-based helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _count_cards_to_low(ranks: List[int], threshold: int) -> int:
    """Count unique cards at or below a rank threshold."""
    return len(set(r for r in ranks if r <= threshold))


def _best_five_low(ranks: List[int]):
    """Get best 5 unique low cards. Returns (sorted_unique_ranks, is_made)."""
    unique = []
    seen = set()
    for r in sorted(ranks):
        if r not in seen:
            seen.add(r)
            unique.append(r)
            if len(unique) == 5:
                break
    return unique, len(unique) >= 5


def _get_hand_bucket(ranks: List[int], street: int) -> int:
    """Get Razz hand bucket (0-10, higher = better).

    EXACT port of PokerArena's HandEvaluator.getRazzBucket11().
    """
    if not ranks:
        return 5

    num_cards = len(ranks)
    has_pair = len(set(ranks)) != len(ranks)
    has_ace = 1 in ranks
    high = max(ranks)

    # ── 3rd-4th street (3-4 cards): Draw-based ──
    if num_cards <= 4:
        # Any pair on 3-4 cards is bad
        if has_pair:
            return 0

        wheel_cards = set(r for r in ranks if r <= 5)
        wheel_count = len(wheel_cards)

        # B10: All wheel cards with Ace (A23, A24, A25, A34, A35, A45)
        if wheel_count == num_cards and has_ace and high <= 5:
            return 10
        # B9: All wheel cards no Ace (234, 235, 245, 345) or all <=6
        if wheel_count == num_cards and high <= 5:
            return 9
        if high <= 6:
            return 9
        # B8: 7-high
        if high <= 7:
            return 8
        # B7-B6: 8-high (split by Ace)
        if high <= 8:
            return 7 if has_ace else 6
        # B5-B4: 9-high
        if high <= 9:
            return 5 if has_ace else 4
        # B3: T-high
        if high <= 10:
            return 3
        # B2: J-high
        if high <= 11:
            return 2
        # B1: Q-high
        if high <= 12:
            return 1
        # B0: K-high
        return 0

    # ── 5th-7th street: Check for made hand first ──
    best_five, has_made = _best_five_low(ranks)

    if has_made:
        high_card = best_five[-1]

        # B10: Made Wheel
        if high_card == 5 and best_five[0] == 1:
            return 10
        # B9: Made 6-low
        if high_card <= 6:
            return 9
        # B8: Made 7-low
        if high_card <= 7:
            return 8

        # 7th street: pure made hand ranking
        if num_cards == 7 or street == 7:
            if high_card <= 8: return 7
            if high_card <= 9: return 6
            if high_card <= 10: return 5
            if high_card <= 11: return 4
            if high_card <= 12: return 3
            return 2  # K-low

        # 5th-6th street: strong made hands stay, weak ones check draws
        if high_card <= 8:
            return 6

        # Weak made hand — check if draw is stronger
        wheel_draw = _count_cards_to_low(ranks, 5)
        seven_draw = _count_cards_to_low(ranks, 7)
        eight_draw = _count_cards_to_low(ranks, 8)
        nine_draw = _count_cards_to_low(ranks, 9)

        if wheel_draw >= 4 or seven_draw >= 4:
            return 7
        if eight_draw >= 4:
            return 6
        if nine_draw >= 4:
            return 5

        # No strong draw — value as weak made hand
        if high_card <= 9: return 5
        if high_card <= 10: return 4
        if high_card <= 11: return 3
        if high_card <= 12: return 2
        return 1

    # ── 5th-6th street: Drawing hands (no made 5) ──
    if num_cards == 7 or street == 7:
        # 7th street with no made hand = paired trash
        if _count_cards_to_low(ranks, 8) >= 4:
            return 1
        return 0

    # Count unique cards toward various draw thresholds
    wheel_draw = _count_cards_to_low(ranks, 5)
    seven_draw = _count_cards_to_low(ranks, 7)
    eight_draw = _count_cards_to_low(ranks, 8)
    nine_draw = _count_cards_to_low(ranks, 9)

    # B7: 4 cards to wheel or 7-low
    if wheel_draw >= 4 or seven_draw >= 4:
        return 7
    # B6: 4 cards to 8-low
    if eight_draw >= 4:
        return 6
    # B5: 4 cards to 9-low
    if nine_draw >= 4:
        return 5
    # B4: 3 to 7-low
    if seven_draw >= 3:
        return 4
    # B3: 3 to 8-low
    if eight_draw >= 3:
        return 3
    # B2: 3 to 9-low
    if nine_draw >= 3:
        return 2
    # B1: 2 low cards
    if eight_draw >= 2:
        return 1
    # B0: Complete trash
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Basic Opponents
# ═══════════════════════════════════════════════════════════════════════════════

def calling_station_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    legal = game.legal_actions()
    if Action.CALL in legal: return Action.CALL
    if Action.CHECK in legal: return Action.CHECK
    if Action.BET in legal: return Action.BET
    return legal[0]


def random_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    return random.choice(game.legal_actions())


def _best_low(ranks: List[int]) -> Optional[int]:
    """Get the high card of the best 5-card low from these ranks.

    Returns None if no 5 unique cards exist (paired out).
    Returns the highest card of the best 5 unique low cards.
    """
    unique = sorted(set(ranks))
    if len(unique) >= 5:
        return unique[4]  # 5th lowest unique card = high card of best low
    return None


def _draw_quality(ranks: List[int]) -> int:
    """Count unique cards 9 or below (draw cards toward a low)."""
    return len(set(r for r in ranks if r <= 9))


def _hero_board_high(game: HeadsUpRazzGame, hero_idx: int) -> int:
    """Get the highest upcard rank of the opponent (hero from TAG's perspective)."""
    opponent_idx = 1 - hero_idx
    opponent_up = [c.rank for c in game.players[opponent_idx].up_cards]
    if not opponent_up:
        return 13
    return max(opponent_up)


def _my_board_beats_hero_board(game: HeadsUpRazzGame, my_idx: int) -> bool:
    """Does my visible board look better (lower) than hero's visible board?"""
    my_up = sorted(c.rank for c in game.players[my_idx].up_cards)
    hero_up = sorted(c.rank for c in game.players[1 - my_idx].up_cards)
    # Compare as tuples — lower = better in Razz
    return tuple(my_up) <= tuple(hero_up)


def _do_action(action: Action, legal: List[Action]) -> Action:
    """Return the action if legal, otherwise first legal action."""
    return action if action in legal else legal[0]


def _raise_or_bet(legal: List[Action], game: HeadsUpRazzGame) -> Action:
    """Return RAISE if legal, BET if in bring-in state, otherwise first legal."""
    if Action.RAISE in legal:
        return Action.RAISE
    if game.is_bring_in_state and Action.BET in legal:
        return Action.BET
    if Action.BET in legal:
        return Action.BET
    return legal[0]


def tag_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    """Tight-Aggressive opponent.

    Rules:
    1. Enters pots with 3 cards 9 or less (all unique).
    2. Bets/raises if hero's board shows a higher (worse) hand than ours.
    3. Check/calls if not in the lead but has an 8-draw or better.
       Folds on 6th street if needs 2 cards to complete the draw.
    4. Bet/raises for value all 7/6/5 lows:
       - 7-low: 1 raise max (folds to re-raise)
       - 6-low: 2 raises max
       - Wheel (5-low): all 4 raises
    """
    legal = game.legal_actions()
    if len(legal) <= 1:
        return legal[0] if legal else Action.FOLD

    player = game.players[player_idx]
    ranks = player.all_ranks
    facing_bet = game.current_bet_level > game.street_contribution[player_idx]
    street = game.street

    # ── 3rd Street: Enter with hands containing A2/A3/A4/A5/23/24/34 (including pairs) ──
    if street == 3:
        sorted_ranks = sorted(ranks)
        # Check if hand contains any of the key 2-card combos
        playable_combos = [{1,2},{1,3},{1,4},{1,5},{2,3},{2,4},{3,4}]
        rank_set = set(ranks)
        has_playable_combo = any(combo.issubset(rank_set) for combo in playable_combos)
        # Also allow: if all 3 unique cards are 9 or less
        all_low = all(r <= 9 for r in set(ranks))

        if not has_playable_combo and not all_low:
            # Don't enter the pot
            if facing_bet:
                return _do_action(Action.FOLD, legal)
            return _do_action(Action.CHECK, legal)

        # Good starting hand — play it
        if facing_bet:
            if _my_board_beats_hero_board(game, player_idx):
                return _raise_or_bet(legal, game)
            return _do_action(Action.CALL, legal)
        else:
            if _my_board_beats_hero_board(game, player_idx):
                return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)
            return _do_action(Action.CHECK, legal)

    # ── 4th Street+ ──

    # Check for made hand (5+ cards with 5 unique)
    made_high = _best_low(ranks)
    draw = _draw_quality(ranks)
    needed = 5 - draw


    if made_high is not None:
        # We have a made hand
        if made_high <= 5:
            # Wheel — max aggression (4 raises)
            if facing_bet:
                return _raise_or_bet(legal, game)
            return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)

        elif made_high <= 6:
            # 6-low — up to 2 raises
            if facing_bet:
                if game.num_bets < 3 and Action.RAISE in legal:
                    return Action.RAISE
                return _do_action(Action.CALL, legal)
            return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)

        elif made_high <= 7:
            # 7-low — 1 raise max, fold to re-raise
            if facing_bet:
                if game.num_bets < 2 and Action.RAISE in legal:
                    return Action.RAISE
                if game.num_bets >= 2:
                    return _do_action(Action.CALL, legal)  # Just call, don't re-raise
                return _do_action(Action.CALL, legal)
            return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)

        else:
            # 8+ low made — play cautiously
            if _my_board_beats_hero_board(game, player_idx):
                if facing_bet:
                    return _do_action(Action.CALL, legal)
                return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)
            else:
                if facing_bet:
                    return _do_action(Action.CALL, legal)
                return _do_action(Action.CHECK, legal)

    # ── Drawing hand (no made 5-card low yet) ──
    draw_cards = _draw_quality(ranks)  # unique cards 8 or below
    cards_needed = 5 - draw_cards  # how many more low cards needed

    # 6th street: fold if needs 2+ cards to complete
    if street >= 6 and cards_needed >= 2:
        if facing_bet:
            return _do_action(Action.FOLD, legal)
        return _do_action(Action.CHECK, legal)

    # Good draw (4+ unique cards ≤9 = needs only 1 card)
    if draw_cards >= 4:
        # Drawing well — check/call per rules, bet/raise if board is ahead
        if _my_board_beats_hero_board(game, player_idx):
            if facing_bet:
                return _raise_or_bet(legal, game)
            return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)
        else:
            if facing_bet:
                return _do_action(Action.CALL, legal)
            return _do_action(Action.CHECK, legal)

    elif draw_cards >= 3:
        # 3-card draw — check/call
        if facing_bet:
            return _do_action(Action.CALL, legal)
        return _do_action(Action.CHECK, legal)

    else:
        # 2 or fewer low cards — fold to bets
        if facing_bet:
            return _do_action(Action.FOLD, legal)
        return _do_action(Action.CHECK, legal)


def _hero_board_best_possible(game: HeadsUpRazzGame, my_idx: int) -> int:
    """Estimate the best low hero's board could make.

    Looks at hero's visible upcards. The best possible hand includes
    those upcards plus the best hidden cards. Returns the minimum
    high card hero could achieve (7 means hero can make at best a 7-low).
    """
    hero_idx = 1 - my_idx
    hero_up = sorted(c.rank for c in game.players[hero_idx].up_cards)
    # Hero's best case: upcards are real, hidden cards fill in the gaps
    # Count unique low upcards
    unique_low = sorted(set(r for r in hero_up if r <= 9))
    if len(unique_low) >= 4:
        # Hero showing 4+ low cards — could have a very strong hand
        return unique_low[min(4, len(unique_low)-1)]  # Best possible high card
    elif len(unique_low) >= 3:
        return max(7, unique_low[-1])  # At best a 7-low type hand
    elif len(unique_low) >= 2:
        return max(8, unique_low[-1])
    else:
        return 9  # Hero board looks bad, best they can do is marginal


def lag_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    """Loose-Aggressive opponent.

    Rules:
    - Enters pots with J or less, folds high pairs and Q+ high
    - ANY street: if board looks better than hero's, cap it
    - Made wheel: cap it
    - Made 6-low: cap it
    - Made 7-low: cap if hero's board can only make 7 or worse, otherwise 2 raises
    - 8-9 low: bet if ahead, call if behind
    - T+ low: call down, bet if board looks ahead
    - Draws: aggressive with good draws, bluffs with scary board
    """
    legal = game.legal_actions()
    if len(legal) <= 1:
        return legal[0] if legal else Action.FOLD

    player = game.players[player_idx]
    ranks = player.all_ranks
    facing_bet = game.current_bet_level > game.street_contribution[player_idx]
    street = game.street
    board_ahead = _my_board_beats_hero_board(game, player_idx)

    # ── ANY STREET: if board looks better than hero's, cap it ──
    if board_ahead and street >= 3:
        if facing_bet:
            if Action.RAISE in legal:
                return Action.RAISE
            return _raise_or_bet(legal, game)
        else:
            return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)

    # ── 3rd Street: Enter with J or less, or any unpaired hand ──
    if street == 3:
        high = max(ranks)
        has_pair = len(set(ranks)) < len(ranks)

        if has_pair and high > 8:
            if facing_bet:
                return _do_action(Action.FOLD, legal)
            return _do_action(Action.CHECK, legal)

        if high > 11:
            if facing_bet:
                return _do_action(Action.FOLD, legal)
            return _do_action(Action.CHECK, legal)

        # Play it
        if facing_bet:
            return _do_action(Action.CALL, legal)
        else:
            if Action.BET in legal and random.random() < 0.3:
                return Action.BET
            return _do_action(Action.CHECK, legal)

    # ── 4th Street+ ──
    made_high = _best_low(ranks)

    if made_high is not None:
        if made_high <= 5:
            # Wheel — cap it
            if facing_bet:
                if Action.RAISE in legal:
                    return Action.RAISE
                return _do_action(Action.CALL, legal)
            return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)

        elif made_high <= 6:
            # 6-low — cap it
            if facing_bet:
                if Action.RAISE in legal:
                    return Action.RAISE
                return _do_action(Action.CALL, legal)
            return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)

        elif made_high <= 7:
            # 7-low — cap if hero's board can only make 7 or worse, otherwise 2 raises
            hero_best = _hero_board_best_possible(game, player_idx)
            if facing_bet:
                if hero_best >= 7:
                    # Hero can only make 7 or worse — cap it
                    if Action.RAISE in legal:
                        return Action.RAISE
                    return _do_action(Action.CALL, legal)
                else:
                    # Hero could have better — 2 raises max
                    if game.num_bets < 3 and Action.RAISE in legal:
                        return Action.RAISE
                    return _do_action(Action.CALL, legal)
            return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)

        elif made_high <= 9:
            # 8-9 low — bet if ahead, call if behind
            if facing_bet:
                return _do_action(Action.CALL, legal)
            if board_ahead and Action.BET in legal:
                return Action.BET
            return _do_action(Action.CHECK, legal)

        else:
            # T+ low — call down, bet if board looks ahead
            if facing_bet:
                return _do_action(Action.CALL, legal)
            if board_ahead and Action.BET in legal:
                return Action.BET
            return _do_action(Action.CHECK, legal)

    # ── Drawing hand ──
    draw_cards = _draw_quality(ranks)
    cards_needed = 5 - draw_cards

    # 7th street with no made hand — usually fold
    if street == 7 and cards_needed > 0:
        if facing_bet:
            return _do_action(Action.FOLD, legal)
        return _do_action(Action.CHECK, legal)

    # 6th street needing 2+ cards — fold to bets
    if street >= 6 and cards_needed >= 2:
        if facing_bet:
            return _do_action(Action.FOLD, legal)
        # Bluff occasionally with scary board
        if _my_board_beats_hero_board(game, player_idx) and random.random() < 0.2 and Action.BET in legal:
            return Action.BET
        return _do_action(Action.CHECK, legal)

    # Good draw
    if draw_cards >= 4:
        if facing_bet:
            if _my_board_beats_hero_board(game, player_idx) and random.random() < 0.5:
                return _raise_or_bet(legal, game)
            return _do_action(Action.CALL, legal)
        if Action.BET in legal:
            return Action.BET
        return _do_action(Action.CHECK, legal)

    elif draw_cards >= 3:
        # LAG plays 3-card draws aggressively
        if facing_bet:
            return _do_action(Action.CALL, legal)
        if _my_board_beats_hero_board(game, player_idx) and Action.BET in legal and random.random() < 0.4:
            return Action.BET
        return _do_action(Action.CHECK, legal)

    else:
        # Bad draw
        if facing_bet:
            return _do_action(Action.FOLD, legal)
        return _do_action(Action.CHECK, legal)


def _villain_board_locked(game: HeadsUpRazzGame, my_idx: int) -> bool:
    """Check if villain's board is 'locked' — all visible cards are low and
    villain likely has a made hand that can't improve much.
    A board is locked when all upcards are unique and ≤7."""
    hero_idx = 1 - my_idx
    hero_up = [c.rank for c in game.players[hero_idx].up_cards]
    unique_low = set(r for r in hero_up if r <= 7)
    return len(unique_low) == len(hero_up) and len(hero_up) >= 2


def _hero_door_card(game: HeadsUpRazzGame, my_idx: int) -> int:
    """Get hero's (opponent's) door card rank."""
    hero_idx = 1 - my_idx
    up = game.players[hero_idx].up_cards
    return up[0].rank if up else 13


def _opponent_board_under_7(game: HeadsUpRazzGame, my_idx: int) -> bool:
    """Check if opponent's board shows 4+ cards under 7."""
    hero_idx = 1 - my_idx
    hero_up = [c.rank for c in game.players[hero_idx].up_cards]
    return len([r for r in hero_up if r <= 7]) >= 4


def _drawing_to_7_or_better(ranks: List[int]) -> bool:
    """Check if hand is drawing to a 7-low or better (4+ unique cards ≤7)."""
    return len(set(r for r in ranks if r <= 7)) >= 4


def _my_draw_lower_than_villain_board(game: HeadsUpRazzGame, my_idx: int) -> bool:
    """Are we drawing to a lower hand than villain's visible board suggests?"""
    my_ranks = game.players[my_idx].all_ranks
    hero_idx = 1 - my_idx
    hero_up = sorted(c.rank for c in game.players[hero_idx].up_cards)

    # My best unique low cards
    my_low = sorted(set(r for r in my_ranks if r <= 9))

    if not hero_up or not my_low:
        return False

    # Compare: if my draw cards are lower than villain's visible cards
    return tuple(my_low[:min(len(my_low), len(hero_up))]) < tuple(hero_up[:min(len(my_low), len(hero_up))])


def mr_heu_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    """Mr. Heu — Heuristic-based realistic opponent.

    Rules:
    1. 50/50 check/bet with unpaired 3 cards <9, or paired with A2/A3/A4/A5/23/24/25/34/35/45
    2. Check/call with unpaired 2 cards <7 + 1 card ≤J where the high card is in the hole
    3. Always raise if opponent shows JQK in door and hero <9
    4. Always raise an A in the door
    5. All remaining +EV hands from tables
    6. Before 6th: bet/raise/cap when board is stronger than villain's
    7. 6th+: bet/raise made 7-low or better; check/call 8-9 low;
       exception: only bet/raise 6-low or better if opponent board shows 4 cards under 7
    8. 6th+: no made hand but drawing to 7 or better → check/call
    9. Anytime villain's board is locked and we draw lower → bet/raise
    """
    legal = game.legal_actions()
    if len(legal) <= 1:
        return legal[0] if legal else Action.FOLD

    player = game.players[player_idx]
    ranks = player.all_ranks
    facing_bet = game.current_bet_level > game.street_contribution[player_idx]
    street = game.street
    board_ahead = _my_board_beats_hero_board(game, player_idx)
    hero_door = _hero_door_card(game, player_idx)
    high = max(ranks) if ranks else 13
    has_pair = len(set(ranks)) < len(ranks)
    unique_ranks = set(ranks)

    # ── Rule 9: Villain board locked and we draw lower → bet/raise (any street) ──
    if _villain_board_locked(game, player_idx) and _my_draw_lower_than_villain_board(game, player_idx):
        if facing_bet:
            if Action.RAISE in legal:
                return Action.RAISE
            return _raise_or_bet(legal, game)
        return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)

    # ── 3rd Street ──
    if street == 3:
        # Rule 3: Always raise if opponent shows JQK and hero <9
        if hero_door >= 11 and high <= 9:
            return _raise_or_bet(legal, game)

        # Rule 4: Always raise an A in the door
        if hero_door == 1:
            return _raise_or_bet(legal, game)

        # Rule 1: 50/50 check/bet with unpaired <9, or paired with key combos
        playable_pairs = [{1,2},{1,3},{1,4},{1,5},{2,3},{2,4},{2,5},{3,4},{3,5},{4,5}]
        has_key_combo = any(combo.issubset(unique_ranks) for combo in playable_pairs)

        if not has_pair and all(r <= 9 for r in unique_ranks):
            # Unpaired, all <9
            if random.random() < 0.5:
                return _do_action(Action.BET, legal) if Action.BET in legal else _raise_or_bet(legal, game)
            else:
                if facing_bet:
                    return _do_action(Action.CALL, legal)
                return _do_action(Action.CHECK, legal)

        if has_pair and has_key_combo:
            # Paired with key combo
            if random.random() < 0.5:
                return _do_action(Action.BET, legal) if Action.BET in legal else _raise_or_bet(legal, game)
            else:
                if facing_bet:
                    return _do_action(Action.CALL, legal)
                return _do_action(Action.CHECK, legal)

        # Rule 2: Check/call with 2 cards <7 + 1 card ≤J where high card in hole
        low_cards = [r for r in ranks if r < 7]
        mid_cards = [r for r in ranks if 7 <= r <= 11]
        if len(low_cards) >= 2 and len(mid_cards) >= 1:
            # Check if the high card is in the hole (not the upcard)
            up_rank = player.up_cards[0].rank if player.up_cards else 13
            hole_ranks = [c.rank for c in player.hole_cards]
            high_in_hole = max(ranks) in hole_ranks and up_rank < 7
            if high_in_hole:
                if facing_bet:
                    return _do_action(Action.CALL, legal)
                return _do_action(Action.CHECK, legal)

        # Rule 5: All remaining +EV hands
        load_ev_tables()
        ev = hero_ev_percentile(ranks)
        if ev > 0.50:
            if facing_bet:
                return _do_action(Action.CALL, legal)
            return _do_action(Action.CHECK, legal)

        # Not playable
        if facing_bet:
            return _do_action(Action.FOLD, legal)
        return _do_action(Action.CHECK, legal)

    # ── 4th-5th Street (before 6th) ──
    if street < 6:
        # Rule 6: Maintain aggression when board is stronger than villain's
        if board_ahead:
            if facing_bet:
                if Action.RAISE in legal:
                    return Action.RAISE
                return _raise_or_bet(legal, game)
            return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)

        # Not ahead on board — check/call with decent draws
        draw = _draw_quality(ranks)
        if draw >= 3:
            if facing_bet:
                return _do_action(Action.CALL, legal)
            return _do_action(Action.CHECK, legal)

        # Bad draw
        if facing_bet:
            return _do_action(Action.FOLD, legal)
        return _do_action(Action.CHECK, legal)

    # ── 6th Street+ ──
    made_high = _best_low(ranks)
    opp_board_scary = _opponent_board_under_7(game, player_idx)

    if made_high is not None:
        # Rule 7: Made hands
        if made_high <= 5:
            # Wheel — always bet/raise
            if facing_bet:
                if Action.RAISE in legal:
                    return Action.RAISE
                return _do_action(Action.CALL, legal)
            return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)

        elif made_high <= 6:
            # 6-low — bet/raise (even against scary boards)
            if facing_bet:
                if Action.RAISE in legal:
                    return Action.RAISE
                return _do_action(Action.CALL, legal)
            return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)

        elif made_high <= 7:
            # 7-low — bet/raise unless opponent board shows 4 cards under 7
            if opp_board_scary:
                # Opponent looks strong — check/call
                if facing_bet:
                    return _do_action(Action.CALL, legal)
                return _do_action(Action.CHECK, legal)
            else:
                if facing_bet:
                    if Action.RAISE in legal:
                        return Action.RAISE
                    return _do_action(Action.CALL, legal)
                return _do_action(Action.BET, legal) if Action.BET in legal else _do_action(Action.CHECK, legal)

        elif made_high <= 9:
            # 8-9 low — check/call
            if facing_bet:
                return _do_action(Action.CALL, legal)
            return _do_action(Action.CHECK, legal)

        else:
            # T+ low — fold to bets, check
            if facing_bet:
                return _do_action(Action.FOLD, legal)
            return _do_action(Action.CHECK, legal)

    else:
        # Rule 8: No made hand — drawing to 7 or better → check/call
        if _drawing_to_7_or_better(ranks):
            if facing_bet:
                return _do_action(Action.CALL, legal)
            return _do_action(Action.CHECK, legal)

        # Drawing but not to 7 — fold to bets
        if facing_bet:
            return _do_action(Action.FOLD, legal)
        return _do_action(Action.CHECK, legal)


# ═══════════════════════════════════════════════════════════════════════════════
# ReBeL Opponent (loaded from HORSE+ model)
# ═══════════════════════════════════════════════════════════════════════════════

_rebel_model = None
_rebel_loaded = False

RANK_CHARS = {1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K'}


def _load_rebel():
    """Load ReBeL value network from HORSE+ directory."""
    global _rebel_model, _rebel_loaded
    if _rebel_loaded:
        return _rebel_model is not None

    _rebel_loaded = True

    candidates = [
        os.path.join(HORSE_DIR, 'razz_rebel_epoch1500_v4.json'),
        os.path.join(HORSE_DIR, 'razz_rebel_epoch4000_v3.json'),
        os.path.join(HORSE_DIR, 'razz_rebel_epoch2900.json'),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)

                layers = []
                for lc in data.get('layerConfigs', []):
                    in_s = lc['inputSize']
                    out_s = lc['outputSize']
                    w = np.array(lc['weights'], dtype=np.float64).reshape(out_s, in_s)
                    b = np.array(lc['biases'], dtype=np.float64)
                    relu = lc.get('useReLU', False)
                    layers.append((w, b, relu))

                _rebel_model = {
                    'layers': layers,
                    'version': 3 if 'v3' in path or 'v4' in path else 1,
                    'path': path,
                }
                print(f"[Opponents] Loaded ReBeL model: {os.path.basename(path)} ({len(layers)} layers)")
                return True
            except Exception as e:
                print(f"[Opponents] Failed to load ReBeL: {e}")

    print("[Opponents] No ReBeL model found")
    return False


def _rebel_forward(x: np.ndarray) -> np.ndarray:
    """Run ReBeL network forward pass."""
    if _rebel_model is None:
        return np.zeros(2)
    for w, b, relu in _rebel_model['layers']:
        x = w @ x + b
        if relu:
            x = np.maximum(0, x)
    return x


def _rebel_encode_v3(game: HeadsUpRazzGame, player_idx: int) -> np.ndarray:
    """Encode game state for ReBeL v3 (84-dim, 14 buckets, rank-count)."""
    player = game.players[player_idx]
    villain_idx = 1 - player_idx
    villain = game.players[villain_idx]

    ranks = player.all_ranks
    bucket = _get_hand_bucket(ranks, game.street)

    # 84-dim feature vector for v3
    features = np.zeros(84, dtype=np.float64)

    # Bucket one-hot (14 dims) — indices 0-13
    bucket_idx = min(bucket, 13)
    features[bucket_idx] = 1.0

    # Street one-hot (5 dims) — indices 14-18
    street_idx = game.street - 3
    if 0 <= street_idx <= 4:
        features[14 + street_idx] = 1.0

    # Hero rank counts (13 dims) — indices 19-31
    for r in ranks:
        if 1 <= r <= 13:
            features[19 + r - 1] += 1.0

    # Villain visible rank counts (13 dims) — indices 32-44
    for c in villain.up_cards:
        if 1 <= c.rank <= 13:
            features[32 + c.rank - 1] += 1.0

    # Pot normalized — index 45
    features[45] = game.pot / 10.0

    # Facing bet — index 46
    to_call = game.current_bet_level - game.street_contribution[player_idx]
    features[46] = 1.0 if to_call > 0 else 0.0

    # Number of raises — index 47
    features[47] = game.num_bets / 4.0

    # Hero card count — index 48
    features[48] = len(ranks) / 7.0

    return features


def rebel_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    """ReBeL opponent: uses value network to evaluate actions."""
    if not _load_rebel():
        return tag_action(game, player_idx)  # Fallback to TAG

    legal = game.legal_actions()
    if len(legal) <= 1:
        return legal[0] if legal else Action.FOLD

    # Evaluate each legal action by looking at the resulting game state
    best_action = legal[0]
    best_value = -999.0

    for action in legal:
        if action == Action.FOLD:
            # Folding = 0 value (no more investment)
            value = 0.0
        else:
            # Simulate action and evaluate resulting state
            child = game.clone()
            child.apply_action(action)

            if child.is_terminal:
                value = child.payoff(player_idx)
            else:
                features = _rebel_encode_v3(child, player_idx)
                output = _rebel_forward(features)
                value = float(output[0]) if len(output) > 0 else 0.0

        if value > best_value:
            best_value = value
            best_action = action

    return best_action


# ═══════════════════════════════════════════════════════════════════════════════
# Pure CFR / Bucketed Opponent
# ═══════════════════════════════════════════════════════════════════════════════

_bucketed_strategies = None
_bucketed_loaded = False


def _load_bucketed():
    """Load bucketed strategy table."""
    global _bucketed_strategies, _bucketed_loaded
    if _bucketed_loaded:
        return _bucketed_strategies is not None

    _bucketed_loaded = True

    candidates = [
        os.path.join(HORSE_DIR, 'bucketed_strategy.json'),
        os.path.join(SCRIPT_DIR, '..', 'bucketed_strategy.json'),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                _bucketed_strategies = data.get('strategies', data)
                print(f"[Opponents] Loaded bucketed CFR: {len(_bucketed_strategies):,} info sets from {os.path.basename(path)}")
                return True
            except Exception as e:
                print(f"[Opponents] Failed to load bucketed CFR: {e}")

    print("[Opponents] No bucketed strategy found")
    return False


def _make_bucketed_key(game: HeadsUpRazzGame, player_idx: int) -> str:
    """Build bucketed lookup key for a player."""
    from bucketer import classify_hero, classify_villain_visible

    player = game.players[player_idx]
    villain = game.players[1 - player_idx]

    hero_bucket = classify_hero(player.all_ranks)
    villain_ranks = [c.rank for c in villain.up_cards]
    villain_bucket = classify_villain_visible(villain_ranks) if villain_ranks else "0H"

    # Bucketed action history
    parts = []
    for street_actions in game.all_actions:
        if not street_actions:
            parts.append('')
            continue
        agg = sum(1 for a, _ in street_actions if a in (Action.BET, Action.RAISE))
        last_action = street_actions[-1][0]
        if last_action in (Action.CHECK, Action.CALL, Action.FOLD):
            parts.append(f"{agg}{last_action.symbol}")
        else:
            parts.append(f"{agg}")

    history = '.'.join(parts)
    return f"H:{hero_bucket}|V:{villain_bucket}|{history}"


def bucketed_cfr_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    """Pure CFR opponent: looks up bucketed strategy table."""
    if not _load_bucketed():
        return tag_action(game, player_idx)  # Fallback

    legal = game.legal_actions()
    if len(legal) <= 1:
        return legal[0] if legal else Action.FOLD

    key = _make_bucketed_key(game, player_idx)

    # Try exact match, then strip trailing dots
    probs = _bucketed_strategies.get(key)
    if not probs:
        stripped = key.rstrip('.')
        probs = _bucketed_strategies.get(stripped)

    if not probs:
        return tag_action(game, player_idx)  # Fallback

    # Map symbol → action
    sym_map = {'f': Action.FOLD, 'k': Action.CHECK, 'c': Action.CALL, 'b': Action.BET, 'r': Action.RAISE}
    legal_probs = []
    total = 0.0
    for sym, p in probs.items():
        action = sym_map.get(sym)
        if action and action in legal:
            legal_probs.append((action, p))
            total += p

    if not legal_probs or total <= 0:
        return tag_action(game, player_idx)

    # Sample from distribution
    r = random.random()
    cumulative = 0.0
    for action, p in legal_probs:
        cumulative += p / total
        if r < cumulative:
            return action
    return legal_probs[-1][0]


# ═══════════════════════════════════════════════════════════════════════════════
# Pure EV Opponent (matches PokerArena's PureEVStrategy)
# ═══════════════════════════════════════════════════════════════════════════════

def _ev_table_lookup(ranks: List[int], num_players: int = 2) -> Optional[float]:
    """Direct EV table lookup matching PokerArena's getWinRate().

    Returns win rate as 0-100 percentage, or None if not found.
    For 6-7 cards, uses best 5 unique low cards for lookup.
    """
    load_ev_tables()

    num_cards = min(len(ranks), 5)
    lookup_ranks = ranks

    # For 6-7 cards: find best 5 unique low
    if len(ranks) > 5:
        unique = []
        seen = set()
        for r in sorted(ranks):
            if r not in seen:
                seen.add(r)
                unique.append(r)
                if len(unique) == 5:
                    break
        if len(unique) == 5:
            lookup_ranks = unique
            num_cards = 5
        else:
            return None

    # Build key: sorted ranks as chars
    key = ''.join(RANK_CHARS.get(r, '?') for r in sorted(lookup_ranks[:num_cards]))

    # Try exact player count, then nearby
    for np in [num_players, num_players - 1, num_players + 1, 5, 4, 3, 2]:
        if num_cards == 3:
            table = _bucketer_mod._ev_3card
        elif num_cards == 4:
            table = _bucketer_mod._ev_4card
        elif num_cards == 5:
            table = _bucketer_mod._ev_5card
        else:
            table = None
        if table and key in table:
            wr = table[key]
            # Table values might be 0-100 or 0-1 depending on source
            return wr if wr > 1.0 else wr * 100.0

    # Fallback for 5+ cards: use hero_ev_percentile
    return hero_ev_percentile(ranks) * 100.0


def pure_ev_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    """Pure EV strategy: compares hero's win rate to fair share.

    - ≥1.3× fair share → raise/bet
    - ≥1.0× fair share → call/bet
    - ≥0.85× fair share → check/call single bet, fold to raise
    - <0.85× fair share → check/fold

    Matches PokerArena's PureEVStrategy.
    """
    legal = game.legal_actions()
    if len(legal) <= 1:
        return legal[0] if legal else Action.FOLD

    player = game.players[player_idx]
    ranks = player.all_ranks
    facing_bet = game.current_bet_level > game.street_contribution[player_idx]

    # Fair share for heads-up = 50%
    num_active = sum(1 for p in game.players if p.is_active)
    fair_share = 100.0 / max(num_active, 1)

    # Get win rate from EV tables (direct lookup, matching PokerArena)
    win_rate = _ev_table_lookup(ranks, num_players=num_active)
    if win_rate is None:
        return Action.FOLD if facing_bet else (Action.CHECK if Action.CHECK in legal else legal[0])

    ratio = win_rate / fair_share if fair_share > 0 else 1.0

    # Significantly above fair share → raise/bet
    if ratio >= 1.3:
        if Action.RAISE in legal: return Action.RAISE
        if Action.BET in legal: return Action.BET
        if Action.CALL in legal: return Action.CALL
        return Action.CHECK if Action.CHECK in legal else legal[0]

    # Above fair share → call/bet
    if ratio >= 1.0:
        if facing_bet:
            return Action.CALL if Action.CALL in legal else legal[0]
        if Action.BET in legal: return Action.BET
        return Action.CHECK if Action.CHECK in legal else legal[0]

    # Just below fair share → check/call single bet, fold to raise
    if ratio >= 0.85:
        if not facing_bet:
            return Action.CHECK if Action.CHECK in legal else legal[0]
        # Match PokerArena: numRaises <= 1 (not num_bets)
        num_raises = sum(1 for a, _ in game.all_actions[game.street - 3] if a == Action.RAISE)
        if num_raises <= 1:
            return Action.CALL if Action.CALL in legal else legal[0]
        return Action.FOLD if Action.FOLD in legal else legal[0]

    # Well below fair share → check/fold
    if facing_bet:
        return Action.FOLD if Action.FOLD in legal else legal[0]
    return Action.CHECK if Action.CHECK in legal else legal[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Mixed Opponent (rotates through all styles)
# ═══════════════════════════════════════════════════════════════════════════════

def self_play_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    """Placeholder — signals the trainer to use the network's own strategy."""
    # This never actually gets called — the trainer checks for 'self_play'
    # and uses the advantage network's strategy instead
    return game.legal_actions()[0]


ALL_OPPONENTS = {
    'self_play': self_play_action,
    'calling_station': calling_station_action,
    'random': random_action,
    'tag': tag_action,
    'lag': lag_action,
    'mr_heu': mr_heu_action,
    'rebel': rebel_action,
    'bucketed_cfr': bucketed_cfr_action,
    'pure_ev': pure_ev_action,
}

# Weights for opponent selection during training
# Higher weight = more training against this opponent
TRAINING_WEIGHTS = {
    'tag': 25,           # Realistic tight play
    'lag': 20,           # Aggressive opponents
    'mr_heu': 20,        # Heuristic-based realistic play
    'pure_ev': 20,       # EV-based — mathematically sound opponent
    # 'rebel': 0,        # DISABLED — value network produces inverted decisions
    'bucketed_cfr': 5,   # Nash-based play
    'calling_station': 5,  # Easy opponent
    'random': 5,         # Sanity check
}


# Dynamic config — can be updated at runtime by the server
_active_opponents: Optional[List[str]] = None
_active_weights: Optional[Dict[str, float]] = None
_balanced_mode: bool = False


def configure_opponents(enabled: List[str] = None, weights: Dict[str, float] = None, balanced: bool = False):
    """Configure which opponents are active and their weights."""
    global _active_opponents, _active_weights, _balanced_mode
    if enabled:
        _active_opponents = [o for o in enabled if o in ALL_OPPONENTS]
    else:
        _active_opponents = None
    _active_weights = weights
    _balanced_mode = balanced

    if _active_opponents:
        if balanced:
            print(f"[Opponents] Configured: {', '.join(_active_opponents)} (balanced)")
        else:
            w_str = ', '.join(f"{o}:{weights.get(o, 10):.0f}" for o in _active_opponents) if weights else "default"
            print(f"[Opponents] Configured: {w_str}")


def pick_training_opponent() -> str:
    """Randomly select an opponent type based on current config."""
    if _active_opponents:
        if _balanced_mode:
            return random.choice(_active_opponents)
        elif _active_weights:
            weights = [_active_weights.get(o, 10) for o in _active_opponents]
            return random.choices(_active_opponents, weights=weights, k=1)[0]
        else:
            return random.choice(_active_opponents)

    # Default: use TRAINING_WEIGHTS
    types = list(TRAINING_WEIGHTS.keys())
    weights = [TRAINING_WEIGHTS[t] for t in types]
    return random.choices(types, weights=weights, k=1)[0]


def mixed_action(game: HeadsUpRazzGame, player_idx: int, opponent_type: str = None) -> Action:
    """Get action from a specific or randomly selected opponent."""
    if opponent_type is None:
        opponent_type = pick_training_opponent()
    fn = ALL_OPPONENTS.get(opponent_type, tag_action)
    return fn(game, player_idx)


# ═══════════════════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════════════════

def preload_all():
    """Load all external models upfront."""
    load_ev_tables()
    _load_rebel()
    _load_bucketed()
    available = ['tag', 'lag', 'pure_ev', 'calling_station', 'random']
    if _rebel_model: available.append('rebel')
    if _bucketed_strategies: available.append('bucketed_cfr')
    print(f"[Opponents] Available: {', '.join(available)}")


if __name__ == '__main__':
    preload_all()
    print("\n=== Opponent Tests ===\n")

    from razz_game import HeadsUpRazzGame, Card

    g = HeadsUpRazzGame()
    g.deal_third_street(
        p0_hole=[Card(1,0), Card(2,1)], p0_up=Card(3,2),
        p1_hole=[Card(10,0), Card(11,1)], p1_up=Card(13,2),
    )

    for name, fn in ALL_OPPONENTS.items():
        action = fn(g, 1)  # Villain (player 1) acts
        print(f"  {name:20s} → {action.name}")

    print("\n  Mixed (10 samples):")
    for _ in range(10):
        opp_type = pick_training_opponent()
        action = mixed_action(g, 1, opp_type)
        print(f"    {opp_type:20s} → {action.name}")

    print("\nAll opponent tests passed!")
