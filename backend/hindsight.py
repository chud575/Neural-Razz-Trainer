"""
Hindsight Correction for Deep CFR

After each MCCFR traversal, replays the hand with perfect information
to identify "missed bets" — situations where the CFR strategy was too
passive given the actual cards.

The correction samples are added to the advantage reservoir with
amplified weight, teaching the network to be more aggressive when
the math supports it.
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from razz_game import HeadsUpRazzGame, Card, Action
from razz_eval import evaluate as razz_evaluate, penalize_pairs
from features import extract_features


# ─── Razz Equity Calculator (Perfect Information) ───────────────────────────

def compute_equity(hero_ranks: List[int], villain_ranks: List[int],
                   remaining_deck_ranks: List[int], street: int) -> float:
    """Compute hero's equity (probability of winning) with perfect information.

    Args:
        hero_ranks: Hero's current card ranks (all known)
        villain_ranks: Villain's current card ranks (all known — hindsight)
        remaining_deck_ranks: Ranks still in the deck (for draw-out calc)
        street: Current street (3-7)

    Returns:
        Float 0.0 to 1.0 — hero's probability of winning at showdown.
    """
    hero_cards_left = 7 - len(hero_ranks)
    villain_cards_left = 7 - len(villain_ranks)

    if hero_cards_left == 0 and villain_cards_left == 0:
        # Both complete — direct comparison
        hero_score = razz_evaluate(hero_ranks)[1]
        villain_score = razz_evaluate(villain_ranks)[1]
        if hero_score < villain_score:
            return 1.0
        elif hero_score > villain_score:
            return 0.0
        else:
            return 0.5  # Tie

    if hero_cards_left + villain_cards_left > 3:
        # Too many cards to enumerate — use Monte Carlo sampling
        # Fewer samples for early streets (speed) — more for later (accuracy)
        samples = 50 if hero_cards_left >= 3 else (100 if hero_cards_left >= 2 else 200)
        return _mc_equity(hero_ranks, villain_ranks, remaining_deck_ranks,
                          hero_cards_left, villain_cards_left, samples=samples)

    # Enumerate all possible completions
    return _enumerate_equity(hero_ranks, villain_ranks, remaining_deck_ranks,
                             hero_cards_left, villain_cards_left)


def _enumerate_equity(hero_ranks, villain_ranks, deck_ranks,
                      hero_left, villain_left) -> float:
    """Exact equity by enumerating all possible runouts."""
    total_cards_needed = hero_left + villain_left
    if total_cards_needed == 0:
        h_score = razz_evaluate(hero_ranks)[1]
        v_score = razz_evaluate(villain_ranks)[1]
        return 1.0 if h_score < v_score else (0.0 if h_score > v_score else 0.5)

    wins = 0.0
    total = 0

    # For small number of cards, enumerate
    if total_cards_needed == 1:
        for r in deck_ranks:
            if hero_left > 0:
                h = hero_ranks + [r]
                v = villain_ranks
            else:
                h = hero_ranks
                v = villain_ranks + [r]
            h_score = razz_evaluate(h)[1]
            v_score = razz_evaluate(v)[1]
            if h_score < v_score:
                wins += 1.0
            elif h_score == v_score:
                wins += 0.5
            total += 1
    elif total_cards_needed == 2:
        for i in range(len(deck_ranks)):
            for j in range(i + 1, len(deck_ranks)):
                r1, r2 = deck_ranks[i], deck_ranks[j]
                if hero_left == 2:
                    h = hero_ranks + [r1, r2]
                    v = villain_ranks
                elif hero_left == 1:
                    h = hero_ranks + [r1]
                    v = villain_ranks + [r2]
                    # Also try the reverse
                    h2 = hero_ranks + [r2]
                    v2 = villain_ranks + [r1]
                    # Both are equally likely
                    for hh, vv in [(h, v), (h2, v2)]:
                        h_score = razz_evaluate(hh)[1]
                        v_score = razz_evaluate(vv)[1]
                        if h_score < v_score:
                            wins += 1.0
                        elif h_score == v_score:
                            wins += 0.5
                        total += 1
                    continue
                else:
                    h = hero_ranks
                    v = villain_ranks + [r1, r2]
                h_score = razz_evaluate(h)[1]
                v_score = razz_evaluate(v)[1]
                if h_score < v_score:
                    wins += 1.0
                elif h_score == v_score:
                    wins += 0.5
                total += 1
    else:
        # Fall back to MC for 3+ cards
        return _mc_equity(hero_ranks, villain_ranks, deck_ranks,
                          hero_left, villain_left, samples=1000)

    return wins / max(total, 1)


def _mc_equity(hero_ranks, villain_ranks, deck_ranks,
               hero_left, villain_left, samples=500) -> float:
    """Monte Carlo equity estimation."""
    wins = 0.0
    for _ in range(samples):
        pool = list(deck_ranks)
        random.shuffle(pool)
        h = hero_ranks + pool[:hero_left]
        v = villain_ranks + pool[hero_left:hero_left + villain_left]
        h_score = razz_evaluate(h)[1]
        v_score = razz_evaluate(v)[1]
        if h_score < v_score:
            wins += 1.0
        elif h_score == v_score:
            wins += 0.5
    return wins / samples


# ─── Outs Calculator ────────────────────────────────────────────────────────

@dataclass
class OutsInfo:
    """Detailed outs breakdown for a player."""
    total_outs: int
    improving_ranks: List[int]  # Ranks that improve the hand
    equity: float               # Win probability


def count_outs(player_ranks: List[int], opponent_ranks: List[int],
               deck_ranks: List[int]) -> OutsInfo:
    """Count hero's outs — cards that improve hero's hand relative to opponent.

    An 'out' is a card where, if hero draws it, hero's hand becomes better
    than opponent's current best hand.
    """
    current_hero_best = razz_evaluate(player_ranks)[1]
    current_opp_best = razz_evaluate(opponent_ranks)[1]

    improving_ranks = []
    for rank in set(deck_ranks):
        new_hand = player_ranks + [rank]
        new_score = razz_evaluate(new_hand)[1]
        if new_score < current_opp_best:
            improving_ranks.append(rank)

    # Count actual cards (not just unique ranks) that are outs
    total_outs = sum(1 for r in deck_ranks if r in improving_ranks)

    equity = total_outs / max(len(deck_ranks), 1)

    return OutsInfo(
        total_outs=total_outs,
        improving_ranks=sorted(improving_ranks),
        equity=equity,
    )


# ─── Hindsight Pass ─────────────────────────────────────────────────────────

@dataclass
class HindsightCorrection:
    """A correction sample from the hindsight pass."""
    features: List[float]        # Game state features at the decision point
    advantages: List[float]      # Corrected advantages (5 actions)
    street: int
    hero_equity: float
    correction_type: str         # 'missed_bet', 'bad_fold', 'good_fold'


def hindsight_pass(game_history: List[dict], deal_info: dict,
                   hero_seat: int) -> List[HindsightCorrection]:
    """Replay a completed hand with perfect information.

    Args:
        game_history: List of decision points from the traversal.
            Each: {'game': HeadsUpRazzGame, 'action_taken': Action, 'strategy': dict}
        deal_info: Full deal information:
            {'hero_ranks': [int], 'villain_ranks': [int], 'all_deck_ranks': [int]}
        hero_seat: Which player is hero (0 or 1)

    Returns:
        List of HindsightCorrection samples for decision points where
        the CFR strategy was significantly wrong.
    """
    corrections = []

    hero_all_ranks = deal_info['hero_ranks']
    villain_all_ranks = deal_info['villain_ranks']
    all_used_ranks = hero_all_ranks + villain_all_ranks

    # Remaining deck ranks (remove all dealt cards)
    full_deck_ranks = []
    used_counts = {}
    for r in all_used_ranks:
        used_counts[r] = used_counts.get(r, 0) + 1
    for r in range(1, 14):
        available = 4 - used_counts.get(r, 0)
        full_deck_ranks.extend([r] * available)

    for node in game_history:
        game = node['game']
        strategy = node['strategy']
        acting = game.current_player

        if acting != hero_seat:
            continue  # Only correct hero's decisions

        street = game.street

        # Skip 3rd street — too early, equity is noisy with 8 cards to come
        if street <= 3:
            continue
        hero = game.players[hero_seat]
        villain = game.players[1 - hero_seat]

        hero_current = hero.all_ranks
        villain_current = villain.all_ranks

        # Compute perfect-info equity at this node
        # Remaining deck = full deck minus cards dealt SO FAR at this street
        cards_seen = hero_current + villain_current
        deck_remaining = list(full_deck_ranks)  # Copy
        seen_counts = {}
        for r in cards_seen:
            seen_counts[r] = seen_counts.get(r, 0) + 1
        remaining = []
        avail = {}
        for r in range(1, 14):
            count = 4 - used_counts.get(r, 0)  # Total available in this deal
            already_dealt = seen_counts.get(r, 0) - (1 if r in hero_all_ranks[:len(hero_current)] else 0)
            # Simpler: just use the full info
            pass

        # Use the full deal to compute equity
        equity = compute_equity(
            hero_all_ranks[:len(hero_current)],
            villain_all_ranks[:len(villain_current)],
            full_deck_ranks,
            street,
        )

        # Determine what the correct action should be given equity
        legal = game.legal_actions()
        features = extract_features(game, hero_seat)

        # Compute hindsight advantages
        # High equity → bet/raise has high advantage
        # Low equity → fold has high advantage
        # Medium equity → call/check
        advantages = [0.0] * 5  # fold, check, call, bet, raise

        facing_bet = any(a == Action.CALL for a in legal)

        if equity >= 0.65:
            # Strong — should be betting/raising
            if facing_bet:
                advantages[Action.RAISE.value] = equity * 2.0
                advantages[Action.CALL.value] = equity * 0.5
                advantages[Action.FOLD.value] = -2.0
            else:
                advantages[Action.BET.value] = equity * 2.0
                advantages[Action.CHECK.value] = -0.5
        elif equity >= 0.45:
            # Medium — should be calling/checking, maybe betting
            if facing_bet:
                advantages[Action.CALL.value] = equity
                advantages[Action.RAISE.value] = (equity - 0.5) * 2.0
                advantages[Action.FOLD.value] = -(1.0 - equity) * 0.5
            else:
                advantages[Action.BET.value] = (equity - 0.45) * 3.0
                advantages[Action.CHECK.value] = 0.3
        elif equity >= 0.25:
            # Weak — should be checking/calling cautiously or folding
            if facing_bet:
                pot_odds = 0.25  # Approximate
                if equity > pot_odds:
                    advantages[Action.CALL.value] = 0.3
                    advantages[Action.FOLD.value] = -0.1
                else:
                    advantages[Action.FOLD.value] = 0.5
                    advantages[Action.CALL.value] = -0.3
            else:
                advantages[Action.CHECK.value] = 0.5
                advantages[Action.BET.value] = -0.5
        else:
            # Dead — fold or check
            if facing_bet:
                advantages[Action.FOLD.value] = 1.0
                advantages[Action.CALL.value] = -1.0
            else:
                advantages[Action.CHECK.value] = 0.5

        # Check if CFR strategy was significantly wrong
        best_cfr_action = max(strategy, key=strategy.get) if strategy else None
        best_hindsight_action = None

        if facing_bet:
            if equity >= 0.65:
                best_hindsight_action = Action.RAISE
            elif equity >= 0.25:
                best_hindsight_action = Action.CALL
            else:
                best_hindsight_action = Action.FOLD
        else:
            if equity >= 0.55:
                best_hindsight_action = Action.BET
            else:
                best_hindsight_action = Action.CHECK

        # Determine correction type
        correction_type = 'neutral'
        if best_cfr_action and best_hindsight_action:
            cfr_passive = best_cfr_action in (Action.CHECK, Action.CALL, Action.FOLD)
            hindsight_aggressive = best_hindsight_action in (Action.BET, Action.RAISE)

            if cfr_passive and hindsight_aggressive and equity >= 0.55:
                correction_type = 'missed_bet'
            elif best_cfr_action == Action.FOLD and equity >= 0.30:
                correction_type = 'bad_fold'
            elif best_cfr_action == Action.FOLD and equity < 0.20:
                correction_type = 'good_fold'

        # Only add correction if it's meaningful (missed bet or bad fold)
        if correction_type in ('missed_bet', 'bad_fold'):
            corrections.append(HindsightCorrection(
                features=features,
                advantages=advantages,
                street=street,
                hero_equity=equity,
                correction_type=correction_type,
            ))

    return corrections


def collect_decision_nodes(game: HeadsUpRazzGame, deal, hero_seat: int,
                           strategy_net, advantage_net) -> Tuple[List[dict], dict]:
    """Replay a game tree collecting decision nodes for hindsight analysis.

    This is called AFTER a normal MCCFR traversal. We already have the deal,
    so we replay the game collecting the strategy at each hero decision point.

    Returns:
        (game_history, deal_info) suitable for hindsight_pass()
    """
    import torch
    from features import extract_features

    # Build deal_info from the FullDeal object
    hero_all_ranks = [c.rank for c in deal.hero_hole] + [c.rank for c in deal.hero_up] + [deal.hero_hole_7th.rank]
    villain_all_ranks = [c.rank for c in deal.villain_hole] + [c.rank for c in deal.villain_up] + [deal.villain_hole_7th.rank]

    deal_info = {
        'hero_ranks': hero_all_ranks,
        'villain_ranks': villain_all_ranks,
    }

    return deal_info
