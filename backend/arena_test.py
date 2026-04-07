"""
Arena Test — Run neural network strategy against simple opponents.

Used by the Flask server's /api/test/arena endpoint.
"""

import random
from typing import Dict, List

from razz_game import HeadsUpRazzGame, Card, Action, make_deck
from features import extract_features
from networks import StrategyNetwork


# ─── Opponent Strategies ────────────────────────────────────────────────────

def calling_station_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    """Always calls, never folds, never raises."""
    legal = game.legal_actions()
    if Action.CALL in legal:
        return Action.CALL
    if Action.CHECK in legal:
        return Action.CHECK
    if Action.BET in legal:
        return Action.BET  # Calling stations sometimes bet
    return legal[0]


def random_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    """Random legal action."""
    return random.choice(game.legal_actions())


def _get_hand_bucket(ranks: List[int], street: int) -> int:
    """Get Razz hand bucket (0-10, higher = better). Matches PokerArena's 11-bucket system.

    For draws (3-4 cards): uses EV table win rate tiers.
    For made hands (5+ cards): uses best-5 high card.
    """
    from bucketer import classify_hero, hero_ev_percentile

    if len(ranks) >= 5:
        # Made hand: use best-5 evaluation
        bucket_code = classify_hero(ranks)
        # Map bucket codes to 0-10 scale
        made_map = {'mW': 10, 'm6': 9, 'm7': 8, 'm8': 7, 'm9': 6, 'mT': 4, 'mP': 3}
        return made_map.get(bucket_code, 5)
    else:
        # Draw: use EV percentile
        ev = hero_ev_percentile(ranks)
        if ev >= 0.70: return 10
        if ev >= 0.65: return 9
        if ev >= 0.60: return 8
        if ev >= 0.55: return 7
        if ev >= 0.50: return 6
        if ev >= 0.45: return 5
        if ev >= 0.40: return 4
        if ev >= 0.35: return 3
        if ev >= 0.25: return 2
        return 1


def tag_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    """Tight-aggressive: bucket-based decisions matching PokerArena's TAG.

    3rd street: raise B8+, call B6+, fold below B6
    Later streets: raise B8+ (if < 2 raises), call B5+, fold below B5
    Also bets B6+ when first to act on later streets.
    """
    legal = game.legal_actions()
    player = game.players[player_idx]
    ranks = player.all_ranks

    bucket = _get_hand_bucket(ranks, game.street)
    facing_bet = game.current_bet_level > game.street_contribution[player_idx]

    if game.street == 3:
        # 3rd street — selective but aggressive
        if facing_bet:
            if bucket >= 8 and Action.RAISE in legal:
                return Action.RAISE
            if bucket >= 6:
                return Action.CALL if Action.CALL in legal else legal[0]
            return Action.FOLD if Action.FOLD in legal else legal[0]
        else:
            if bucket >= 7 and Action.BET in legal:
                return Action.BET
            return Action.CHECK if Action.CHECK in legal else legal[0]
    else:
        # Later streets — value bet strong, fold weak
        if facing_bet:
            if bucket >= 8 and Action.RAISE in legal and game.num_bets < 2:
                return Action.RAISE
            if bucket >= 5:
                return Action.CALL if Action.CALL in legal else legal[0]
            return Action.FOLD if Action.FOLD in legal else legal[0]
        else:
            if bucket >= 6 and Action.BET in legal:
                return Action.BET
            return Action.CHECK if Action.CHECK in legal else legal[0]


def lag_action(game: HeadsUpRazzGame, player_idx: int) -> Action:
    """Loose-aggressive: plays wider range aggressively, bucket-aware.

    3rd street: raise B5+, call B3+, fold below B3
    Later streets: raise B6+ (freely), bet B4+, call B3+, fold below B3
    Much wider range than TAG but still bucket-aware.
    """
    legal = game.legal_actions()
    player = game.players[player_idx]
    ranks = player.all_ranks

    bucket = _get_hand_bucket(ranks, game.street)
    facing_bet = game.current_bet_level > game.street_contribution[player_idx]

    if game.street == 3:
        if facing_bet:
            if bucket >= 5 and Action.RAISE in legal and random.random() < 0.6:
                return Action.RAISE
            if bucket >= 3:
                return Action.CALL if Action.CALL in legal else legal[0]
            return Action.FOLD if Action.FOLD in legal else legal[0]
        else:
            if bucket >= 4 and Action.BET in legal:
                return Action.BET
            return Action.CHECK if Action.CHECK in legal else legal[0]
    else:
        if facing_bet:
            if bucket >= 6 and Action.RAISE in legal and random.random() < 0.5:
                return Action.RAISE
            if bucket >= 3:
                return Action.CALL if Action.CALL in legal else legal[0]
            return Action.FOLD if Action.FOLD in legal else legal[0]
        else:
            if bucket >= 4 and Action.BET in legal:
                return Action.BET
            if bucket >= 2 and Action.BET in legal and random.random() < 0.3:
                return Action.BET  # Semi-bluff with marginal hands
            return Action.CHECK if Action.CHECK in legal else legal[0]


from opponents import (
    tag_action as tag_action_proper,
    lag_action as lag_action_proper,
    mr_heu_action,
    rebel_action, bucketed_cfr_action, pure_ev_action,
)

OPPONENTS = {
    'calling_station': calling_station_action,
    'random': random_action,
    'tag': tag_action_proper,
    'lag': lag_action_proper,
    'mr_heu': mr_heu_action,
    'rebel': rebel_action,
    'bucketed_cfr': bucketed_cfr_action,
    'pure_ev': pure_ev_action,
}


# ─── Neural Network Decision ───────────────────────────────────────────────

def neural_action(network: StrategyNetwork, game: HeadsUpRazzGame, hero_seat: int) -> Action:
    """Pick an action using the neural network."""
    legal = game.legal_actions()
    if len(legal) <= 1:
        return legal[0] if legal else Action.FOLD

    features = extract_features(game, hero_seat)
    probs = network.predict(features)

    # Filter to legal actions
    legal_probs = [(a, probs[a.value]) for a in legal]
    total = sum(p for _, p in legal_probs)

    if total <= 0:
        return random.choice(legal)

    # Sample from distribution
    r = random.random()
    cumulative = 0.0
    for action, prob in legal_probs:
        cumulative += prob / total
        if r < cumulative:
            return action

    return legal_probs[-1][0]


# ─── Arena Runner ───────────────────────────────────────────────────────────

def _get_scope_hands(hand_scope: str) -> List[List[int]]:
    """Get starting hand rank patterns for a scope."""
    from trainer_strategy import get_starting_hands
    return get_starting_hands(hand_scope)


def _deal_forced_hand(hero_ranks: List[int], deck: List[Card]) -> List[Card]:
    """Pick cards from deck matching the specified ranks. Returns [hole1, hole2, up]."""
    cards = []
    for rank in hero_ranks:
        for i, card in enumerate(deck):
            if card.rank == rank:
                cards.append(deck.pop(i))
                break
    return cards


def _parse_hand_string(hand_str: str) -> List[int]:
    """Parse a hand string like 'A23' or '369' into rank values."""
    rank_map = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13}
    return [rank_map[c] for c in hand_str.upper() if c in rank_map]


def run_arena(network: StrategyNetwork, num_hands: int = 500,
              opponent_type: str = 'calling_station',
              hand_scope: str = None,
              single_hand: str = None) -> Dict:
    """Run arena test: neural network vs opponent.

    Args:
        hand_scope: If set (e.g. 'premium'), hero only gets hands from that scope.
                    Villain always gets random hands.
        single_hand: If set (e.g. 'A23'), hero always gets this specific hand.
                     Overrides hand_scope.

    Returns dict with win_rate, bb_per_100, fold_rate, etc.
    """
    opponent_fn = OPPONENTS.get(opponent_type, calling_station_action)

    # Determine hero hand pool
    if single_hand:
        parsed = _parse_hand_string(single_hand)
        scope_hands = [parsed] if len(parsed) == 3 else None
    elif hand_scope:
        scope_hands = _get_scope_hands(hand_scope)
    else:
        scope_hands = None

    total_profit = 0.0
    wins = 0
    losses = 0
    folds = 0
    showdowns = 0
    sd_wins = 0
    hand_histories = []  # Last N hands for review

    RANK_CHARS = {1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K'}
    def _fmt_card(c): return RANK_CHARS.get(c.rank, '?') + ['s','h','d','c'][c.suit]
    def _fmt_ranks(ranks): return ''.join(RANK_CHARS.get(r, '?') for r in sorted(ranks))

    for h in range(num_hands):
        hero_seat = h % 2
        villain_seat = 1 - hero_seat

        deck = make_deck()
        game = HeadsUpRazzGame()

        if scope_hands:
            hero_ranks = random.choice(scope_hands)
            hero_cards = _deal_forced_hand(hero_ranks, deck)
            if len(hero_cards) < 3:
                continue
            random.shuffle(deck)
            v_hole = [deck.pop(), deck.pop()]
            v_up = deck.pop()

            if hero_seat == 0:
                game.deal_third_street(
                    [hero_cards[0], hero_cards[1]], hero_cards[2],
                    v_hole, v_up)
            else:
                game.deal_third_street(
                    v_hole, v_up,
                    [hero_cards[0], hero_cards[1]], hero_cards[2])
        else:
            p0_hole = [deck.pop(), deck.pop()]
            p0_up = deck.pop()
            p1_hole = [deck.pop(), deck.pop()]
            p1_up = deck.pop()
            game.deal_third_street(p0_hole, p0_up, p1_hole, p1_up)

        # Record starting hands for history
        hero_start = _fmt_ranks(game.players[hero_seat].all_ranks)
        villain_start = _fmt_ranks(game.players[villain_seat].all_ranks)
        hero_door = _fmt_card(game.players[hero_seat].up_cards[0]) if game.players[hero_seat].up_cards else '?'
        villain_door = _fmt_card(game.players[villain_seat].up_cards[0]) if game.players[villain_seat].up_cards else '?'

        action_log = []  # Per-action log for this hand
        action_count = 0
        hero_folded = False

        def _board_summary():
            """Show both hands: hero sees all own cards, villain hole cards hidden."""
            h = game.players[hero_seat]
            v = game.players[villain_seat]
            # Hero: (hole1 hole2) up1 up2 ...
            hero_hole = [_fmt_card(c) for c in h.hole_cards]
            hero_up = [_fmt_card(c) for c in h.up_cards]
            hero_str = f"({' '.join(hero_hole)}) {' '.join(hero_up)}" if hero_hole else ' '.join(hero_up)
            # Villain: (?? ??) up1 up2 ...
            v_up = [_fmt_card(c) for c in v.up_cards]
            v_hole_count = len(v.hole_cards)
            villain_str = f"({'?? ' * v_hole_count}){' '.join(v_up)}" if v_up else '??'
            return f"  Hero: [{hero_str}]  Opp: [{villain_str}]  Pot: {game.pot:.1f}"

        # Log 3rd street starting board
        action_log.append(f"── Street 3 ──")
        action_log.append(_board_summary())

        while not game.is_terminal and action_count < 60:
            action_count += 1
            cp = game.current_player
            is_hero = (cp == hero_seat)
            street = game.street

            if is_hero:
                # Get probabilities for logging
                features = extract_features(game, hero_seat)
                probs = network.predict(features)
                action = neural_action(network, game, hero_seat)
                action_names = ['fold', 'check', 'call', 'bet', 'raise']
                prob_str = '/'.join(f"{action_names[i][0]}:{p:.0%}" for i, p in enumerate(probs) if p > 0.01)
                action_log.append(f"  Hero: {action.name} ({prob_str})")
            else:
                action = opponent_fn(game, cp)
                action_log.append(f"  Opp:  {action.name}")

            if action == Action.FOLD and is_hero:
                hero_folded = True

            prev_street = game.street
            game.apply_action(action)

            # Deal next street
            if game.street != prev_street and not game.is_terminal:
                if game.street in [4, 5, 6]:
                    for p in range(2):
                        if game.players[p].is_active:
                            card = deck.pop()
                            game.deal_card(p, card, is_hole=False)
                    action_log.append(f"── Street {game.street} ──")
                    action_log.append(_board_summary())
                elif game.street == 7:
                    for p in range(2):
                        if game.players[p].is_active:
                            card = deck.pop()
                            game.deal_card(p, card, is_hole=True)
                    action_log.append(f"── Street 7 (river) ──")
                    action_log.append(_board_summary())

        payoff = game.payoff(hero_seat)
        total_profit += payoff

        # Show final hands and result in action log
        hero_final_cards = [_fmt_card(c) for c in game.players[hero_seat].all_cards]
        villain_final_cards = [_fmt_card(c) for c in game.players[villain_seat].all_cards]
        hero_final_ranks = _fmt_ranks(game.players[hero_seat].all_ranks)
        villain_final_ranks = _fmt_ranks(game.players[villain_seat].all_ranks)

        if hero_folded:
            folds += 1
            result_str = f"FOLD (lost {abs(payoff):.1f})"
            action_log.append(f"── RESULT: Hero folded. Lost {abs(payoff):.1f} ──")
        elif game.winner is not None:
            if game.winner == hero_seat:
                wins += 1
                showdowns += 1
                sd_wins += 1
                result_str = f"WIN +{payoff:.1f}"
                action_log.append(f"── SHOWDOWN ──")
                action_log.append(f"  Hero:    {' '.join(hero_final_cards)} → {hero_final_ranks}")
                action_log.append(f"  Villain: {' '.join(villain_final_cards)} → {villain_final_ranks}")
                action_log.append(f"  HERO WINS +{payoff:.1f} (pot {game.pot:.1f})")
            else:
                losses += 1
                showdowns += 1
                result_str = f"LOSS {payoff:.1f}"
                action_log.append(f"── SHOWDOWN ──")
                action_log.append(f"  Hero:    {' '.join(hero_final_cards)} → {hero_final_ranks}")
                action_log.append(f"  Villain: {' '.join(villain_final_cards)} → {villain_final_ranks}")
                action_log.append(f"  HERO LOSES {payoff:.1f} (pot {game.pot:.1f})")
        else:
            showdowns += 1
            result_str = f"CHOP {payoff:.1f}"
            action_log.append(f"── SHOWDOWN (CHOP) ──")
            action_log.append(f"  Hero:    {' '.join(hero_final_cards)} → {hero_final_ranks}")
            action_log.append(f"  Villain: {' '.join(villain_final_cards)} → {villain_final_ranks}")
            action_log.append(f"  CHOP {payoff:.1f}")

        hero_final = hero_final_ranks
        villain_final = villain_final_ranks
        hand_record = {
            'hand_num': h + 1,
            'hero_start': hero_start,
            'villain_start': villain_start,
            'hero_door': hero_door,
            'villain_door': villain_door,
            'hero_final': hero_final,
            'villain_final': villain_final,
            'result': result_str,
            'payoff': round(payoff, 2),
            'street_ended': game.street,
            'hero_folded': hero_folded,
            'actions': action_log,
            'pot': round(game.pot, 2),
        }

        # Keep ALL hands — no cap
        hand_histories.append(hand_record)

    bb_per_100 = (total_profit / num_hands) * 100 / 2.0
    win_rate = wins / max(num_hands, 1)
    fold_rate = folds / max(num_hands, 1)
    sd_win_rate = sd_wins / max(showdowns, 1)

    # Keep chronological order

    return {
        'num_hands': num_hands,
        'opponent': opponent_type,
        'win_rate': round(win_rate * 100, 1),
        'bb_per_100': round(bb_per_100, 1),
        'fold_rate': round(fold_rate * 100, 1),
        'sd_win_rate': round(sd_win_rate * 100, 1),
        'total_profit': round(total_profit, 2),
        'wins': wins,
        'losses': losses,
        'folds': folds,
        'showdowns': showdowns,
        'hand_histories': hand_histories,  # All hands
    }


# ─── Quick test ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Arena Test ===\n")

    # Create an untrained network (should play ~uniformly)
    net = StrategyNetwork()
    result = run_arena(net, num_hands=1000, opponent_type='calling_station')

    print(f"Results vs Calling Station (1K hands, untrained network):")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n✅ Arena test complete")
