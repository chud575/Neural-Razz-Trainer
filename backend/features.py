"""
Feature Extraction for Neural Razz Trainer

Extracts a 32-dimensional feature vector from a Razz game state.
This vector is the input to all 3 neural network modes.

Feature layout (32 dimensions):
  [0-4]   Street one-hot (3rd-7th)
  [5]     Pot size / big bet
  [6]     Facing bet (binary)
  [7]     Raises this street / 4
  [8-21]  Hero bucket one-hot (14 buckets)
  [22]    Hero EV percentile (0-1)
  [23]    Villain visible card count / 4
  [24]    Villain best visible rank / 13
  [25]    Hero has pair (binary)
  [26-30] Villain aggression per street (5 floats, one per street 3-7)
  [31]    Hero card count / 7
"""

from typing import List

from razz_game import HeadsUpRazzGame, Action, BIG_BET, MAX_BETS_PER_STREET
from bucketer import (
    classify_hero, classify_villain_visible, hero_ev_percentile,
    ALL_HERO_BUCKETS, HERO_BUCKET_INDEX, load_ev_tables,
)

FEATURE_DIM = 32


def extract_features(game: HeadsUpRazzGame, hero_seat: int) -> List[float]:
    """Extract 32-dimensional feature vector from a game state.

    Args:
        game: Current game state
        hero_seat: Which player is hero (0 or 1)

    Returns:
        32-element list of floats, ready for neural network input.
    """
    load_ev_tables()

    villain_seat = 1 - hero_seat
    hero = game.players[hero_seat]
    villain = game.players[villain_seat]

    features = [0.0] * FEATURE_DIM

    # ── [0-4] Street one-hot ────────────────────────────────────────────
    street_idx = game.street - 3  # 0-4 for streets 3-7
    if 0 <= street_idx <= 4:
        features[street_idx] = 1.0

    # ── [5] Pot size normalized ─────────────────────────────────────────
    features[5] = game.pot / BIG_BET  # Normalize by big bet

    # ── [6] Facing bet ──────────────────────────────────────────────────
    my_contrib = game.street_contribution[hero_seat]
    to_call = game.current_bet_level - my_contrib
    features[6] = 1.0 if to_call > 0 else 0.0

    # ── [7] Raises this street ──────────────────────────────────────────
    features[7] = game.num_bets / MAX_BETS_PER_STREET

    # ── [8-21] Hero bucket one-hot ──────────────────────────────────────
    hero_ranks = hero.all_ranks
    if hero_ranks:
        bucket = classify_hero(hero_ranks)
        idx = HERO_BUCKET_INDEX.get(bucket, 0)
        features[8 + idx] = 1.0

    # ── [22] Hero EV percentile ─────────────────────────────────────────
    if hero_ranks:
        features[22] = hero_ev_percentile(hero_ranks)

    # ── [23] Villain visible card count ─────────────────────────────────
    villain_up_ranks = [c.rank for c in villain.up_cards]
    features[23] = len(villain_up_ranks) / 4.0

    # ── [24] Villain best visible rank ──────────────────────────────────
    if villain_up_ranks:
        features[24] = min(villain_up_ranks) / 13.0  # Lower = better for villain

    # ── [25] Hero has pair ──────────────────────────────────────────────
    if hero_ranks:
        features[25] = 1.0 if len(set(hero_ranks)) < len(hero_ranks) else 0.0

    # ── [26-30] Villain aggression per street ────────────────────────────
    for i in range(5):
        features[26 + i] = game.player_agg[villain_seat][i] / MAX_BETS_PER_STREET

    # ── [31] Hero card count ────────────────────────────────────────────
    features[31] = hero.card_count / 7.0

    return features


def feature_names() -> List[str]:
    """Return human-readable names for each feature dimension."""
    names = []
    # [0-4] Street
    for s in [3, 4, 5, 6, 7]:
        names.append(f'street_{s}')
    # [5] Pot
    names.append('pot_normalized')
    # [6] Facing bet
    names.append('facing_bet')
    # [7] Raises
    names.append('raises_normalized')
    # [8-21] Hero bucket
    for b in ALL_HERO_BUCKETS:
        names.append(f'hero_{b}')
    # [22] EV
    names.append('hero_ev_pct')
    # [23-24] Villain
    names.append('villain_count')
    names.append('villain_best_rank')
    # [25] Pair
    names.append('hero_has_pair')
    # [26-30] Villain aggression
    for s in [3, 4, 5, 6, 7]:
        names.append(f'villain_agg_street_{s}')
    # [31] Card count
    names.append('hero_card_count')

    assert len(names) == FEATURE_DIM
    return names


# ─── Quick validation ───────────────────────────────────────────────────────

if __name__ == '__main__':
    from razz_game import Card
    print("=== Feature Extraction Tests ===\n")

    # Setup a 3rd street game
    g = HeadsUpRazzGame()
    g.deal_third_street(
        p0_hole=[Card(1, 0), Card(2, 1)], p0_up=Card(3, 2),   # A23
        p1_hole=[Card(10, 0), Card(11, 1)], p1_up=Card(13, 2), # TJK
    )

    feats = extract_features(g, hero_seat=0)
    names = feature_names()

    assert len(feats) == FEATURE_DIM
    print(f"Feature vector ({FEATURE_DIM} dims):")
    for i, (name, val) in enumerate(zip(names, feats)):
        if val != 0:
            print(f"  [{i:2d}] {name:20s} = {val:.3f}")

    # Verify key features
    assert feats[0] == 1.0, "Should be 3rd street"
    assert feats[6] == 1.0 or feats[6] == 0.0, "Facing bet should be binary"
    assert feats[22] > 0.5, "A23 should have high EV percentile"
    assert feats[31] == 3/7, "Hero has 3 cards"
    print("\n✅ Feature extraction tests passed!")

    # Test after some actions
    g.apply_action(Action.BET)  # Complete
    g.apply_action(Action.CALL)  # Call → advance to 4th
    g.deal_card(0, Card(4, 0), is_hole=False)
    g.deal_card(1, Card(12, 0), is_hole=False)

    feats4 = extract_features(g, hero_seat=0)
    print(f"\n4th street features (non-zero):")
    for i, (name, val) in enumerate(zip(names, feats4)):
        if val != 0:
            print(f"  [{i:2d}] {name:20s} = {val:.3f}")

    assert feats4[1] == 1.0, "Should be 4th street"
    assert feats4[31] == 4/7, "Hero has 4 cards"
    # Hero was the aggressor on 3rd, so villain aggression should be 0
    assert feats4[26] == 0, "3rd street villain aggression should be 0 (hero bet)"
    print("\n✅ 4th street feature tests passed!")

    print("\nAll feature tests passed!")
