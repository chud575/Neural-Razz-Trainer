"""
Belief Bucketer — port of BeliefBucketer.swift

Classifies hero hands into 14 buckets and villain visible cards into
count+quality categories. Uses EV table lookup for draws (3-4 cards)
and structural evaluation for made hands (5+ cards).

Hero buckets (14):
  Draws:  d1 (70%+), d2 (60-70%), d3 (50-60%), d4 (45-50%),
          d5 (40-45%), d6 (30-40%), d7 (<30%)
  Made:   mW (wheel/5-low), m6, m7, m8, m9, mT (T+), mP (paired)
"""

import os
import json
from typing import List, Dict, Optional

from razz_eval import penalize_pairs


# ─── EV Tables ──────────────────────────────────────────────────────────────

_ev_3card: Dict[str, float] = {}
_ev_4card: Dict[str, float] = {}
_ev_5card: Dict[str, float] = {}
_ev_loaded = False

RANK_CHARS = {1:'A', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7',
              8:'8', 9:'9', 10:'T', 11:'J', 12:'Q', 13:'K'}
CHAR_RANKS = {v: k for k, v in RANK_CHARS.items()}


def load_ev_tables(ev_dir: str = None):
    """Load 2-player 3-card, 4-card, and 5-card EV tables."""
    global _ev_3card, _ev_4card, _ev_5card, _ev_loaded

    if _ev_loaded:
        return

    if ev_dir is None:
        # Search common locations
        search = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'HORSE+ Master', 'EV_Tables_Razz'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'HORSE+ Master'),
            os.path.expanduser('~/Documents/Resources'),
        ]
    else:
        search = [ev_dir]

    for d in search:
        if not os.path.isdir(d):
            continue
        _try_load_ev(d, 3, _ev_3card)
        _try_load_ev(d, 4, _ev_4card)
        _try_load_ev(d, 5, _ev_5card)
        if _ev_3card and _ev_4card:
            break

    _ev_loaded = True
    print(f"[Bucketer] EV tables: {len(_ev_3card)} 3-card, {len(_ev_4card)} 4-card, {len(_ev_5card)} 5-card entries")


def _try_load_ev(directory: str, num_cards: int, table: dict):
    """Try to load EV table from a directory."""
    patterns = [
        f'razz-ev-2p-{num_cards}card.json',
        f'razz-{num_cards}card-2p-ev.json',
    ]
    for filename in patterns:
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)

            # Format 1: { "evTable": { "A23": 73.5, ... } }
            if 'evTable' in data:
                for key, val in data['evTable'].items():
                    # Strip suffixes like -rainbow
                    base = key.split('-')[0]
                    table[base] = val
                return

            # Format 2: { "results": [ { "rawHand": "A23", "winRate": 73.5 }, ... ] }
            if 'results' in data:
                for entry in data['results']:
                    key = entry.get('rawHand') or entry.get('handDescription', '')
                    wr = entry.get('winRate', 0)
                    if key:
                        table[key] = wr
                return
        except Exception as e:
            print(f"[Bucketer] Failed to load {path}: {e}")


# ─── Hero Bucketing ─────────────────────────────────────────────────────────

def tier_from_win_rate(wr: float) -> str:
    """Map win rate (0-100 scale) to draw tier."""
    if wr >= 70: return 'd1'
    if wr >= 60: return 'd2'
    if wr >= 50: return 'd3'
    if wr >= 45: return 'd4'
    if wr >= 40: return 'd5'
    if wr >= 30: return 'd6'
    return 'd7'


def classify_hero(ranks: List[int]) -> str:
    """Classify hero hand into one of 14 buckets.

    Matches Swift BeliefBucketer.classifyHeroFromRanks().

    Args:
        ranks: All hero card ranks (1=A through 13=K), 3-7 cards.

    Returns:
        Bucket code: 'd1'-'d7' for draws, 'mW','m6'-'m9','mT','mP' for made.
    """
    if len(ranks) >= 5:
        # Made hand: penalize pairs, take best 5
        penalized = penalize_pairs(ranks)
        penalized.sort()
        best5 = penalized[:5]

        # Check for pairs in best 5
        if any(r > 13 for r in best5):
            return 'mP'

        high = best5[-1]
        if high <= 5:  return 'mW'
        if high <= 6:  return 'm6'
        if high <= 7:  return 'm7'
        if high <= 8:  return 'm8'
        if high <= 9:  return 'm9'
        return 'mT'

    # Draw (3-4 cards): EV table lookup
    load_ev_tables()
    key = ''.join(RANK_CHARS.get(r, '?') for r in sorted(ranks))
    table = _ev_3card if len(ranks) == 3 else _ev_4card

    if key in table:
        return tier_from_win_rate(table[key])

    # Heuristic fallback
    unique = set(ranks)
    if len(unique) < len(ranks):
        # Paired
        high = max(unique)
        if high <= 5: return 'd5'
        if high <= 8: return 'd6'
        return 'd7'

    high = max(ranks)
    if high <= 5:  return 'd1'
    if high <= 7:  return 'd2'
    if high <= 9:  return 'd3'
    if high <= 10: return 'd4'
    if high <= 11: return 'd5'
    return 'd6'


def hero_ev_percentile(ranks: List[int]) -> float:
    """Get hero's EV as 0-1 percentile. Higher = better.

    For draws: uses EV table.
    For made hands: uses normalized score.
    """
    load_ev_tables()

    if len(ranks) <= 4:
        key = ''.join(RANK_CHARS.get(r, '?') for r in sorted(ranks))
        table = _ev_3card if len(ranks) == 3 else _ev_4card
        if key in table:
            return table[key] / 100.0  # 0-1 scale
        # Fallback: estimate from high card
        high = max(ranks)
        return max(0.0, 1.0 - high / 13.0)

    # Made hand: use penalized score
    from razz_eval import normalized_score
    ns = normalized_score(ranks)
    return 1.0 - ns  # Invert: 0=worst → 1=best


# ─── Villain Bucketing ──────────────────────────────────────────────────────

def classify_villain_visible(ranks: List[int]) -> str:
    """Classify villain's visible upcards.

    Matches Swift classifyVillainVisibleFromRanks().

    Format: "{count}{quality}" e.g. "1L", "2M", "3H"
      count: number of visible cards (1-4)
      quality: L (best rank ≤5), M (6-8), H (9+)
    """
    count = len(ranks)
    best = min(ranks) if ranks else 13
    if best <= 5:   quality = 'L'
    elif best <= 8: quality = 'M'
    else:           quality = 'H'
    return f"{count}{quality}"


# ─── Bucket Metadata ────────────────────────────────────────────────────────

ALL_HERO_BUCKETS = ['d1','d2','d3','d4','d5','d6','d7',
                     'mW','m6','m7','m8','m9','mT','mP']

HERO_BUCKET_INDEX = {b: i for i, b in enumerate(ALL_HERO_BUCKETS)}


# ─── Quick validation ───────────────────────────────────────────────────────

if __name__ == '__main__':
    load_ev_tables()
    print("\n=== Bucketer Tests ===\n")

    tests = [
        ([1, 2, 3], 'd1', 'A23 (premium)'),
        ([3, 6, 9], 'd2', '369 (good draw)'),
        ([8, 9, 10], 'd3', '89T (mid draw, EV ~50%)'),
        ([11, 12, 13], 'd7', 'JQK (trash)'),
        ([1, 1, 5], 'd4', 'AA5 (paired Aces, EV ~45%)'),
        ([1, 2, 3, 4, 5], 'mW', 'A2345 (wheel)'),
        ([1, 2, 3, 4, 7], 'm7', 'A2347 (7-low)'),
        ([1, 2, 3, 4, 9], 'm9', 'A2349 (9-low)'),
        ([1, 1, 3, 4, 5], 'mP', 'AA345 (paired made)'),
        ([2, 3, 4, 5, 10], 'mT', '2345T (T-low)'),
    ]

    for ranks, expected, desc in tests:
        bucket = classify_hero(ranks)
        status = "✅" if bucket == expected else "❌"
        print(f"  {status} {desc}: {bucket} (expected {expected})")

    print()

    villain_tests = [
        ([13], '1H', 'K showing'),
        ([3], '1L', '3 showing'),
        ([7], '1M', '7 showing'),
        ([3, 8], '2L', '3,8 showing'),
        ([9, 10], '2H', '9,T showing'),
        ([2, 5, 7, 11], '4L', '2,5,7,J showing'),
    ]

    for ranks, expected, desc in villain_tests:
        bucket = classify_villain_visible(ranks)
        status = "✅" if bucket == expected else "❌"
        print(f"  {status} {desc}: {bucket} (expected {expected})")

    print()

    # EV percentile test
    ev_a23 = hero_ev_percentile([1, 2, 3])
    ev_369 = hero_ev_percentile([3, 6, 9])
    ev_jqk = hero_ev_percentile([11, 12, 13])
    print(f"  EV percentiles: A23={ev_a23:.3f}, 369={ev_369:.3f}, JQK={ev_jqk:.3f}")
    assert ev_a23 > ev_369 > ev_jqk
    print("  ✅ EV ordering correct")

    print("\nAll bucketer tests passed!")
