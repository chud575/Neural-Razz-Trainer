"""
Razz Hand Evaluator

Evaluates Razz (lowball) hands. Lower is better.
Ace = 1 (best), King = 13 (worst).

CRITICAL: Pairs are penalized by adding 13 to each duplicate occurrence.
  AA98543 → [1, 14, 9, 8, 5, 4, 3] → best 5 = [1, 3, 4, 5, 8] = 8-low
  KK2345  → [13, 26, 2, 3, 4, 5]   → best 5 = [2, 3, 4, 5, 13] = K-low
  AAKK543 → [1, 14, 13, 26, 5, 4, 3] → best 5 = [1, 3, 4, 5, 13] = K-low

After penalizing pairs, sort ascending, take best (lowest) 5.
"""

from typing import List, Tuple


def penalize_pairs(ranks: List[int]) -> List[int]:
    """Add 13 to each duplicate occurrence of a rank.

    First occurrence stays at face value.
    Second occurrence gets +13.
    Third occurrence gets +26.
    Fourth occurrence gets +39.

    Examples:
        [1, 1, 9, 8, 5, 4, 3] → [1, 14, 9, 8, 5, 4, 3]
        [13, 13, 2, 3, 4, 5] → [13, 26, 2, 3, 4, 5]
        [5, 5, 5, 2, 3, 4, 6] → [5, 18, 31, 2, 3, 4, 6]
    """
    seen_count = {}
    result = []
    for r in ranks:
        count = seen_count.get(r, 0)
        result.append(r + count * 13)
        seen_count[r] = count + 1
    return result


def evaluate(ranks: List[int]) -> Tuple[List[int], float]:
    """Evaluate a Razz hand.

    Args:
        ranks: List of rank values (1=Ace through 13=King), 3-7 cards.

    Returns:
        (best_5, score) where:
        - best_5: the 5 lowest penalized ranks (sorted ascending)
        - score: numeric score (lower = better). Comparable across hands.

    For fewer than 5 cards, returns all cards sorted (no truncation).
    """
    penalized = penalize_pairs(ranks)
    penalized.sort()

    # Take best 5 (or all if fewer)
    best = penalized[:5] if len(penalized) >= 5 else penalized[:]

    # Score: weighted sum. Highest card in best 5 is most significant.
    # best is sorted ascending: best[-1] is the highest (worst) card.
    # In Razz, compare high card first, then next highest, etc.
    # So best[-1] (highest) gets the largest weight (14^4).
    # Score = best[-1]*14^4 + best[-2]*14^3 + ... + best[0]*14^0
    score = 0.0
    n = len(best)
    for i in range(n):
        # best[i] at position i from low end → weight = 14^i (low weight for low cards)
        score += best[i] * (14 ** i)

    return best, score


def normalized_score(ranks: List[int]) -> float:
    """Return a 0-1 score where 0 = best possible (A2345 wheel) and 1 = worst.

    Used for feature extraction. Normalized against REALISTIC range of hands
    that actually appear in Razz, not theoretical worst.
    """
    _, raw = evaluate(ranks)

    # Best possible: A2345 wheel
    best_score = 5 * 14**4 + 4 * 14**3 + 3 * 14**2 + 2 * 14 + 1  # ~203,391

    # Worst REALISTIC unpaired: 9TJQK
    # evaluate([9,10,11,12,13]) = 13*14^4+12*14^3+11*14^2+10*14+9 = ~530,841
    worst_score = 13 * 14**4 + 12 * 14**3 + 11 * 14**2 + 10 * 14 + 9  # ~530,841

    # Clamp to range — paired hands can exceed worst_score
    return min(1.0, max(0.0, (raw - best_score) / (worst_score - best_score)))


def hand_description(best_5: List[int]) -> str:
    """Human-readable description of evaluated hand."""
    rank_names = {1:'A', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7',
                  8:'8', 9:'9', 10:'T', 11:'J', 12:'Q', 13:'K'}

    # Check if any penalized ranks > 13 (indicates pairs)
    has_pair = any(r > 13 for r in best_5)

    # Show the actual card values
    display = []
    for r in best_5:
        if r > 13:
            base_rank = ((r - 1) % 13) + 1
            display.append(f"({rank_names.get(base_rank, '?')})")  # parentheses = paired
        else:
            display.append(rank_names.get(r, '?'))

    high = best_5[-1] if best_5 else 0
    if high <= 13:
        return f"{rank_names.get(high, '?')}-low: {' '.join(display)}"
    else:
        base = ((high - 1) % 13) + 1
        return f"Paired ({rank_names.get(base, '?')}): {' '.join(display)}"


# ─── Quick validation ───────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Razz Evaluator Tests ===\n")

    tests = [
        # (ranks, expected_best_5, description)
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], "Wheel (A-5)"),
        ([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5], "Wheel from 7 cards"),
        ([1, 1, 9, 8, 5, 4, 3], [1, 3, 4, 5, 8], "Pair of Aces → 8-low"),
        ([13, 13, 2, 3, 4, 5], [2, 3, 4, 5, 13], "Pair of Kings → K-low"),
        ([1, 1, 13, 13, 5, 4, 3], [1, 3, 4, 5, 13], "AA KK → K-low"),
        ([5, 5, 5, 2, 3, 4, 6], [2, 3, 4, 5, 6], "Trips 5s → 6-low"),
        ([3, 6, 9], [3, 6, 9], "369 (3 cards)"),
        ([1, 2, 3], [1, 2, 3], "A23 (3 cards)"),
        ([1, 2, 3, 4], [1, 2, 3, 4], "A234 (4 cards)"),
        ([8, 9, 10, 11, 12, 13, 13], [8, 9, 10, 11, 12], "Bad hand → Q-low"),
    ]

    all_pass = True
    for ranks, expected, desc in tests:
        best, score = evaluate(ranks)
        passed = best == expected
        status = "✅" if passed else "❌"
        print(f"  {status} {desc}")
        print(f"     Input: {ranks}")
        print(f"     Best5: {best} (expected {expected})")
        print(f"     Score: {score:.0f} | {hand_description(best)}")
        if not passed:
            all_pass = False
        print()

    # Comparison test: A2345 < A2346 < 23456
    _, s1 = evaluate([1, 2, 3, 4, 5])
    _, s2 = evaluate([1, 2, 3, 4, 6])
    _, s3 = evaluate([2, 3, 4, 5, 6])
    assert s1 < s2 < s3, f"Ordering failed: {s1} < {s2} < {s3}"
    print("✅ Ordering: A2345 < A2346 < 23456")

    # Pair penalty test: AA543 best5=[1,3,4,5,14] should be WORSE than clean 8-low
    _, sp = evaluate([1, 1, 5, 4, 3])
    _, sn = evaluate([8, 6, 5, 4, 3])
    print(f"\n  AA543 score: {sp:.0f} (best5: {evaluate([1,1,5,4,3])[0]})")
    print(f"  86543 score: {sn:.0f} (best5: {evaluate([8,6,5,4,3])[0]})")
    # AA543 → [1, 14, 5, 4, 3] → best5 [1, 3, 4, 5, 14] → high card is 14 (paired A)
    # 86543 → [3, 4, 5, 6, 8] → 8-low, high card is 8
    # 14 > 8, so AA543 is WORSE — the pair penalty makes the Ace count as 14
    assert sp > sn, "AA543 (paired, high=14) should be worse than 86543 (8-low)"
    print("✅ Pair penalty: AA543 (high=14) worse than 86543 (8-low)")

    # In Razz, ANY unpaired hand beats ANY paired hand.
    # So KQJT9 (K-high, no pair) beats AA543 (paired).
    _, sk = evaluate([13, 12, 11, 10, 9])
    assert sp > sk, "AA543 (paired) should LOSE to KQJT9 (K-high, no pair)"
    print("✅ KQJT9 (K-high) beats AA543 (paired) — correct Razz rule")

    # Two pair test: AAKK5 should be very bad
    _, s2p = evaluate([1, 1, 13, 13, 5])
    print(f"  AAKK5 score: {s2p:.0f} (best5: {evaluate([1,1,13,13,5])[0]})")
    assert s2p > sn, "AAKK5 should be worse than 86543"
    print("✅ Two pair AAKK5 worse than 86543")

    print(f"\n{'All tests passed!' if all_pass else 'SOME TESTS FAILED'}")
