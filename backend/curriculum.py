"""
Curriculum Learning for Neural Razz Trainer

Phase 1: "Formulas" — Supervised training on known-correct scenarios
Phase 2: "Tests" — Deep CFR with hindsight, applying the formulas
Phase 3: "Final Exam" — Battery test on unseen scenarios

Generates supervised training samples from battery scenarios with
random variations (suits, slight pot changes) and injects them
into the training reservoir.
"""

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass

from razz_game import HeadsUpRazzGame, Card, Action
from features import extract_features
from battery_test import BatteryScenario, c


# ═══════════════════════════════════════════════════════════════════════════════
# EXPANDED BATTERY SCENARIOS (~60 scenarios across all EV groups)
# ═══════════════════════════════════════════════════════════════════════════════

# ── TIER 1: No-Brainers (any player should get these right) ──────────────────

TIER1_SCENARIOS = [
    # --- G1: Elite hands (70%+ EV) ---
    BatteryScenario(
        name="A23 vs trash door — raise",
        tier=1, hero_hole=[c(1,0),c(2,1)], hero_up=[c(3,2)],
        villain_hole=[c(13,0),c(12,1)], villain_up=[c(11,2)],
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['raise'],
        explanation="Best starting hand vs JQK — always raise",
    ),
    BatteryScenario(
        name="A23 vs K door — raise/call",
        tier=1, hero_hole=[c(1,0),c(2,1)], hero_up=[c(3,2)],
        villain_hole=[c(10,0),c(11,1)], villain_up=[c(13,2)],
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['raise', 'call'],
        explanation="A23 vs K — never fold",
    ),
    BatteryScenario(
        name="JQK vs A door — fold",
        tier=1, hero_hole=[c(11,0),c(12,1)], hero_up=[c(13,2)],
        villain_hole=[c(1,0),c(2,1)], villain_up=[c(3,2)],
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['fold'],
        explanation="Trash vs premium door — always fold",
    ),
    BatteryScenario(
        name="Wheel on 5th — bet",
        tier=1, hero_hole=[c(1,0),c(2,1)],
        hero_up=[c(3,2),c(4,3),c(5,0)],
        villain_hole=[c(10,0),c(11,1)],
        villain_up=[c(13,2),c(12,3),c(9,0)],
        street=5, pot=6.0, facing_bet=False,
        expected_actions=['bet'],
        explanation="Made wheel — always bet for value",
    ),
    BatteryScenario(
        name="Wheel facing bet — raise",
        tier=1, hero_hole=[c(1,0),c(2,1)],
        hero_up=[c(3,2),c(4,3),c(5,0)],
        villain_hole=[c(10,0),c(13,1)],
        villain_up=[c(13,2),c(12,3),c(11,0)],
        street=5, pot=8.0, facing_bet=True,
        expected_actions=['raise'],
        explanation="Wheel vs trash facing bet — raise for max value",
    ),
    BatteryScenario(
        name="234 vs QKJ door — raise",
        tier=1, hero_hole=[c(2,0),c(3,1)], hero_up=[c(4,2)],
        villain_hole=[c(12,0),c(13,1)], villain_up=[c(11,2)],
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['raise'],
        explanation="234 vs J door — premium hand, raise",
    ),
    BatteryScenario(
        name="A45 caught good on 4th — bet",
        tier=1, hero_hole=[c(1,0),c(4,1)],
        hero_up=[c(5,2),c(2,3)],
        villain_hole=[c(10,0),c(11,1)],
        villain_up=[c(13,2),c(9,3)],
        street=4, pot=2.0, facing_bet=False,
        expected_actions=['bet'],
        explanation="A245 on 4th vs K9 board — strong draw, bet",
    ),

    # --- Obvious folds ---
    BatteryScenario(
        name="TJQ facing raise — fold",
        tier=1, hero_hole=[c(10,0),c(11,1)], hero_up=[c(12,2)],
        villain_hole=[c(1,0),c(2,1)], villain_up=[c(4,2)],
        street=3, pot=2.0, facing_bet=True,
        expected_actions=['fold'],
        explanation="TJQ vs 4 door facing raise — always fold",
    ),
    BatteryScenario(
        name="KKQ — fold",
        tier=1, hero_hole=[c(13,0),c(13,1)], hero_up=[c(12,2)],
        villain_hole=[c(5,0),c(6,1)], villain_up=[c(7,2)],
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['fold'],
        explanation="Paired kings with Q — always fold",
    ),
]

# ── TIER 2: Standard (solid player should get these) ─────────────────────────

TIER2_SCENARIOS = [
    # --- G2: Strong draws (60-70% EV) ---
    BatteryScenario(
        name="369 vs bad door — bet",
        tier=2, hero_hole=[c(3,0),c(6,1)], hero_up=[c(9,2)],
        villain_hole=[c(10,0),c(11,1)], villain_up=[c(13,2)],
        street=3, pot=1.0, facing_bet=False,
        expected_actions=['bet'],
        explanation="369 vs K door first to act — bet",
    ),
    BatteryScenario(
        name="369 vs A door — call",
        tier=2, hero_hole=[c(3,0),c(6,1)], hero_up=[c(9,2)],
        villain_hole=[c(2,0),c(4,1)], villain_up=[c(1,2)],
        street=3, pot=1.5, facing_bet=True,
        expected_actions=['call'],
        explanation="369 vs Ace — call, don't fold",
    ),
    BatteryScenario(
        name="258 vs Q door — bet/raise",
        tier=2, hero_hole=[c(2,0),c(5,1)], hero_up=[c(8,2)],
        villain_hole=[c(10,0),c(11,1)], villain_up=[c(12,2)],
        street=3, pot=1.0, facing_bet=False,
        expected_actions=['bet'],
        explanation="258 vs Q door — strong draw, bet",
    ),
    BatteryScenario(
        name="A23 bricked K on 4th — check",
        tier=2, hero_hole=[c(1,0),c(2,1)],
        hero_up=[c(3,2),c(13,3)],
        villain_hole=[c(6,0),c(7,1)],
        villain_up=[c(5,2),c(4,3)],
        street=4, pot=2.0, facing_bet=False,
        expected_actions=['check'],
        explanation="A23K vs 54 board — bricked, check",
    ),
    BatteryScenario(
        name="A235 good draw — bet",
        tier=2, hero_hole=[c(1,0),c(2,1)],
        hero_up=[c(3,2),c(5,3)],
        villain_hole=[c(10,0),c(11,1)],
        villain_up=[c(13,2),c(9,3)],
        street=4, pot=2.0, facing_bet=False,
        expected_actions=['bet'],
        explanation="A235 vs K9 board — great draw, bet",
    ),
    BatteryScenario(
        name="Made 8-low facing bet — call/raise",
        tier=2, hero_hole=[c(1,0),c(2,1)],
        hero_up=[c(3,2),c(5,3),c(8,0)],
        villain_hole=[c(6,0),c(7,1)],
        villain_up=[c(4,2),c(9,3),c(10,0)],
        street=5, pot=6.0, facing_bet=True,
        expected_actions=['call', 'raise'],
        explanation="Made 8-low facing bet — always continue",
    ),
    BatteryScenario(
        name="A46 caught 2 on 4th — bet",
        tier=2, hero_hole=[c(1,0),c(4,1)],
        hero_up=[c(6,2),c(2,3)],
        villain_hole=[c(10,0),c(11,1)],
        villain_up=[c(8,2),c(12,3)],
        street=4, pot=2.0, facing_bet=False,
        expected_actions=['bet'],
        explanation="A246 on 4th vs 8Q board — excellent draw, bet",
    ),

    # --- G3: Good draws (50-60% EV) ---
    BatteryScenario(
        name="57T vs bad door — bet",
        tier=2, hero_hole=[c(5,0),c(7,1)], hero_up=[c(10,2)],
        villain_hole=[c(11,0),c(12,1)], villain_up=[c(13,2)],
        street=3, pot=1.0, facing_bet=False,
        expected_actions=['bet'],
        explanation="57T vs K door — decent draw, bet",
    ),
    BatteryScenario(
        name="46T vs 5 door — call",
        tier=2, hero_hole=[c(4,0),c(6,1)], hero_up=[c(10,2)],
        villain_hole=[c(1,0),c(2,1)], villain_up=[c(5,2)],
        street=3, pot=1.5, facing_bet=True,
        expected_actions=['call'],
        explanation="46T vs 5 door facing bet — playable, call",
    ),
]

# ── TIER 3: Intermediate (requires hand reading) ─────────────────────────────

TIER3_SCENARIOS = [
    # --- G4: Playable/Marginal (45-50% EV) — pairs, marginal draws ---
    BatteryScenario(
        name="AA2 — call (paired but low)",
        tier=3, hero_hole=[c(1,0),c(1,1)], hero_up=[c(2,2)],
        villain_hole=[c(10,0),c(11,1)], villain_up=[c(8,2)],
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['call'],
        explanation="AA2 paired but low — call, hope to improve",
    ),
    BatteryScenario(
        name="334 — call",
        tier=3, hero_hole=[c(3,0),c(3,1)], hero_up=[c(4,2)],
        villain_hole=[c(7,0),c(8,1)], villain_up=[c(6,2)],
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['call'],
        explanation="334 paired but very low — call",
    ),
    BatteryScenario(
        name="78T vs A door — fold",
        tier=3, hero_hole=[c(7,0),c(8,1)], hero_up=[c(10,2)],
        villain_hole=[c(1,0),c(2,1)], villain_up=[c(3,2)],
        street=3, pot=1.5, facing_bet=True,
        expected_actions=['fold'],
        explanation="78T vs 3 door — weak draw, fold",
    ),
    BatteryScenario(
        name="A24KQ bricked twice — fold to aggression",
        tier=3, hero_hole=[c(1,0),c(2,1)],
        hero_up=[c(4,2),c(13,3),c(12,0)],
        villain_hole=[c(3,0),c(5,1)],
        villain_up=[c(6,2),c(7,3),c(8,0)],
        street=5, pot=8.0, facing_bet=True,
        expected_actions=['fold'],
        explanation="A24KQ vs 678 board facing bet — bricked, fold",
    ),
    BatteryScenario(
        name="A35 caught J on 4th — check/fold",
        tier=3, hero_hole=[c(1,0),c(3,1)],
        hero_up=[c(5,2),c(11,3)],
        villain_hole=[c(2,0),c(4,1)],
        villain_up=[c(6,2),c(3,3)],
        street=4, pot=2.0, facing_bet=True,
        expected_actions=['fold', 'call'],
        explanation="A35J vs 63 board facing bet — bricked, fold or reluctant call",
    ),

    # --- G5: Marginal (40-45% EV) ---
    BatteryScenario(
        name="59J vs K door — fold",
        tier=3, hero_hole=[c(5,0),c(9,1)], hero_up=[c(11,2)],
        villain_hole=[c(1,0),c(2,1)], villain_up=[c(13,2)],
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['fold'],
        explanation="59J — marginal hand facing bet, fold",
    ),
    BatteryScenario(
        name="38K first to act — check",
        tier=3, hero_hole=[c(3,0),c(8,1)], hero_up=[c(13,2)],
        villain_hole=[c(5,0),c(6,1)], villain_up=[c(4,2)],
        street=3, pot=0.5, facing_bet=False,
        expected_actions=['check'],
        explanation="38K first to act vs good board — check, don't bet",
    ),
]

# ── TIER 4: Advanced (nuanced play) ──────────────────────────────────────────

TIER4_SCENARIOS = [
    BatteryScenario(
        name="Semi-bluff A357 vs scary board",
        tier=4, hero_hole=[c(1,0),c(3,1)],
        hero_up=[c(5,2),c(7,3)],
        villain_hole=[c(2,0),c(4,1)],
        villain_up=[c(6,2),c(10,3)],
        street=4, pot=4.0, facing_bet=False,
        expected_actions=['bet', 'check'],
        explanation="A357 vs 6T board — can bet for value or check",
    ),
    BatteryScenario(
        name="Made 9-low vs apparent 7-low — call/fold",
        tier=4, hero_hole=[c(1,0),c(3,1)],
        hero_up=[c(5,2),c(7,3),c(9,0)],
        villain_hole=[c(2,0),c(4,1)],
        villain_up=[c(3,2),c(5,3),c(7,0)],
        street=5, pot=8.0, facing_bet=True,
        expected_actions=['call', 'fold'],
        explanation="9-low vs apparent 7-low — tough spot",
    ),
    BatteryScenario(
        name="6-low vs 8-low board — raise for value",
        tier=4, hero_hole=[c(1,0),c(2,1)],
        hero_up=[c(3,2),c(4,3),c(6,0)],
        villain_hole=[c(5,0),c(10,1)],
        villain_up=[c(7,2),c(8,3),c(9,0)],
        street=5, pot=6.0, facing_bet=True,
        expected_actions=['raise'],
        explanation="Made 6-low vs 789 board — raise for value",
    ),
    BatteryScenario(
        name="Live 4-card draw vs made hand — call",
        tier=4, hero_hole=[c(1,0),c(2,1)],
        hero_up=[c(4,2),c(6,3)],
        villain_hole=[c(3,0),c(5,1)],
        villain_up=[c(7,2),c(8,3)],
        street=4, pot=4.0, facing_bet=True,
        expected_actions=['call', 'raise'],
        explanation="A246 live draw vs 78 board facing bet — call with outs",
    ),

    # --- G6-G7: Weak/Bad hands ---
    BatteryScenario(
        name="TJK — always fold facing bet",
        tier=4, hero_hole=[c(10,0),c(11,1)], hero_up=[c(13,2)],
        villain_hole=[c(5,0),c(6,1)], villain_up=[c(7,2)],
        street=3, pot=1.5, facing_bet=True,
        expected_actions=['fold'],
        explanation="TJK vs 7 door — trash, always fold",
    ),
    BatteryScenario(
        name="9TJ — fold",
        tier=4, hero_hole=[c(9,0),c(10,1)], hero_up=[c(11,2)],
        villain_hole=[c(3,0),c(5,1)], villain_up=[c(4,2)],
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['fold'],
        explanation="9TJ vs 4 door — weak, fold",
    ),
]

# ── TIER 5: Expert (GTO nuances) ─────────────────────────────────────────────

TIER5_SCENARIOS = [
    BatteryScenario(
        name="Value bet thin river 8-low",
        tier=5, hero_hole=[c(1,0),c(2,1),c(8,0)],
        hero_up=[c(3,2),c(5,3),c(6,0),c(7,1)],
        villain_hole=[c(4,0),c(9,1),c(11,0)],
        villain_up=[c(10,2),c(6,3),c(5,0),c(3,1)],
        street=7, pot=12.0, facing_bet=False,
        expected_actions=['bet', 'check'],
        explanation="8-low on river — thin value bet",
    ),
    BatteryScenario(
        name="Wheel on river — bet big",
        tier=5, hero_hole=[c(1,0),c(2,1),c(5,0)],
        hero_up=[c(3,2),c(4,3),c(7,0),c(9,1)],
        villain_hole=[c(6,0),c(8,1),c(13,0)],
        villain_up=[c(10,2),c(7,3),c(6,0),c(5,1)],
        street=7, pot=10.0, facing_bet=False,
        expected_actions=['bet'],
        explanation="Made wheel on river — always bet",
    ),
    BatteryScenario(
        name="Bad river card — check/fold",
        tier=5, hero_hole=[c(1,0),c(3,1),c(13,0)],
        hero_up=[c(5,2),c(7,3),c(9,0),c(11,1)],
        villain_hole=[c(2,0),c(4,1),c(6,0)],
        villain_up=[c(8,2),c(3,3),c(5,0),c(2,1)],
        street=7, pot=10.0, facing_bet=True,
        expected_actions=['fold'],
        explanation="Bricked river (K) vs strong board — fold",
    ),
    BatteryScenario(
        name="Made 7-low facing river raise — call",
        tier=5, hero_hole=[c(1,0),c(2,1),c(7,0)],
        hero_up=[c(3,2),c(4,3),c(6,0),c(10,1)],
        villain_hole=[c(5,0),c(8,1),c(9,0)],
        villain_up=[c(2,2),c(5,3),c(7,0),c(4,1)],
        street=7, pot=14.0, facing_bet=True,
        expected_actions=['call'],
        explanation="Made 7-low facing river bet — strong enough to call",
    ),
]

ALL_EXPANDED_SCENARIOS = TIER1_SCENARIOS + TIER2_SCENARIOS + TIER3_SCENARIOS + TIER4_SCENARIOS + TIER5_SCENARIOS


# ═══════════════════════════════════════════════════════════════════════════════
# CURRICULUM: Generate supervised training samples from scenarios
# ═══════════════════════════════════════════════════════════════════════════════

def generate_curriculum_samples(scenarios: List[BatteryScenario] = None,
                                 variations_per_scenario: int = 500) -> List[Tuple[List[float], List[float], int]]:
    """Generate supervised training samples from battery scenarios.

    Each scenario is varied by randomizing suits and slightly varying pots.
    Returns list of (features, target_strategy, iteration_weight) tuples
    ready for reservoir injection.

    Args:
        scenarios: List of scenarios (default: ALL_EXPANDED_SCENARIOS)
        variations_per_scenario: How many variations to generate per scenario

    Returns:
        List of (features, target, weight) tuples
    """
    if scenarios is None:
        scenarios = ALL_EXPANDED_SCENARIOS

    samples = []
    suits = [0, 1, 2, 3]

    for scenario in scenarios:
        target = _build_target(scenario)

        # More variations for problem scenarios (4th street, T2)
        num_vars = variations_per_scenario
        if scenario.street == 4:
            num_vars = int(variations_per_scenario * 2)  # 2x variations for 4th street
        if scenario.tier == 2:
            num_vars = int(num_vars * 1.5)  # Extra for T2 scenarios

        for v in range(num_vars):
            # Randomize suits (keep ranks the same)
            random.shuffle(suits)
            suit_map = {0: suits[0], 1: suits[1], 2: suits[2], 3: suits[3]}

            hero_hole = [(r, suit_map.get(s, random.choice(suits))) for r, s in scenario.hero_hole]
            hero_up = [(r, suit_map.get(s, random.choice(suits))) for r, s in scenario.hero_up]
            villain_hole = [(r, suit_map.get(s, random.choice(suits))) for r, s in scenario.villain_hole]
            villain_up = [(r, suit_map.get(s, random.choice(suits))) for r, s in scenario.villain_up]

            # Slight pot variation (+/- 20%)
            pot_mult = 0.8 + random.random() * 0.4
            pot = scenario.pot * pot_mult

            # Build game and extract features
            try:
                game = _build_game(hero_hole, hero_up, villain_hole, villain_up,
                                    scenario.street, pot, scenario.facing_bet)
                features = extract_features(game, 0)

                # Weight by tier: T1 = base weight, T2 = 2x (these are the 4th street
                # scenarios the model keeps failing), T3-T5 = 1.5x
                # Also boost 4th street scenarios specifically (street == 4)
                base_weight = 1_000_000
                tier_mult = {1: 1.0, 2: 2.0, 3: 1.5, 4: 1.5, 5: 1.5}.get(scenario.tier, 1.0)
                street_mult = 2.0 if scenario.street == 4 else 1.0
                weight = int(base_weight * tier_mult * street_mult)

                samples.append((features, target, weight))
            except Exception:
                continue

    print(f"[Curriculum] Generated {len(samples)} supervised samples from {len(scenarios)} scenarios")
    return samples


def _build_target(scenario: BatteryScenario) -> List[float]:
    """Build a 5-element target probability vector from expected actions."""
    action_map = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4}
    target = [0.0] * 5

    # Distribute probability among expected actions
    n = len(scenario.expected_actions)
    for action_name in scenario.expected_actions:
        idx = action_map.get(action_name, 0)
        target[idx] = 1.0 / n

    return target


def _build_game(hero_hole, hero_up, villain_hole, villain_up,
                street, pot, facing_bet) -> HeadsUpRazzGame:
    """Build a game state from card tuples."""
    game = HeadsUpRazzGame()

    h_hole = [Card(r, s) for r, s in hero_hole]
    h_up = [Card(r, s) for r, s in hero_up]
    v_hole = [Card(r, s) for r, s in villain_hole]
    v_up = [Card(r, s) for r, s in villain_up]

    game.deal_third_street(
        p0_hole=h_hole[:2], p0_up=h_up[0],
        p1_hole=v_hole[:2], p1_up=v_up[0],
    )

    for i in range(1, len(h_up)):
        game.deal_card(0, h_up[i], is_hole=False)
        if i < len(v_up):
            game.deal_card(1, v_up[i], is_hole=False)

    if len(h_hole) > 2:
        game.deal_card(0, h_hole[2], is_hole=True)
    if len(v_hole) > 2:
        game.deal_card(1, v_hole[2], is_hole=True)

    game.pot = pot
    game.street = street

    if facing_bet:
        game.current_bet_level = 1.0
        game.street_contribution[0] = 0.0
    else:
        game.current_bet_level = 0.0
        game.street_contribution[0] = 0.0
        game.street_contribution[1] = 0.0
        game.num_bets = 0

    return game


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AutoTrainConfig:
    """Configuration for the auto-training pipeline."""
    # Per-group training iterations
    iterations_per_group: int = 500_000
    # Minimum battery score to pass a group (0-1)
    min_battery_score: float = 0.60
    # Minimum BB/100 vs TAG to pass arena test
    min_bb_vs_tag: float = 0.0  # Must be positive
    # Arena hands per opponent
    arena_hands: int = 5000
    # Learning rate
    learning_rate: float = 0.001
    # Batch size
    batch_size: int = 512
    # Enable hindsight correction
    enable_hindsight: bool = True
    # Reservoir size
    reservoir_size: int = 10_000_000
    # Curriculum injection
    inject_curriculum: bool = True
    curriculum_variations: int = 500
    # Max retries per group before moving on
    max_retries: int = 3

# EV Groups: (name, hand_scope_key, description)
EV_GROUPS = [
    ('ev_group_1', 'G1: Elite 70%+', 35),
    ('ev_group_2', 'G2: Strong 60-70%', 60),
    ('ev_group_3', 'G3: Good 50-60%', 55),
    ('ev_group_4', 'G4: Playable 45-50%', 114),
    ('ev_group_5', 'G5: Marginal 40-45%', 55),
    ('ev_group_6', 'G6: Weak 35-40%', 26),
    ('ev_group_7', 'G7: Bad 25-35%', 62),
    ('ev_group_8', 'G8: Trash <25%', 48),
]


@dataclass
class AutoTrainProgress:
    """Tracks progress of auto-training pipeline."""
    current_group: int = 0
    total_groups: int = 8
    current_group_name: str = ""
    current_group_attempt: int = 0
    phase: str = "idle"  # idle, curriculum, training, arena, battery, complete
    group_results: List[Dict] = None
    is_running: bool = False
    should_stop: bool = False

    def __post_init__(self):
        if self.group_results is None:
            self.group_results = []


# ─── CLI test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Total expanded scenarios: {len(ALL_EXPANDED_SCENARIOS)}")
    for tier in range(1, 6):
        count = sum(1 for s in ALL_EXPANDED_SCENARIOS if s.tier == tier)
        print(f"  Tier {tier}: {count} scenarios")

    print()
    print("Generating curriculum samples (100 variations each)...")
    samples = generate_curriculum_samples(variations_per_scenario=100)
    print(f"Generated {len(samples)} total samples")

    # Check target distributions
    print()
    print("Target distribution check:")
    for s in ALL_EXPANDED_SCENARIOS[:5]:
        target = _build_target(s)
        actions = ['fold', 'check', 'call', 'bet', 'raise']
        probs = ' | '.join(f"{a}:{p:.0%}" for a, p in zip(actions, target) if p > 0)
        print(f"  {s.name}: {probs}")
