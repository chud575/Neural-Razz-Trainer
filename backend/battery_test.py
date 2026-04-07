"""
Battery Test for Neural Razz Trainer

Tests the neural network against specific known-correct scenarios
across 5 tiers of difficulty. Returns pass/fail for each scenario
plus an overall score.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from razz_game import HeadsUpRazzGame, Card, Action
from features import extract_features


@dataclass
class BatteryScenario:
    name: str
    tier: int                           # 1-5 (1=no-brainer, 5=expert)
    hero_hole: List[Tuple[int, int]]    # [(rank, suit), ...]
    hero_up: List[Tuple[int, int]]      # upcards
    villain_hole: List[Tuple[int, int]]
    villain_up: List[Tuple[int, int]]
    street: int                         # 3-7
    pot: float
    facing_bet: bool
    expected_actions: List[str]         # acceptable actions: ['raise'], ['call','raise'], etc.
    explanation: str


@dataclass
class BatteryResult:
    scenario: BatteryScenario
    passed: bool
    predicted_action: str
    probabilities: Dict[str, float]
    confidence: float


@dataclass
class BatteryReport:
    results: List[BatteryResult]

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def total_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def score(self) -> float:
        return self.total_passed / max(self.total_tests, 1)

    @property
    def tier_scores(self) -> Dict[int, Tuple[int, int]]:
        """Returns {tier: (passed, total)} for each tier."""
        scores = {}
        for r in self.results:
            t = r.scenario.tier
            if t not in scores:
                scores[t] = [0, 0]
            scores[t][1] += 1
            if r.passed:
                scores[t][0] += 1
        return {t: tuple(v) for t, v in scores.items()}


# ─── Helper ─────────────────────────────────────────────────────────────────

def c(rank: int, suit: int) -> Tuple[int, int]:
    """Shorthand for card tuple. rank: 1=A,...,13=K. suit: 0-3."""
    return (rank, suit)


# ─── Scenarios ──────────────────────────────────────────────────────────────

# TIER 1: No-Brainers — any competent player gets these right
tier1 = [
    BatteryScenario(
        name="Premium vs trash — raise",
        tier=1,
        hero_hole=[c(1,0), c(2,1)], hero_up=[c(3,2)],       # A23
        villain_hole=[c(13,0), c(12,1)], villain_up=[c(11,2)], # KQJ
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['raise'],
        explanation="A23 vs JQK door card — always raise",
    ),
    BatteryScenario(
        name="Premium vs bad door — raise/call",
        tier=1,
        hero_hole=[c(1,0), c(2,1)], hero_up=[c(3,2)],       # A23
        villain_hole=[c(10,0), c(11,1)], villain_up=[c(13,2)], # TJK
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['raise', 'call'],
        explanation="A23 vs K door — raise or call, never fold",
    ),
    BatteryScenario(
        name="Trash vs premium — fold",
        tier=1,
        hero_hole=[c(11,0), c(12,1)], hero_up=[c(13,2)],    # JQK
        villain_hole=[c(1,0), c(2,1)], villain_up=[c(3,2)],   # A23
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['fold'],
        explanation="JQK vs A door card — always fold",
    ),
    BatteryScenario(
        name="Wheel on 5th — bet",
        tier=1,
        hero_hole=[c(1,0), c(2,1)], hero_up=[c(3,2), c(4,3), c(5,0)],  # A2345
        villain_hole=[c(10,0), c(11,1)], villain_up=[c(13,2), c(12,3), c(9,0)],
        street=5, pot=6.0, facing_bet=False,
        expected_actions=['bet'],
        explanation="Made wheel on 5th — always bet for value",
    ),
    BatteryScenario(
        name="Nut lock on 5th facing bet — raise",
        tier=1,
        hero_hole=[c(1,0), c(2,1)], hero_up=[c(3,2), c(4,3), c(5,0)],  # A2345
        villain_hole=[c(10,0), c(13,1)], villain_up=[c(13,2), c(12,3), c(11,0)],
        street=5, pot=8.0, facing_bet=True,
        expected_actions=['raise'],
        explanation="Wheel vs KQJ board facing bet — raise for max value",
    ),
]

# TIER 2: Standard — solid player should get these
tier2 = [
    BatteryScenario(
        name="Good draw vs bad door — bet/raise",
        tier=2,
        hero_hole=[c(3,0), c(6,1)], hero_up=[c(9,2)],       # 369
        villain_hole=[c(10,0), c(11,1)], villain_up=[c(13,2)], # TJK
        street=3, pot=1.0, facing_bet=False,
        expected_actions=['bet'],
        explanation="369 vs K door when first to act — bet",
    ),
    BatteryScenario(
        name="369 facing bet from A door — call",
        tier=2,
        hero_hole=[c(3,0), c(6,1)], hero_up=[c(9,2)],       # 369
        villain_hole=[c(2,0), c(4,1)], villain_up=[c(1,2)],   # 24A
        street=3, pot=1.5, facing_bet=True,
        expected_actions=['call'],
        explanation="369 vs Ace door — call, don't fold or raise",
    ),
    BatteryScenario(
        name="Bricked 4th — check",
        tier=2,
        hero_hole=[c(1,0), c(2,1)], hero_up=[c(3,2), c(13,3)],  # A23 + K
        villain_hole=[c(6,0), c(7,1)], villain_up=[c(5,2), c(4,3)],  # 67 + 54
        street=4, pot=2.0, facing_bet=False,
        expected_actions=['check'],
        explanation="A23K vs 5-4 board — bricked, check",
    ),
    BatteryScenario(
        name="Good 4th street draw — bet",
        tier=2,
        hero_hole=[c(1,0), c(2,1)], hero_up=[c(3,2), c(5,3)],  # A235
        villain_hole=[c(10,0), c(11,1)], villain_up=[c(13,2), c(9,3)],
        street=4, pot=2.0, facing_bet=False,
        expected_actions=['bet'],
        explanation="A235 vs K-9 board — strong draw, bet",
    ),
    BatteryScenario(
        name="Made 8-low facing bet — call/raise",
        tier=2,
        hero_hole=[c(1,0), c(2,1)], hero_up=[c(3,2), c(5,3), c(8,0)],  # A2358
        villain_hole=[c(6,0), c(7,1)], villain_up=[c(4,2), c(9,3), c(10,0)],
        street=5, pot=6.0, facing_bet=True,
        expected_actions=['call', 'raise'],
        explanation="Made 8-low facing bet — call or raise, don't fold",
    ),
]

# TIER 3: Intermediate — requires understanding of position and draw quality
tier3 = [
    BatteryScenario(
        name="Marginal draw vs scary board — fold",
        tier=3,
        hero_hole=[c(7,0), c(8,1)], hero_up=[c(10,2)],      # 78T
        villain_hole=[c(1,0), c(2,1)], villain_up=[c(3,2)],   # A23
        street=3, pot=1.5, facing_bet=True,
        expected_actions=['fold'],
        explanation="78T vs A door with raise — fold weak draw",
    ),
    BatteryScenario(
        name="Paired but low — call (not fold, not raise)",
        tier=3,
        hero_hole=[c(1,0), c(1,1)], hero_up=[c(2,2)],       # AA2
        villain_hole=[c(10,0), c(11,1)], villain_up=[c(8,2)],
        street=3, pot=1.0, facing_bet=True,
        expected_actions=['call'],
        explanation="AA2 paired but low — call, hope to improve",
    ),
    BatteryScenario(
        name="Good draw bricked twice — fold to aggression",
        tier=3,
        hero_hole=[c(1,0), c(2,1)], hero_up=[c(4,2), c(13,3), c(12,0)],  # A24KQ
        villain_hole=[c(3,0), c(5,1)], villain_up=[c(6,2), c(7,3), c(8,0)],
        street=5, pot=8.0, facing_bet=True,
        expected_actions=['fold'],
        explanation="A24KQ vs 6-7-8 board facing bet — bricked, fold",
    ),
]

# TIER 4: Advanced — nuanced decisions
tier4 = [
    BatteryScenario(
        name="Semi-bluff with 4-card draw vs scary board",
        tier=4,
        hero_hole=[c(1,0), c(3,1)], hero_up=[c(5,2), c(7,3)],  # A357
        villain_hole=[c(2,0), c(4,1)], villain_up=[c(6,2), c(10,3)],  # 246T
        street=4, pot=4.0, facing_bet=False,
        expected_actions=['bet', 'check'],
        explanation="A357 vs 6-T board — can bet for value or check, both reasonable",
    ),
    BatteryScenario(
        name="Made 9-low vs apparent 7-low — call/fold",
        tier=4,
        hero_hole=[c(1,0), c(3,1)], hero_up=[c(5,2), c(7,3), c(9,0)],  # A3579
        villain_hole=[c(2,0), c(4,1)], villain_up=[c(3,2), c(5,3), c(7,0)],  # 2,4 + 3,5,7 showing
        street=5, pot=8.0, facing_bet=True,
        expected_actions=['call', 'fold'],
        explanation="9-low vs apparent 7-low — tough spot, call or fold",
    ),
]

# TIER 5: Expert — GTO nuances
tier5 = [
    BatteryScenario(
        name="Value bet thin on river with 8-low",
        tier=5,
        hero_hole=[c(1,0), c(2,1), c(8,0)],  # A,2 hole + 8 river
        hero_up=[c(3,2), c(5,3), c(6,0), c(7,1)],  # 3,5,6,7 up
        villain_hole=[c(4,0), c(9,1), c(11,0)],  # 4,9 hole + J river
        villain_up=[c(10,2), c(6,3), c(5,0), c(3,1)],
        street=7, pot=12.0, facing_bet=False,
        expected_actions=['bet', 'check'],
        explanation="8-low on river vs mixed board — thin value bet or check",
    ),
]


ALL_SCENARIOS = tier1 + tier2 + tier3 + tier4 + tier5


# ─── Runner ─────────────────────────────────────────────────────────────────

def run_battery(network, scenarios: List[BatteryScenario] = None) -> BatteryReport:
    """Run all battery scenarios against the neural network.

    Args:
        network: A StrategyNetwork (or any object with .predict(features) -> list)
        scenarios: Custom scenarios, or None for ALL_SCENARIOS

    Returns: BatteryReport with pass/fail for each scenario
    """
    if scenarios is None:
        scenarios = ALL_SCENARIOS

    results = []
    for scenario in scenarios:
        result = _evaluate_scenario(network, scenario)
        results.append(result)

    return BatteryReport(results=results)


def _evaluate_scenario(network, scenario: BatteryScenario) -> BatteryResult:
    """Evaluate a single scenario."""
    # Build game state
    game = HeadsUpRazzGame()
    hero_seat = 0

    hero_hole = [Card(r, s) for r, s in scenario.hero_hole]
    hero_up = [Card(r, s) for r, s in scenario.hero_up]
    villain_hole = [Card(r, s) for r, s in scenario.villain_hole]
    villain_up = [Card(r, s) for r, s in scenario.villain_up]

    # Deal 3rd street
    game.deal_third_street(
        p0_hole=hero_hole[:2], p0_up=hero_up[0],
        p1_hole=villain_hole[:2], p1_up=villain_up[0],
    )

    # Deal remaining streets
    for i in range(1, len(hero_up)):
        game.deal_card(hero_seat, hero_up[i], is_hole=False)
        if i < len(villain_up):
            game.deal_card(1, villain_up[i], is_hole=False)

    # Deal hole cards for 7th street
    if len(hero_hole) > 2:
        game.deal_card(hero_seat, hero_hole[2], is_hole=True)
    if len(villain_hole) > 2:
        game.deal_card(1, villain_hole[2], is_hole=True)

    # Set pot and street
    game.pot = scenario.pot
    game.street = scenario.street
    if scenario.facing_bet:
        game.current_bet_level = 1.0
        game.street_contribution[hero_seat] = 0.0
    else:
        # Explicitly clear any bet state from deal_third_street
        game.current_bet_level = 0.0
        game.street_contribution[0] = 0.0
        game.street_contribution[1] = 0.0
        game.num_bets = 0

    # Extract features and predict
    features = extract_features(game, hero_seat)
    probs = network.predict(features)

    action_names = ['fold', 'check', 'call', 'bet', 'raise']
    prob_dict = {name: float(p) for name, p in zip(action_names, probs)}

    # Filter to legal actions based on facing_bet
    if scenario.facing_bet:
        legal = {k: v for k, v in prob_dict.items() if k in ['fold', 'call', 'raise']}
    else:
        legal = {k: v for k, v in prob_dict.items() if k in ['check', 'bet', 'fold']}

    # Renormalize
    total = sum(legal.values())
    if total > 0:
        legal = {k: v / total for k, v in legal.items()}

    # Best action
    best = max(legal, key=legal.get) if legal else 'fold'
    confidence = legal.get(best, 0)

    # Check if passed
    passed = best in scenario.expected_actions

    return BatteryResult(
        scenario=scenario,
        passed=passed,
        predicted_action=best,
        probabilities=legal,
        confidence=confidence,
    )


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    from networks import StrategyNetwork, load_from_json

    # Load model
    model_path = '../neural_razz_strategy.json'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    network = load_from_json(model_path)
    if not network:
        print(f"Failed to load model from {model_path}")
        sys.exit(1)

    print(f"Loaded model from {model_path}")
    print()

    report = run_battery(network)

    # Print results
    tier_names = {1: 'No-Brainer', 2: 'Standard', 3: 'Intermediate', 4: 'Advanced', 5: 'Expert'}

    for tier in sorted(report.tier_scores.keys()):
        passed, total = report.tier_scores[tier]
        print(f"\n{'='*60}")
        print(f"  TIER {tier}: {tier_names.get(tier, '?')} — {passed}/{total}")
        print(f"{'='*60}")

        for r in report.results:
            if r.scenario.tier != tier:
                continue
            icon = '✅' if r.passed else '❌'
            probs_str = ' | '.join(f"{a}:{p:.0%}" for a, p in sorted(r.probabilities.items(), key=lambda x: -x[1]))
            expected = '/'.join(r.scenario.expected_actions)
            print(f"  {icon} {r.scenario.name}")
            print(f"     Expected: {expected}  Got: {r.predicted_action} ({r.confidence:.0%})")
            print(f"     Probs: {probs_str}")
            if not r.passed:
                print(f"     Why: {r.scenario.explanation}")

    print(f"\n{'='*60}")
    print(f"  OVERALL: {report.total_passed}/{report.total_tests} ({report.score:.0%})")
    print(f"{'='*60}")
