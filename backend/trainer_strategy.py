"""
Mode 1: Strategy Network Trainer

Runs external-sampling MCCFR to build a tabular strategy, then distills
it into a neural network via KL divergence training.

Flow:
1. Run MCCFR traversals (same as Swift solver)
2. At each info set, record (features, regret-matched strategy)
3. Collect into reservoir buffer
4. Periodically train network on reservoir batches
5. The network learns to generalize across similar game states

This gives ~100% hit rate because the network can interpolate to unseen states.
"""

import random
import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from razz_game import (
    HeadsUpRazzGame, Action, Card, make_deck,
    ANTE, BRING_IN, SMALL_BET, BIG_BET, MAX_BETS_PER_STREET,
    bet_size,
)
from features import extract_features, FEATURE_DIM
from networks import StrategyNetwork
from reservoir import ReservoirBuffer


# ─── Info Set (tabular, for MCCFR) ─────────────────────────────────────────

class InfoSet:
    """Tabular info set for MCCFR. Stores cumulative regrets and strategy sums."""
    __slots__ = ['regret_sum', 'strategy_sum', 'visit_count']

    def __init__(self):
        self.regret_sum = [0.0] * 5   # fold, check, call, bet, raise
        self.strategy_sum = [0.0] * 5
        self.visit_count = 0

    def get_strategy(self, legal_actions: List[Action]) -> Dict[Action, float]:
        """Current strategy via regret matching (positive regrets normalized)."""
        strategy = {}
        positive_sum = 0.0
        for a in legal_actions:
            r = max(0.0, self.regret_sum[a.value])
            strategy[a] = r
            positive_sum += r

        if positive_sum > 0:
            for a in legal_actions:
                strategy[a] /= positive_sum
        else:
            uniform = 1.0 / len(legal_actions)
            for a in legal_actions:
                strategy[a] = uniform

        return strategy

    def get_average_strategy(self, legal_actions: List[Action]) -> Dict[Action, float]:
        """Average strategy (converges to Nash)."""
        strategy = {}
        total = 0.0
        for a in legal_actions:
            s = self.strategy_sum[a.value]
            strategy[a] = s
            total += s

        if total > 0:
            for a in legal_actions:
                strategy[a] /= total
        else:
            uniform = 1.0 / len(legal_actions)
            for a in legal_actions:
                strategy[a] = uniform

        return strategy


# ─── Pre-dealt cards ────────────────────────────────────────────────────────

@dataclass
class FullDeal:
    """All cards for a complete hand, pre-dealt before traversal."""
    hero_hole: List[Card]         # [card1, card2]
    hero_up: List[Card]           # [3rd, 4th, 5th, 6th]
    hero_hole_7th: Card
    villain_hole: List[Card]      # [card1, card2]
    villain_up: List[Card]        # [3rd, 4th, 5th, 6th]
    villain_hole_7th: Card


def deal_hand(hero_start_ranks: List[int], deck: List[Card] = None) -> Optional[FullDeal]:
    """Deal a complete hand with hero getting specific starting ranks."""
    if deck is None:
        deck = make_deck()
    else:
        random.shuffle(deck)

    pool = list(deck)

    # Pick hero's 3 starting cards with matching ranks
    hero_cards = []
    for rank in hero_start_ranks:
        found = None
        for i, c in enumerate(pool):
            if c.rank == rank:
                found = i
                break
        if found is None:
            return None
        hero_cards.append(pool.pop(found))

    hero_hole = [hero_cards[0], hero_cards[1]]
    hero_up_3rd = hero_cards[2]

    # Villain's cards (random)
    villain_hole = [pool.pop(0), pool.pop(0)]
    villain_up_3rd = pool.pop(0)

    # 4th-6th street upcards
    hero_up_4th = pool.pop(0)
    villain_up_4th = pool.pop(0)
    hero_up_5th = pool.pop(0)
    villain_up_5th = pool.pop(0)
    hero_up_6th = pool.pop(0)
    villain_up_6th = pool.pop(0)

    # 7th street hole cards
    hero_hole_7th = pool.pop(0)
    villain_hole_7th = pool.pop(0)

    return FullDeal(
        hero_hole=hero_hole,
        hero_up=[hero_up_3rd, hero_up_4th, hero_up_5th, hero_up_6th],
        hero_hole_7th=hero_hole_7th,
        villain_hole=villain_hole,
        villain_up=[villain_up_3rd, villain_up_4th, villain_up_5th, villain_up_6th],
        villain_hole_7th=villain_hole_7th,
    )


def deal_if_street_advanced(game: HeadsUpRazzGame, deal: FullDeal,
                             prev_street: int, hero_seat: int):
    """If the street advanced, deal the pre-dealt cards."""
    if game.street == prev_street or game.is_terminal:
        return

    villain_seat = 1 - hero_seat
    street = game.street

    if street == 4:
        game.deal_card(hero_seat, deal.hero_up[1], is_hole=False)
        game.deal_card(villain_seat, deal.villain_up[1], is_hole=False)
    elif street == 5:
        game.deal_card(hero_seat, deal.hero_up[2], is_hole=False)
        game.deal_card(villain_seat, deal.villain_up[2], is_hole=False)
    elif street == 6:
        game.deal_card(hero_seat, deal.hero_up[3], is_hole=False)
        game.deal_card(villain_seat, deal.villain_up[3], is_hole=False)
    elif street == 7:
        game.deal_card(hero_seat, deal.hero_hole_7th, is_hole=True)
        game.deal_card(villain_seat, deal.villain_hole_7th, is_hole=True)


# ─── MCCFR Traversal ───────────────────────────────────────────────────────

def mccfr_traverse(
    game: HeadsUpRazzGame,
    deal: FullDeal,
    traversing_player: int,
    hero_seat: int,
    hero_info_sets: Dict[str, InfoSet],
    villain_info_sets: Dict[str, InfoSet],
    feature_reservoir: ReservoirBuffer,
    min_visits_for_collection: int = 50,
    iteration: int = 0,
) -> float:
    """External-sampling MCCFR traversal.

    Traversing player explores ALL actions. Opponent samples ONE action.
    Matches RazzCFRSolver.mccfrTraverse() in Swift.
    """
    if game.is_terminal:
        return game.payoff(traversing_player)

    acting_player = game.current_player
    is_hero = (acting_player == hero_seat)
    legal = game.legal_actions()
    if not legal:
        return 0.0

    # Build info set key
    if is_hero:
        key = make_hero_key(game, hero_seat)
        info_sets = hero_info_sets
    else:
        key = make_villain_key(game, hero_seat)
        info_sets = villain_info_sets

    if key not in info_sets:
        info_sets[key] = InfoSet()
    info_set = info_sets[key]
    info_set.visit_count += 1

    strategy = info_set.get_strategy(legal)

    if acting_player == traversing_player:
        # TRAVERSING PLAYER: try all actions
        action_values = {}
        node_value = 0.0

        for action in legal:
            child = game.clone()
            prev_street = child.street
            child.apply_action(action)
            deal_if_street_advanced(child, deal, prev_street, hero_seat)

            ev = mccfr_traverse(child, deal, traversing_player, hero_seat,
                               hero_info_sets, villain_info_sets,
                               feature_reservoir, min_visits_for_collection, iteration)
            action_values[action] = ev
            node_value += strategy.get(action, 0) * ev

        # Update regrets
        for action in legal:
            regret = action_values.get(action, 0) - node_value
            info_set.regret_sum[action.value] += regret

        # Collect training data for hero info sets
        if is_hero and info_set.visit_count >= min_visits_for_collection:
            features = extract_features(game, hero_seat)
            # Target: current regret-matched strategy as probability vector
            target = [0.0] * 5
            for a, p in strategy.items():
                target[a.value] = p
            feature_reservoir.add(features, target, iteration)

        return node_value

    else:
        # OPPONENT: sample one action, update strategy sum
        for action in legal:
            info_set.strategy_sum[action.value] += strategy.get(action, 0)

        # Sample action
        r = random.random()
        cumulative = 0.0
        sampled = legal[-1]
        for action in legal:
            cumulative += strategy.get(action, 0)
            if r < cumulative:
                sampled = action
                break

        child = game.clone()
        prev_street = child.street
        child.apply_action(sampled)
        deal_if_street_advanced(child, deal, prev_street, hero_seat)

        return mccfr_traverse(child, deal, traversing_player, hero_seat,
                             hero_info_sets, villain_info_sets,
                             feature_reservoir, min_visits_for_collection, iteration)


# ─── Key Construction ───────────────────────────────────────────────────────

RANK_CHARS = {1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',
              8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K'}


def make_hero_key(game: HeadsUpRazzGame, hero_seat: int) -> str:
    """Build hero info set key. Matches RazzCFRSolver.makeHeroKey()."""
    villain_seat = 1 - hero_seat
    hero = game.players[hero_seat]
    villain = game.players[villain_seat]

    # Hero ranks: sorted, best-5 truncated
    ranks = sorted(c.rank for c in hero.all_cards)
    if len(ranks) > 5:
        ranks = ranks[:5]
    hero_str = ''.join(sorted(RANK_CHARS.get(r, '?') for r in ranks))

    # Villain visible ranks, sorted
    villain_str = ''.join(sorted(RANK_CHARS.get(c.rank, '?') for c in villain.up_cards))

    # Bucketed action history
    history = game.bucketed_action_history

    return f"h:{hero_str}|v:{villain_str}|{history}"


def make_villain_key(game: HeadsUpRazzGame, hero_seat: int) -> str:
    """Build villain info set key (bucketed)."""
    villain_seat = 1 - hero_seat
    from bucketer import classify_hero, classify_villain_visible

    villain = game.players[villain_seat]
    hero = game.players[hero_seat]

    # Villain bucket
    villain_ranks = villain.all_ranks
    bucket = classify_hero(villain_ranks)  # Same bucketing for villain's own hand

    # Hero visible (what villain can see)
    hero_visible = ''.join(sorted(RANK_CHARS.get(c.rank, '?') for c in hero.up_cards))

    history = game.bucketed_action_history

    return f"b:{bucket}|h:{hero_visible}|{history}"


# ─── Training Config ────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    iterations: int = 100_000
    learning_rate: float = 0.001
    batch_size: int = 256
    reservoir_size: int = 200_000
    collect_interval: int = 100       # Collect every N iterations
    train_interval: int = 5_000       # Train every N iterations
    min_visits: int = 50              # Min visits before collecting from info set
    report_interval: int = 1_000
    hand_scope: str = 'premium'       # 'premium', 'top50', 'allUnpaired', 'allHands'


# ─── Hand Scopes ────────────────────────────────────────────────────────────

def get_starting_hands(scope: str) -> List[List[int]]:
    """Generate starting hand rank patterns for a scope."""
    if scope == 'premium':
        # Top 35 hands (high card ≤ 8, no pairs)
        hands = []
        for a in range(1, 14):
            for b in range(a+1, 14):
                for c in range(b+1, 14):
                    if c <= 8:
                        hands.append([a, b, c])
        return hands
    elif scope == 'top50':
        hands = []
        for a in range(1, 14):
            for b in range(a+1, 14):
                for c in range(b+1, 14):
                    if c <= 10:
                        hands.append([a, b, c])
        return hands
    elif scope == 'allUnpaired':
        hands = []
        for a in range(1, 14):
            for b in range(a+1, 14):
                for c in range(b+1, 14):
                    hands.append([a, b, c])
        return hands
    elif scope.startswith('ev_group_'):
        # EV-based groupings: ev_group_1 through ev_group_8
        return _get_ev_group_hands(int(scope.split('_')[-1]))
    else:  # allHands
        hands = []
        # Unpaired
        for a in range(1, 14):
            for b in range(a+1, 14):
                for c in range(b+1, 14):
                    hands.append([a, b, c])
        # Paired
        for pair in range(1, 14):
            for kicker in range(1, 14):
                if kicker != pair:
                    hands.append(sorted([pair, pair, kicker]))
        # Trips
        for r in range(1, 14):
            hands.append([r, r, r])
        return hands


# ─── EV-Based Hand Groups ──────────────────────────────────────────────────

_EV_TABLE_CACHE = None

def _load_ev_table():
    """Load the 3-card 2-player EV table for hand grouping."""
    global _EV_TABLE_CACHE
    if _EV_TABLE_CACHE is not None:
        return _EV_TABLE_CACHE

    import os, json
    # Search multiple locations
    search_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'HORSE+ Master', 'EV_Tables_Razz', 'razz-ev-2p-3card.json'),
        os.path.expanduser('~/Desktop/Poker Apps/HORSE+ Master/EV_Tables_Razz/razz-ev-2p-3card.json'),
    ]
    for path in search_paths:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            results = data.get('results', [])
            ev_table = {}
            for r in results:
                hand_str = r.get('rawHand', r.get('handDescription', ''))
                wr = r.get('winRate', 0)
                if hand_str and wr > 0:
                    ev_table[hand_str] = wr
            _EV_TABLE_CACHE = ev_table
            print(f"[EV Groups] Loaded {len(ev_table)} hands from {path}")
            return ev_table

    print("[EV Groups] WARNING: No EV table found, using heuristic grouping")
    _EV_TABLE_CACHE = {}
    return {}


def _hand_to_ev_key(hand: List[int]) -> str:
    """Convert [1, 2, 3] to 'A23' for EV table lookup."""
    rank_chars = {1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K'}
    return ''.join(rank_chars.get(r, '?') for r in sorted(hand))


def _get_all_hands_with_ev() -> List[tuple]:
    """Get all 455 hands with their EV win rates, sorted by EV descending."""
    ev_table = _load_ev_table()
    all_hands = get_starting_hands('allHands')
    result = []
    for hand in all_hands:
        key = _hand_to_ev_key(hand)
        wr = ev_table.get(key, 0)
        result.append((hand, wr))
    result.sort(key=lambda x: -x[1])
    return result


# EV Group boundaries (win rate thresholds)
EV_GROUP_BOUNDS = {
    1: (70, 100),    # Elite — 35 hands
    2: (60, 70),     # Strong — 60 hands
    3: (50, 60),     # Good — 55 hands
    4: (45, 50),     # Playable — 114 hands
    5: (40, 45),     # Marginal — 55 hands
    6: (35, 40),     # Weak — 26 hands
    7: (25, 35),     # Bad — 62 hands
    8: (0, 25),      # Trash — 48 hands
}

EV_GROUP_NAMES = {
    1: "Elite (70%+)",
    2: "Strong (60-70%)",
    3: "Good (50-60%)",
    4: "Playable (45-50%)",
    5: "Marginal (40-45%)",
    6: "Weak (35-40%)",
    7: "Bad (25-35%)",
    8: "Trash (<25%)",
}


def _get_ev_group_hands(group: int) -> List[List[int]]:
    """Get hands for a specific EV group (1-8)."""
    lo, hi = EV_GROUP_BOUNDS.get(group, (0, 100))
    all_with_ev = _get_all_hands_with_ev()
    return [hand for hand, wr in all_with_ev if lo <= wr < hi]


def get_ev_group_info() -> List[dict]:
    """Get info about all EV groups for the UI."""
    result = []
    for g in range(1, 9):
        hands = _get_ev_group_hands(g)
        lo, hi = EV_GROUP_BOUNDS[g]
        result.append({
            'group': g,
            'name': EV_GROUP_NAMES[g],
            'count': len(hands),
            'ev_range': f"{lo}-{hi}%",
        })
    return result


# ─── Main Training Loop ────────────────────────────────────────────────────

@dataclass
class TrainingState:
    """Mutable training state, readable by the server for progress updates."""
    iteration: int = 0
    total_iterations: int = 0
    loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    hero_info_set_count: int = 0
    villain_info_set_count: int = 0
    reservoir_size: int = 0
    running: bool = False
    should_stop: bool = False
    train_steps: int = 0
    elapsed_seconds: float = 0.0
    hands_in_scope: int = 0


def train_strategy(config: TrainingConfig,
                   state: TrainingState,
                   on_progress: Callable = None) -> StrategyNetwork:
    """Run Mode 1 training: MCCFR + strategy distillation.

    Args:
        config: Training configuration
        state: Mutable state object for progress tracking
        on_progress: Optional callback(state) called at each report interval

    Returns:
        Trained StrategyNetwork
    """
    # Setup
    network = StrategyNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    reservoir = ReservoirBuffer(max_size=config.reservoir_size)

    hero_info_sets: Dict[str, InfoSet] = {}
    villain_info_sets: Dict[str, InfoSet] = {}

    starting_hands = get_starting_hands(config.hand_scope)
    state.hands_in_scope = len(starting_hands)
    state.total_iterations = config.iterations
    state.running = True

    print(f"[Strategy Trainer] Starting: {config.iterations} iterations, "
          f"{len(starting_hands)} hands ({config.hand_scope})")

    start_time = time.time()

    for i in range(config.iterations):
        if state.should_stop:
            break

        state.iteration = i

        # Alternate traversing player
        traversing_player = i % 2

        # Random starting hand
        hero_start_ranks = random.choice(starting_hands)

        # Deal
        deal = deal_hand(hero_start_ranks)
        if deal is None:
            continue

        # Setup game
        game = HeadsUpRazzGame()
        game.deal_third_street(
            p0_hole=deal.hero_hole, p0_up=deal.hero_up[0],
            p1_hole=deal.villain_hole, p1_up=deal.villain_up[0],
        )

        # Run MCCFR traversal
        mccfr_traverse(
            game, deal, traversing_player, hero_seat=0,
            hero_info_sets=hero_info_sets,
            villain_info_sets=villain_info_sets,
            feature_reservoir=reservoir,
            min_visits_for_collection=config.min_visits,
            iteration=i,
        )

        # ── Train network periodically ──────────────────────────────────
        if i > 0 and i % config.train_interval == 0 and reservoir.size >= config.batch_size:
            loss = _train_step(network, optimizer, reservoir, config.batch_size)
            state.loss = loss
            state.loss_history.append(loss)
            state.train_steps += 1

        # ── Progress report ─────────────────────────────────────────────
        if i > 0 and i % config.report_interval == 0:
            state.hero_info_set_count = len(hero_info_sets)
            state.villain_info_set_count = len(villain_info_sets)
            state.reservoir_size = reservoir.size
            state.elapsed_seconds = time.time() - start_time

            if on_progress:
                on_progress(state)

            if i % (config.report_interval * 10) == 0:
                iters_per_sec = i / max(state.elapsed_seconds, 0.01)
                print(f"  [{i}/{config.iterations}] hero={len(hero_info_sets):,} "
                      f"villain={len(villain_info_sets):,} reservoir={reservoir.size:,} "
                      f"loss={state.loss:.4f} ({iters_per_sec:.0f} iter/s)")

    # Final training pass with larger batch
    if reservoir.size >= config.batch_size:
        for _ in range(20):
            loss = _train_step(network, optimizer, reservoir, min(config.batch_size * 4, reservoir.size))
        state.loss = loss
        state.loss_history.append(loss)
        state.train_steps += 20

    state.running = False
    state.elapsed_seconds = time.time() - start_time
    state.hero_info_set_count = len(hero_info_sets)
    state.villain_info_set_count = len(villain_info_sets)

    print(f"[Strategy Trainer] Complete: {state.iteration} iterations, "
          f"{len(hero_info_sets):,} hero info sets, "
          f"{state.train_steps} train steps, "
          f"final loss={state.loss:.4f}, "
          f"time={state.elapsed_seconds:.1f}s")

    return network


def _train_step(network: StrategyNetwork, optimizer: torch.optim.Optimizer,
                reservoir: ReservoirBuffer, batch_size: int) -> float:
    """One training step: sample from reservoir, compute KL div loss, update network."""
    features_batch, targets_batch, _ = reservoir.sample(batch_size)

    features_t = torch.tensor(features_batch, dtype=torch.float32)
    targets_t = torch.tensor(targets_batch, dtype=torch.float32)

    # Clamp targets to avoid log(0)
    targets_t = targets_t.clamp(min=1e-8)

    # Forward
    predicted = network(features_t)

    # KL divergence: sum(target * log(target / predicted))
    loss = F.kl_div(predicted.log(), targets_t, reduction='batchmean')

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ─── Quick validation ───────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Strategy Trainer Quick Test ===\n")

    config = TrainingConfig(
        iterations=5_000,
        train_interval=1_000,
        report_interval=1_000,
        min_visits=10,
        hand_scope='premium',
        reservoir_size=50_000,
    )
    state = TrainingState()

    network = train_strategy(config, state)

    print(f"\nFinal state:")
    print(f"  Iterations: {state.iteration}")
    print(f"  Hero info sets: {state.hero_info_set_count:,}")
    print(f"  Villain info sets: {state.villain_info_set_count:,}")
    print(f"  Reservoir: {state.reservoir_size:,}")
    print(f"  Loss: {state.loss:.4f}")
    print(f"  Train steps: {state.train_steps}")
    print(f"  Time: {state.elapsed_seconds:.1f}s")

    # Test a prediction
    from razz_game import Card
    test_game = HeadsUpRazzGame()
    test_game.deal_third_street(
        p0_hole=[Card(1,0), Card(2,1)], p0_up=Card(3,2),
        p1_hole=[Card(10,0), Card(11,1)], p1_up=Card(13,2),
    )
    features = extract_features(test_game, hero_seat=0)
    probs = network.predict(features)
    action_names = ['fold', 'check', 'call', 'bet', 'raise']
    print(f"\nA23 vs K on 3rd street:")
    for name, p in zip(action_names, probs):
        print(f"  {name}: {p:.3f}")

    # Should lean toward bet/raise (it's a premium hand)
    aggressive = probs[3] + probs[4]  # bet + raise
    print(f"  Aggressive total: {aggressive:.3f}")
    print(f"  {'✅' if aggressive > 0.3 else '⚠️'} {'Good' if aggressive > 0.3 else 'Needs more training'}")

    print("\nStrategy trainer test complete!")
