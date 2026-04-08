"""
Mode 3: Value Network Trainer

Trains a network to predict showdown equity (probability of winning at
showdown) for any game state, using pure Monte Carlo random rollouts.

No strategy network is required — both players act randomly (check/call)
through remaining streets, cards are dealt randomly, and the showdown
result is recorded.

For each iteration:
  1. Pick a random starting hand from the configured scope
  2. Pre-deal all cards for both players
  3. Walk through each street, collecting (features, equity) at each
     decision point
  4. Equity is estimated by MC rollouts: randomly complete remaining
     cards + passive play to showdown, measure win rate

The value network learns to predict equity from the 32-dim feature vector.
"""

import random
import time
import threading
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

from razz_game import (
    HeadsUpRazzGame, Action, Card, make_deck,
    ANTE, BRING_IN, SMALL_BET, BIG_BET,
)
from razz_eval import evaluate as razz_evaluate
from features import extract_features, FEATURE_DIM
from networks import ValueNetwork
from reservoir import ReservoirBuffer
from trainer_strategy import (
    FullDeal, deal_hand, deal_if_street_advanced,
    get_starting_hands, TrainingState,
)


# ─── Value Config ─────────────────────────────────────────────────────────

@dataclass
class ValueConfig:
    iterations: int = 100_000
    learning_rate: float = 0.001
    batch_size: int = 512
    reservoir_size: int = 2_000_000
    train_interval: int = 1_000       # Retrain network every N iterations
    train_steps: int = 200            # SGD steps per retrain
    report_interval: int = 5_000
    hand_scope: str = 'premium'
    mc_samples: int = 200             # Monte Carlo rollouts per state


# ─── Monte Carlo Equity ──────────────────────────────────────────────────

def mc_equity(
    hero_ranks: List[int],
    villain_ranks: List[int],
    hero_cards_remaining: int,
    villain_cards_remaining: int,
    used_cards: set,
    num_samples: int = 200,
) -> float:
    """Estimate hero's showdown equity via Monte Carlo rollouts.

    Given the cards each player currently holds, randomly deal the
    remaining cards and evaluate the showdown.  Returns win probability
    in [0, 1] (ties count as 0.5).

    Args:
        hero_ranks: ranks hero already holds (len 3-7)
        villain_ranks: ranks villain already holds (len 3-7)
        hero_cards_remaining: how many more cards hero needs (0-4)
        villain_cards_remaining: how many more cards villain needs (0-4)
        used_cards: set of (rank, suit) tuples already dealt
        num_samples: number of rollouts
    """
    if hero_cards_remaining == 0 and villain_cards_remaining == 0:
        # Already at showdown — evaluate directly
        _, h_score = razz_evaluate(hero_ranks)
        _, v_score = razz_evaluate(villain_ranks)
        if h_score < v_score:
            return 1.0
        elif h_score > v_score:
            return 0.0
        return 0.5

    # Build available card pool (just ranks — suits don't matter for eval)
    available = []
    for r in range(1, 14):
        for s in range(4):
            if (r, s) not in used_cards:
                available.append(r)

    total_needed = hero_cards_remaining + villain_cards_remaining
    if len(available) < total_needed:
        return 0.5  # Edge case — not enough cards

    wins = 0.0
    for s in range(num_samples):
        random.shuffle(available)
        h_final = list(hero_ranks) + available[:hero_cards_remaining]
        v_final = list(villain_ranks) + available[hero_cards_remaining:total_needed]

        _, h_score = razz_evaluate(h_final)
        _, v_score = razz_evaluate(v_final)

        if h_score < v_score:
            wins += 1.0
        elif h_score == v_score:
            wins += 0.5

        # Yield GIL every 50 samples so Flask threads can serve requests
        if s % 50 == 49:
            time.sleep(0)

    return wins / num_samples


# ─── Collect states from a hand ──────────────────────────────────────────

def _collect_hand_states(
    deal: FullDeal,
    hero_seat: int,
    mc_samples: int,
) -> List[Tuple[List[float], float]]:
    """Play through a hand using PureEV opponent for both sides and collect
    (features, equity) at each hero decision point.

    PureEV makes EV-based bet/fold/raise decisions, so the network sees
    realistic game states with varied pot sizes and fold scenarios.

    Returns list of (feature_vector, equity) pairs.
    """
    from opponents import pure_ev_action

    villain_seat = 1 - hero_seat
    game = HeadsUpRazzGame()
    game.deal_third_street(
        p0_hole=deal.hero_hole if hero_seat == 0 else deal.villain_hole,
        p0_up=deal.hero_up[0] if hero_seat == 0 else deal.villain_up[0],
        p1_hole=deal.villain_hole if hero_seat == 0 else deal.hero_hole,
        p1_up=deal.villain_up[0] if hero_seat == 0 else deal.hero_up[0],
    )

    samples = []

    # Build set of all dealt cards for MC exclusion
    all_dealt = set()
    for c in deal.hero_hole:
        all_dealt.add((c.rank, c.suit))
    for c in deal.hero_up:
        all_dealt.add((c.rank, c.suit))
    all_dealt.add((deal.hero_hole_7th.rank, deal.hero_hole_7th.suit))
    for c in deal.villain_hole:
        all_dealt.add((c.rank, c.suit))
    for c in deal.villain_up:
        all_dealt.add((c.rank, c.suit))
    all_dealt.add((deal.villain_hole_7th.rank, deal.villain_hole_7th.suit))

    max_actions = 100  # Safety limit
    action_count = 0

    while not game.is_terminal and action_count < max_actions:
        acting = game.current_player
        legal = game.legal_actions()
        if not legal:
            break

        # Collect feature + equity for hero's decision points
        if acting == hero_seat:
            features = extract_features(game, hero_seat)

            hero = game.players[hero_seat]
            villain = game.players[villain_seat]
            hero_ranks = hero.all_ranks
            villain_ranks = villain.all_ranks

            # Cards remaining: total 7 per player
            hero_remaining = 7 - hero.card_count
            villain_remaining = 7 - villain.card_count

            equity = mc_equity(
                hero_ranks, villain_ranks,
                hero_remaining, villain_remaining,
                all_dealt, mc_samples,
            )
            samples.append((features, equity))

        # Both players use PureEV for realistic play
        prev_street = game.street
        action = pure_ev_action(game, acting)
        if action not in legal:
            # Fallback: check > call > first legal
            if Action.CHECK in legal:
                action = Action.CHECK
            elif Action.CALL in legal:
                action = Action.CALL
            else:
                action = legal[0]
        game.apply_action(action)

        deal_if_street_advanced(game, deal, prev_street, hero_seat)
        action_count += 1

    return samples


# ─── Network Training ────────────────────────────────────────────────────

def _train_value_network(
    net: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    reservoir: ReservoirBuffer,
    batch_size: int,
    num_steps: int,
) -> float:
    """Train the value network on sampled (features, equity) pairs.

    Loss: Iteration-weighted MSE with sigmoid output (equity in [0,1]).
    """
    total_loss = 0.0
    for step in range(num_steps):
        features_batch, targets_batch, iters_batch = reservoir.sample(batch_size)
        if not features_batch:
            break

        features_t = torch.tensor(features_batch, dtype=torch.float32)
        targets_t = torch.tensor(targets_batch, dtype=torch.float32)  # shape [B, 1]

        # Linear CFR weighting
        iters_t = torch.tensor(iters_batch, dtype=torch.float32)
        max_iter = iters_t.max().clamp(min=1.0)
        weights = (iters_t / max_iter).clamp(min=0.1)

        predicted = torch.sigmoid(net(features_t))  # [0, 1]
        per_sample_loss = ((predicted - targets_t) ** 2).mean(dim=1)
        loss = (per_sample_loss * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(num_steps, 1)


# ─── Main Training Loop ──────────────────────────────────────────────────

def train_value(
    config: ValueConfig,
    state: TrainingState,
    resume_state: dict = None,
) -> dict:
    """Run value network training via Monte Carlo equity estimation.

    Returns dict with 'value_net', 'value_reservoir', 'base_iteration'.
    """
    # Resume or create fresh
    if resume_state:
        value_net = resume_state['value_net']
        value_reservoir = resume_state['value_reservoir']
        base_iteration = resume_state.get('base_iteration', 0)

        if value_reservoir.max_size != config.reservoir_size:
            value_reservoir.resize(config.reservoir_size)

        print(f"[Value] Resuming from iteration {base_iteration}, "
              f"reservoir: {len(value_reservoir)}/{value_reservoir.max_size}")
    else:
        value_net = ValueNetwork()
        value_reservoir = ReservoirBuffer(max_size=config.reservoir_size)
        base_iteration = 0

    optimizer = torch.optim.Adam(value_net.parameters(), lr=config.learning_rate)

    starting_hands = get_starting_hands(config.hand_scope)
    state.hands_in_scope = len(starting_hands)
    state.total_iterations = base_iteration + config.iterations
    state.running = True

    print(f"[Value] Starting: {config.iterations} iterations, "
          f"{len(starting_hands)} hands ({config.hand_scope}), "
          f"{config.mc_samples} MC samples/state")
    print(f"[Value] Network: {sum(p.numel() for p in value_net.parameters()):,} params")

    start_time = time.time()
    loss = 0.0

    for i in range(config.iterations):
        if state.should_stop:
            break

        state.iteration = base_iteration + i

        # Random starting hand
        hero_start_ranks = random.choice(starting_hands)

        # Deal full hand
        deal = deal_hand(hero_start_ranks)
        if deal is None:
            continue

        # Collect (features, equity) pairs from this hand
        hero_seat = 0
        hand_samples = _collect_hand_states(deal, hero_seat, config.mc_samples)

        for features, equity in hand_samples:
            value_reservoir.add(features, [equity], i)

        # Yield GIL periodically so Flask can serve status requests
        if i % 10 == 9:
            time.sleep(0)

        # ── Retrain network periodically ───────────────────────────────
        if i > 0 and i % config.train_interval == 0:
            if value_reservoir.size >= config.batch_size:
                loss = _train_value_network(
                    value_net, optimizer, value_reservoir,
                    config.batch_size, config.train_steps,
                )
                state.loss = loss
                state.loss_history.append(loss)
                state.train_steps += 1

        # ── Progress report ─────────────────────────────────────────────
        if i > 0 and i % config.report_interval == 0:
            state.reservoir_size = value_reservoir.size
            state.hero_info_set_count = value_reservoir.size
            state.elapsed_seconds = time.time() - start_time

            iters_per_sec = i / max(state.elapsed_seconds, 0.01)
            print(f"  [{i}/{config.iterations}] reservoir={value_reservoir.size:,} "
                  f"loss={loss:.6f} ({iters_per_sec:.0f} iter/s)")

        # ── Periodic checkpoint (every 100K) ──────────────────────────
        if i > 0 and i % 100_000 == 0:
            from checkpoint import save_value_checkpoint as _save_vcp
            _cp_state = {
                'value_net': value_net,
                'value_reservoir': value_reservoir,
                'base_iteration': base_iteration + i,
            }
            threading.Thread(target=_save_vcp, args=(_cp_state,), daemon=True).start()
            print(f"  [Checkpoint] Auto-saving at iteration {base_iteration + i}...")

    # Final training passes
    print("[Value] Final training passes...")
    if value_reservoir.size >= config.batch_size:
        for _ in range(50):
            loss = _train_value_network(
                value_net, optimizer, value_reservoir,
                min(config.batch_size * 2, value_reservoir.size),
                config.train_steps,
            )

    state.running = False
    state.loss = loss
    state.elapsed_seconds = time.time() - start_time

    print(f"[Value] Complete: {state.iteration} iterations, "
          f"reservoir={value_reservoir.size:,}, "
          f"loss={loss:.6f}, time={state.elapsed_seconds:.1f}s")

    return {
        'value_net': value_net,
        'value_reservoir': value_reservoir,
        'base_iteration': base_iteration + i + 1,
    }


# ─── Quick validation ───────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Value Network Quick Test ===\n")

    config = ValueConfig(
        iterations=5_000,
        train_interval=1_000,
        report_interval=2_500,
        train_steps=50,
        batch_size=256,
        hand_scope='premium',
        reservoir_size=500_000,
        mc_samples=100,
    )
    state = TrainingState()

    result = train_value(config, state)
    value_net = result['value_net']

    print(f"\nFinal state:")
    print(f"  Iterations: {state.iteration:,}")
    print(f"  Reservoir: {state.reservoir_size:,}")
    print(f"  Loss: {state.loss:.6f}")
    print(f"  Time: {state.elapsed_seconds:.1f}s")

    # Test predictions
    import torch
    from razz_game import Card

    def predict_equity(net, game, hero_seat):
        feats = extract_features(game, hero_seat)
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            return torch.sigmoid(net(x)).item()

    # A23 vs KQJ — hero should have high equity
    g1 = HeadsUpRazzGame()
    g1.deal_third_street(
        p0_hole=[Card(1, 0), Card(2, 1)], p0_up=Card(3, 2),
        p1_hole=[Card(11, 0), Card(12, 1)], p1_up=Card(13, 2),
    )
    eq1 = predict_equity(value_net, g1, 0)
    print(f"\nA23 vs KQJ equity: {eq1:.1%}")

    # KQJ vs A23 — hero should have low equity
    g2 = HeadsUpRazzGame()
    g2.deal_third_street(
        p0_hole=[Card(11, 0), Card(12, 1)], p0_up=Card(13, 2),
        p1_hole=[Card(1, 0), Card(2, 1)], p1_up=Card(3, 2),
    )
    eq2 = predict_equity(value_net, g2, 0)
    print(f"KQJ vs A23 equity: {eq2:.1%}")

    # 456 vs 789 — moderate edge
    g3 = HeadsUpRazzGame()
    g3.deal_third_street(
        p0_hole=[Card(4, 0), Card(5, 1)], p0_up=Card(6, 2),
        p1_hole=[Card(7, 0), Card(8, 1)], p1_up=Card(9, 2),
    )
    eq3 = predict_equity(value_net, g3, 0)
    print(f"456 vs 789 equity: {eq3:.1%}")

    print(f"\nExpected: A23 > 456 > KQJ")
    print(f"Actual:   {eq1:.1%} {'>' if eq1 > eq3 else '<'} {eq3:.1%} {'>' if eq3 > eq2 else '<'} {eq2:.1%}")

    print("\nValue network test complete!")
