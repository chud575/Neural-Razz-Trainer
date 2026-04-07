"""
Mode 2: Deep CFR Trainer

Based on Brown et al. 2019 "Deep Counterfactual Regret Minimization"

Key difference from Mode 1: NO tabular info sets. The neural network
IS the solver. It predicts regrets directly from game features, and
regret matching converts those predictions into strategies.

Two networks:
  - Advantage (regret) network: predicts counterfactual advantages per action
  - Strategy network: trained on the average strategy produced over time

Two reservoirs:
  - Advantage reservoir: (features, advantages, iteration) triples
  - Strategy reservoir: (features, strategy) pairs for the average strategy

Flow per iteration:
  1. Use the advantage network to get current strategy (via regret matching)
  2. Traverse the game tree (external sampling)
  3. At traversing player's nodes: compute advantages, store in reservoir
  4. At opponent's nodes: store current strategy in strategy reservoir
  5. Periodically retrain both networks from their reservoirs

At deployment: only the strategy network is needed.
"""

import random
import time
import math
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from razz_game import HeadsUpRazzGame, Action, Card, make_deck
from features import extract_features, FEATURE_DIM
from networks import RegretNetwork, StrategyNetwork
from reservoir import ReservoirBuffer
from trainer_strategy import (
    FullDeal, deal_hand, deal_if_street_advanced,
    get_starting_hands, TrainingState,
)


# ─── Deep CFR Config ────────────────────────────────────────────────────────

@dataclass
class DeepCFRConfig:
    iterations: int = 100_000
    advantage_lr: float = 0.001
    strategy_lr: float = 0.001
    batch_size: int = 512
    advantage_reservoir_size: int = 2_000_000
    strategy_reservoir_size: int = 2_000_000
    train_interval: int = 1_000        # Retrain networks every N iterations
    advantage_train_steps: int = 200   # SGD steps per retrain
    strategy_train_steps: int = 200
    report_interval: int = 5_000
    hand_scope: str = 'premium'
    enable_hindsight: bool = False      # Enable hindsight correction pass
    hindsight_weight: float = 1.5       # Weight multiplier for hindsight samples


# ─── Deep CFR Traversal ────────────────────────────────────────────────────

def deep_cfr_traverse(
    game: HeadsUpRazzGame,
    deal: FullDeal,
    traversing_player: int,
    hero_seat: int,
    advantage_net: RegretNetwork,
    advantage_reservoir: ReservoirBuffer,
    strategy_reservoir: ReservoirBuffer,
    iteration: int,
    decision_log: List = None,
) -> float:
    """External-sampling MCCFR traversal using neural network for strategy.

    If decision_log is provided, records hero's decision points for hindsight analysis.

    Instead of looking up a tabular info set, we query the advantage network
    to get regret estimates, then apply regret matching for the strategy.

    Traversing player: explore ALL actions, compute advantages.
    Opponent: sample ONE action from the network's strategy.
    """
    if game.is_terminal:
        return game.payoff(traversing_player)

    acting_player = game.current_player
    legal = game.legal_actions()
    if not legal:
        return 0.0

    # Get features for current game state from acting player's perspective
    features = extract_features(game, acting_player)

    # Get strategy from advantage network via regret matching
    strategy = _get_network_strategy(advantage_net, features, legal)

    if acting_player == traversing_player:
        # TRAVERSING PLAYER: explore all actions, compute advantages
        action_values = {}
        node_value = 0.0

        for action in legal:
            child = game.clone()
            prev_street = child.street
            child.apply_action(action)
            deal_if_street_advanced(child, deal, prev_street, hero_seat)

            ev = deep_cfr_traverse(
                child, deal, traversing_player, hero_seat,
                advantage_net, advantage_reservoir, strategy_reservoir,
                iteration, decision_log,
            )
            action_values[action] = ev
            node_value += strategy.get(action, 0) * ev

        # Compute advantages (counterfactual regrets)
        advantages = [0.0] * 5
        for action in legal:
            advantages[action.value] = action_values.get(action, 0) - node_value

        # Store in advantage reservoir with iteration number for Linear CFR weighting
        advantage_reservoir.add(features, advantages, iteration)

        # Store TRAVERSING PLAYER's strategy in strategy reservoir.
        strategy_target = [0.0] * 5
        for a, p in strategy.items():
            strategy_target[a.value] = p
        strategy_reservoir.add(features, strategy_target, iteration)

        # Log decision point for hindsight analysis
        if decision_log is not None:
            decision_log.append({
                'game': game.clone(),
                'strategy': dict(strategy),
                'features': features,
                'street': game.street,
            })

        return node_value

    else:
        # OPPONENT: pick from configured opponent mix (includes self-play as an option)
        from opponents import pick_training_opponent, mixed_action

        opp_type = pick_training_opponent()

        if opp_type == 'self_play':
            # Self-play: sample from network strategy
            r = random.random()
            cumulative = 0.0
            sampled = legal[-1]
            for action in legal:
                cumulative += strategy.get(action, 0)
                if r < cumulative:
                    sampled = action
                    break
        else:
            # External opponent
            sampled = mixed_action(game, acting_player, opp_type)
            if sampled not in legal:
                sampled = legal[0]

        child = game.clone()
        prev_street = child.street
        child.apply_action(sampled)
        deal_if_street_advanced(child, deal, prev_street, hero_seat)

        return deep_cfr_traverse(
            child, deal, traversing_player, hero_seat,
            advantage_net, advantage_reservoir, strategy_reservoir,
            iteration, decision_log,
        )


def _get_network_strategy(net: RegretNetwork, features: List[float],
                           legal: List[Action]) -> Dict[Action, float]:
    """Get strategy from the advantage network via regret matching."""
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        regrets = net(x).squeeze(0).tolist()

    # Regret matching: positive regrets normalized to legal actions only
    strategy = {}
    positive_sum = 0.0
    for a in legal:
        r = max(0.0, regrets[a.value])
        strategy[a] = r
        positive_sum += r

    if positive_sum > 0:
        for a in legal:
            strategy[a] /= positive_sum
    else:
        uniform = 1.0 / len(legal)
        for a in legal:
            strategy[a] = uniform

    return strategy


# ─── Network Training ──────────────────────────────────────────────────────

def _train_advantage_network(net: RegretNetwork, optimizer: torch.optim.Optimizer,
                              reservoir: ReservoirBuffer, batch_size: int,
                              num_steps: int) -> float:
    """Train the advantage network on sampled (features, advantages) pairs.

    Loss: Iteration-weighted MSE (Linear CFR — later samples weighted more).
    """
    total_loss = 0.0
    for step in range(num_steps):
        features_batch, targets_batch, iters_batch = reservoir.sample(batch_size)
        if not features_batch:
            break

        features_t = torch.tensor(features_batch, dtype=torch.float32)
        targets_t = torch.tensor(targets_batch, dtype=torch.float32)

        # Linear CFR weighting: weight = iteration / max_iteration
        iters_t = torch.tensor(iters_batch, dtype=torch.float32)
        max_iter = iters_t.max().clamp(min=1.0)
        weights = (iters_t / max_iter).clamp(min=0.1)  # Floor at 0.1 so early data isn't ignored

        predicted = net(features_t)
        per_sample_loss = ((predicted - targets_t) ** 2).mean(dim=1)  # MSE per sample
        loss = (per_sample_loss * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(num_steps, 1)


def _train_strategy_network(net: StrategyNetwork, optimizer: torch.optim.Optimizer,
                             reservoir: ReservoirBuffer, batch_size: int,
                             num_steps: int) -> float:
    """Train the strategy network on sampled (features, strategy) pairs.

    Loss: Iteration-weighted cross-entropy (later iterations = better strategy).
    This network learns the AVERAGE strategy, which converges to Nash.
    """
    total_loss = 0.0
    for step in range(num_steps):
        features_batch, targets_batch, iters_batch = reservoir.sample(batch_size)
        if not features_batch:
            break

        features_t = torch.tensor(features_batch, dtype=torch.float32)
        targets_t = torch.tensor(targets_batch, dtype=torch.float32).clamp(min=1e-8)

        # Linear CFR weighting
        iters_t = torch.tensor(iters_batch, dtype=torch.float32)
        max_iter = iters_t.max().clamp(min=1.0)
        weights = (iters_t / max_iter).clamp(min=0.1)

        predicted = net(features_t).clamp(min=1e-8)
        # Cross-entropy: -sum(target * log(predicted)) per sample
        per_sample_loss = -(targets_t * predicted.log()).sum(dim=1)
        loss = (per_sample_loss * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(num_steps, 1)


# ─── Main Training Loop ────────────────────────────────────────────────────

def train_deep_cfr(config: DeepCFRConfig,
                   state: TrainingState,
                   on_progress: Callable = None,
                   resume_state: dict = None) -> dict:
    """Run Deep CFR training.

    Args:
        resume_state: If provided, resume from existing networks/reservoirs.
            Keys: 'advantage_net', 'strategy_net', 'advantage_reservoir',
                  'strategy_reservoir', 'base_iteration'

    Returns dict with 'strategy_net', 'advantage_net', 'advantage_reservoir',
    'strategy_reservoir' — for persistence between runs.
    """
    # Resume or create fresh
    if resume_state:
        advantage_net = resume_state['advantage_net']
        strategy_net = resume_state['strategy_net']
        advantage_reservoir = resume_state['advantage_reservoir']
        strategy_reservoir = resume_state['strategy_reservoir']
        base_iteration = resume_state.get('base_iteration', 0)

        # Resize reservoirs if config changed
        if advantage_reservoir.max_size != config.advantage_reservoir_size:
            advantage_reservoir.resize(config.advantage_reservoir_size)
        if strategy_reservoir.max_size != config.strategy_reservoir_size:
            strategy_reservoir.resize(config.strategy_reservoir_size)

        print(f"[Deep CFR] Resuming from iteration {base_iteration}, "
              f"adv reservoir: {len(advantage_reservoir)}/{advantage_reservoir.max_size}, "
              f"strat reservoir: {len(strategy_reservoir)}/{strategy_reservoir.max_size}")
    else:
        advantage_net = RegretNetwork()
        strategy_net = StrategyNetwork()
        advantage_reservoir = ReservoirBuffer(max_size=config.advantage_reservoir_size)
        strategy_reservoir = ReservoirBuffer(max_size=config.strategy_reservoir_size)
        base_iteration = 0

    advantage_opt = torch.optim.Adam(advantage_net.parameters(), lr=config.advantage_lr)
    strategy_opt = torch.optim.Adam(strategy_net.parameters(), lr=config.strategy_lr)

    starting_hands = get_starting_hands(config.hand_scope)
    state.hands_in_scope = len(starting_hands)
    state.total_iterations = base_iteration + config.iterations
    state.running = True

    # Preload external opponent models
    from opponents import preload_all as _preload_opponents
    _preload_opponents()

    print(f"[Deep CFR] Starting: {config.iterations} iterations, "
          f"{len(starting_hands)} hands ({config.hand_scope})")
    print(f"[Deep CFR] Advantage net: {sum(p.numel() for p in advantage_net.parameters()):,} params")
    print(f"[Deep CFR] Strategy net: {sum(p.numel() for p in strategy_net.parameters()):,} params")
    from opponents import _active_opponents, _balanced_mode
    if _active_opponents:
        opp_str = ', '.join(_active_opponents)
        mode_str = 'balanced' if _balanced_mode else 'weighted'
        print(f"[Deep CFR] Opponents: {opp_str} ({mode_str})")
    else:
        print(f"[Deep CFR] Opponents: default mix (TAG/LAG/PureEV/ReBeL/CFR/CS/Random + self-play)")

    start_time = time.time()
    adv_loss = 0.0
    strat_loss = 0.0

    for i in range(config.iterations):
        if state.should_stop:
            break

        state.iteration = base_iteration + i

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

        # Run Deep CFR traversal
        decision_log = [] if config.enable_hindsight else None
        deep_cfr_traverse(
            game, deal, traversing_player, hero_seat=0,
            advantage_net=advantage_net,
            advantage_reservoir=advantage_reservoir,
            strategy_reservoir=strategy_reservoir,
            iteration=i,
            decision_log=decision_log,
        )

        # Hindsight correction pass — replay with perfect info
        if config.enable_hindsight and decision_log:
            from hindsight import hindsight_pass
            hero_all = [c.rank for c in deal.hero_hole] + [c.rank for c in deal.hero_up] + [deal.hero_hole_7th.rank]
            villain_all = [c.rank for c in deal.villain_hole] + [c.rank for c in deal.villain_up] + [deal.villain_hole_7th.rank]
            deal_info = {'hero_ranks': hero_all, 'villain_ranks': villain_all}

            corrections = hindsight_pass(decision_log, deal_info, hero_seat=0)
            for corr in corrections:
                # Add hindsight corrections to advantage reservoir with boosted weight
                # Use iteration * hindsight_weight so they're weighted higher
                boosted_iter = int(i * config.hindsight_weight)
                advantage_reservoir.add(corr.features, corr.advantages, boosted_iter)

        # ── Retrain networks periodically ───────────────────────────────
        if i > 0 and i % config.train_interval == 0:
            if advantage_reservoir.size >= config.batch_size:
                adv_loss = _train_advantage_network(
                    advantage_net, advantage_opt, advantage_reservoir,
                    config.batch_size, config.advantage_train_steps,
                )

            if strategy_reservoir.size >= config.batch_size:
                strat_loss = _train_strategy_network(
                    strategy_net, strategy_opt, strategy_reservoir,
                    config.batch_size, config.strategy_train_steps,
                )

            state.loss = strat_loss  # Report strategy loss as primary
            state.loss_history.append(strat_loss)
            state.train_steps += 1

        # ── Progress report ─────────────────────────────────────────────
        if i > 0 and i % config.report_interval == 0:
            state.reservoir_size = advantage_reservoir.size + strategy_reservoir.size
            state.hero_info_set_count = advantage_reservoir.size  # Repurpose for display
            state.villain_info_set_count = strategy_reservoir.size
            state.elapsed_seconds = time.time() - start_time

            if on_progress:
                on_progress(state)

            iters_per_sec = i / max(state.elapsed_seconds, 0.01)
            print(f"  [{i}/{config.iterations}] adv_reservoir={advantage_reservoir.size:,} "
                  f"strat_reservoir={strategy_reservoir.size:,} "
                  f"adv_loss={adv_loss:.4f} strat_loss={strat_loss:.4f} "
                  f"({iters_per_sec:.0f} iter/s)")

        # ── Periodic checkpoint (every 100K) ──────────────────────────────
        if i > 0 and i % 100_000 == 0:
            import threading as _thr
            _cp_state = {
                'strategy_net': strategy_net,
                'advantage_net': advantage_net,
                'advantage_reservoir': advantage_reservoir,
                'strategy_reservoir': strategy_reservoir,
                'base_iteration': base_iteration + i,
            }
            from checkpoint import save_checkpoint as _save_cp
            _thr.Thread(target=_save_cp, args=(_cp_state,), daemon=True).start()
            print(f"  [Checkpoint] Auto-saving at iteration {base_iteration + i}...")

    # Final training passes
    print("[Deep CFR] Final training passes...")
    if advantage_reservoir.size >= config.batch_size:
        for _ in range(50):
            adv_loss = _train_advantage_network(
                advantage_net, advantage_opt, advantage_reservoir,
                min(config.batch_size * 2, advantage_reservoir.size),
                config.advantage_train_steps,
            )

    if strategy_reservoir.size >= config.batch_size:
        for _ in range(50):
            strat_loss = _train_strategy_network(
                strategy_net, strategy_opt, strategy_reservoir,
                min(config.batch_size * 2, strategy_reservoir.size),
                config.strategy_train_steps,
            )

    state.running = False
    state.loss = strat_loss
    state.elapsed_seconds = time.time() - start_time

    print(f"[Deep CFR] Complete: {state.iteration} iterations, "
          f"adv_reservoir={advantage_reservoir.size:,}, "
          f"strat_reservoir={strategy_reservoir.size:,}, "
          f"adv_loss={adv_loss:.4f}, strat_loss={strat_loss:.4f}, "
          f"time={state.elapsed_seconds:.1f}s")

    return {
        'strategy_net': strategy_net,
        'advantage_net': advantage_net,
        'advantage_reservoir': advantage_reservoir,
        'strategy_reservoir': strategy_reservoir,
        'base_iteration': base_iteration + i + 1,
    }


# ─── Quick validation ───────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Deep CFR Quick Test ===\n")

    config = DeepCFRConfig(
        iterations=10_000,
        train_interval=2_000,
        report_interval=5_000,
        advantage_train_steps=50,
        strategy_train_steps=50,
        batch_size=256,
        hand_scope='premium',
        advantage_reservoir_size=500_000,
        strategy_reservoir_size=500_000,
    )
    state = TrainingState()

    strategy_net = train_deep_cfr(config, state)

    print(f"\nFinal state:")
    print(f"  Iterations: {state.iteration:,}")
    print(f"  Advantage reservoir: {state.hero_info_set_count:,}")
    print(f"  Strategy reservoir: {state.villain_info_set_count:,}")
    print(f"  Strategy loss: {state.loss:.4f}")
    print(f"  Train steps: {state.train_steps}")
    print(f"  Time: {state.elapsed_seconds:.1f}s")

    # Test predictions
    from razz_game import Card
    from features import extract_features

    action_names = ['fold', 'check', 'call', 'bet', 'raise']

    # A23 vs K
    g = HeadsUpRazzGame()
    g.deal_third_street(
        p0_hole=[Card(1,0), Card(2,1)], p0_up=Card(3,2),
        p1_hole=[Card(10,0), Card(11,1)], p1_up=Card(13,2),
    )
    feats = extract_features(g, 0)
    probs = strategy_net.predict(feats)
    print(f"\nA23 vs K (3rd st): {' '.join(f'{n}={p:.1%}' for n,p in zip(action_names, probs))}")

    # JQK vs A
    g2 = HeadsUpRazzGame()
    g2.deal_third_street(
        p0_hole=[Card(11,0), Card(12,1)], p0_up=Card(13,2),
        p1_hole=[Card(2,0), Card(3,1)], p1_up=Card(1,2),
    )
    feats2 = extract_features(g2, 0)
    probs2 = strategy_net.predict(feats2)
    print(f"JQK vs A (3rd st): {' '.join(f'{n}={p:.1%}' for n,p in zip(action_names, probs2))}")

    # Arena test
    print("\n=== Arena Tests ===")
    from arena_test import run_arena
    for opp in ['calling_station', 'tag', 'lag', 'random']:
        # Arena uses network.predict() — works with StrategyNetwork
        result = run_arena(strategy_net, num_hands=2000, opponent_type=opp)
        print(f"  vs {opp:20s}: Win={result['win_rate']}% BB/100={result['bb_per_100']:+.1f} "
              f"Fold={result['fold_rate']}% SD_Win={result['sd_win_rate']}%")

    print("\nDeep CFR test complete!")
