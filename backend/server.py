"""
Neural Razz Trainer — Flask API Server

Endpoints:
  POST /api/train/start   — Start training (mode, iterations, etc.)
  POST /api/train/stop    — Stop training
  GET  /api/train/status  — Training progress
  POST /api/test/arena    — Run arena test
  GET  /api/model/info    — Current model info
  POST /api/model/export  — Export model to JSON
  POST /api/model/predict — Single inference
"""

import os
import sys
import json
import time
import threading
from flask import Flask, request, jsonify

from trainer_strategy import TrainingConfig, TrainingState, train_strategy
from trainer_regret import DeepCFRConfig, train_deep_cfr
from networks import StrategyNetwork, export_to_json
from features import extract_features, FEATURE_DIM
from razz_game import HeadsUpRazzGame, Card, Action
from checkpoint import save_checkpoint, load_checkpoint, has_checkpoint, delete_checkpoint

app = Flask(__name__)

# ─── Global State ───────────────────────────────────────────────────────────

_training_state = TrainingState()
_current_network: StrategyNetwork = None
_training_thread: threading.Thread = None
_lock = threading.Lock()

# Persistent state for resume across training runs
_deep_cfr_state: dict = None  # Holds networks + reservoirs between runs
_current_mode: str = None


# ─── Training Endpoints ────────────────────────────────────────────────────

@app.route('/api/train/start', methods=['POST'])
def train_start():
    global _training_state, _current_network, _training_thread, _deep_cfr_state, _current_mode

    if _training_state.running:
        return jsonify({'error': 'Training already running'}), 400

    data = request.json or {}
    mode = data.get('mode', 'strategy')
    resuming = False

    # Create fresh training state but preserve cumulative iteration count
    base_iter = 0
    if _deep_cfr_state and mode == 'deep_cfr' and _current_mode == 'deep_cfr':
        base_iter = _deep_cfr_state.get('base_iteration', 0)
        resuming = True

    _training_state = TrainingState(total_iterations=base_iter + data.get('iterations', 100_000))
    _training_state.iteration = base_iter
    _current_mode = mode

    if mode == 'deep_cfr':
        config = DeepCFRConfig(
            iterations=data.get('iterations', 100_000),
            advantage_lr=data.get('learning_rate', 0.001),
            strategy_lr=data.get('learning_rate', 0.001),
            batch_size=data.get('batch_size', 512),
            train_interval=data.get('train_interval', 1_000),
            advantage_train_steps=data.get('advantage_train_steps', 200),
            strategy_train_steps=data.get('strategy_train_steps', 200),
            report_interval=data.get('report_interval', 5_000),
            hand_scope=data.get('hand_scope', 'premium'),
            advantage_reservoir_size=data.get('reservoir_size', 2_000_000),
            strategy_reservoir_size=data.get('reservoir_size', 2_000_000),
            enable_hindsight=data.get('enable_hindsight', False),
            hindsight_weight=data.get('hindsight_weight', 1.5),
        )

        # Resume from in-memory state, or disk checkpoint, or fresh start
        resume = None
        if resuming:
            resume = _deep_cfr_state
        elif has_checkpoint():
            resume = load_checkpoint()
            if resume:
                base_iter = resume.get('base_iteration', 0)
                _training_state.total_iterations = base_iter + config.iterations
                _training_state.iteration = base_iter
                resuming = True

        # Configure opponents from request
        opp_config = data.get('opponent_config')
        if opp_config:
            from opponents import configure_opponents
            configure_opponents(
                enabled=opp_config.get('enabled'),
                weights=opp_config.get('weights'),
                balanced=opp_config.get('balanced', False),
            )

        def _run():
            global _current_network, _deep_cfr_state
            result = train_deep_cfr(config, _training_state, resume_state=resume)
            with _lock:
                _current_network = result['strategy_net']
                _deep_cfr_state = result
            # Save checkpoint on background thread so it doesn't block health checks
            threading.Thread(target=save_checkpoint, args=(result,), daemon=True).start()
    else:
        config = TrainingConfig(
            iterations=data.get('iterations', 100_000),
            learning_rate=data.get('learning_rate', 0.001),
            batch_size=data.get('batch_size', 256),
            train_interval=data.get('train_interval', 5_000),
            report_interval=data.get('report_interval', 1_000),
            min_visits=data.get('min_visits', 50),
            hand_scope=data.get('hand_scope', 'premium'),
            reservoir_size=data.get('reservoir_size', 200_000),
        )

        def _run():
            global _current_network
            network = train_strategy(config, _training_state)
            with _lock:
                _current_network = network

    _training_thread = threading.Thread(target=_run, daemon=True)
    _training_thread.start()

    from trainer_strategy import get_starting_hands
    return jsonify({
        'status': 'resumed' if resuming else 'started',
        'mode': mode,
        'iterations': data.get('iterations', 100_000),
        'base_iteration': base_iter,
        'hand_scope': data.get('hand_scope', 'premium'),
        'hands_count': len(get_starting_hands(data.get('hand_scope', 'premium'))),
    })


@app.route('/api/train/reset', methods=['POST'])
def train_reset():
    global _training_state, _current_network, _deep_cfr_state, _current_mode
    if _training_state.running:
        return jsonify({'error': 'Cannot reset while training is running'}), 400
    _training_state = TrainingState()
    _current_network = None
    _deep_cfr_state = None
    _current_mode = None
    delete_checkpoint()
    return jsonify({'status': 'reset', 'message': 'All networks, reservoirs, state, and disk checkpoint cleared'})


@app.route('/api/checkpoint/load', methods=['POST'])
def checkpoint_load():
    global _training_state, _current_network, _deep_cfr_state, _current_mode

    if _training_state.running:
        return jsonify({'error': 'Cannot load while training is running'}), 400

    if not has_checkpoint():
        return jsonify({'status': 'not_found', 'message': 'No checkpoint found on disk'})

    loaded = load_checkpoint()
    if not loaded:
        return jsonify({'status': 'error', 'message': 'Failed to load checkpoint'})

    _deep_cfr_state = loaded
    _current_network = loaded['strategy_net']
    _current_mode = 'deep_cfr'
    base_iter = loaded.get('base_iteration', 0)
    _training_state.iteration = base_iter
    _training_state.total_iterations = base_iter

    return jsonify({
        'status': 'loaded',
        'base_iteration': base_iter,
        'advantage_reservoir': len(loaded.get('advantage_reservoir', [])),
        'strategy_reservoir': len(loaded.get('strategy_reservoir', [])),
    })


@app.route('/api/hands/groups', methods=['GET'])
def hands_groups():
    from trainer_strategy import get_ev_group_info
    return jsonify({'groups': get_ev_group_info()})


@app.route('/api/train/stop', methods=['POST'])
def train_stop():
    _training_state.should_stop = True
    return jsonify({'status': 'stopping'})


@app.route('/api/train/status', methods=['GET'])
def train_status():
    s = _training_state
    return jsonify({
        'running': s.running,
        'iteration': s.iteration,
        'total_iterations': s.total_iterations,
        'loss': s.loss,
        'loss_history': s.loss_history[-100:],  # Last 100 points
        'hero_info_sets': s.hero_info_set_count,
        'villain_info_sets': s.villain_info_set_count,
        'reservoir_size': s.reservoir_size,
        'train_steps': s.train_steps,
        'elapsed_seconds': s.elapsed_seconds,
        'hands_in_scope': s.hands_in_scope,
        'has_model': _current_network is not None,
    })


# ─── Curriculum Injection ──────────────────────────────────────────────────

@app.route('/api/train/inject-curriculum', methods=['POST'])
def inject_curriculum():
    global _deep_cfr_state
    if _deep_cfr_state is None:
        return jsonify({'error': 'No training state — train first or load checkpoint'}), 400
    if _training_state.running:
        return jsonify({'error': 'Cannot inject while training is running'}), 400

    data = request.json or {}
    variations = data.get('variations', 500)

    from curriculum import generate_curriculum_samples, ALL_EXPANDED_SCENARIOS
    samples = generate_curriculum_samples(
        scenarios=ALL_EXPANDED_SCENARIOS,
        variations_per_scenario=variations
    )

    # Inject into strategy reservoir
    strat_reservoir = _deep_cfr_state.get('strategy_reservoir')
    if strat_reservoir:
        injected = 0
        for features, target, weight in samples:
            strat_reservoir.add(features, target, weight)
            injected += 1
        print(f"[Curriculum] Injected {injected} samples into strategy reservoir (now {strat_reservoir.size:,})")
        return jsonify({
            'status': 'injected',
            'samples': injected,
            'reservoir_size': strat_reservoir.size,
        })

    return jsonify({'error': 'No strategy reservoir found'}), 400


# ─── Auto-Training Pipeline ───────────────────────────────────────────────

_auto_train_progress = None

@app.route('/api/auto-train/start', methods=['POST'])
def auto_train_start():
    global _auto_train_progress, _training_state, _current_network, _deep_cfr_state, _current_mode

    if _training_state.running:
        return jsonify({'error': 'Training already running'}), 400

    data = request.json or {}

    from curriculum import AutoTrainConfig, AutoTrainProgress, EV_GROUPS, generate_curriculum_samples, ALL_EXPANDED_SCENARIOS
    from trainer_regret import DeepCFRConfig, train_deep_cfr
    from battery_test import run_battery
    from arena_test import run_arena

    config = AutoTrainConfig(
        iterations_per_group=data.get('iterations_per_group', 500_000),
        min_battery_score=data.get('min_battery_score', 0.60),
        min_bb_vs_tag=data.get('min_bb_vs_tag', 0.0),
        arena_hands=data.get('arena_hands', 5000),
        learning_rate=data.get('learning_rate', 0.001),
        batch_size=data.get('batch_size', 512),
        enable_hindsight=data.get('enable_hindsight', True),
        reservoir_size=data.get('reservoir_size', 10_000_000),
        inject_curriculum=data.get('inject_curriculum', True),
        curriculum_variations=data.get('curriculum_variations', 500),
        max_retries=data.get('max_retries', 3),
    )

    start_group = data.get('start_group', 0)  # 0-indexed, skip groups already trained

    _auto_train_progress = AutoTrainProgress(
        total_groups=len(EV_GROUPS),
        is_running=True,
    )
    progress = _auto_train_progress

    def _auto_run():
        global _current_network, _deep_cfr_state, _current_mode
        _current_mode = 'deep_cfr'

        # Phase 1: Inject curriculum (if enabled and first run)
        if config.inject_curriculum and _deep_cfr_state and not progress.should_stop:
            progress.phase = 'curriculum'
            progress.current_group_name = "Injecting curriculum..."
            print(f"\n[Auto-Train] Phase 1: Injecting curriculum samples...")
            samples = generate_curriculum_samples(
                scenarios=ALL_EXPANDED_SCENARIOS,
                variations_per_scenario=config.curriculum_variations
            )
            strat_res = _deep_cfr_state.get('strategy_reservoir')
            if strat_res:
                for features, target, weight in samples:
                    strat_res.add(features, target, weight)
                print(f"[Auto-Train] Injected {len(samples)} curriculum samples")

        # Phase 2: Train each group — run the full cycle TWICE
        # No retries, no skipping — every group trains every pass
        num_passes = 2
        group_iterations = {
            0: 100_000,   # G1: Elite
            1: 500_000,   # G2: Strong
            2: 500_000,   # G3: Good
            3: 500_000,   # G4: Playable
            4: 100_000,   # G5: Marginal
            5: 100_000,   # G6: Weak
            6: 100_000,   # G7: Bad
            7: 100_000,   # G8: Trash
        }

        for pass_num in range(num_passes):
            print(f"\n[Auto-Train] ════ Pass {pass_num + 1}/{num_passes} ════")

            first_group = start_group if pass_num == 0 else 0
            for group_idx in range(first_group, len(EV_GROUPS)):
                if progress.should_stop:
                    break

                group_key, group_name, group_size = EV_GROUPS[group_idx]
                progress.current_group = group_idx + 1
                progress.current_group_name = group_name
                progress.phase = 'training'

                iters = group_iterations.get(group_idx, config.iterations_per_group)
                print(f"\n[Auto-Train] Pass {pass_num+1} | Group {group_idx+1}/{len(EV_GROUPS)}: {group_name} ({iters//1000}K iters)")

                # Configure Deep CFR for this group
                dcfr_config = DeepCFRConfig(
                    iterations=iters,
                    advantage_lr=config.learning_rate,
                    strategy_lr=config.learning_rate,
                    batch_size=config.batch_size,
                    train_interval=1_000,
                    advantage_train_steps=200,
                    strategy_train_steps=200,
                    report_interval=5_000,
                    hand_scope=group_key,
                    advantage_reservoir_size=config.reservoir_size,
                    strategy_reservoir_size=config.reservoir_size,
                    enable_hindsight=config.enable_hindsight,
                    hindsight_weight=1.5,
                )

                # Training state
                base_iter = _deep_cfr_state.get('base_iteration', 0) if _deep_cfr_state else 0
                _training_state.total_iterations = base_iter + iters
                _training_state.iteration = base_iter
                _training_state.running = True
                _training_state.should_stop = False

                resume = _deep_cfr_state if _deep_cfr_state else None
                result = train_deep_cfr(dcfr_config, _training_state, resume_state=resume)

                with _lock:
                    _current_network = result['strategy_net']
                    _deep_cfr_state = result

                # Save checkpoint
                save_checkpoint(result)

                # Battery test
                progress.phase = 'battery'
                print(f"[Auto-Train] Running battery test...")
                from battery_test import run_battery as run_bat
                from curriculum import ALL_EXPANDED_SCENARIOS as scenarios
                bat_report = run_bat(_current_network, scenarios)
                battery_score = bat_report.score
                print(f"[Auto-Train] Battery: {bat_report.total_passed}/{bat_report.total_tests} ({battery_score:.0%})")

                # Arena test
                progress.phase = 'arena'
                print(f"[Auto-Train] Running arena test vs TAG...")
                arena_result = run_arena(_current_network, num_hands=config.arena_hands,
                                         opponent_type='tag', hand_scope=group_key)
                bb_vs_tag = arena_result.get('bb_per_100', -999)
                print(f"[Auto-Train] Arena vs TAG: {arena_result.get('win_rate')}% win, {bb_vs_tag:+.1f} BB/100")

                group_result = {
                    'group': group_name,
                    'pass': pass_num + 1,
                    'battery_score': battery_score,
                    'battery_passed': bat_report.total_passed,
                    'battery_total': bat_report.total_tests,
                    'bb_vs_tag': bb_vs_tag,
                    'win_vs_tag': arena_result.get('win_rate', 0),
                    'passed': True,  # Always pass — no retries
                }
                progress.group_results.append(group_result)
                print(f"[Auto-Train] ✅ {group_name} done (battery={battery_score:.0%}, BB/100={bb_vs_tag:+.1f})")

            if progress.should_stop:
                break
            print(f"\n[Auto-Train] ════ Pass {pass_num + 1}/{num_passes} complete ════")

        progress.phase = 'complete'
        progress.is_running = False
        _training_state.running = False
        print(f"\n[Auto-Train] Complete! {num_passes} passes done.")

    _training_thread = threading.Thread(target=_auto_run, daemon=True)
    _training_thread.start()

    return jsonify({
        'status': 'started',
        'total_groups': len(EV_GROUPS),
        'start_group': start_group,
        'config': {
            'iterations_per_group': config.iterations_per_group,
            'min_battery_score': config.min_battery_score,
            'enable_hindsight': config.enable_hindsight,
            'inject_curriculum': config.inject_curriculum,
        }
    })


@app.route('/api/auto-train/stop', methods=['POST'])
def auto_train_stop():
    global _auto_train_progress
    _training_state.should_stop = True
    if _auto_train_progress:
        _auto_train_progress.should_stop = True
    return jsonify({'status': 'stopping'})


@app.route('/api/auto-train/status', methods=['GET'])
def auto_train_status():
    if _auto_train_progress is None:
        return jsonify({'running': False, 'phase': 'idle'})

    p = _auto_train_progress
    return jsonify({
        'running': p.is_running,
        'phase': p.phase,
        'current_group': p.current_group,
        'total_groups': p.total_groups,
        'current_group_name': p.current_group_name,
        'current_group_attempt': p.current_group_attempt,
        'group_results': p.group_results,
        'iteration': _training_state.iteration,
        'total_iterations': _training_state.total_iterations,
        'loss': _training_state.loss,
    })


# ─── Arena Test ─────────────────────────────────────────────────────────────

@app.route('/api/test/arena', methods=['POST'])
def test_arena():
    print(f"[Arena] Request received — has_model={_current_network is not None}")
    if _current_network is None:
        print("[Arena] REJECTED — no model loaded")
        return jsonify({'error': 'No trained model. Train first.'}), 400

    data = request.json or {}
    num_hands = data.get('num_hands', 500)
    opponent = data.get('opponent', 'calling_station')
    hand_scope = data.get('hand_scope', None)
    single_hand = data.get('single_hand', None)

    print(f"[Arena] Running: {num_hands} hands vs {opponent}, scope={hand_scope}, single={single_hand}")

    from arena_test import run_arena
    result = run_arena(_current_network, num_hands=num_hands, opponent_type=opponent,
                       hand_scope=hand_scope, single_hand=single_hand)

    print(f"[Arena] Complete: Win={result.get('win_rate')}% BB/100={result.get('bb_per_100')}")
    return jsonify(result)


@app.route('/api/test/battery', methods=['POST'])
def test_battery():
    print(f"[Battery] Request received — has_model={_current_network is not None}")
    if _current_network is None:
        return jsonify({'error': 'No trained model. Train first.'}), 400

    from battery_test import run_battery
    from curriculum import ALL_EXPANDED_SCENARIOS
    # Use expanded scenarios (covers all EV groups)
    report = run_battery(_current_network, scenarios=ALL_EXPANDED_SCENARIOS)

    results = []
    for r in report.results:
        results.append({
            'name': r.scenario.name,
            'tier': r.scenario.tier,
            'passed': r.passed,
            'predicted_action': r.predicted_action,
            'expected_actions': r.scenario.expected_actions,
            'probabilities': r.probabilities,
            'confidence': r.confidence,
            'explanation': r.scenario.explanation,
        })

    tier_scores = {}
    for tier, (passed, total) in report.tier_scores.items():
        tier_scores[str(tier)] = {'passed': passed, 'total': total}

    print(f"[Battery] Complete: {report.total_passed}/{report.total_tests} ({report.score:.0%})")
    return jsonify({
        'total_passed': report.total_passed,
        'total_tests': report.total_tests,
        'score': report.score,
        'tier_scores': tier_scores,
        'results': results,
    })


# ─── Model Info & Export ────────────────────────────────────────────────────

@app.route('/api/model/info', methods=['GET'])
def model_info():
    if _current_network is None:
        return jsonify({'loaded': False})

    params = sum(p.numel() for p in _current_network.parameters())
    return jsonify({
        'loaded': True,
        'type': 'strategy',
        'input_dim': FEATURE_DIM,
        'parameters': params,
        'train_steps': _training_state.train_steps,
        'loss': _training_state.loss,
    })


@app.route('/api/model/export', methods=['POST'])
def model_export():
    if _current_network is None:
        return jsonify({'error': 'No model to export'}), 400

    data = request.json or {}
    filename = data.get('filename', 'neural_razz_strategy.json')

    # Export to the project's parent directory
    export_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(export_dir, filename)

    metadata = {
        'mode': 'strategy',
        'train_steps': _training_state.train_steps,
        'iterations': _training_state.iteration,
        'hero_info_sets': _training_state.hero_info_set_count,
        'hands_in_scope': _training_state.hands_in_scope,
        'loss': _training_state.loss,
        'learning_rate': 0.001,
    }

    export_to_json(_current_network, path, metadata)

    size_mb = os.path.getsize(path) / 1_000_000
    return jsonify({
        'path': path,
        'size_mb': round(size_mb, 2),
        'parameters': sum(p.numel() for p in _current_network.parameters()),
    })


@app.route('/api/model/predict', methods=['POST'])
def model_predict():
    if _current_network is None:
        return jsonify({'error': 'No model loaded'}), 400

    data = request.json or {}
    features = data.get('features', [])

    if len(features) != FEATURE_DIM:
        return jsonify({'error': f'Expected {FEATURE_DIM} features, got {len(features)}'}), 400

    probs = _current_network.predict(features)
    action_names = ['fold', 'check', 'call', 'bet', 'raise']

    return jsonify({
        'probabilities': dict(zip(action_names, probs)),
        'recommended': action_names[probs.index(max(probs))],
    })


# ─── Health ─────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': '1.0.0'})


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Neural Razz Trainer — Backend Server")
    print("=" * 60)
    print(f"  Feature dim: {FEATURE_DIM}")
    print(f"  PyTorch: {__import__('torch').__version__}")

    # Auto-load checkpoint from disk if available
    if has_checkpoint():
        _deep_cfr_state = load_checkpoint()
        if _deep_cfr_state:
            _current_network = _deep_cfr_state['strategy_net']
            _current_mode = 'deep_cfr'
            _training_state.iteration = _deep_cfr_state.get('base_iteration', 0)
            _training_state.total_iterations = _deep_cfr_state.get('base_iteration', 0)
            _training_state.has_model = True
            print(f"  Model ready from checkpoint (iteration {_training_state.iteration})")
    else:
        print("  No checkpoint found — start fresh")

    print()
    app.run(host='127.0.0.1', port=5050, debug=False, threaded=True)
