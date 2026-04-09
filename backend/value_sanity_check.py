"""
Value Network Sanity Check

Loads the value checkpoint and compares predicted equity vs true MC equity
for a variety of hand matchups. Run this after training to see how well
the value network learned showdown equity.

Usage:
    python3 value_sanity_check.py
"""

import torch

from razz_game import HeadsUpRazzGame, Card
from features import extract_features
from checkpoint import load_value_checkpoint
from trainer_value import mc_equity


def predict(net, p0_hole, p0_up, p1_hole, p1_up):
    """Run the value network on a 3rd-street game state."""
    g = HeadsUpRazzGame()
    g.deal_third_street(
        p0_hole=p0_hole, p0_up=p0_up,
        p1_hole=p1_hole, p1_up=p1_up,
    )
    feats = extract_features(g, 0)
    with torch.no_grad():
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        return torch.sigmoid(net(x)).item()


def true_eq(h_ranks, v_ranks):
    """Ground-truth equity via high-sample Monte Carlo."""
    return mc_equity(h_ranks, v_ranks, 4, 4, set(), num_samples=5000)


def main():
    state = load_value_checkpoint()
    if not state:
        print("ERROR: No value checkpoint found in checkpoints_value/")
        return

    net = state['value_net']
    net.eval()
    print(f"Loaded checkpoint: iteration {state['base_iteration']:,}, "
          f"reservoir {state['value_reservoir'].size:,}")
    print()

    # (name, p0_hole, p0_up, p1_hole, p1_up, hero_ranks, villain_ranks)
    tests = [
        # Dominant hero
        ('A23 vs KQJ',  [Card(1,0), Card(2,1)], Card(3,2),
                        [Card(11,0), Card(12,1)], Card(13,2),
                        [1,2,3], [11,12,13]),
        ('A24 vs TJQ',  [Card(1,0), Card(2,1)], Card(4,2),
                        [Card(10,0), Card(11,1)], Card(12,2),
                        [1,2,4], [10,11,12]),
        ('A25 vs 89T',  [Card(1,0), Card(2,1)], Card(5,2),
                        [Card(8,0), Card(9,1)], Card(10,2),
                        [1,2,5], [8,9,10]),

        # Mid-range matchups
        ('345 vs 678',  [Card(3,0), Card(4,1)], Card(5,2),
                        [Card(6,0), Card(7,1)], Card(8,2),
                        [3,4,5], [6,7,8]),
        ('456 vs 789',  [Card(4,0), Card(5,1)], Card(6,2),
                        [Card(7,0), Card(8,1)], Card(9,2),
                        [4,5,6], [7,8,9]),
        ('567 vs 89T',  [Card(5,0), Card(6,1)], Card(7,2),
                        [Card(8,0), Card(9,1)], Card(10,2),
                        [5,6,7], [8,9,10]),
        ('678 vs 9TJ',  [Card(6,0), Card(7,1)], Card(8,2),
                        [Card(9,0), Card(10,1)], Card(11,2),
                        [6,7,8], [9,10,11]),

        # Close matchups
        ('A23 vs 456',  [Card(1,0), Card(2,1)], Card(3,2),
                        [Card(4,0), Card(5,1)], Card(6,2),
                        [1,2,3], [4,5,6]),
        ('A23 vs A24',  [Card(1,0), Card(2,1)], Card(3,2),
                        [Card(1,3), Card(2,3)], Card(4,2),
                        [1,2,3], [1,2,4]),

        # Dominated hero (weak hand as hero)
        ('KQJ vs A23',  [Card(13,0), Card(12,1)], Card(11,2),
                        [Card(1,0), Card(2,1)], Card(3,2),
                        [13,12,11], [1,2,3]),
        ('TJQ vs 345',  [Card(10,0), Card(11,1)], Card(12,2),
                        [Card(3,0), Card(4,1)], Card(5,2),
                        [10,11,12], [3,4,5]),
        ('89T vs A23',  [Card(8,0), Card(9,1)], Card(10,2),
                        [Card(1,0), Card(2,1)], Card(3,2),
                        [8,9,10], [1,2,3]),
    ]

    print('=' * 65)
    print(f'{"Hand Matchup":<30} {"Predicted":>10} {"True MC":>10} {"Error":>8}')
    print('=' * 65)

    total_err = 0.0
    for name, p0h, p0u, p1h, p1u, hr, vr in tests:
        pred = predict(net, p0h, p0u, p1h, p1u)
        true = true_eq(hr, vr)
        err = pred - true
        total_err += abs(err)
        print(f'{name:<30} {pred:>9.1%} {true:>9.1%} {err:>+7.1%}')

    print('=' * 65)
    print(f'Mean absolute error: {total_err/len(tests):.1%}')


if __name__ == '__main__':
    main()
