"""Independent scorer for Exp 5: MinAtar 1.0.15, 600-32-32-6 tanh MLP, softmax sampling, sticky actions 0.1,
episodes cut at 10,000 steps, mean over N episodes.

Observation: 10x10x6 (row, col, channel) flattened as i = (row*10+col)*6 + channel, channels
  0 player, 1 danger, 2 reward, 3 projectile, 4 trail, 5 gauge, built from the game's own channels as stated in the
  prompts (a cell is 1 if any source channel is 1).

Usage:
  python minatar_eval.py <game> <weights.npz|edits.md> [--episodes 1000] [--seed 0] [--workers 8]
npz keys accepted: W1,b1,W2,b2,W3,b3 or the authors' actor/dense1/kernel ... names.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from edits import apply_edits, last_edits_block, new_mlp  # noqa: E402

# game -> list over the 6 output channels of source channel names in the MinAtar Env.channels dict
CHANNEL_MAPS = {
    "breakout": [["paddle"], [], ["brick"], ["ball"], ["trail"], []],
    "asterix": [["player"], ["enemy"], ["gold"], [], ["trail"], []],
    "seaquest": [
        ["sub_front", "sub_back"],
        ["enemy_bullet", "enemy_fish", "enemy_sub"],
        ["enemy_fish", "enemy_sub", "diver"],
        ["friendly_bullet"],
        ["trail", "sub_back"],
        ["oxygen_guage", "diver_guage"],
    ],
}


def load_weights(path):
    if path.endswith(".npz"):
        z = np.load(path)
        keys = list(z.keys())
        if "W1" in keys:
            return {k: np.array(z[k], dtype=np.float64) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
        m = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel",
             "b2": "actor/dense2/bias", "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}
        return {k: np.array(z[v], dtype=np.float64) for k, v in m.items()}
    text = open(path, encoding="utf-8").read()
    T = new_mlp()
    rep = apply_edits(T, last_edits_block(text))
    print("edit report:", json.dumps({k: v for k, v in rep.items() if k != "errors"}), file=sys.stderr)
    for e in rep["errors"]:
        print("  edit error:", e, file=sys.stderr)
    return T


def make_obs_builder(game):
    from minatar.environments import asterix, breakout, seaquest  # noqa: F401
    env_mod = {"breakout": breakout, "asterix": asterix, "seaquest": seaquest}[game]
    env = env_mod.Env()
    src_idx = [[env.channels[n] for n in names] for names in CHANNEL_MAPS[game]]
    return env, src_idx


def build_obs(state, src_idx):
    # state: (10,10,C) bool -> (10,10,6) -> flatten in (row, col, channel) order
    obs = np.zeros((10, 10, 6), dtype=np.float64)
    for ch, srcs in enumerate(src_idx):
        for s in srcs:
            obs[:, :, ch] = np.logical_or(obs[:, :, ch], state[:, :, s])
    return obs.reshape(600)


def policy_logits(W, x):
    h1 = np.tanh(x @ W["W1"] + W["b1"])
    h2 = np.tanh(h1 @ W["W2"] + W["b2"])
    return h2 @ W["W3"] + W["b3"]


def run_episodes(args):
    game, W, n, seed, max_steps, sticky = args
    rng = np.random.RandomState(seed)
    env, src_idx = make_obs_builder(game)
    env.random = np.random.RandomState(seed + 1_000_003)
    scores, lengths = [], []
    for _ in range(n):
        env.reset()
        total, steps, last_a = 0.0, 0, 0
        terminal = False
        while not terminal and steps < max_steps:
            x = build_obs(env.state(), src_idx)
            z = policy_logits(W, x)
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            a = int(rng.choice(6, p=p))
            if rng.rand() < sticky:
                a = last_a
            last_a = a
            r, terminal = env.act(a)
            total += r
            steps += 1
        scores.append(total)
        lengths.append(steps)
    return scores, lengths


def evaluate(game, W, episodes=1000, seed=0, workers=8, max_steps=10_000, sticky=0.1):
    per = [episodes // workers + (1 if i < episodes % workers else 0) for i in range(workers)]
    jobs = [(game, W, per[i], seed * 1000 + i, max_steps, sticky) for i in range(workers) if per[i] > 0]
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(run_episodes, jobs)
    else:
        res = [run_episodes(j) for j in jobs]
    scores = np.array(sum((r[0] for r in res), []))
    lengths = np.array(sum((r[1] for r in res), []))
    return {
        "mean": float(scores.mean()),
        "se": float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0,
        "episodes": int(len(scores)),
        "max": float(scores.max()),
        "mean_length": float(lengths.mean()),
        "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("weights")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    W = load_weights(a.weights)
    nz = {k: int(np.count_nonzero(v)) for k, v in W.items()}
    print("nonzero:", nz, "total", sum(nz.values()), file=sys.stderr)
    res = evaluate(a.game, W, a.episodes, a.seed, a.workers)
    res["game"], res["weights"] = a.game, a.weights
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
