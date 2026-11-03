"""Exp 4 scorer: run an actor (Kinetix parameter names in an .npz) on the car-on-ramp level `l/h2_car_ramp`
with sampled actions, N episodes, max 256 steps, and report success / red / timeout like the authors' records.

Usage: python exp4_score.py <actor.npz> [--episodes 1024] [--seed 0] [--out result.json]
Independent re-implementation; validated by reproducing 0% on the authors' trained actor and 100% on their edit.
"""
import argparse
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# jax2d 1.0.1 still calls jnp.clip(x, a_min=..., a_max=...); newer JAX only accepts min=/max=.
_clip = jnp.clip


def _clip_compat(x, *args, a_min=None, a_max=None, **kw):
    if a_min is not None:
        kw["min"] = a_min
    if a_max is not None:
        kw["max"] = a_max
    return _clip(x, *args, **kw)


jnp.clip = _clip_compat

from kinetix.environment import ActionType, ObservationType, make_kinetix_env  # noqa: E402
from kinetix.util import load_from_json_file

P = "params/GeneralActorCriticRNN_0/"
NAMES = {"embed.W": "params/Dense_0/kernel", "embed.b": "params/Dense_0/bias",
         "head.W": P + "Dense_0/kernel", "head.b": P + "Dense_0/bias"}
for k in range(5):
    NAMES[f"a{k}.W"] = P + f"DenseAndActivation_{k}/Dense_0/kernel"
    NAMES[f"a{k}.b"] = P + f"DenseAndActivation_{k}/Dense_0/bias"
ENTRY_SLICES = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 14), (14, 16)]


def load_actor(path):
    z = np.load(path)
    if "embed.W" in z.files:  # the authors' edited_actor.npz naming
        alt = {"embed.W": "embed.W", "embed.b": "embed.b", "head.W": "actor_head.W", "head.b": "actor_head.b"}
        for k in range(5):
            alt[f"a{k}.W"], alt[f"a{k}.b"] = f"actor{k}.W", f"actor{k}.b"
        return {n: np.array(z[v], dtype=np.float64) for n, v in alt.items()}
    return {n: np.array(z[v], dtype=np.float64) for n, v in NAMES.items()}


def logits(T, obs):
    e = np.maximum(0, obs @ T["embed.W"] + T["embed.b"])
    a = np.tanh(e @ T["a0.W"] + T["a0.b"])
    for k in range(1, 5):
        a = np.tanh(a @ T[f"a{k}.W"] + T[f"a{k}.b"])
    return a @ T["head.W"] + T["head.b"]


def sample_actions(lg, rng):
    n = lg.shape[0]
    out = np.zeros((n, 6), dtype=np.int32)
    for j, (s, t) in enumerate(ENTRY_SLICES):
        z = lg[:, s:t]
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        c = p.cumsum(1)
        u = rng.rand(n, 1)
        out[:, j] = (u > c).sum(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--episodes", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace", action="store_true", help="print the first episode every 16 steps")
    a = ap.parse_args()
    T = load_actor(a.actor)

    level, static_env_params, env_params = load_from_json_file("l/h2_car_ramp.json")
    env = make_kinetix_env(action_type=ActionType.MULTI_DISCRETE, observation_type=ObservationType.SYMBOLIC_FLAT,
                           reset_fn=lambda rng: level, env_params=env_params, static_env_params=static_env_params)
    N = a.episodes
    key = jax.random.PRNGKey(a.seed)
    key, kr = jax.random.split(key)
    reset_v = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step_v = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    obs, state = reset_v(jax.random.split(kr, N), env_params)
    obs = np.asarray(obs)
    print(f"obs shape {obs.shape}; action space {env.action_space(env_params)}", file=sys.stderr)
    rng = np.random.RandomState(a.seed)
    alive = np.ones(N, dtype=bool)
    outcome = np.zeros(N, dtype=np.int32)  # 0 timeout, 1 success, -1 red
    length = np.full(N, 256, dtype=np.int32)
    ret = np.zeros(N)
    t0 = time.time()
    for t in range(256):
        lg = logits(T, obs)
        act = sample_actions(lg, rng)
        if a.trace and t % 16 == 0:
            p = np.exp(lg[0, :3] - lg[0, :3].max()); p /= p.sum()
            print(f"  step {t:3d} | body x {obs[0,52]:.2f} y {obs[0,53]:.2f} | green x {obs[0,234]:.2f} y {obs[0,235]:.2f} | action {act[0,0]} | p {np.round(p,2)}", file=sys.stderr)
        key, ks = jax.random.split(key)
        obs_j, state, reward, done, info = step_v(jax.random.split(ks, N), state, jnp.asarray(act), env_params)
        obs = np.asarray(obs_j)
        reward = np.asarray(reward, dtype=np.float64)
        done = np.asarray(done, dtype=bool)
        ret[alive] += reward[alive]
        just = alive & done
        if just.any():
            outcome[just & (reward > 0.5)] = 1
            outcome[just & (reward < -0.5)] = -1
            length[just] = t + 1
            alive &= ~done
        if not alive.any():
            break
    res = {"actor": a.actor, "episodes": N, "success": float((outcome == 1).mean()), "red": float((outcome == -1).mean()),
           "timeout": float((outcome == 0).mean()), "mean_return": float(ret.mean()), "mean_length": float(length.mean()),
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
