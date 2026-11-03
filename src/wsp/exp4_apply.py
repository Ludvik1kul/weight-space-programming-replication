"""Exp 4: apply the edits in answers/exp4_fix_car.md to the released Kinetix actor and unit-test the controller
logic on synthetic observations (no physics; Kinetix is needed to score the real episodes).
Usage: python exp4_apply.py <logs_repo_root> <project_root>
Writes results/exp4_edited_actor_claude.npz (Kinetix parameter names, actor only) for scoring in Kinetix."""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from edits import apply_edits, last_edits_block  # noqa: E402

repo, proj = Path(sys.argv[1]), Path(sys.argv[2])
z = np.load(repo / "exp4_fix_car/networks/ppo_final_seed0.npz")
P = "params/GeneralActorCriticRNN_0/"
names = {"embed.W": "params/Dense_0/kernel", "embed.b": "params/Dense_0/bias",
         "actor_head.W": P + "Dense_0/kernel", "actor_head.b": P + "Dense_0/bias"}
for k in range(5):
    names[f"actor{k}.W"] = P + f"DenseAndActivation_{k}/Dense_0/kernel"
    names[f"actor{k}.b"] = P + f"DenseAndActivation_{k}/Dense_0/bias"
T = {n: np.array(z[v], dtype=np.float64) for n, v in names.items()}
orig = {n: v.copy() for n, v in T.items()}

text = (proj / "answers/exp4_fix_car.md").read_text(encoding="utf-8")
block = last_edits_block(text)
# support the prompt's "ZERO <tensor>" statement
zero_lines = re.findall(r"^\s*ZERO\s+([A-Za-z0-9_.]+)", block, re.M)
block = re.sub(r"^\s*ZERO\s+[A-Za-z0-9_.]+.*$", "", block, flags=re.M)
for n in zero_lines:
    T[n][...] = 0
rep = apply_edits(T, block)
print("edit report:", {k: v for k, v in rep.items() if k != "errors"})
for e in rep["errors"]:
    print("  error:", e)
changed = sum(int((T[n] != orig[n]).sum()) for n in T)
total = sum(v.size for v in T.values())
print(f"changed {changed} of {total} actor weights ({100*changed/total:.2f}%)")


def act(obs):
    e = np.maximum(0, obs @ T["embed.W"] + T["embed.b"])
    a = np.tanh(e @ T["actor0.W"] + T["actor0.b"])
    for k in range(1, 5):
        a = np.tanh(a @ T[f"actor{k}.W"] + T[f"actor{k}.b"])
    return a @ T["actor_head.W"] + T["actor_head.b"]


def obs_with(x, vx, gx, base_scale=1.0):
    o = np.random.RandomState(0).uniform(-1, 1, 679) * base_scale  # arbitrary other entries
    o[52], o[54], o[234] = x, vx, gx
    return o


print("\ncontroller logic on synthetic observations (entry-0 logits [off, +1, -1] and argmax):")
cases = [(2.80, 0.0, 2.54, "start: back up -> expect -1 (idx 2)"),
         (3.50, 1.5, 3.2, "backing up fast, x<3.8 -> -1 (idx 2)"),
         (3.85, 1.0, 3.6, "x>=3.8 -> +1 (idx 1)"),
         (4.30, -0.5, 4.0, "returning, vx<0 -> +1 (idx 1)"),
         (2.20, -2.5, 1.9, "on the ramp at speed -> +1 (idx 1)"),
         (0.80, -0.3, 0.60, "landed near box -> +1 (idx 1)"),
         (0.70, 0.4, 0.45, "pushing box, rolling back but wheel<0.95 -> +1 (idx 1)"),
         (2.00, 0.0, 1.75, "stalled on ramp as in the record -> -1 (idx 2): retry"),
         (1.60, 0.3, 1.35, "rolled back off ramp, wheel>0.95 -> -1 (idx 2)")]
for x, vx, gx, note in cases:
    for sc in (0.0, 1.0, 5.0):
        lg = act(obs_with(x, vx, gx, sc))[:3]
        print(f"  x={x:4.2f} vx={vx:+4.1f} green_x={gx:4.2f} other|obs|<={sc}: {np.round(lg,1)} argmax={lg.argmax()}   {note if sc == 0 else ''}")

out = {v: (T[n] if n in T else z[v]) for n, v in names.items()}
np.savez(proj / "results/exp4_edited_actor_claude.npz", **{v: out[v].astype(np.float32) for v in out})
print("\nsaved results/exp4_edited_actor_claude.npz (Kinetix parameter names)")
