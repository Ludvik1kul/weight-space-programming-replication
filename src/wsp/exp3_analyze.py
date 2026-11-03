"""Exp 3: weight-only analysis of the four transformer blocks (no forward pass, no embeddings).

Pre-registered pipeline (written before looking at any network):
  1. For every head (l,h), W_QK = W_q[h]^T W_k[h] and W_OV = W_o[:,h] W_v[h]  (d x d, rank 64).
  2. K-composition of head (l,h), l>=1, with each layer-0 head g (Elhage et al. 2021):
         ||W_QK^(l,h) W_OV^(0,g)||_F / (||W_QK^(l,h)||_F ||W_OV^(0,g)||_F)
     A copy (induction) head must read, through its keys, what a previous-token head wrote, so it shows an
     outlying K-composition with one layer-0 head. Q-composition is computed too for contrast.
  3. OV "copying" score of each head: eigenvalues of W_OV; score = sum(Re lambda) / sum(|lambda|).
     A copying head moves information along the same directions it reads (positive eigenvalues).
  4. Copy-head call: K-composition z-score (over all (head, layer-0 head) pairs of layers 1-3) >= 3 with some
     layer-0 head AND OV copying score above the median of its layer.
  5. Corpus signals: effective rank (participation ratio of singular values) of the last block's output
     matrices W_proj.3 and W_o.3 and of the whole-network concatenations; the paper's cue was that the romaji
     model's last block writes in fewer distinct directions. Also printed: the same for block 0, and LayerNorm
     statistics, so the three networks can be compared side by side.
Usage: python exp3_analyze.py <blind_dir>   (expects <blind_dir>/A/blocks.npz, B/..., C/...)
"""
import sys
from pathlib import Path

import numpy as np

HEADS, LAYERS, D = 6, 4, 384
HD = D // HEADS


def load(path):
    z = np.load(path)
    return {k: np.array(z[k], dtype=np.float64) for k in z.files}


def head_mats(W, l, h):
    s = slice(h * HD, (h + 1) * HD)
    Wq, Wk, Wv, Wo = W[f"W_q.{l}"][s], W[f"W_k.{l}"][s], W[f"W_v.{l}"][s], W[f"W_o.{l}"][:, s]
    return Wq.T @ Wk, Wo @ Wv  # W_QK (d x d), W_OV (d x d)


def fro(a):
    return np.linalg.norm(a)


def eff_rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    p = s**2 / (s**2).sum()
    return float(1.0 / (p**2).sum())


def analyze(name, W):
    print(f"\n########## network {name} ##########")
    QK, OV = {}, {}
    for l in range(LAYERS):
        for h in range(HEADS):
            QK[(l, h)], OV[(l, h)] = head_mats(W, l, h)
    # copying score
    copy_score = {}
    for key, M in OV.items():
        lam = np.linalg.eigvals(M)
        idx = np.argsort(-np.abs(lam))[:HD]
        lam = lam[idx]
        copy_score[key] = float(lam.real.sum() / np.abs(lam).sum())
    # composition with layer-0 heads
    kcomp, qcomp = {}, {}
    for l in range(1, LAYERS):
        for h in range(HEADS):
            for g in range(HEADS):
                A, B = QK[(l, h)], OV[(0, g)]
                kcomp[(l, h, g)] = fro(A @ B) / (fro(A) * fro(B))
                qcomp[(l, h, g)] = fro(B.T @ A) / (fro(A) * fro(B))
    kv = np.array(list(kcomp.values()))
    qv = np.array(list(qcomp.values()))
    kz = {k: (v - kv.mean()) / kv.std() for k, v in kcomp.items()}
    qz = {k: (v - qv.mean()) / qv.std() for k, v in qcomp.items()}
    print(f"K-composition over all pairs: mean {kv.mean():.4f} sd {kv.std():.4f}; Q-composition mean {qv.mean():.4f} sd {qv.std():.4f}")
    print("\nhead   copyOV   maxKcomp(z, with L0H)   maxQcomp(z, with L0H)   call")
    calls = {}
    for l in range(LAYERS):
        med = np.median([copy_score[(l, h)] for h in range(HEADS)])
        for h in range(HEADS):
            if l == 0:
                print(f"L0H{h}   {copy_score[(0,h)]:+.3f}   (layer 0: composition n/a)")
                continue
            gk = max(range(HEADS), key=lambda g: kz[(l, h, g)])
            gq = max(range(HEADS), key=lambda g: qz[(l, h, g)])
            is_copy = kz[(l, h, gk)] >= 3.0 and copy_score[(l, h)] > med
            calls[(l, h)] = "copy" if is_copy else "unclear"
            print(f"L{l}H{h}   {copy_score[(l,h)]:+.3f}   {kcomp[(l,h,gk)]:.4f} (z={kz[(l,h,gk)]:+.1f}, L0H{gk})   "
                  f"{qcomp[(l,h,gq)]:.4f} (z={qz[(l,h,gq)]:+.1f}, L0H{gq})   {calls[(l,h)]}")
    # which layer-0 head do the copy candidates compose with?
    partners = [max(range(HEADS), key=lambda g: kz[(l, h, g)]) for (l, h), c in calls.items() if c == "copy"]
    print("\ncopy heads:", [f"{l}.{h}" for (l, h), c in calls.items() if c == "copy"],
          " their layer-0 K-composition partners:", [f"0.{g}" for g in partners])
    # corpus signals
    print("\neffective rank (participation ratio of singular values):")
    for l in range(LAYERS):
        Wo_all = W[f"W_o.{l}"]
        print(f"  layer {l}: W_proj {eff_rank(W[f'W_proj.{l}']):7.1f}   W_o {eff_rank(Wo_all):7.1f}   "
              f"W_fc {eff_rank(W[f'W_fc.{l}']):7.1f}   W_v {eff_rank(W[f'W_v.{l}']):7.1f}   W_q {eff_rank(W[f'W_q.{l}']):7.1f}   W_k {eff_rank(W[f'W_k.{l}']):7.1f}")
    # norms
    print("\nFrobenius norms per layer (W_q, W_k, W_v, W_o, W_fc, W_proj):")
    for l in range(LAYERS):
        print("  layer", l, [round(fro(W[f"{n}.{l}"]), 2) for n in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]])
    print("\nLayerNorm: mean/std of ln1_w, ln2_w per layer; number of |ln_w| > 2 ('amplified dims'):")
    for l in range(LAYERS):
        a, b = W[f"ln1_w.{l}"], W[f"ln2_w.{l}"]
        print(f"  layer {l}: ln1 {a.mean():.3f}/{a.std():.3f} ({int((np.abs(a)>2).sum())})   ln2 {b.mean():.3f}/{b.std():.3f} ({int((np.abs(b)>2).sum())})")
    # MLP: how many hidden units of the last block write in near-duplicate directions? cluster W_proj.3 columns
    P = W["W_proj.3"]  # (384, 1536): column = write direction of hidden unit
    Pn = P / (np.linalg.norm(P, axis=0, keepdims=True) + 1e-9)
    C = Pn.T @ Pn
    np.fill_diagonal(C, 0)
    print(f"\nlast-block MLP write directions: fraction of unit pairs with |cos| > 0.8: {(np.abs(C) > 0.8).mean():.5f}; "
          f"max |cos| {np.abs(C).max():.3f}; mean top-5 |cos| per unit {np.sort(np.abs(C), axis=1)[:, -5:].mean():.3f}")
    # distinct write directions of the last block (attention + MLP): rank of [W_o.3 | W_proj.3] by 90% energy
    M = np.concatenate([W["W_o.3"], W["W_proj.3"]], axis=1)
    s = np.linalg.svd(M, compute_uv=False)
    cum = np.cumsum(s**2) / (s**2).sum()
    print(f"last block [W_o|W_proj]: dims for 50/80/90/95% energy: {int((cum<0.5).sum())+1} {int((cum<0.8).sum())+1} {int((cum<0.9).sum())+1} {int((cum<0.95).sum())+1}")
    M0 = np.concatenate([W["W_o.0"], W["W_proj.0"]], axis=1)
    s0 = np.linalg.svd(M0, compute_uv=False)
    cum0 = np.cumsum(s0**2) / (s0**2).sum()
    print(f"first block [W_o|W_proj]: dims for 50/80/90/95% energy: {int((cum0<0.5).sum())+1} {int((cum0<0.8).sum())+1} {int((cum0<0.9).sum())+1} {int((cum0<0.95).sum())+1}")
    return calls


if __name__ == "__main__":
    bd = Path(sys.argv[1])
    for name in ["A", "B", "C"]:
        analyze(name, load(bd / name / "blocks.npz"))
