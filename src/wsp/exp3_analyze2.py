"""Exp 3, refined weight-only test (still no forward pass).

Matching score between a later head (l,h) and an earlier head g (any earlier layer):
    M = W_QK^(l,h) @ W_OV^g            (d x d)
    T = trace(M) / (||W_QK^(l,h)||_F * ||W_OV^g||_F)
trace(M) = <W_QK^T, W_OV^g>_F, i.e. how well the earlier head's output map is aligned with the later head's
query-key bilinear form. For an induction (copy) head whose keys read the previous-token information written by
head g, and whose queries read the same token subspace, M is close to a positive multiple of a projector, so T is
large and positive; for unrelated heads T is near 0 with either sign. The query/key side is folded with the later
layer's ln1 gain (a diagonal matrix), which is all we can do without the LayerNorm statistics.
Copy call: T z-score >= 3 among all (later head, earlier head) pairs of the network, with T > 0.
Also reported: the best partner g (candidate previous-token head) and, as a corpus cue, the spectrum of the
last block's writes.
Usage: python exp3_analyze2.py <blind_dir>
"""
import sys
from pathlib import Path

import numpy as np

HEADS, LAYERS, D = 6, 4, 384
HD = D // HEADS


def load(path):
    z = np.load(path)
    return {k: np.array(z[k], dtype=np.float64) for k in z.files}


def mats(W, l, h, fold_ln=True):
    s = slice(h * HD, (h + 1) * HD)
    g = np.diag(W[f"ln1_w.{l}"]) if fold_ln else np.eye(D)
    Wq, Wk, Wv, Wo = W[f"W_q.{l}"][s] @ g, W[f"W_k.{l}"][s] @ g, W[f"W_v.{l}"][s], W[f"W_o.{l}"][:, s]
    return Wq.T @ Wk, Wo @ Wv


def analyze(name, W):
    print(f"\n########## network {name} ##########")
    QK, OV = {}, {}
    for l in range(LAYERS):
        for h in range(HEADS):
            QK[(l, h)], OV[(l, h)] = mats(W, l, h)
    T = {}
    for l in range(1, LAYERS):
        for h in range(HEADS):
            for lg in range(l):
                for g in range(HEADS):
                    A, B = QK[(l, h)], OV[(lg, g)]
                    T[(l, h, lg, g)] = np.trace(A @ B) / (np.linalg.norm(A) * np.linalg.norm(B))
    vals = np.array(list(T.values()))
    mu, sd = vals.mean(), vals.std()
    print(f"matching score over {len(vals)} pairs: mean {mu:+.5f} sd {sd:.5f}")
    calls = {}
    print("\nhead    best partner   T        z      2nd best         call")
    for l in range(1, LAYERS):
        for h in range(HEADS):
            cands = sorted([(T[(l, h, lg, g)], lg, g) for lg in range(l) for g in range(HEADS)], reverse=True)
            t1, lg1, g1 = cands[0]
            t2, lg2, g2 = cands[1]
            z = (t1 - mu) / sd
            calls[(l, h)] = "copy" if (z >= 3 and t1 > 0) else "unclear"
            print(f"L{l}H{h}    L{lg1}H{g1}        {t1:+.4f}  {z:+5.1f}   L{lg2}H{g2} {t2:+.4f}    {calls[(l,h)]}")
    print("\ncopy heads:", [f"{l}.{h}" for (l, h), c in calls.items() if c == "copy"])
    # the previous-token candidates: earlier heads that are best partners of the copy heads
    print("top-10 pairs overall:")
    for (l, h, lg, g), t in sorted(T.items(), key=lambda kv: -kv[1])[:10]:
        print(f"   L{l}H{h} <- L{lg}H{g}: T={t:+.4f} z={(t-mu)/sd:+.1f}")
    # negative side for symmetry
    print("bottom-3 pairs:")
    for (l, h, lg, g), t in sorted(T.items(), key=lambda kv: kv[1])[:3]:
        print(f"   L{l}H{h} <- L{lg}H{g}: T={t:+.4f} z={(t-mu)/sd:+.1f}")
    # corpus cue: singular value spectrum of the last block's total write [W_o.3 | W_proj.3]
    M = np.concatenate([W["W_o.3"], W["W_proj.3"]], axis=1)
    s = np.linalg.svd(M, compute_uv=False)
    cum = np.cumsum(s**2) / (s**2).sum()
    print(f"last block write spectrum: top-10 singular values {np.round(s[:10],2)}; dims for 50/80/90% energy: "
          f"{int((cum<0.5).sum())+1} {int((cum<0.8).sum())+1} {int((cum<0.9).sum())+1}; s1/s20 {s[0]/s[19]:.2f}")
    P = W["W_proj.3"]
    sp = np.linalg.svd(P, compute_uv=False)
    cp = np.cumsum(sp**2) / (sp**2).sum()
    print(f"W_proj.3 alone: dims for 50/80/90% energy: {int((cp<0.5).sum())+1} {int((cp<0.8).sum())+1} {int((cp<0.9).sum())+1}; top-5 {np.round(sp[:5],2)}")
    return calls


if __name__ == "__main__":
    bd = Path(sys.argv[1])
    for name in ["A", "B", "C"]:
        analyze(name, load(bd / name / "blocks.npz"))
