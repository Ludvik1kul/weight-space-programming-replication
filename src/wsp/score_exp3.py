"""Unblind and score the Exp 3 answers: corpus per label, and copy heads vs the authors' induction scores."""
import json, re, sys
from pathlib import Path

repo, proj = Path(sys.argv[1]), Path(sys.argv[2])
key = json.loads((proj / "blind/exp3/.key.json").read_text())
heads = json.loads((repo / "exp3_language/eval/heads.json").read_text())
txt = (proj / "answers/exp3_answers.md").read_text(encoding="utf-8")
blocks = re.split(r"^## Network ", txt, flags=re.M)[1:]
print("label -> truth:", key)
for b in blocks:
    lab = b[0]
    corpus = re.search(r"^corpus:\s*(\w+)", b, re.M).group(1)
    copies = re.findall(r"L(\d)H(\d) copy", b)
    truth = key[lab]
    hk = heads.get(truth) or heads[f"{truth}_L4_H6_d384_s0"]
    # find the induction scores structure
    def induct_of(l, h):
        for cand in (f"L{l}H{h}", f"{l}.{h}"):
            if isinstance(hk, dict) and cand in hk:
                v = hk[cand]
                return v["induct"] if isinstance(v, dict) else v
        return None
    true_copy = []
    if isinstance(hk, dict):
        for k, v in hk.items():
            ind = v.get("induct") if isinstance(v, dict) else None
            if ind is not None and ind > 0.2:
                true_copy.append((k, round(ind, 3)))
    mine = [f"{l}.{h}" for l, h in copies]
    print(f"\nlabel {lab}: truth corpus = {truth}; Claude said {corpus} -> {'CORRECT' if corpus == truth else 'WRONG'}")
    print(f"  Claude copy heads: {mine}  with induct scores {[induct_of(l, h) for l, h in copies]}")
    print(f"  true copy heads (induct > 0.2): {true_copy}")
