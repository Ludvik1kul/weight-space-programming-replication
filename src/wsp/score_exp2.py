"""Score the blind Exp 2 answers against the authors' answer key.
Usage: python score_exp2.py <logs_repo_root> <project_root>"""
import json, re, sys
from pathlib import Path

repo, proj = Path(sys.argv[1]), Path(sys.argv[2])
key = json.loads((proj / "blind/exp2/.key.json").read_text())
akey = json.loads((repo / "exp2_diffusion/eval/answer_key.json").read_text())
scores = json.loads((repo / "exp2_diffusion/eval/scores.json").read_text())
print("answer_key.json:", json.dumps(akey)[:900])
print()
for lab in sorted(key):
    txt = (proj / f"answers/exp2_{lab}.md").read_text(encoding="utf-8")
    img = re.search(r"^Images:\s*(.+)$", txt, re.M).group(1).strip()
    conf = re.search(r"^Confidence:\s*(\d+)", txt, re.M).group(1)
    print(f"{lab}: truth={key[lab]:9} Claude: {img[:72]}  ({conf}%)")
print()
print("Astra (scores.json):", json.dumps(scores, indent=1)[:1600])
