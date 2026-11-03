"""Score the blind Exp 1 answers against the authors' answer key.
Usage: python score_exp1.py <logs_repo_root> <project_root>"""
import json, re, sys
from pathlib import Path

repo, proj = Path(sys.argv[1]), Path(sys.argv[2])
key = json.loads((proj / "blind/exp1/.key.json").read_text())        # label -> run name
akey = json.loads((repo / "exp1_which_game/eval/answer_key.json").read_text())
scores = json.loads((repo / "exp1_which_game/eval/scores.json").read_text())
print("answer_key.json:", json.dumps(akey))
print()
rows = []
for lab in sorted(key):
    run = key[lab]
    txt = (proj / f"answers/exp1_{lab}.md").read_text(encoding="utf-8")
    game = re.search(r"^Game:\s*(.+)$", txt, re.M).group(1).strip()
    conf = re.search(r"^Confidence:\s*(\d+)", txt, re.M).group(1)
    truth = akey.get(run, akey) if isinstance(akey, dict) else akey
    rows.append((lab, run, game, conf))
print(f"{'label':5} {'run (truth)':12} {'Claude answer':60} conf")
for lab, run, game, conf in rows:
    print(f"{lab:5} {run:12} {game[:60]:60} {conf}%")
print()
print("authors' scores.json (Astra):")
print(json.dumps(scores, indent=1)[:3000])
