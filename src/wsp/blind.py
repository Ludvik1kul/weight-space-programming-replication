"""Blind-prompt generator.

Copies the authors' exact prompts into blind/<exp>/<LABEL>.txt with shuffled
labels.  The label -> run mapping is written to blind/<exp>/.key.json, which
the model under test (this Claude session) must never open.  Only scorer.py
reads the key.

Usage: python blind.py <exp1|exp2|exp3> <logs_repo_root> <project_root>
"""
import json, random, re, shutil, sys
from pathlib import Path

LEAK_WORDS = {
    "exp1": ["breakout", "pong", "pac", "freeway"],
    "exp2": ["circle", "square", "triangle"],
    "exp3": ["python", "romaji", "japanese", "english", "wikipedia", "aozora"],
}

def leak_count(text, words):
    t = text.lower()
    return {w: len(re.findall(w, t)) for w in words}

def main(exp, repo, proj):
    repo, proj = Path(repo), Path(proj)
    out = proj / "blind" / exp
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261103)
    if exp == "exp1":
        runs = sorted(d for d in (repo / "exp1_which_game/runs").iterdir() if d.is_dir())
        files = [d / "prompt.txt" for d in runs]
    elif exp == "exp2":
        runs = sorted(d for d in (repo / "exp2_diffusion/runs").iterdir() if d.is_dir())
        files = [d / "prompt.txt" for d in runs]
    elif exp == "exp3":
        runs = sorted(d for d in (repo / "exp3_language/networks").iterdir() if d.is_dir())
        files = runs
    else:
        raise SystemExit("unknown exp")
    labels = [chr(ord("A") + i) for i in range(len(runs))]
    rng.shuffle(labels)
    key = {}
    for lab, run, f in zip(labels, runs, files):
        key[lab] = run.name
        if f.is_dir():
            dst = out / lab
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(f, dst)
            for p in dst.rglob("*"):
                if p.is_file() and p.suffix in (".txt", ".md", ".json"):
                    leaks = leak_count(p.read_text(encoding="utf-8", errors="replace"), LEAK_WORDS[exp])
                    if any(leaks.values()):
                        print(f"WARNING leak words in {p.relative_to(out)}: {leaks}")
        else:
            text = f.read_text(encoding="utf-8")
            leaks = leak_count(text, LEAK_WORDS[exp])
            print(f"label {lab}: {len(text):,} chars, leak-word counts {leaks}")
            (out / f"{lab}.txt").write_text(text, encoding="utf-8")
    (out / ".key.json").write_text(json.dumps(key, indent=1))
    print("wrote", out, "labels", sorted(labels), "(key hidden in .key.json)")

if __name__ == "__main__":
    main(*sys.argv[1:4])
