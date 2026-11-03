"""Parser for the small `edits` language used in Exp 4 and Exp 5 of the weight-space-programming post.

Statements (one per line, '#' comments, a bracketed list may continue over following lines):
    W1[12, 0] = 2.5
    W1[0:6, 3] = [1, 0, -1, 0, 2, 0]
    b1[0:4] = -1
    W1[3:600:60, 0] = [...]
    W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
    W3[:, 2] *= 2          (also += and -=)
Only the LAST ```edits block of a reply is read. A statement that cannot be read is skipped.
Written independently from the authors' reader; validated by reproducing their networks/<game>/hand.npz.
"""
import ast
import re
from typing import Dict, List, Tuple

import numpy as np

_STMT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\[(.*?)\]\s*(=|\+=|-=|\*=)\s*(.+?)\s*$", re.S)


def last_edits_block(text: str) -> str:
    blocks = re.findall(r"```edits[ \t]*\n(.*?)```", text, re.S)
    if not blocks:
        raise ValueError("no ```edits block found")
    return blocks[-1]


def _strip_comment(line: str) -> str:
    out, in_str = [], False
    for ch in line:
        if ch == "#" and not in_str:
            break
        out.append(ch)
    return "".join(out)


def split_statements(block: str) -> List[str]:
    """Join continuation lines: a statement is complete when its brackets balance."""
    stmts, cur, depth = [], "", 0
    for raw in block.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip() and depth == 0:
            continue
        cur += (" " if cur else "") + line.strip()
        depth += line.count("[") + line.count("(") - line.count("]") - line.count(")")
        if depth <= 0:
            if cur.strip():
                stmts.append(cur.strip())
            cur, depth = "", 0
    if cur.strip():
        stmts.append(cur.strip())
    return stmts


def _parse_index(idx: str) -> Tuple:
    parts = [p.strip() for p in idx.split(",")]
    out = []
    for p in parts:
        if ":" in p:
            bits = [b.strip() for b in p.split(":")]
            if len(bits) > 3:
                raise ValueError(f"bad slice {p}")
            vals = [int(b) if b else None for b in bits]
            while len(vals) < 3:
                vals.append(None)
            out.append(slice(*vals))
        else:
            out.append(int(p))
    return tuple(out)


def apply_edits(tensors: Dict[str, np.ndarray], block: str) -> Dict:
    """Apply an edits block in place. Returns a report dict."""
    stmts = split_statements(block)
    report = {"statements": len(stmts), "applied": 0, "errors": [], "entries_written": 0}
    for s in stmts:
        m = _STMT.match(s)
        if not m:
            report["errors"].append(f"unreadable: {s[:80]}")
            continue
        name, idx, op, rhs = m.groups()
        if name not in tensors:
            report["errors"].append(f"unknown tensor {name}: {s[:80]}")
            continue
        try:
            index = _parse_index(idx)
            value = ast.literal_eval(rhs.replace(" ", "").replace("\n", "")) if rhs.strip()[0] in "[(" else float(rhs)
            value = np.array(value, dtype=np.float64)
            if np.any(np.abs(value) > 1e6):
                raise ValueError("|v| > 1e6")
            T = tensors[name]
            view = T[index]
            if value.ndim and value.size != view.size and value.shape != view.shape:
                raise ValueError(f"shape mismatch: target {view.shape}, value {value.shape}")
            if value.ndim and value.shape != view.shape:
                value = value.reshape(view.shape)
            if op == "=":
                T[index] = value
            elif op == "+=":
                T[index] = view + value
            elif op == "-=":
                T[index] = view - value
            elif op == "*=":
                T[index] = view * value
            report["applied"] += 1
            report["entries_written"] += int(np.size(view))
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"{type(e).__name__}: {e} in: {s[:80]}")
    return report


def new_mlp(sizes=(600, 32, 32, 6)) -> Dict[str, np.ndarray]:
    d = {}
    for i in range(len(sizes) - 1):
        d[f"W{i+1}"] = np.zeros((sizes[i], sizes[i + 1]))
        d[f"b{i+1}"] = np.zeros(sizes[i + 1])
    return d
