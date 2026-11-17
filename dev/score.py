"""Score a forecast file against SullyGnome ground truth, same metric as the house.

Pinball loss on the log scale, per RULES.txt:
  loss item = 2 * max(tau * d, (tau - 1) * d), d = ln(y) - ln(q)
  loss = mean over hours 2..24 x 5 levels.
Also reports typical miss and 90% coverage.

Usage: python3 dev/score.py <outdir>   (reads <outdir>/p00.out .. p03.out)
"""
import json
import math
import sys
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ROOT = Path(__file__).resolve().parent.parent
GT = json.loads((ROOT / "dev" / "ground_truth.json").read_text())


def pinball(y, q, tau):
    d = math.log(max(1.0, y)) - math.log(max(1.0, q))
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_case(forecast, actual):
    """forecast: {"hours": [{"hour": h, "q": [...]}]}, actual: 24 hourly prints (h1..h24)."""
    rows = {r["hour"]: r["q"] for r in forecast["hours"]}
    total, n, abslog, covered = 0.0, 0, 0.0, 0
    for h in range(2, 25):
        y = actual[h - 1]
        q = rows[h]
        for tau, qv in zip(LEVELS, q):
            total += pinball(y, qv, tau)
            n += 1
        abslog += abs(math.log(max(1.0, y)) - math.log(max(1.0, q[2])))
        if q[0] <= y <= q[4]:
            covered += 1
    return {
        "loss": total / n,
        "typical_miss_pct": 100.0 * (math.exp(abslog / 23.0) - 1.0),
        "coverage90": covered / 23.0,
    }


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out"
    losses = []
    for cid in ("p00", "p01", "p02", "p03"):
        fc = json.loads((outdir / f"{cid}.out").read_text())
        s = score_case(fc, GT[cid])
        losses.append(s["loss"])
        print(f"{cid}: loss={s['loss']:.6f} typical_miss={s['typical_miss_pct']:.1f}% coverage90={s['coverage90']:.3f}")
    print(f"MEAN loss={sum(losses) / len(losses):.6f}  (house baseline ~0.387)")


if __name__ == "__main__":
    main()
