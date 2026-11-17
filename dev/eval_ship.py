"""Faithful LOO evaluation of the SHIPPED main.py model.

Reuses main.ratio_quantiles directly: each comparable is treated as a
pseudo-target (its hour-1 print known, hours 2..24 predicted from the other
comparables' raw dicts, exactly as the grader's case would carry them).
Reports overall mean, big-launch means (h1 >= 100k / >= 300k), and per-target
losses. Also scores the 4 practice cases through main.forecast_case, with an
optional extra pool of the other 3 practice launches (mimics the final case,
whose comparables include all 4 practice launches).
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import main  # noqa: E402

KIT = Path("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit")
GT = json.loads((ROOT / "dev" / "ground_truth.json").read_text())
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def pinball(y, q, tau):
    d = math.log(max(1.0, y)) - math.log(max(1.0, q))
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_logratios(h1, log_qs, actual24):
    total, n = 0.0, 0
    for i, h in enumerate(range(2, 25)):
        y = actual24[h - 1]
        for tau, lr in zip(LEVELS, log_qs[i]):
            total += pinball(y, math.exp(lr) * h1, tau)
            n += 1
    return total / n


def load_raw_comps():
    case = json.loads((KIT / "input" / "p00.in").read_text())
    return case["comparables"]


def curve_of(comp):
    return main.release_day_curve(comp)


def loo():
    comps = load_raw_comps()
    usable = []
    for c in comps:
        curve = curve_of(c)
        if curve is not None:
            usable.append((c, curve))
    print(f"comparables with full release-day curves: {len(usable)} / {len(comps)}")
    rows = []
    for c, curve in usable:
        pool = [o for o, _ in usable if o is not c]
        lq = main.ratio_quantiles({"comparables": pool}, curve[0])
        loss = score_logratios(curve[0], lq, curve)
        rows.append((c["name"], curve[0], loss))
    return rows


def practice(extra=False):
    losses = {}
    for cid in ("p00", "p01", "p02", "p03"):
        case = json.loads((KIT / "input" / f"{cid}.in").read_text())
        if extra:
            for ocid in ("p00", "p01", "p02", "p03"):
                if ocid == cid:
                    continue
                curve = GT[ocid]
                ds = {"p00": "2026-09-15", "p01": "2018-10-26",
                      "p02": "2022-11-09", "p03": "2023-10-20"}[ocid]
                comp = {
                    "name": f"practice-{ocid}",
                    "day_start": ds + "T00:00:00Z",
                    "unlock": "Tier: comparable. Rolling midnight local.",
                    "hourly": [{"from": f"{ds}T{h:02d}:00:00Z", "viewers": v,
                                "channels": 100} for h, v in enumerate(curve)],
                }
                case["comparables"].append(comp)
        fc = main.forecast_case(case, cid)
        actual = GT[cid]
        total, n = 0.0, 0
        for row in fc["hours"]:
            y = actual[row["hour"] - 1]
            for tau, q in zip(LEVELS, row["q"]):
                total += pinball(y, q, tau)
                n += 1
        losses[cid] = total / n
    return losses


def main_report():
    rows = loo()
    mean = sum(l for _, _, l in rows) / len(rows)
    big100 = [(n, h, l) for n, h, l in rows if h >= 100_000]
    big300 = [(n, h, l) for n, h, l in rows if h >= 300_000]
    m100 = sum(l for _, _, l in big100) / len(big100) if big100 else float("nan")
    m300 = sum(l for _, _, l in big300) / len(big300) if big300 else float("nan")
    print(f"SHIPPED model LOO: overall={mean:.4f} (n={len(rows)})  "
          f"h1>=100k={m100:.4f} (n={len(big100)})  h1>=300k={m300:.4f} (n={len(big300)})")
    print("  big100 per target:", ", ".join(f"{n}({l:.3f})" for n, _, l in sorted(big100, key=lambda x: -x[1])))
    print("  worst overall:", ", ".join(f"{n}({l:.3f})" for n, _, l in sorted(rows, key=lambda x: -x[2])[:6]))
    pl = practice(extra=False)
    ple = practice(extra=True)
    print(f"SHIPPED practice: mean={sum(pl.values())/4:.4f} "
          f"[{' '.join(f'{k}={v:.3f}' for k, v in pl.items())}]")
    print(f"SHIPPED practice+3: mean={sum(ple.values())/4:.4f} "
          f"[{' '.join(f'{k}={v:.3f}' for k, v in ple.items())}]")


if __name__ == "__main__":
    main_report()
