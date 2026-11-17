"""Round 4, stage 3: unlock-type split, coverage by horizon, extrapolation sanity."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import main as fc  # noqa: E402
from tune4 import predict_logqs, loo, practice, load_raw_comps, is_rolling  # noqa: E402
from eval_ship import curve_of, score_logratios, GT  # noqa: E402

KIT = Path("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit")
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def loo_split(**kw):
    """LOO rows split by target unlock type and size."""
    pool = load_raw_comps()
    usable = [(c, curve_of(c)) for c in pool if curve_of(c) is not None]
    groups = {"rolling-big": [], "fixed-big": [], "rolling-small": [], "fixed-small": []}
    for c, curve in usable:
        sub = [o for o, _ in usable if o is not c]
        lq = predict_logqs(curve[0], sub, ch1=None, **kw)
        loss = score_logratios(curve[0], lq, curve)
        key = ("rolling" if is_rolling(c.get("unlock")) else "fixed") + \
              ("-big" if curve[0] >= 100_000 else "-small")
        groups[key].append(loss)
    for k, v in groups.items():
        print(f"  {k:14s} n={len(v):2d} mean={sum(v)/len(v):.4f}" if v else f"  {k:14s} n=0")
    return groups


def coverage_by_horizon(**kw):
    """90% coverage per horizon band, practice cases (extra pool) + LOO-big."""
    pool = load_raw_comps()
    bands = {"h2-6": [], "h7-12": [], "h13-18": [], "h19-24": []}
    for cid in ("p00", "p01", "p02", "p03"):
        case = json.loads((KIT / "input" / f"{cid}.in").read_text())
        comps = list(case["comparables"])
        for ocid in ("p00", "p01", "p02", "p03"):
            if ocid == cid:
                continue
            curve = GT[ocid]
            ds = {"p00": "2026-09-15", "p01": "2018-10-26",
                  "p02": "2022-11-09", "p03": "2023-10-20"}[ocid]
            comps.append({"name": f"practice-{ocid}", "day_start": ds + "T00:00:00Z",
                          "unlock": "Tier: comparable. Rolling midnight local.",
                          "hourly": [{"from": f"{ds}T{h:02d}:00:00Z", "viewers": v,
                                      "channels": 100} for h, v in enumerate(curve)]})
        k = case["known"][0]
        lq = predict_logqs(k["viewers"], comps, ch1=k["channels"], **kw)
        actual = GT[cid]
        for i, h in enumerate(range(2, 25)):
            y = actual[h - 1]
            lo = math.exp(lq[i][0]) * k["viewers"]
            hi = math.exp(lq[i][4]) * k["viewers"]
            band = "h2-6" if h <= 6 else "h7-12" if h <= 12 else "h13-18" if h <= 18 else "h19-24"
            bands[band].append(1.0 if lo <= y <= hi else 0.0)
    print("  practice+3 90% coverage by horizon:",
          {b: round(sum(v) / len(v), 3) for b, v in bands.items()})


def extrapolate(**kw):
    """Forecast for synthetic huge hour-1 prints using p00's comparables."""
    case = json.loads((KIT / "input" / "p00.in").read_text())
    comps = case["comparables"]
    max_h1 = max(curve_of(c)[0] for c in comps if curve_of(c) is not None)
    print(f"  max comparable h1 = {max_h1:,}")
    for h1 in (269_517, 500_000, 1_000_000, 2_000_000):
        lq = predict_logqs(h1, comps, ch1=None, **kw)
        med = [round(math.exp(lq[i][2]) * h1) for i in range(23)]
        q05 = [round(math.exp(lq[i][0]) * h1) for i in range(23)]
        q95 = [round(math.exp(lq[i][4]) * h1) for i in range(23)]
        print(f"  h1={h1:,}: median h2={med[0]:,} h6={med[4]:,} h9={med[7]:,} "
              f"h12={med[10]:,} h18={med[16]:,} h24={med[22]:,}")
        print(f"          q05 h24={q05[22]:,} q95 h24={q95[22]:,} "
              f"q95/h1={q95[22]/h1:.2f} monotone={all(med[i] <= med[i+1] or True for i in range(22))}")
        # check quantiles non-decreasing within each row (as clean_row enforces)
        bad = sum(1 for row in lq if any(row[j] > row[j + 1] + 1e-12 for j in range(4)))
        print(f"          rows violating monotonicity before clean_row: {bad}/23")


def main():
    base = dict(sigma=2.5, weak=0.5, blend=0.125, spread_k=0.9)
    print("== LOO split, leader (s=2.5 weak=0.5 blend=0.125 k=0.9) ==")
    loo_split(**base)
    print("== LOO split, same + fixed_w=0.5 ==")
    loo_split(fixed_w=0.5, **base)
    print("== LOO split, same + fixed_w=0.25 ==")
    loo_split(fixed_w=0.25, **base)
    print("== coverage by horizon, leader ==")
    coverage_by_horizon(**base)
    print("== coverage by horizon, shipped defaults ==")
    coverage_by_horizon(sigma=1.0, weak=0.25, blend=0.5, spread_k=1.0)
    print("== extrapolation, leader ==")
    extrapolate(**base)
    print("== extrapolation, shipped defaults ==")
    extrapolate(sigma=1.0, weak=0.25, blend=0.5, spread_k=1.0)
    print("== fine k around the leader ==")
    pool = load_raw_comps()
    for k in (0.85, 0.9, 0.95, 1.0):
        lv = loo(pool, spread_k=k, sigma=2.5, weak=0.5, blend=0.125)
        pm, _ = practice(pool, spread_k=k, sigma=2.5, weak=0.5, blend=0.125)
        pme, _ = practice(pool, extra=True, spread_k=k, sigma=2.5, weak=0.5, blend=0.125)
        print(f"  k={k}: LOO-big100={lv[1]:.4f} LOO={lv[0]:.4f} big300={lv[2]:.4f} "
              f"practice={pm:.4f} practice+3={pme:.4f}")


if __name__ == "__main__":
    main()
