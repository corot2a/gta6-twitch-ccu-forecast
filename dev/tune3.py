"""Round 3: local-linear (LOESS-style) centre vs plain kernel-weighted quantiles.

Plain kernel averaging cannot extrapolate a size trend: if the target's hour-1 is
bigger than every comparable's, its neighbours are all smaller launches and the
average sits at their level. A local-linear fit of log-ratio on log hour-1 gap,
evaluated at the target (gap 0), corrects that. Spread comes from the weighted
quantiles of the fit residuals.
"""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import load_comps, wquantile, tier_weight  # noqa: E402
from tune import score_from_logratios  # noqa: E402
from tune2 import extra_pool, spread_scale  # noqa: E402

KIT = Path("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit")
GT = json.loads(Path(__file__).resolve().parent.joinpath("ground_truth.json").read_text())
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def predict_ll(h1, pool, sigma=1.0):
    lh1 = math.log(h1)
    out = []
    for h in range(23):
        xs, ys, ws = [], [], []
        for c in pool:
            if "early-access-contaminated" in c["unlock"]:
                continue
            x = math.log(c["h1"]) - lh1
            w = math.exp(-(x / sigma) ** 2)
            if "Tier: weak" in c["unlock"]:
                w *= 0.25
            xs.append(x)
            ys.append(c["ratios"][h])
            ws.append(w)
        # Weighted least squares y ~ a + b x.
        sw = sum(ws)
        mx = sum(w * x for w, x in zip(ws, xs)) / sw
        my = sum(w * y for w, y in zip(ws, ys)) / sw
        sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
        sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
        b = sxy / sxx if sxx > 1e-12 else 0.0
        a = my - b * mx
        # Centre at the target (x = 0) is a; spread from residuals around the fit.
        resid = [(y - (a + b * x), w) for x, y, w in zip(xs, ys, ws)]
        out.append([a + wquantile(resid, tau) for tau in LEVELS])
    return out


def practice_score(pool, extra=False):
    losses = {}
    for cid in ("p00", "p01", "p02", "p03"):
        case = json.loads((KIT / "input" / f"{cid}.in").read_text())
        k = case["known"][0]
        p = list(pool)
        if extra:
            p = p + extra_pool(cid)
        lq = predict_ll(k["viewers"], p)
        losses[cid] = score_from_logratios(k["viewers"], lq, GT[cid])
    return sum(losses.values()) / 4, losses


def loo(pool, big_only=False):
    losses = []
    for t in pool:
        if big_only and t["h1"] < 100_000:
            continue
        sub = [c for c in pool if c is not t]
        lq = predict_ll(t["h1"], sub)
        losses.append((t["name"], t["h1"], score_from_logratios(t["h1"], lq, t["curve"])))
    mean = sum(l for _, _, l in losses) / len(losses)
    return mean, losses


def main():
    pool = load_comps()
    pm, pl = practice_score(pool)
    pme, ple = practice_score(pool, extra=True)
    lv, _ = loo(pool)
    lvb, big = loo(pool, big_only=True)
    print(f"local-linear: practice={pm:.4f} [{ ' '.join(f'{k}={v:.3f}' for k, v in pl.items()) }]")
    print(f"local-linear: practice+3={pme:.4f} [{ ' '.join(f'{k}={v:.3f}' for k, v in ple.items()) }]")
    print(f"local-linear: LOO={lv:.4f} LOO-big={lvb:.4f}")
    print("  LOO-big per target:", ", ".join(f"{n}({l:.3f})" for n, _, l in sorted(big, key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
