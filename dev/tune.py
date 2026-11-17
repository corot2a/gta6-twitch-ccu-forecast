"""Tune the forecaster: kernel width, tier weights, pre-unlock filter, channels.
Evaluated two ways: LOO-CV over comparables, and direct scoring on the 4 practice
cases vs SullyGnome ground truth (the closest analogue of the final GTA VI case:
all four are rolling-midnight launches with sizeable hour-1 prints)."""
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import load_comps, wquantile, pinball, tier_weight, is_fixed  # noqa: E402

KIT = Path("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit")
GT = json.loads(Path(__file__).resolve().parent.joinpath("ground_truth.json").read_text())
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def peak_frac(c):
    return c["h1"] / max(c["curve"])


def predict_ratios(h1, ch1, pool, sigma=1.0, use_ch=False, min_peak_frac=0.0,
                   drop_contaminated=False, drop_weak=False):
    lh1 = math.log(h1)
    lch = math.log(ch1) if ch1 else None
    out = []
    for h in range(23):
        pairs = []
        for c in pool:
            if peak_frac(c) < min_peak_frac:
                continue
            tw = tier_weight(c["unlock"])
            if drop_contaminated and "early-access-contaminated" in c["unlock"]:
                continue
            if drop_weak and "Tier: weak" in c["unlock"]:
                continue
            d = math.log(c["h1"]) - lh1
            w = tw * math.exp(-(d / sigma) ** 2)
            if use_ch and lch and c.get("ch1"):
                dc = math.log(c["ch1"]) - lch
                w *= math.exp(-(dc / 1.0) ** 2)
            pairs.append((c["ratios"][h], w))
        out.append([wquantile(pairs, tau) for tau in LEVELS])
    return out


def score_from_logratios(h1, log_qs, actual24):
    total, n = 0.0, 0
    for i, h in enumerate(range(2, 25)):
        y = actual24[h - 1]
        for tau, lr in zip(LEVELS, log_qs[i]):
            total += pinball(y, math.exp(lr) * h1, tau)
            n += 1
    return total / n


def practice_score(pool, **kw):
    losses = {}
    for cid in ("p00", "p01", "p02", "p03"):
        case = json.loads((KIT / "input" / f"{cid}.in").read_text())
        k = case["known"][0]
        lq = predict_ratios(k["viewers"], k["channels"], pool, **kw)
        losses[cid] = score_from_logratios(k["viewers"], lq, GT[cid])
    mean = sum(losses.values()) / 4
    return mean, losses


def loo(pool, **kw):
    losses = []
    for t in pool:
        sub = [c for c in pool if c is not t]
        lq = predict_ratios(t["h1"], t.get("ch1"), sub, **kw)
        losses.append(score_from_logratios(t["h1"], lq, t["curve"]))
    return sum(losses) / len(losses)


def main():
    pool = load_comps()
    print(f"pool={len(pool)}")
    configs = [
        ("sim s=1.0", dict(sigma=1.0)),
        ("sim s=0.5", dict(sigma=0.5)),
        ("sim s=0.75", dict(sigma=0.75)),
        ("sim s=1.5", dict(sigma=1.5)),
        ("sim s=2.0", dict(sigma=2.0)),
        ("sim s=1.0 +ch", dict(sigma=1.0, use_ch=True)),
        ("sim s=0.75 +ch", dict(sigma=0.75, use_ch=True)),
        ("sim s=1.0 pf>=0.05", dict(sigma=1.0, min_peak_frac=0.05)),
        ("sim s=1.0 pf>=0.10", dict(sigma=1.0, min_peak_frac=0.10)),
        ("sim s=0.75 pf>=0.05", dict(sigma=0.75, min_peak_frac=0.05)),
        ("sim s=0.75 pf>=0.10", dict(sigma=0.75, min_peak_frac=0.10)),
        ("sim s=1.0 drop-contam", dict(sigma=1.0, drop_contaminated=True)),
        ("sim s=1.0 drop-weak", dict(sigma=1.0, drop_weak=True)),
        ("sim s=1.0 pf>=0.05 drop-contam", dict(sigma=1.0, min_peak_frac=0.05, drop_contaminated=True)),
        ("sim s=0.75 pf>=0.05 +ch", dict(sigma=0.75, min_peak_frac=0.05, use_ch=True)),
    ]
    for label, kw in configs:
        pm, pl = practice_score(pool, **kw)
        lv = loo(pool, **kw)
        detail = " ".join(f"{k}={v:.3f}" for k, v in pl.items())
        print(f"{label:32s} practice mean={pm:.4f} [{detail}]  LOO={lv:.4f}")


if __name__ == "__main__":
    main()
