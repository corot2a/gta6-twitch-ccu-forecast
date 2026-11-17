"""Round 2 tuning: sigma, spread calibration, and adding the other practice launches
to the pool (mimics the final case, whose comparables include the 4 practice launches)."""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import load_comps, wquantile, pinball, tier_weight  # noqa: E402
from tune import predict_ratios, score_from_logratios  # noqa: E402

KIT = Path("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit")
GT = json.loads(Path(__file__).resolve().parent.joinpath("ground_truth.json").read_text())
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def extra_pool(exclude_cid):
    """The other practice launches as pseudo-comparables (their true day curves)."""
    out = []
    for cid in ("p00", "p01", "p02", "p03"):
        if cid == exclude_cid:
            continue
        curve = GT[cid]
        out.append({
            "name": f"practice-{cid}",
            "unlock": "Tier: comparable.",
            "curve": curve,
            "h1": curve[0],
            "ch1": None,
            "ratios": [math.log(curve[h] / curve[0]) for h in range(1, 24)],
        })
    return out


def spread_scale(log_qs, k):
    out = []
    for qs in log_qs:
        med = qs[2]
        out.append([med + k * (q - med) for q in qs])
    return out


def practice_score(pool, extra=False, k=1.0, **kw):
    losses = {}
    for cid in ("p00", "p01", "p02", "p03"):
        case = json.loads((KIT / "input" / f"{cid}.in").read_text())
        kk = case["known"][0]
        p = list(pool)
        if extra:
            p = p + extra_pool(cid)
        lq = predict_ratios(kk["viewers"], kk["channels"], p, **kw)
        lq = spread_scale(lq, k)
        losses[cid] = score_from_logratios(kk["viewers"], lq, GT[cid])
    return sum(losses.values()) / 4, losses


def loo(pool, k=1.0, big_only=False, **kw):
    losses = []
    for t in pool:
        if big_only and t["h1"] < 100_000:
            continue
        sub = [c for c in pool if c is not t]
        lq = predict_ratios(t["h1"], t.get("ch1"), sub, **kw)
        lq = spread_scale(lq, k)
        losses.append(score_from_logratios(t["h1"], lq, t["curve"]))
    return sum(losses) / len(losses), len(losses)


def main():
    pool = [c for c in load_comps() if "early-access-contaminated" not in c["unlock"]]
    print(f"pool (contaminated dropped)={len(pool)}")
    for sigma in (0.75, 1.0, 1.25, 1.5):
        pm, pl = practice_score(pool, sigma=sigma, drop_contaminated=True)
        pme, ple = practice_score(pool, extra=True, sigma=sigma, drop_contaminated=True)
        lv, _ = loo(pool, sigma=sigma, drop_contaminated=True)
        lvb, nb = loo(pool, big_only=True, sigma=sigma, drop_contaminated=True)
        print(f"s={sigma}: practice={pm:.4f} practice+3={pme:.4f} "
              f"[{ ' '.join(f'{k}={v:.3f}' for k, v in ple.items()) }] LOO={lv:.4f} LOO-big={lvb:.4f}(n={nb})")
    print()
    for k in (0.7, 0.85, 1.0, 1.15, 1.3):
        pm, pl = practice_score(pool, extra=True, sigma=1.0, drop_contaminated=True, k=k)
        lvb, nb = loo(pool, big_only=True, sigma=1.0, drop_contaminated=True, k=k)
        print(f"spread k={k}: practice+3={pm:.4f} [{ ' '.join(f'{kk}={v:.3f}' for kk, v in pl.items()) }] LOO-big={lvb:.4f}")


if __name__ == "__main__":
    main()
