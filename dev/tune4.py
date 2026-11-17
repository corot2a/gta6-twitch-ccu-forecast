"""Round 4 tuning: parametrised replica of main.ratio_quantiles.

Grid over kernel width, weak-analogue weight, kernel-vs-linear blend, spread
scale, unlock-type weighting and channels. Selection metrics, in order:
LOO-big100 (h1 >= 100k, the GTA VI regime, n=16), LOO overall, practice mean,
practice+3 mean (extra pool of the other 3 practice launches, mimics final).
A config must not worsen practice vs the shipped defaults to be adopted.
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import main as fc  # noqa: E402
from eval_ship import load_raw_comps, curve_of, pinball, score_logratios, GT  # noqa: E402

KIT = Path("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit")
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def is_rolling(unlock):
    t = (unlock or "").lower()
    return ("rolling" in t or "midnight local" in t or "midnight in each" in t
            or "local midnight" in t)


def hour0_channels(comp):
    ds = (comp.get("day_start") or "")[:10]
    for row in comp.get("hourly") or []:
        if (row.get("from") or "")[:10] == ds and (row.get("from") or "")[11:13] == "00":
            return row.get("channels")
    return None


def predict_logqs(h1, pool_raw, ch1=None, sigma=1.0, weak=0.25, blend=0.5,
                  spread_k=1.0, fixed_w=1.0, use_ch=False):
    """blend = weight of the kernel estimator; (1-blend) = local-linear."""
    log_h1 = math.log(h1)
    log_ch1 = math.log(ch1) if ch1 else None
    pool = []
    for comp in pool_raw:
        tier = fc.tier_of(comp)
        if tier in ("target-history", "early-access-contaminated"):
            continue
        curve = fc.release_day_curve(comp)
        if curve is None:
            continue
        gap = math.log(curve[0]) - log_h1
        w = math.exp(-((gap / sigma) ** 2))
        if tier == "weak":
            w *= weak
        if not is_rolling(comp.get("unlock")):
            w *= fixed_w
        if use_ch and log_ch1:
            cch = hour0_channels(comp)
            if cch:
                dc = math.log(cch) - log_ch1
                w *= math.exp(-(dc / 1.0) ** 2)
        ratios = [math.log(curve[h] / curve[0]) for h in range(1, 24)]
        pool.append((w, gap, ratios))
    if not pool:
        return []
    out = []
    for h in range(23):
        pairs = [(ratios[h], w) for w, _, ratios in pool]
        kqs = [fc.weighted_quantile(pairs, tau) for tau in LEVELS]
        tw = sum(w for w, _, _ in pool)
        mx = sum(w * x for w, x, _ in pool) / tw
        my = sum(w * ratios[h] for w, _, ratios in pool) / tw
        sxx = sum(w * (x - mx) ** 2 for w, x, _ in pool)
        sxy = sum(w * (x - mx) * (ratios[h] - my) for w, x, ratios in pool)
        slope = sxy / sxx if sxx > 1e-12 else 0.0
        centre = my - slope * mx
        resid = [(ratios[h] - (centre + slope * x), w) for w, x, ratios in pool]
        lqs = [centre + fc.weighted_quantile(resid, tau) for tau in LEVELS]
        merged = [blend * kq + (1 - blend) * lq for kq, lq in zip(kqs, lqs)]
        med = merged[2]
        out.append([med + spread_k * (q - med) for q in merged])
    return out


def loo(pool_raw, **kw):
    usable = []
    for c in pool_raw:
        curve = curve_of(c)
        if curve is not None:
            usable.append((c, curve))
    rows = []
    for c, curve in usable:
        sub = [o for o, _ in usable if o is not c]
        lq = predict_logqs(curve[0], sub, ch1=None, **kw)
        rows.append((c["name"], curve[0], score_logratios(curve[0], lq, curve)))
    mean = sum(l for _, _, l in rows) / len(rows)
    b100 = [l for _, h, l in rows if h >= 100_000]
    b300 = [l for _, h, l in rows if h >= 300_000]
    return (mean, sum(b100) / len(b100), sum(b300) / len(b300), rows)


def practice(pool_raw, extra=False, **kw):
    losses = {}
    for cid in ("p00", "p01", "p02", "p03"):
        case = json.loads((KIT / "input" / f"{cid}.in").read_text())
        comps = list(case["comparables"])
        if extra:
            for ocid in ("p00", "p01", "p02", "p03"):
                if ocid == cid:
                    continue
                curve = GT[ocid]
                ds = {"p00": "2026-09-15", "p01": "2018-10-26",
                      "p02": "2022-11-09", "p03": "2023-10-20"}[ocid]
                comps.append({
                    "name": f"practice-{ocid}",
                    "day_start": ds + "T00:00:00Z",
                    "unlock": "Tier: comparable. Rolling midnight local.",
                    "hourly": [{"from": f"{ds}T{h:02d}:00:00Z", "viewers": v,
                                "channels": 100} for h, v in enumerate(curve)],
                })
        k = case["known"][0]
        lq = predict_logqs(k["viewers"], comps, ch1=k["channels"], **kw)
        losses[cid] = score_logratios(k["viewers"], lq, GT[cid])
    return sum(losses.values()) / 4, losses


def main():
    pool = load_raw_comps()
    t0 = time.time()
    # Sanity: defaults must reproduce the shipped numbers.
    lv = loo(pool)
    pm, _ = practice(pool)
    pme, _ = practice(pool, extra=True)
    print(f"defaults check: LOO={lv[0]:.4f} big100={lv[1]:.4f} big300={lv[2]:.4f} "
          f"practice={pm:.4f} practice+3={pme:.4f} (expect 0.4201/0.2292/0.1337/0.1924/0.1891)")
    results = []
    for sigma in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        for weak in (0.0, 0.25, 0.5):
            for blend in (0.0, 0.25, 0.5, 0.75, 1.0):
                lv = loo(pool, sigma=sigma, weak=weak, blend=blend)
                pm, _ = practice(pool, sigma=sigma, weak=weak, blend=blend)
                pme, _ = practice(pool, extra=True, sigma=sigma, weak=weak, blend=blend)
                results.append((lv[1], lv[0], lv[2], pm, pme, sigma, weak, blend))
    results.sort()
    print(f"\nstage 1 (sigma x weak x blend, k=1.0), {len(results)} configs, {time.time()-t0:.0f}s")
    print("top 12 by LOO-big100:")
    for b100, overall, b300, pm, pme, s, w, b in results[:12]:
        print(f"  s={s:<4} weak={w:<4} blend={b:<4} LOO-big100={b100:.4f} "
              f"LOO={overall:.4f} big300={b300:.4f} practice={pm:.4f} practice+3={pme:.4f}")
    print("defaults rank:", [i for i, r in enumerate(results)
                             if r[5] == 1.0 and r[6] == 0.25 and r[7] == 0.5][0] + 1,
          f"of {len(results)}")


if __name__ == "__main__":
    main()
