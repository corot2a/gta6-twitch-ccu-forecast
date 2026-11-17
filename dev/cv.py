"""Leave-one-launch-out cross-validation over the 38 comparables.

Each comparable is treated as a pseudo-target: its hour-1 print is known, hours 2..24
are predicted from the other comparables' log-ratio curves r_h = ln(v_h / v_1).
Scored with the house pinball loss on the log scale.
"""
import json
import math
from pathlib import Path

KIT = Path("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit")
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def load_comps():
    case = json.loads((KIT / "input" / "p00.in").read_text())
    out = []
    for c in case["comparables"]:
        if "target-history" in c.get("unlock", ""):
            continue
        ds = c["day_start"][:10]
        by_hour = {}
        ch = {}
        for row in c.get("hourly", []):
            f = row.get("from", "")
            if f[:10] == ds:
                by_hour[int(f[11:13])] = row.get("viewers")
                ch[int(f[11:13])] = row.get("channels")
        if len(by_hour) != 24 or by_hour.get(0) is None:
            continue
        curve = [by_hour[h] for h in range(24)]
        out.append({
            "name": c["name"],
            "unlock": c.get("unlock", ""),
            "curve": curve,
            "h1": curve[0],
            "ch1": ch.get(0),
            "ratios": [math.log(curve[h] / curve[0]) for h in range(1, 24)],
        })
    return out


def tier_weight(unlock):
    t = unlock or ""
    if "Tier: weak" in t:
        return 0.25
    if "Tier: early-access-contaminated" in t:
        return 0.5
    return 1.0


def is_rolling(unlock):
    t = (unlock or "").lower()
    return ("rolling" in t or "midnight local" in t or "midnight in each" in t
            or "local midnight" in t)


def is_fixed(unlock):
    t = (unlock or "").lower()
    return ("simultaneous" in t or "single global" in t or "global unlock" in t)


def wquantile(pairs, tau):
    """pairs: list of (value, weight). Weighted quantile with linear interpolation."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return pairs[len(pairs) // 2][0]
    cum, pts = 0.0, []
    for v, w in pairs:
        cum += w
        pts.append((cum / tot, v))
    if tau <= pts[0][0]:
        return pts[0][1]
    if tau >= pts[-1][0]:
        return pts[-1][1]
    for i in range(1, len(pts)):
        if pts[i][0] >= tau:
            p0, v0 = pts[i - 1]
            p1, v1 = pts[i]
            if p1 == p0:
                return v1
            return v0 + (v1 - v0) * (tau - p0) / (p1 - p0)
    return pts[-1][1]


def predict(target, pool, scheme):
    """Return per-hour (2..24) list of 5 log-quantiles of the ratio r_h."""
    lh1 = math.log(target["h1"])
    out = []
    for h in range(23):
        pairs = []
        for c in pool:
            w = tier_weight(c["unlock"])
            if scheme in ("rolling", "rolling_sim"):
                if not is_rolling(c["unlock"]):
                    continue
            if scheme in ("sim", "rolling_sim"):
                d = math.log(c["h1"]) - lh1
                w *= math.exp(-(d / 1.0) ** 2)
            if scheme == "nofixed":
                if is_fixed(c["unlock"]):
                    continue
            pairs.append((c["ratios"][h], w))
        qs = [wquantile(pairs, tau) for tau in LEVELS]
        out.append(qs)
    return out


def pinball(y, q, tau):
    d = math.log(max(1.0, y)) - math.log(max(1.0, q))
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_target(target, log_qs):
    total, n = 0.0, 0
    for i, h in enumerate(range(2, 25)):
        y = target["curve"][h - 1]
        for tau, lr in zip(LEVELS, log_qs[i]):
            total += pinball(y, math.exp(lr) * target["h1"], tau)
            n += 1
    return total / n


def main():
    comps = load_comps()
    print(f"comparables with full release-day curves: {len(comps)}")
    schemes = ["all", "nofixed", "rolling", "sim", "rolling_sim"]
    for scheme in schemes:
        losses = []
        for t in comps:
            pool = [c for c in comps if c is not t]
            lq = predict(t, pool, scheme)
            losses.append((t["name"], t["h1"], score_target(t, lq)))
        mean = sum(l for _, _, l in losses) / len(losses)
        big = [(n, h1, l) for n, h1, l in losses if h1 >= 300_000]
        bigmean = sum(l for _, _, l in big) / len(big) if big else float("nan")
        worst = sorted(losses, key=lambda x: -x[2])[:5]
        print(f"\n[{scheme}] mean={mean:.4f}  big-h1(>=300k) mean={bigmean:.4f} n={len(big)}")
        print("  worst:", ", ".join(f"{n}({l:.3f})" for n, _, l in worst))


if __name__ == "__main__":
    main()
