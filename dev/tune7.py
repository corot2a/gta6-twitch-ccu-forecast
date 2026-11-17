"""Round 4, stage 4: fixed_w sweep + diagnosis + horizon-dependent spread."""
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


def loo_rows(**kw):
    pool = load_raw_comps()
    usable = [(c, curve_of(c)) for c in pool if curve_of(c) is not None]
    rows = []
    for c, curve in usable:
        sub = [o for o, _ in usable if o is not c]
        lq = predict_logqs(curve[0], sub, ch1=None, **kw)
        rows.append((c["name"], curve[0], is_rolling(c.get("unlock")),
                     score_logratios(curve[0], lq, curve)))
    return rows


def horizon_scale(log_qs, early=0.0, late=0.0):
    """Scale (q - median) per hour: +early for h2-6, -late for h13-24."""
    out = []
    for i, qs in enumerate(log_qs):
        h = i + 2
        m = 1.0 + (early if h <= 6 else 0.0) - (late if h >= 13 else 0.0)
        med = qs[2]
        out.append([med + m * (q - med) for q in qs])
    return out


def practice_horizon(pool_raw, early=0.0, late=0.0, extra=False, **kw):
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
                comps.append({"name": f"practice-{ocid}", "day_start": ds + "T00:00:00Z",
                              "unlock": "Tier: comparable. Rolling midnight local.",
                              "hourly": [{"from": f"{ds}T{h:02d}:00:00Z", "viewers": v,
                                          "channels": 100} for h, v in enumerate(curve)]})
        k = case["known"][0]
        lq = horizon_scale(predict_logqs(k["viewers"], comps, ch1=k["channels"], **kw),
                           early, late)
        losses[cid] = score_logratios(k["viewers"], lq, GT[cid])
    return sum(losses.values()) / 4, losses


def loo_horizon(early=0.0, late=0.0, **kw):
    pool = load_raw_comps()
    usable = [(c, curve_of(c)) for c in pool if curve_of(c) is not None]
    rows = []
    for c, curve in usable:
        sub = [o for o, _ in usable if o is not c]
        lq = horizon_scale(predict_logqs(curve[0], sub, ch1=None, **kw), early, late)
        rows.append((c["name"], curve[0], score_logratios(curve[0], lq, curve)))
    mean = sum(l for _, _, l in rows) / len(rows)
    b100 = [l for _, h, l in rows if h >= 100_000]
    b300 = [l for _, h, l in rows if h >= 300_000]
    return mean, sum(b100) / len(b100), sum(b300) / len(b300)


def main():
    base = dict(sigma=2.5, weak=0.5, blend=0.125, spread_k=0.9)
    print("== diagnosis: LOO loss under leader, by unlock class ==")
    rows = loo_rows(**base)
    n_roll = sum(1 for _, _, r, _ in rows if r)
    print(f"rolling: {n_roll} / {len(rows)} comparables")
    print("  worst 8:", ", ".join(f"{n}({'R' if r else 'F'} {l:.2f})"
                                  for n, _, r, l in sorted(rows, key=lambda x: -x[3])[:8]))
    print("  best 8:", ", ".join(f"{n}({'R' if r else 'F'} {l:.2f})"
                                 for n, _, r, l in sorted(rows, key=lambda x: x[3])[:8]))
    print("\n== fixed_w sweep (leader otherwise) ==")
    pool = load_raw_comps()
    for fw in (1.0, 0.75, 0.5, 0.375, 0.25, 0.125, 0.0):
        rows = loo_rows(fixed_w=fw, **base)
        rb = [l for _, h, r, l in rows if r and h >= 100_000]
        fb = [l for _, h, r, l in rows if not r and h >= 100_000]
        b100 = [l for _, h, _, l in rows if h >= 100_000]
        pm, _ = practice(pool, fixed_w=fw, **base)
        pme, _ = practice(pool, extra=True, fixed_w=fw, **base)
        print(f"  fw={fw:<5} rolling-big={sum(rb)/len(rb):.4f} (n={len(rb)}) "
              f"fixed-big={sum(fb)/len(fb):.4f} (n={len(fb)}) "
              f"LOO-big100={sum(b100)/len(b100):.4f} practice={pm:.4f} practice+3={pme:.4f}")
    print("\n== horizon spread (early +, late -) on leader, fw=1.0 and fw=0.5 ==")
    for fw in (1.0, 0.5):
        for early, late in ((0.0, 0.0), (0.15, 0.0), (0.3, 0.0), (0.0, 0.15),
                            (0.0, 0.3), (0.15, 0.15), (0.3, 0.15), (0.3, 0.3)):
            lv = loo_horizon(early, late, fixed_w=fw, **base)
            pm, _ = practice_horizon(pool, early, late, fixed_w=fw, **base)
            pme, _ = practice_horizon(pool, early, late, extra=True, fixed_w=fw, **base)
            print(f"  fw={fw} early=+{early} late=-{late}: "
                  f"LOO-big100={lv[1]:.4f} LOO={lv[0]:.4f} big300={lv[2]:.4f} "
                  f"practice={pm:.4f} practice+3={pme:.4f}")


if __name__ == "__main__":
    main()
