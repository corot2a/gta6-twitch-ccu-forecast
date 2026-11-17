"""Round 4, stage 5: re-tune sigma/blend/k jointly with fixed_w=0.25,
scored on rolling-big LOO (the GTA VI regime), LOO-big100, practice, practice+3."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tune4 import loo, practice, load_raw_comps  # noqa: E402
from tune7 import loo_rows  # noqa: E402


def main():
    pool = load_raw_comps()
    print("fw=0.25 joint grid (sigma x blend x k):")
    results = []
    for sigma in (1.5, 2.0, 2.5, 3.0):
        for blend in (0.0, 0.125, 0.25):
            for k in (0.9, 1.0):
                kw = dict(sigma=sigma, weak=0.5, blend=blend, spread_k=k, fixed_w=0.25)
                rows = loo_rows(**kw)
                rb = [l for _, h, r, l in rows if r and h >= 100_000]
                rb_mean = sum(rb) / len(rb)
                lv = loo(pool, **kw)
                pm, _ = practice(pool, **kw)
                pme, _ = practice(pool, extra=True, **kw)
                results.append((rb_mean, lv[1], lv[0], pm, pme, sigma, blend, k))
    results.sort()
    for rb, b100, overall, pm, pme, s, b, k in results:
        print(f"  s={s:<4} blend={b:<5} k={k}: rolling-big={rb:.4f} "
              f"LOO-big100={b100:.4f} LOO={overall:.4f} practice={pm:.4f} practice+3={pme:.4f}")
    print("\nshipped defaults for reference: rolling-big=? LOO-big100=0.2292 "
          "LOO=0.4201 practice=0.1924 practice+3=0.1891")
    print("leader without fw (s=2.5 weak=0.5 blend=0.125 k=0.9): "
          "rolling-big=0.1747 LOO-big100=0.2172 LOO=0.3863 practice=0.1772 practice+3=0.1681")


if __name__ == "__main__":
    main()
