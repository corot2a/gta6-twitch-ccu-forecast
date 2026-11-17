"""Round 4, stages 1b+2: wider kernels / pure-linear, then spread scale,
unlock-type weighting and channels on the leaders."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tune4 import loo, practice, load_raw_comps  # noqa: E402


def show(label, **kw):
    pool = load_raw_comps()
    lv = loo(pool, **kw)
    pm, pl = practice(pool, **kw)
    pme, ple = practice(pool, extra=True, **kw)
    print(f"{label:44s} LOO-big100={lv[1]:.4f} LOO={lv[0]:.4f} big300={lv[2]:.4f} "
          f"practice={pm:.4f} practice+3={pme:.4f}")
    return lv, pm, pme


def main():
    t0 = time.time()
    print("== stage 1b: wider sigma, linear-heavy ==")
    for sigma in (2.0, 2.5, 3.0, 4.0, 1e9):
        for blend in (0.0, 0.125, 0.25):
            for weak in (0.25, 0.5):
                show(f"s={sigma} weak={weak} blend={blend}",
                     sigma=sigma, weak=weak, blend=blend)
    print(f"({time.time()-t0:.0f}s)")
    print("\n== stage 2: spread k on the leaders ==")
    leaders = [
        ("s=2.0 weak=0.5 blend=0.25", dict(sigma=2.0, weak=0.5, blend=0.25)),
        ("s=2.5 weak=0.5 blend=0.125", dict(sigma=2.5, weak=0.5, blend=0.125)),
        ("s=3.0 weak=0.5 blend=0.0", dict(sigma=3.0, weak=0.5, blend=0.0)),
        ("s=2.0 weak=0.5 blend=0.0", dict(sigma=2.0, weak=0.5, blend=0.0)),
    ]
    for label, kw in leaders:
        for k in (0.8, 0.9, 1.0, 1.1, 1.2):
            show(f"{label} k={k}", spread_k=k, **kw)
    print("\n== stage 2b: unlock-type weight and channels on the leaders ==")
    for label, kw in leaders:
        show(f"{label} fixed_w=0.5", fixed_w=0.5, **kw)
        show(f"{label} fixed_w=0.25", fixed_w=0.25, **kw)
        show(f"{label} +channels", use_ch=True, **kw)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
