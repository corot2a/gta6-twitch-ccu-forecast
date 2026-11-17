"""Explore the comparables in the practice cases: unlock types, ratio curves, coverage."""
import json
import math
import re
from pathlib import Path

KIT = Path("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit")
CASES = ["p00", "p01", "p02", "p03"]


def load_case(cid):
    return json.loads((KIT / "input" / f"{cid}.in").read_text())


def day_curve(comp):
    """24 hourly viewer counts for the release day (hours 1..24), None if missing.
    Comparable shape: {"day_start": "2022-02-25T00:00:00Z", "hourly": [{"from": ..., "viewers": ...}]}"""
    ds = comp.get("day_start", "")[:10]
    by_hour = {}
    for row in comp.get("hourly", []):
        f = row.get("from", "")
        if f[:10] == ds:
            by_hour[int(f[11:13])] = row.get("viewers")
    return [by_hour.get(h) for h in range(24)]


def classify(unlock):
    t = (unlock or "").lower()
    if any(k in t for k in ("rolling", "midnight local", "12 am local", "12am local",
                            "midnight in each", "local midnight", "midnight (local",
                            "local time")):
        return "rolling"
    if any(k in t for k in ("fixed", "global unlock", "simultaneous", "single global",
                            "worldwide", "all regions", "same time")):
        return "fixed"
    return "unknown"


def main():
    all_comps = {}
    for cid in CASES:
        case = load_case(cid)
        for c in case["comparables"]:
            all_comps.setdefault(c["name"], c)
    print(f"unique comparables across practice cases: {len(all_comps)}")
    kinds = {}
    for name, c in sorted(all_comps.items()):
        kinds.setdefault(classify(c.get("unlock")), []).append(name)
    for k, v in kinds.items():
        print(f"\n== {k} ({len(v)}) ==")
        for n in v:
            print(f"  {n}: {all_comps[n].get('unlock')!r}")
    # completeness of release-day curves
    print("\n== release-day curve completeness (hours 1..24 present) ==")
    complete, partial = [], []
    for name, c in sorted(all_comps.items()):
        curve = day_curve(c)
        n = sum(1 for x in curve if x is not None)
        (complete if n == 24 else partial).append((name, n, curve[0]))
    print(f"complete: {len(complete)} / {len(all_comps)}")
    for name, n, h1 in complete:
        print(f"  {name}: h1={h1}")
    print(f"partial: {len(partial)}")
    for name, n, h1 in partial:
        print(f"  {name}: {n}/24 h1={h1}")


if __name__ == "__main__":
    main()
