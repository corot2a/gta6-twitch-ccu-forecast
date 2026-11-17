"""Forecast GTA VI launch-day Twitch viewers, hour by hour (hours 2..24).

Model: learn the shape of a big launch day from past launches.
For every comparable past launch we take its release day's 24 hourly viewer
counts and turn them into log-ratios against its own hour-1 print:

    r_h = ln(viewers_h / viewers_1)          for h = 2..24

Hour 1 is the anchor because it is the one number the case actually gives us
for the target launch. A launch's ratio curve is mostly a clock: midnight
rolling unlocks bring New Zealand and Asia first, then Europe, then the big
jumps when the US East (05:00Z) and US West (08:00Z) midnights hit, then a
broad evening peak. Launches of a similar size share a similar curve, so each
comparable is weighted by how close its own hour-1 print is to the target's,
with a wide Gaussian kernel on log hour-1 (width 2.5):

    w = exp(-((ln h1_comp - ln h1_target) / 2.5) ** 2)

A wide kernel matters: with a narrow one the pool is a handful of
near-neighbours and the average sits inside their range, while the wide
kernel pools the whole launch population and lets the size trend (below) do
the work. Weak analogues keep half weight.

Unlock type matters as much as size. Rolling-midnight launches (one region's
midnight after another) share one shape; simultaneous global unlocks share
another, and mixing the two families blurs both. So a comparable whose unlock
note does not describe a rolling midnight keeps only a quarter of its weight.
GTA VI itself unlocks on rolling midnights, and on the eight big rolling
launches in leave-one-out validation this weighting cuts the loss by about
8%, while the four rolling practice launches improve by about a fifth.

Two kinds of comparables are left out of the pool: the target game's own
pre-release history (not a launch at all), and launches whose release day was
contaminated by a days-earlier early-access opening (their "release day" curve
is really day three of play, so its shape does not match a fresh launch).

For each forecast hour we take the weighted quantiles of the pooled log-ratios
at the five scored levels and exponentiate back against the target's hour-1
print. The centre comes from two estimators, averaged in log space: the plain
kernel-weighted quantiles (weight 0.125), and a local-linear fit of log-ratio
on the log hour-1 gap (evaluated at the target, gap zero) whose spread is the
weighted quantiles of the fit's residuals (weight 0.875). The linear term
matters because the target may be bigger than every comparable: a plain
average can only sit inside the range of past launches, while the fit carries
the size trend to the target's own scale. Averaging the two keeps either one's
failure mode from dominating. Finally the quantiles are pulled 10% toward the
median: the pooled spreads run slightly wide on big launches.
No randomness anywhere: same input, same bytes out.

Validation (leave-one-launch-out over the 38 comparables, and the four
practice launches against SullyGnome ground truth): LOO 0.408, LOO on
hour-1 >= 100k launches 0.232, LOO on big rolling launches 0.157, practice
mean pinball loss 0.153 (0.146 with the final pool of 42 comparables, which
includes the other practice launches). House persistence baseline:
practice 0.387.

Run by the grader as `python3 -P -s /bot/main.py` (Python 3.13, stdlib only,
no network). Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand
in for /input and /output.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Width of the log hour-1 similarity kernel (natural-log units).
KERNEL_WIDTH = 2.5
# Weight kept for comparables marked a weak analogue.
WEAK_WEIGHT = 0.5
# Weight of the plain kernel-quantile estimator; the local-linear fit takes
# the rest (1 - KERNEL_BLEND).
KERNEL_BLEND = 0.125
# Spread scale: quantiles are pulled this far toward the median.
SPREAD_SCALE = 0.9
# Weight kept for comparables whose unlock is not a rolling midnight.
FIXED_WEIGHT = 0.25
# Hard caps the rules put on any forecast value.
MAX_VIEWERS = 1e9
# Fallback spread (log units per sqrt hour) if no comparable survives filtering.
FALLBACK_SIGMA = 0.15
FALLBACK_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up (the way JavaScript rounds it).

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def tier_of(comp: dict) -> str:
    """The comparability tier recorded in the comparable's unlock note."""
    text = comp.get("unlock") or ""
    if "Tier: target-history" in text:
        return "target-history"
    if "Tier: early-access-contaminated" in text:
        return "early-access-contaminated"
    if "Tier: weak" in text:
        return "weak"
    return "comparable"


def is_rolling(comp: dict) -> bool:
    """True when the comparable's unlock note describes a rolling midnight.

    The note is free text, so this is keyword matching; anything it cannot
    recognise counts as not rolling and is downweighted, not dropped.
    """
    text = (comp.get("unlock") or "").lower()
    return ("rolling" in text or "midnight local" in text
            or "midnight in each" in text or "local midnight" in text)


def release_day_curve(comp: dict) -> list[float] | None:
    """The comparable's 24 hourly viewer counts for its release day.

    Returns None unless all 24 hours are present and hour 1 is positive:
    a ratio curve needs a real anchor and no holes.
    """
    day = (comp.get("day_start") or "")[:10]
    if not day:
        return None
    by_hour: dict[int, float] = {}
    for row in comp.get("hourly") or []:
        stamp = row.get("from") or ""
        if stamp[:10] == day:
            try:
                by_hour[int(stamp[11:13])] = float(row["viewers"])
            except (TypeError, ValueError):
                return None
    if len(by_hour) != 24 or by_hour.get(0, 0) <= 0:
        return None
    return [by_hour[h] for h in range(24)]


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """Weighted tau-quantile of the values, linear interpolation on cumulative weight."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    running = 0.0
    points: list[tuple[float, float]] = []
    for value, weight in ordered:
        running += weight
        points.append((running / total, value))
    if tau <= points[0][0]:
        return points[0][1]
    if tau >= points[-1][0]:
        return points[-1][1]
    for i in range(1, len(points)):
        if points[i][0] >= tau:
            p0, v0 = points[i - 1]
            p1, v1 = points[i]
            if p1 == p0:
                return v1
            return v0 + (v1 - v0) * (tau - p0) / (p1 - p0)
    return points[-1][1]


def ratio_quantiles(case: dict, hour1: float) -> list[list[float]]:
    """Per forecast hour, the five quantiles of log(viewers_h / viewers_1).

    Two estimators, averaged in log space. (1) Kernel-weighted quantiles of
    the pooled log-ratios. (2) A local-linear fit of log-ratio on the log
    hour-1 gap, read at the target (gap zero), with spread from the weighted
    quantiles of the fit residuals. Both use the same similarity weights.
    Non-rolling comparables keep only FIXED_WEIGHT: their shape family
    differs from a rolling-midnight launch's. The merged quantiles are
    pulled SPREAD_SCALE of the way toward the median.
    """
    log_h1 = math.log(hour1)
    pool: list[tuple[float, float, list[float]]] = []  # (weight, log-h1 gap, 23 log-ratios)
    for comp in case.get("comparables") or []:
        tier = tier_of(comp)
        if tier in ("target-history", "early-access-contaminated"):
            continue
        curve = release_day_curve(comp)
        if curve is None:
            continue
        gap = math.log(curve[0]) - log_h1
        weight = math.exp(-((gap / KERNEL_WIDTH) ** 2))
        if tier == "weak":
            weight *= WEAK_WEIGHT
        if not is_rolling(comp):
            weight *= FIXED_WEIGHT
        ratios = [math.log(curve[h] / curve[0]) for h in range(1, 24)]
        pool.append((weight, gap, ratios))
    if not pool:
        return []
    out: list[list[float]] = []
    for h in range(23):
        # (1) Kernel-weighted quantiles.
        pairs = [(ratios[h], weight) for weight, _, ratios in pool]
        kernel_qs = [weighted_quantile(pairs, tau) for tau in LEVELS]
        # (2) Local-linear centre: weighted least squares of ratio on gap.
        total_w = sum(w for w, _, _ in pool)
        mean_x = sum(w * x for w, x, _ in pool) / total_w
        mean_y = sum(w * ratios[h] for w, _, ratios in pool) / total_w
        sxx = sum(w * (x - mean_x) ** 2 for w, x, _ in pool)
        sxy = sum(w * (x - mean_x) * (ratios[h] - mean_y)
                  for w, x, ratios in pool)
        slope = sxy / sxx if sxx > 1e-12 else 0.0
        centre = mean_y - slope * mean_x  # fitted value at gap 0
        resid = [(ratios[h] - (centre + slope * x), w)
                 for w, x, ratios in pool]
        linear_qs = [centre + weighted_quantile(resid, tau) for tau in LEVELS]
        merged = [KERNEL_BLEND * kq + (1 - KERNEL_BLEND) * lq
                  for kq, lq in zip(kernel_qs, linear_qs)]
        median = merged[2]
        out.append([median + SPREAD_SCALE * (q - median) for q in merged])
    return out


def clean_row(values: list[float]) -> list[float | int]:
    """Round, then enforce the output rules: > 0, <= 1e9, non-decreasing."""
    rounded = [round2(v) for v in values]
    fixed: list[float | int] = []
    prev = 0.0
    for v in rounded:
        v = max(1.0, min(MAX_VIEWERS, float(v)))
        if v < prev:
            v = prev
        fixed.append(int(v) if float(v).is_integer() else v)
        prev = float(v)
    return fixed


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_qs = ratio_quantiles(case, p1)
    rows = []
    for i, row in enumerate(case["forecast"]):
        h = row["hour"]
        if log_qs:
            values = [p1 * math.exp(lq) for lq in log_qs[i]]
        else:
            # No usable comparable: fall back to persistence with a widening spread.
            values = [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1)) for z in FALLBACK_Z]
        rows.append({"hour": h, "q": clean_row(values)})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
