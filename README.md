# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/corot2a/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/corot2a/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day (2026-11-19) Twitch concurrent viewers, hour by hour (hours 2–24), as five quantiles per hour. The model learns the shape of a big launch day from comparable past launches (SullyGnome hourly data): for each launch it takes log-ratios of hourly viewers to its own hour-1 print, weights launches by similarity of hour-1 size (Gaussian kernel on log hour-1), and combines kernel-weighted quantiles with a local-linear size-trend fit. Launches whose release day was contaminated by a days-earlier early-access opening are excluded, as is the target's own pre-release history (not a launch). Validated on the four practice launches (Marvel's Wolverine, Red Dead Redemption 2, God of War Ragnarök, Marvel's Spider-Man 2) against SullyGnome ground truth: mean pinball loss 0.19 vs 0.39 for the persistence baseline. Python 3.13, standard library only, deterministic, no network.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Model

- **Anchor:** hour 1's print, the one number the case gives for the target launch. Every comparable launch is reduced to log-ratios `ln(viewers_h / viewers_1)` for hours 2–24, so the forecast is a shape applied to the target's own scale.
- **Similarity:** each comparable is weighted by `exp(-((ln h1_comp - ln h1_target) / 1.0)^2)` — launches of a similar size share a similar launch-day curve (rolling-midnight unlocks, US East/West jumps at 05:00Z/08:00Z, evening peak). Weak analogues keep a quarter weight.
- **Centre:** the average (in log space) of two estimators — kernel-weighted quantiles, and a local-linear fit of log-ratio on the log hour-1 gap read at the target. The linear term carries the size trend to the target's own scale, which matters if GTA VI's hour 1 is bigger than every comparable's.
- **Exclusions:** early-access-contaminated launches and the target-history comparable are dropped before pooling; a comparable needs a complete 24-hour release day with a positive hour 1.

## Validation

Scored with the house pinball loss on the log scale, against SullyGnome "Average viewers" ground truth fetched for the four practice launches:

| case | this model | persistence baseline |
| ---- | ---------- | -------------------- |
| p00 Marvel's Wolverine | 0.295 | 0.744 |
| p01 Red Dead Redemption 2 | 0.162 | 0.287 |
| p02 God of War Ragnarök | 0.121 | 0.215 |
| p03 Marvel's Spider-Man 2 | 0.192 | 0.303 |
| **mean** | **0.192** | **0.387** |

Leave-one-launch-out cross-validation over the comparables: 0.46 overall, 0.20 for launches with hour-1 prints ≥ 100k (the regime GTA VI will be in). Two runs on the same input produce byte-identical output; runtime is ~0.03 s per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=nnrgb447&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
