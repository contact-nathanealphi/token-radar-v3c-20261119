# token-radar-v3c-20261119

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/contact-nathanealphi/token-radar-v3c-20261119/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/contact-nathanealphi/token-radar-v3c-20261119/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

A token launch radar that ranks pre-token protocols by launch-odds-weighted expected value. The value model fits peer multiples from the snapshot's traded protocols (geometric mean per category, shrunk, with a fitted size tilt); a logistic regression fitted on all 758 protocols scores how much each pre-token protocol resembles the ones that already record a token; the board ranks by raw value × launch odds. Validation is computed at build time and shown on the dashboard: 5-fold cross-validation for the odds model (AUC 0.740) and leave-one-out validation for the value model, each beside its baseline. Every farm plan is a dated, costed timeline, and every one of the 758 snapshot protocols is accounted for.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input <snapshot.json>
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=9ncgmbd6&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
