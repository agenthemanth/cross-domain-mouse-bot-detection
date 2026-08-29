# Re-baseline log

## 2026-08-30 — `circles_human_fast` normalised-coordinate rescale

### What changed
`BalabitValidationPipeline.parseDelbotPoints` now reads each DELBOT file's
`resolution:W,H` header and, when **every** coordinate in the file lies inside
the unit square while the header reports a `W x H` pixel screen, multiplies the
coordinates back to pixels.

Only one folder is affected: `circles_human_fast` (9 of 3,453 training rows).
Its files store coordinates as screen fractions in `[0,1]`; read as pixels they
gave `mean_velocity ~= 0.0066` (a near-stationary "slow human"), which is a unit
error, not a real movement profile. Verified by probing coordinate ranges of all
10 DELBOT folders: every `circles_bot_*` folder and every other `circles_human_*`
folder stores pixels (max |x| in the hundreds); `circles_human_fast` is the only
one with max |x|,|y| <= 1. Headers vary per file (1536x864 and 1536x754) so the
scale is read per file.

`DelbotValidationPipeline` and `FalsePositiveEvaluator` had their own inline
copies of the DELBOT parse loop that silently missed this rescale; both now
delegate to `parseDelbotPoints`.

### Before / after (authoritative — measured by toggling the change with
`git stash` on the frozen reference tools, same JVM, same classpath)

| metric | pre-rescale | post-rescale |
|---|---|---|
| Balabit human FPR @0.5 — `BalabitValidationPipeline` | 9.85% (2381/24182) | **9.79% (2368/24182)** |
| mightymerge micro-avg FPR — `FalsePositiveEvaluator` | 5.45% | **5.28%** |
| Cross-domain pooled bot AUC — `BalabitCrossDomainEval` | 0.5326 | **0.4417** |
| Cross-domain Balabit FPR @0.5 | 9.85% | 9.79% |
| DELBOT training instances | 3453 | 3453 (rows rescaled, none dropped) |

### Interpretation
- **No qualitative conclusion changes.** Zero-shot DELBOT->Balabit transfer was
  already "no transfer" (pooled bot AUC ~0.53, TPR@<=1%FPR ~0). Post-fix it is
  0.44 — still no transfer, if anything slightly *worse* ranking. The 9 mis-scaled
  rows were the only near-stationary "human" examples in training, so removing
  that artefact tightened the human region and a few borderline synthetic bots
  now fall inside it.
- Human FPR moves by 0.06 pp — negligible.
- Every downstream tier (2, 4, 5) that mixes DELBOT rows into training is
  dominated by ~25k Balabit-derived instances vs 3,453 DELBOT, so those numbers
  move by less than this.

### Dropped: `DelbotFastFileSensitivity.java`
A standalone 3-way sensitivity harness (AS_SHIPPED / RESCALED / EXCLUDED) was
written for this but its AS_SHIPPED arm did **not** reproduce the known
pre-rescale baseline (reported pooled AUC 0.2188 vs the reference tool's 0.5326
on an identical training set), so its deltas were not trustworthy. Its RESCALED
arm happened to match (0.4417). Rather than ship a harness that fails its own
control, the before/after above was measured directly by `git stash` toggling on
`BalabitCrossDomainEval` / `BalabitValidationPipeline` / `FalsePositiveEvaluator`,
which are the frozen, reproducible references. Tool and its result file removed.

### Still open (unchanged by this)
- Raw max velocity ~1277 px/ms (1 ms dt floor too coarse for large jumps) — a
  deliberate, announced Tier-1 re-baseline, not done here.
- No GAN / captured-bot trajectories.
- Adversarial bot matches only ~10/18 features.
