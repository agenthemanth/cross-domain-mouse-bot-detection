# CAPTCHA_PROJECT — orientation

**Read this first.** It is the index: what the project is, how to build and run it,
what every file is for, and what is still open. It deliberately carries almost no
result numbers.

> ## Where the numbers live
> **`CROSS_DOMAIN_STUDY.html` is the single canonical writeup.** Every finding,
> every table, every quotable figure lives there — extend it, do not fork it.
> This file carries only the four headline numbers below, as a sanity anchor.
> If a number here and a number there ever disagree, the study HTML wins.
>
> This project has been re-baselined twice and several published figures were
> **retracted**. Before quoting any number from any other file, check it against
> the study HTML. See *Document status* at the bottom for which files are stale.

---

## 1. What this is

A mouse-dynamics bot detector and, more importantly, a **negative-result study**
about it. Java 17 / Weka 3.8.6.

A Weka `RandomForest` (100 trees, seed 1) is trained on the **DELBOT** dataset
(`delbot_data/`, 3,453 instances — bot folders `gan` / `naturalmousemotion` /
`pyhm` / `pynput`, plus human folders) and then evaluated **out of domain** on the
**Balabit Mouse Dynamics Challenge** (`balabit_data/`, 1,676 files, 10 users,
continuous admin sessions, all human) and on **mightymerge.io** browsing humans
(via the precomputed `human_mouse_reference_features.csv`).

Balabit has no bots, so bots are **synthesised inside the Balabit domain** as
matched twins of real human chunks (same start point, end point, point count and
duration, so those columns cannot leak the label). That is what makes a two-sided
cross-domain evaluation possible at all.

The work is organised as seven **tiers**, each an attack or a defence in an arms
race. The story: hand-crafted features + trees do not transfer, synthetic-bot
training augmentation fixes the easy case, and every new evasion needs its own
training data.

**Headline anchors (post-rescale, 2026-08-30):**

| quantity | value |
|---|---|
| Balabit human FPR @0.5 (zero-shot) | 9.79% |
| mightymerge FPR (micro / macro) | 5.28% / 3.92% |
| Zero-shot cross-domain bot ROC-AUC | 0.4417 — *no transfer* |
| Best defended model (Tier 5, honest CALIB/REPORT split) | ~74% evasive recall @ ~3.4% report FPR |

**The second channel.** Tier 5 showed the reject option is nearly free but is
only useful if deferred traffic has somewhere to go. `CaptchaChallenge` is that
destination: a sequential-reveal CAPTCHA where the route is released one hop at a
time on the server's clock, so it cannot be precomputed or replayed. It has its
own scorer — the forest is deliberately not applied to it. See open item 10 for
what it does and does not buy.

Segmentation: Balabit raw sessions are split on 3 s inactivity gaps into
"gap-chunks" (median 13.6 s / 66 points), min 20 distinct-timestamp points →
24,182 human chunks. Per-*action* segmentation was tried (Tier 3) and is worse.

---

## 2. Build & run

No Maven runner available — build with `javac` plus the jars already in
`%USERPROFILE%\.m2\repository`. Let `M = %USERPROFILE%\.m2\repository`.

```
javac -cp "M\nz\ac\waikato\cms\weka\weka-stable\3.8.6\weka-stable-3.8.6.jar;M\nz\ac\waikato\cms\weka\thirdparty\bounce\0.18\bounce-0.18.jar;M\tech\tablesaw\tablesaw-core\0.43.1\tablesaw-core-0.43.1.jar" -d target\classes src\main\java\*.java
```

```
java -Xmx2g -cp "target\classes;M\nz\ac\waikato\cms\weka\weka-stable\3.8.6\weka-stable-3.8.6.jar;M\nz\ac\waikato\cms\weka\thirdparty\bounce\0.18\bounce-0.18.jar" <MainClass> [args]
```

**Gotchas — all of these have bitten before:**

- **Compile ALL sources together** (`src\main\java\*.java`). Single-file `javac`
  fails to resolve the cross-class references and *silently leaves stale
  `.class` files* — you then measure yesterday's code.
- `-Xmx2g` is needed for the augmented evals (`Tier2AugmentedEval` ≈ 85 s).
- `tablesaw` is compile-time only; it is not on the run classpath.
- Determinism is deliberate and load-bearing: `rf.setSeed(1)`, `Arrays.sort()` on
  every `File.listFiles()`, per-chunk-seeded synthesis. Keep it that way — the
  before/after tables depend on it.
- Re-baselines are measured by `git stash`-toggling the change and re-running the
  **frozen reference tools** in the same JVM, not by a bespoke sensitivity
  harness. (One such harness failed its own control and was deleted; see
  `REBASELINE_NOTES.md`.)

---

## 3. File map — `src/main/java/`

### Core pipeline (frozen references — Tier-1 numbers must keep reproducing)

| file | what it does |
|---|---|
| `BalabitValidationPipeline.java` | The spine. DELBOT parsing (`parseDelbotPoints`), Balabit session parsing + 3 s gap-chunking, timestamp collapse, the 7-feature `computeFeatures`, schema building. Everything else reuses these methods rather than copying them. → `balabit_validation_results.txt` |
| `BalabitCrossDomainEval.java` | Two-sided zero-shot cross-domain eval: DELBOT-trained RF vs 24,182 real Balabit human chunks + 4 synthetic bot twins each. Confusion matrix, ROC-AUC, EER, TPR@≤1%/≤5% FPR, per-user FPR. → `balabit_crossdomain_eval_results.txt` |
| `FalsePositiveEvaluator.java` | Same-domain FPR on mightymerge browsing humans. → `samedomain_falsepositive_results.txt` **Confounded** — see Open items. |
| `DelbotValidationPipeline.java` | In-domain DELBOT validation. |
| `HumanReferenceFeatureExtraction.java` | Builds `human_mouse_reference_features.csv` from mightymerge. Checked, no clamp bug. |
| `Phase2FeatureExtraction.java` | Click-fraud CSV feature engineering (`click_fraud_*`). Side branch. |
| `Main.java` | Entry point / scratch driver. |

**The 7 baseline features:** `num_points`, `duration_ms`, `mean_velocity`,
`std_velocity`, `mean_acceleration`, `mean_jerk`, `path_efficiency`. Class `"1"` = bot.

### Bot synthesis

| file | what it does |
|---|---|
| `BalabitBotSynthesizer.java` | **Naive** bots. `BotType` = `MODERATE_LINEAR`, `MODERATE_LINEAR_VP`, `ADVANCED_BEZIER`, `ADVANCED_BEZIER_VP_JITTER`. `TimingModel` = `MATCHED_SPREAD` \| `BURST` (BURST packs ~85% of points into 4–12% of the duration). Matched-pair design; deterministic. |
| `AdversarialBotSynthesizer.java` | **Evasive** bots. Reuses the victim chunk's own (step-distance, dt) pairs so speed statistics match *by construction*; path = line + sine deviations bisected to match path length. `Ordering` = `SHUFFLE` (Tier 4 — random permutation) \| `BALLISTIC` (Tier 7 — sorted onto a rise-then-fall speed envelope). |
| `BalabitActionSegmenter.java` | Splits Balabit into MM / PC / DD mouse *actions* (139,636 total, median 1.6 s / 11 pts) instead of gap-chunks. Used only by Tier 3. |

### Feature sets

`Tier2Features.java` — one shared extractor, `Mode` enum:

| mode | #feat | contents |
|---|---|---|
| `BASELINE` | 7 | the Tier-1 seven (delegates to `BalabitValidationPipeline.computeFeatures`, reproduces it exactly) |
| `AUGMENTED` | 18 | 7 + 11 domain-bridging: `velocity_cv`, `accel_to_vel_ratio`, `jerk_to_vel_ratio`, `time_to_peak_vel_ratio`, `accel_fraction`, `dir_reversal_rate`, `mean_turning_angle`, `std_turning_angle`, `mean_curvature`, `pause_ratio`, `vel_p90_p50_ratio` |
| `SCALEFREE` | 14 | drops absolute velocity/accel/jerk, dimensionless only |
| `AUGMENTED_SEQ` | 20 | AUGMENTED + `velocity_lag1_autocorr`, `velocity_step_roughness` (Tier 6) |
| `SCALEFREE_SEQ` | 16 | SCALEFREE + the same two |

### Tier experiments

| tool | args | question | result file |
|---|---|---|---|
| `Tier2CrossDomainEval` | `MATCHED_SPREAD` \| `BURST` | Do richer features fix zero-shot transfer? (**No — they make it worse**) | `tier2_crossdomain_matched_results.txt`, `tier2_crossdomain_burst_results.txt` |
| `Tier2AugmentedEval` | — | Does adding synthetic bots to training fix it? (**Yes, for naive bots**) | `tier2_augmented_results.txt` |
| `Tier3ActionEval` | — | Does per-action segmentation help? (**No**) | `tier3_action_results.txt` |
| `Tier4AdversarialEval` | — | Does a feature-matched evasive bot defeat it? (**Yes; adversarial training + 18 feats recovers**) | `tier4_adversarial_results.txt` |
| `Tier5OperatingPointEval` | a `Tier2Features.Mode` | Is the high human FPR a model defect or a threshold artefact? (**Threshold artefact**) | `tier5_operating_point_results.txt` (AUG 18), `tier5_operating_point_seq_results.txt` (SEQ 20) |
| `Tier6FeatureProbe` | — | Do temporal-ordering features catch the shuffle? (**Marginally; the assumed mechanism isn't there**) | `tier6_feature_probe_results.txt` |
| `Tier7ArmsRaceEval` | — | Does a second evasion strategy (BALLISTIC) defeat the Tier-5 model? (**Yes; defences are attack-specific**) | `tier7_arms_race_results.txt` |

**User split** (Tiers 2–7, sorted users, alternating):
train `user12,16,21,29,7` / test `user15,20,23,35,9`.
Tier 5 subdivides the test half into CALIB `user20,35` (threshold only) and
REPORT `user15,23,9` (all quoted rates) — this is what removed the same-data
optimism, and it is why the honest headline is ~3.4% FPR, not ~1%.

### Demo front end

| file | what it does |
|---|---|
| `DemoModelTrainer.java` | One-time: freezes the Tier-4 `AUGMENTED` / `AUG_NAIVE_PLUS` model into `demo/model/` — `rf.model`, `schema.arff`, `operating_point.txt` (threshold 0.87), `golden_vectors.json`. ⚠️ The 0.87 is picked on the **same** 5 held-out users it then reports 0.9% FPR against — same-data optimistic; see open item 4, real unseen-user FPR ~1.5–2%. |
| `DemoServer.java` | `com.sun.net.httpserver` on 127.0.0.1:8787. Serves `demo/` statically and `POST /score` with `{"features":[<18 numbers>]}` → `{"score","label","threshold"}`. |
| `demo/index.html` | The **only** page — the live detector. Capture arena, P(bot) gauge, movable decision line, 4 bot-sim buttons. |
| `demo/features.js` | Hand port of `Tier2Features.compute(AUGMENTED)` to JS. |
| `demo/parity.js` | `node demo/parity.js` — recomputes the 25 golden vectors and asserts a Java match. Must print `PARITY PASS` (worst rel err ~7e-12). **Run before demoing**; without it the live score is untrustworthy even with a correct model. |
| `demo/demo.js` | Capture → chunking → feature → `/score` wiring, gauge, bot sims. |

### CAPTCHA channel (the reject option's destination)

| file | what it does |
|---|---|
| `CaptchaChallenge.java` | Sequential-reveal CAPTCHA. 12 server-rendered distorted glyph PNGs (`java.awt`, no new deps), 5 hops, **one target revealed at a time** with each release stamped on the server clock — so the route cannot be precomputed or replayed. Owns its own rule-based scorer and `Rules` thresholds. |
| `demo/challenge.html`, `demo/challenge.js` | The challenge page and its four in-page attack sims. |
| `demo/attack_probe.py` | `python demo/attack_probe.py <port>` — drives `/challenge/*` end to end and prints the verdict each attacker profile gets. Stdlib only. → `captcha_channel_results.txt` |

Routes on `DemoServer`: `POST /challenge/new`, `GET /challenge/tile`,
`GET /challenge/prompt`, `POST /challenge/arrive`, `POST /challenge/verify`, and
`GET /challenge/oracle` (**demo affordance — hands out the answer so a sim can
play without an OCR model; must not ship**).

**Two channels, never one score.** The forest is deliberately *not* applied to
CAPTCHA traces — Tier 3 measured that short goal-directed segments push it from
9.79% to 32.8% human FPR and AUC 0.53 → 0.36. See open item 10.

```
java -cp "target\classes;<weka>;<bounce>" DemoServer      # then open http://127.0.0.1:8787/
```

See `demo/README.md` for the full demo notes, including the threshold caveat.

---

## 4. Data

| path | what |
|---|---|
| `delbot_data/` | DELBOT training set. 3,453 instances. Bot folders `circles_bot_gan`, `naturalmousemotion`, `pyhm`, `pynput`; human folders `circles_human_*`. Files carry a `resolution:W,H` header. |
| `balabit_data/` | Balabit Mouse Dynamics Challenge. 1,676 files, 10 users, all human. Timestamps in seconds; ~16.5% of consecutive raw samples share a timestamp. |
| `human_mouse_reference_features.csv` | Precomputed mightymerge.io browsing-human features (5,748 people). No raw x/y. |
| `click_fraud_dataset (1).csv`, `click_fraud_phase2_engineered.csv` | Click-fraud side branch. |
| `delbot_features_engineered.csv` | Engineered DELBOT features. |

---

## 5. Open items

**Methodological / must state in the paper**

1. **Not zero-shot from Tier 2 on.** Every augmented model trains on 5 Balabit
   users — this is semi-supervised domain adaptation. Part of the human-FPR drop
   is just having in-domain human examples.
2. **All test bots come from one generator family** (`BalabitBotSynthesizer` /
   `AdversarialBotSynthesizer`: Bezier, min-jerk, shuffle, ballistic). No captured
   Selenium / pyautogui traffic, no GAN trajectories.
3. **The arms race is not won.** The evasive bots match a *subset* of features by
   construction. A Round-8 bot that also matched accel/jerk ordering and curvature —
   or hit the human roughness band (~1.54) exactly instead of overshooting — would
   push detection back toward chance.
4. **The operating point does not transfer across users** (~1.6–2× FPR slippage
   between disjoint user sets). Calibrate at roughly half the FPR you actually need.
   This is why Tier 5's honest headline is ~3.4% FPR, not ~1% — and why the demo's
   threshold 0.87 / "0.9% human FPR" pair is optimistic (chosen on the users it
   reports against). Not fixed for the demo; stated as a caveat in `demo/README.md`.
5. **Latency is asserted, not measured** — the real-time claim has never been timed
   for the 100-tree forest plus 20-feature extraction.

**Known defects, not fixed**

6. **mightymerge is confounded.** The reference CSV's "DELBOT-compatible" columns
   are *not* computed the same way as DELBOT training and cannot be (no raw x/y):
   `mean_velocity_delbot` is the mean of a precomputed velocity field (~1000× scale
   gap, has negatives) and accel/jerk there are true time-derivatives while DELBOT
   training uses plain first differences. Only `num_points`, `duration_ms` and
   `path_efficiency` are comparable. **Recommendation:** treat the Balabit pipeline
   as the sound cross-dataset evaluation; either restrict the mightymerge check to
   those 3 columns or drop mightymerge from the paper's cross-dataset story.
7. **Raw max velocity ~1277 px/ms is unphysical** — the 1 ms `dt` floor is too
   coarse for large position jumps. Fixing it would move the Tier-1 baseline, so do
   it as a deliberate, announced re-baseline, never silently.
8. **Citations:** Iliou et al. 2021 (both entries) and Wei et al. 2019 are
   **UNVERIFIED** — treat their exact numbers as unconfirmed until checked against
   the papers. A prior session mis-cited arXiv 2504.21415 as a "cross-dataset RF
   AUC 0.72" result; it is **not** (within-dataset user authentication, no
   cross-dataset eval anywhere in it). That error propagated into tier docs before
   it was caught. Re-verify every citation before it enters the manuscript.
9. `demo/model/rf.model` is an 18 MB binary now in git history. Regenerable from
   `DemoModelTrainer`; could be gitignored.
10. **The CAPTCHA channel's thresholds are uncalibrated.** Every constant in
    `CaptchaChallenge.Rules` is hand-set from first principles; there is no human
    sample for this challenge and n=1 is not calibration. No rate from
    `captcha_channel_results.txt` is a deployment number. Also open: a jittered
    human-paced bot passes every rule today (by design — the channel buys a
    *throughput tax*, forcing the attacker online and serialised, not a
    classification guarantee); the regularity rule is secondary and its threshold
    sits only ~1.3× clear of that bot; the glyph distortion is a speed bump, not
    the security property; and `/challenge/oracle` must not ship.

**Next real lever**

Not more scalar features — hand-crafted feature engineering on gap-chunk
aggregates is exhausted (Tier 6). A different *representation*: trajectory-image
CNN (Wei et al.), or a velocity-sequence deep model.

---

## 6. Document status

| file | status |
|---|---|
| `CROSS_DOMAIN_STUDY.html` | ✅ **CANONICAL.** Full 7-tier report + baseline + demo sections. Post-rescale. Extend this one. |
| `PROJECT.md` | ✅ this file — index and orientation. |
| `REBASELINE_NOTES.md` | ✅ current. The `circles_human_fast` normalised-coordinate rescale, with the before/after table and how it was measured. |
| `demo/README.md` | ✅ current. Demo build / parity / run / caveats, plus the CAPTCHA channel's design, trust boundary and measured attack table. |
| `captcha_channel_results.txt` | ✅ current. Attack-probe log for the CAPTCHA channel. Regenerate with `python demo/attack_probe.py <port>`. |
| `TIER1-4_EVALUATION.html` | ⚠️ **STALE.** Pre-rescale numbers throughout (0.5326, 9.85%, 12.39%) and still discusses the retracted 1.07% FPR. It is paper-facing, so anyone quoting it quotes retracted figures. Update deliberately or delete. |
| `AGENT_HANDOFF.txt` | ⚠️ **SUPERSEDED** by this file. Tier-1-era; pre-rescale numbers and the retracted AUC-0.72 mis-citation. Kept only for the session narrative. |
| `balabit_data/README.md` | dataset documentation (upstream). |

**Retracted figures — never quote these:** Balabit FPR 1.07% and 16.69%;
mightymerge FPR 0.71% and 2.18%; cross-domain AUC 0.5326; "12.4% human FPR" as a
model defect; "cross-dataset RF AUC 0.72" attributed to arXiv 2504.21415.
