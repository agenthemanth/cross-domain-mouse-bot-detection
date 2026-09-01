# Mouse-dynamics bot detector — live demo

Two pages, served by `DemoServer`:

- **`/`** (`demo/index.html`) — the passive detector. Move the mouse for a live
  P(bot) score; bot-sim buttons; movable decision line.
- **`/challenge.html`** — the **sequential-reveal CAPTCHA**, the active second
  channel. See [The CAPTCHA channel](#the-captcha-channel) below.

(An earlier "Meander" marketing landing page at `/` with the detector at
`/lab.html` was removed on 2026-08-31; it is in git history.)

Front end for the frozen `AUGMENTED-18 / AUG_NAIVE_PLUS` Random Forest. Covers
gaps 1–3 from the plan (model selection + serialization, capture→feature bridge,
live scoring). Gaps 4–5 (per-context recalibration, real captured-bot validation)
are out of scope and stated as limitations on the page.

## One-time: freeze the model

```
javac -cp "<weka>;<bounce>;<tablesaw>" -d target/classes src/main/java/*.java
java -Xmx2g -cp "target/classes;<weka>;<bounce>" DemoModelTrainer
```

Writes `demo/model/`:
- `rf.model` — serialized `weka.classifiers.trees.RandomForest` (100 trees, seed 1)
- `schema.arff` — the 18-attribute + class header the scorer rebuilds instances against
- `operating_point.txt` — the P(bot) threshold and the held-out metrics this build achieves
- `golden_vectors.json` — 25 real held-out Balabit chunks + their Java feature vectors, for the parity test

Locked config: `Tier2Features.Mode.AUGMENTED` (18), trained on DELBOT + one naive
`BalabitBotSynthesizer` twin + one `SHUFFLE` `AdversarialBotSynthesizer` twin per
training-user chunk. 5 Balabit users held out (same split as Tiers 4–5). The model
is byte-identical to Tier 4's `AUGMENTED AUG_NAIVE_PLUS` row (naive AUC 0.9995,
evasive AUC 0.9649).

**Threshold:** `DemoModelTrainer` computes it fresh at a 1% human-FPR budget over
all 5 held-out users and gets **0.87**. Tier 5's reject-option section quotes 0.82
— that was on 2 calibration users only, a different subset. The demo ships 0.87.
At 0.87 on held-out users: 0.9% human FPR, 99.4% naive-bot recall, 58.3%
evasive-bot recall.

> **Caveat (open, not fixed):** that 0.9% is *same-data optimistic*. The threshold
> is chosen on the same 5 held-out users it is then scored against. Tier 6's
> CALIB/REPORT split showed the operating point does **not** transfer across users
> — a threshold set for X% FPR on calibration users lands at ~1.6–2x that on
> disjoint report users. So the demo's true FPR on unseen users is more like
> 1.5–2%, and on the demo page itself (a third domain) it is unmeasured. Quote
> 0.9% as "on the users it was calibrated on", not as a deployment number.

## Verify JS↔Java feature parity (run this before demoing)

```
node demo/parity.js
```

Recomputes all 25 golden vectors with `demo/features.js` and asserts they match
the Java `Tier2Features.compute(AUGMENTED)` output. Must print `PARITY PASS`
(worst relative error is ~7e-12). If it fails, the live score is untrustworthy
even with a correct model.

## Run the demo

```
java -cp "target/classes;<weka>;<bounce>" DemoServer          # port 8787
```

Open <http://127.0.0.1:8787/>. Move the mouse in the box for a live P(bot); the
sim buttons reproduce known bot movement models. Drag the decision line to see the
recall / false-positive trade-off (the Tier 5 finding).

`POST /score` body `{"features":[<18 numbers>]}` → `{"score":…,"label":…,"threshold":…}`.

## Known behaviour

The model was trained on Balabit admin-work sessions and their synthetic twins.
The demo page is a third domain it never saw, and the absolute-velocity features
depend on the viewer's screen / mouse / DPI. So:
- the **human** side can misfire — this is the zero-shot-transfer failure the
  study documents, not a demo bug;
- at the strict 0.87 line the model catches ~58% of evasive bots, so `linear`,
  `step` and `evasive` often score 0.5–0.85 and sit *under* the line. The
  `bezier` (min-jerk) bot reliably clears it. Move the threshold to demonstrate.

---

## The CAPTCHA channel

`/challenge.html` + `CaptchaChallenge.java`. This is the destination for the
**reject option** — Tier 5 found that deferring the uncertain score band is close
to free (±10% band, CALIB/REPORT split: ~94% coverage, ~0.5% kept human FPR, ~69%
evasive recall) but is "only useful if deferred traffic has somewhere to go".
Nothing in tiers 1–7 filled that slot. This does.

### The mechanic

Twelve distorted glyphs on a board, five hops. You are shown **one** target at a
time; the next is released only once you have clicked the current one — and the
release instant is stamped on the **server's** clock. The whole route is
therefore unknowable in advance, so it cannot be precomputed and cannot be
replayed. Glyphs are rendered server-side as PNGs (`java.awt`, no new deps), so
the letters are never in the page as text; the prompt is an image too.

### Two channels, never one score

**The forest is deliberately not applied to CAPTCHA traces.** Tier 3 measured
that mistake: short goal-directed segments took zero-shot human FPR from 9.79% to
32.8% and AUC from 0.53 to 0.36, because they sit even further outside DELBOT's
long circle-drawing distribution. CAPTCHA hops are exactly that shape. So the
challenge has its own scorer and its own thresholds, and the two channels are
never averaged.

### The trust boundary

| | measured by | forgeable? |
|---|---|---|
| **hard** — per-hop reveal→arrive wall time, and its regularity (MAD/median) | server clock only | no — a client can only actually be slow |
| **soft** — trace span, dwell, corrective submovements | client's submitted trace | yes — used only as corroboration |

The reconciliation between them is asymmetric on purpose: a trace may span
**less** time than the server's window (an honest trace always does — it misses
the reaction time at the front and the click latency at the back) but never
**more**. An overrun is not suspicious, it is proof the timestamps were written
rather than measured.

### What it actually buys — stated honestly

**Not** "bots cannot pass". A bot reads the board, OCRs the glyphs and sleeps a
plausible delay in a few lines. What the design buys is that the attacker is
forced **online and serialised**: it must solve each hop live, in order, inside a
window it cannot shrink. That is a throughput tax, not a classification
guarantee — the same "defences are attack-specific" conclusion the seven tiers
reached, one level up.

### Measured behaviour

```
python demo/attack_probe.py 8787       # DemoServer must be running
```

Result log: `captcha_channel_results.txt` (repo root). Every profile is handed
the answer by `/challenge/oracle`, i.e. a perfect OCR attacker is granted for
free so the timing channel is the only thing under test.

| profile | verdict | what tripped |
|---|---|---|
| human-paced | PASS | — (cover 87%, MAD 0.099) |
| bot instant | REJECT | 28 ms hop — under the 250 ms floor; MAD 0.000 |
| bot fixed-delay | REJECT | MAD 0.0009 — near-constant programmed delay |
| bot forged-trace | REJECT | trace claims 13.6 s inside a 5.2 s window |
| **bot jittered** | **PASS** | **nothing — this is the honest limit** |
| wrong tile | REJECT | wrong glyph at hop 3 |
| replay | REJECT | second verify of a completed challenge refused |

Stable over 3 runs.

**Regularity is MAD/median, not the coefficient of variation** — and that change
came out of the measurement, not the design. CV was the first cut and was
measured to be useless here: one ~350 ms scheduling hiccup in five hops took a
bot sleeping *exactly 700 ms* from CV 0.005 to 0.121, straight past the
threshold. Five samples cannot support a variance statistic. MAD ignores the
outlier (same run: 0.0009).

### Caveats

- **The thresholds are uncalibrated.** Every constant in
  `CaptchaChallenge.Rules` is hand-set from first principles. There is no human
  sample for this challenge and n=1 is not calibration. Calibrating them on real
  users is open work, and until then no rate quoted here is a deployment number.
- **The regularity rule is secondary and its threshold is thinly justified.**
  MAD separation is fixed-delay 0.0009 / jittered 0.025–0.029 / human 0.095–0.100.
  The 0.02 threshold is 20× clear of the fixed-delay bot but only ~1.3× clear of
  the jittered one — its placement is not meaningfully justified by three
  synthetic profiles. `MIN_HOP_WALL_MS` is the load-bearing hard signal.
- **The glyph distortion is a speed bump, not the security property.** A prompt
  is drawn in a different typeface from its own tile, so an attacker cannot skip
  reading by pixel-matching the prompt against the twelve tiles — but a shape
  matcher or an OCR model still wins, and is meant to. The sequential reveal and
  the server clock are what the design rests on.
- `/challenge/oracle` hands out the current answer. It exists for the attack
  probe and the in-page sims only and **must not ship**.
- `corrective submovements` is reported for inspection and never fails a
  challenge on its own — it is a client-side signal resting on an untested motor
  hypothesis, and Tier 6 is the cautionary tale for trusting one of those.
- Not verified in a real browser end-to-end (the extension was unavailable);
  the protocol is verified at the HTTP level by `attack_probe.py`, and the page's
  JS parses clean with every element id matched.
