# Live demo — mouse-dynamics bot detector

Browser front end for the frozen `AUGMENTED-18 / AUG_NAIVE_PLUS` Random Forest.
Covers gaps 1–3 from the plan (model selection + serialization, capture→feature
bridge, live scoring). Gaps 4–5 (per-context recalibration, real captured-bot
validation) are out of scope and stated as limitations on the page.

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
