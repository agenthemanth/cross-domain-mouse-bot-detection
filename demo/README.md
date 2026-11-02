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

## The test login page (`/login.html`)

Linked from the header of `/`. It's a throwaway "Test Portal" sign-in (test account
`demo` / `letmein`) where **two checks run at the same time** and the server judges
both at `POST /auth/login`:

1. **Background mouse movement.** Every mousemove on the page is captured, except
   while you're dragging the puzzle piece; each drag boundary starts a new segment.
   The server splits the trace with the 3 s gap rule and scores each chunk with ≥20
   distinct points using the frozen forest (`Tier2Features.compute(AUGMENTED)`,
   server-side, so the client can't lie about its features). The channel verdict is
   the median chunk score against the 0.87 line.
2. **Moving puzzle** (`PerturbationPuzzle.java`). Drag the piece onto the outline
   of the same shape. The board is a **server-rendered PNG** with three decoy
   outlines, so the slot's position is never sent as a number. While you drag, the
   piece position streams to `/puzzle/move` (sequence-numbered, up to 6 in flight)
   and is stamped on the **server clock**. After a secret fraction of the course
   (20–35%), a reply delivers a new board: the slot has moved 60–85 px perpendicular
   to the course and the decoys have shifted. The reply also carries an **ack token**
   that the client echoes on every later report. The server measures the re-aim onset
   **from the ack's arrival**. The ack and the re-aim cross the same uplink, so
   network round trip is not mistaken for reaction time. A person corrects after a
   visuomotor delay. An open-loop bot never corrects. A naive closed-loop bot
   corrects in the ack message itself.

The decision table has no averaging:

| puzzle | background | decision |
|---|---|---|
| fail | any | BOT |
| pass | ≥ 0.87 | BOT (veto) |
| pass | < 0.87 | HUMAN |
| pass | < 20 points | HUMAN (no evidence) |

**Bot ladder.** The in-page buttons find the slot **by vision**: a white-box locator
thresholds the board PNG, takes connected components and matches them against the
piece's outline. It costs ~11–28 ms per board read in headless Chrome, and the page
shows the measured cost. `python demo/login_probe.py <port>` needs
`DemoServer <port> --oracle` and reads the slot from the oracle instead. That gives
every bot free, perfect perception and isolates the timing channel. The probe
writes `login_channel_results.txt`.

| level | what it does | outcome (every simulated link) |
|---|---|---|
| L1 | pyautogui-style lines; drags straight to where the slot *was* | BOT: misses the moved slot |
| L2 | Tier-4 shuffled-speed evasive mouse; pre-planned drag | BOT: misses the moved slot |
| L3 | re-reads the slot every frame, re-aims at once | BOT: onset 0–52 ms after the ack (1 of 12 untimed, caught by the 300 ms ack→place floor) |
| L4 | re-reads the slot, waits a sampled 180–260 ms, then re-aims | **HUMAN: passes** |

**Network.** The probe runs every rung and the human models over four *simulated*
links. Each message gets its own Gaussian one-way delay: 0, 20±5, 60±15 and
120±35 ms. The **old** rule timed the re-aim from when the jump was *sent*. Recomputed
on these trials' raw onsets (the old code itself was not rerun), it lets L3 through
3/3 at ~120 ms and 3/3 at ~240 ms round trip, because the round trip counts as
"reaction". That is a lower bound. With the ack-referenced rule L3 is blocked
12/12, and the human models pass 48/48. Fixing this turned up a second loophole: the ack-carrying
sample had been treated as pre-reaction baseline, so a bot that re-aimed in that
message hid the correction. It now counts as post-reaction.

**In headless Chrome** (1400×1000, 2026-09-24): L1 and L2 missed, L3 was blocked at
3–10 ms, and L4 passed at ~210 ms. Three scripted trusted-event drags
(`Input.dispatchMouseEvent`, aimed with the oracle, smooth delayed re-aim; **not a
person**) passed at 416–439 ms. The passive channel scored everything 0.42–0.67 and
**vetoed nothing**: the Tier 4 result reproduced live.

**Human-shaped drags.** The probe runs motor *models*, not people: drags bowed ±30 px
and ±45 px with ~1.5 px tremor and a 180–260 ms delayed smooth re-aim, plus
miss-pause-re-grab runs.

Two rules changed earlier because of these models:
- **Re-aim threshold raised from 6 px to 15 px.** At 6 px, with a walk-back to
  baseline, curvature alone timed 4 of 10 ±30 px drags as "too fast". The 15 px rule
  reports the crossing time itself, which overstates latency; that errs toward
  passing a person.
- **Maximum-onset rule removed.** It rejected the miss-and-re-grab path.

If a strongly curved drag hides the onset, the latency rule is skipped and noted. The
piece still has to land on the moved slot, ≥ 300 ms after the ack.

### Participant study (what calibrates all of this)

```
run_demo.bat --log-trials                  # or: java ... DemoServer --log-trials
# each person opens  http://127.0.0.1:8787/login.html?study=1&p=P01   (anonymous code)
python demo/analyze_trials.py              # -> pass rates, distributions, calibrated floors
```

Study mode hides the verdict panel and loads a fresh puzzle after every sign-in. The
server appends one line per trial to `demo/study_data/trials.jsonl`, which is
git-ignored. Each line holds the participant code, both verdicts and the puzzle's
raw server-clocked trace, and never the username, password or background trace.
`analyze_trials.py` reports rates per participant (n = people), calibrates each
floor on half the participants and reports it on the other half, following the
Tier 5 lesson. Aim for 20+ people, with both mouse and trackpad users and some
on a remote connection. Get whatever consent your institution requires. Edit
`PerturbationPuzzle.Rules` by hand from its output. **Do not retrain the forest on
study data.**

**Caveats.**
- **Every puzzle threshold is hand-set** until the study has run.
- **Real-hand behaviour is unverified.** In particular, no one has checked whether
  ordinary human background movement on this page trips the 0.87 veto; the study
  log measures exactly that. If it does, that is the domain-shift failure documented
  under *Known behaviour*. Soften the rule openly; don't hide it.
- **L4 passing is the honest limit.** What the puzzle buys is cost. The attacker
  must run a live closed-loop controller with vision and a human latency model; it
  can no longer replay a pre-planned path. Hiding the slot in an image adds a
  perception step, not a wall.
- **Positions are still client-claimed.** Only the timing is server-owned. Network
  effects were measured on a simulated link. Real TCP/HTTP2 and browser scheduling
  are untested, and jitter still blurs the 100 ms floor.
- **`/puzzle/oracle` must not ship.** It returns 404 unless the server was started
  with `--oracle`.
- **No background movement means "no evidence" → HUMAN** on the passive side. A
  silent bot has to beat only the puzzle. In the one study-mode test drag, a scripted
  regular sine sweep in the background *was* vetoed (0.95).

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
that mistake: on the 5 held-out users, short goal-directed segments took zero-shot
human FPR from 10.85% to 28.85% (7 features) and from 32.58% to 71.09% (14), with
bot AUC at chance either way, because they sit even further outside DELBOT's
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
