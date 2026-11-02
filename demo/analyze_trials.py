"""analyze_trials.py -- turn the participant-study log into the numbers a paper
needs: how often real people fail the puzzle, how often the passive channel
wrongly vetoes them, and what the puzzle thresholds SHOULD be.

    python demo/analyze_trials.py [demo/study_data/trials.jsonl] [--budget 0.01]

The log is written by `DemoServer --log-trials` for every sign-in on
/login.html?study=1&p=<code>. Stdlib only.

What it reports
  1. Per-participant and pooled pass rates, with Wilson 95% intervals. Pooled
     trials are NOT independent (one person contributes many), so the
     participant-level mean is the number to quote, and n = participants.
  2. The distributions the rules cut on: net re-aim onset (ms from ack),
     ack -> placed, post-ack sample count, jump->ack round trip.
  3. Why failures failed (first flag), and the passive channel's human
     false-veto rate against the frozen 0.87 line.
  4. Threshold calibration done the way Tier 5 had to be: choose each floor on
     a CALIB half of the PARTICIPANTS, report its failure rate on the disjoint
     REPORT half. Tier 5 measured 1.6-2x FPR slippage when an operating point
     crosses users; this checks whether the puzzle has the same problem.
  5. If bot-sim trials are in the log (sim > 0), their onsets next to the human
     ones, so the margin at the chosen floor is visible.

It never changes a threshold itself. PerturbationPuzzle.Rules is edited by hand,
openly, citing this output.
"""
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict

MIN_PARTICIPANTS_TO_CALIBRATE = 20
CURRENT = {"MIN_ONSET_MS": 100, "MIN_JUMP_TO_DROP_MS": 300, "MIN_POST_JUMP_SAMPLES": 5}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(name, xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        print(f"  {name:28s} (none)")
        return
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95]
    print(f"  {name:28s} n={len(xs):4d}  " + "  ".join(f"p{int(q * 100):02d}={pct(xs, q):7.1f}" for q in qs)
          + f"  min={min(xs):.1f}")


def half(code):
    return int(hashlib.sha256(code.encode()).hexdigest(), 16) % 2


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "demo/study_data/trials.jsonl"
    budget = 0.01
    if "--budget" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--budget") + 1])

    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    humans = [r for r in rows if r.get("sim", 0) == 0 and r.get("participant")]
    sims = [r for r in rows if r.get("sim", 0) > 0]
    anon = [r for r in rows if r.get("sim", 0) == 0 and not r.get("participant")]
    print(f"log: {path}")
    print(f"  {len(rows)} trials: {len(humans)} participant trials, {len(sims)} bot-sim trials, "
          f"{len(anon)} with no participant code (ignored)")
    by = defaultdict(list)
    for r in humans:
        by[r["participant"]].append(r)
    P = len(by)
    print(f"  participants: {P}")
    if P == 0:
        return
    if P < MIN_PARTICIPANTS_TO_CALIBRATE:
        print(f"  !! fewer than {MIN_PARTICIPANTS_TO_CALIBRATE} participants: every number below is INDICATIVE,"
              " not a calibration")

    # ---- 1. pass rates ----
    print("\n1. PASS RATES (current hand-set rules)")
    print(f"  {'participant':14s} {'trials':>6s} {'puzzle pass':>11s} {'decision HUMAN':>15s}")
    part_puzzle, part_decision = [], []
    for code in sorted(by):
        rs = by[code]
        pp = sum(r["result"]["puzzle"]["pass"] for r in rs) / len(rs)
        dp = sum(r["result"]["decision"] == "HUMAN" for r in rs) / len(rs)
        part_puzzle.append(pp)
        part_decision.append(dp)
        print(f"  {code:14s} {len(rs):6d} {pp:11.1%} {dp:15.1%}")
    k = sum(r["result"]["puzzle"]["pass"] for r in humans)
    p, lo, hi = wilson(k, len(humans))
    print(f"\n  participant-level mean puzzle pass rate : {sum(part_puzzle) / P:.1%}   <- quote this (n = {P} people)")
    print(f"  participant-level mean HUMAN decision   : {sum(part_decision) / P:.1%}")
    print(f"  pooled trials puzzle pass              : {p:.1%} [{lo:.1%}, {hi:.1%}] "
          f"(Wilson 95%, {len(humans)} trials; trials are NOT independent)")

    # ---- 2. distributions ----
    pz = [r["result"]["puzzle"] for r in humans]
    print("\n2. DISTRIBUTIONS (ms unless noted; human trials)")
    dist("net onset (from ack)", [x["onsetMs"] for x in pz])
    dist("raw onset (from jump send)", [x["rawOnsetMs"] for x in pz])
    dist("ack -> placed", [x["jumpToDropMs"] for x in pz])
    dist("jump -> ack (round trip)", [x["ackMs"] for x in pz])
    dist("post-ack samples (count)", [x["postJumpSamples"] for x in pz if x["placed"]])
    untimed = sum(1 for x in pz if x["placed"] and x["jumped"] and x["onsetMs"] is None and x["pass"])
    print(f"  re-aim not timeable (curved drag, latency rule skipped): {untimed}/{len(pz)}")

    # ---- 3. failure reasons, passive channel ----
    print("\n3. WHY HUMAN TRIALS FAILED")
    reasons = Counter((x["flags"] or ["?"])[0].split(" -- ")[0][:70] for x in pz if not x["pass"])
    if not reasons:
        print("  none")
    for why, n in reasons.most_common():
        print(f"  {n:4d}  {why}")
    pv = [r["result"]["passive"] for r in humans]
    scored = [x for x in pv if x["verdict"] != "NO_EVIDENCE"]
    vetoed = sum(x["verdict"] == "BOT" for x in scored)
    p, lo, hi = wilson(vetoed, len(scored))
    print(f"\n  passive channel: {len(scored)}/{len(pv)} trials had enough background movement to score")
    if scored:
        print(f"  human false-veto rate at the frozen line : {p:.1%} [{lo:.1%}, {hi:.1%}] ({vetoed}/{len(scored)} trials)")
        dist("passive median P(bot)", [x["median"] for x in scored])
        print("  (the model is frozen; if this rate is high, that is the Tier-1 domain shift, and the")
        print("   honest fix is to drop or loosen the veto openly, not to retrain on the study data)")

    # ---- 4. calibration on CALIB participants, reported on REPORT participants ----
    print(f"\n4. CALIBRATION  (floor = CALIB-half quantile at a {budget:.0%} per-rule failure budget; "
          "reported on the disjoint REPORT half)")
    calib = [r for r in humans if half(r["participant"]) == 0]
    report = [r for r in humans if half(r["participant"]) == 1]
    nc = len({r["participant"] for r in calib})
    nr = len({r["participant"] for r in report})
    print(f"  CALIB: {nc} participants / {len(calib)} trials    REPORT: {nr} participants / {len(report)} trials")
    for key, rule in (("onsetMs", "MIN_ONSET_MS"), ("jumpToDropMs", "MIN_JUMP_TO_DROP_MS")):
        cv = [r["result"]["puzzle"][key] for r in calib if r["result"]["puzzle"][key] is not None]
        rv = [r["result"]["puzzle"][key] for r in report if r["result"]["puzzle"][key] is not None]
        if len(cv) < 10 or not rv:
            print(f"  {rule:22s} not enough data (calib {len(cv)}, report {len(rv)})")
            continue
        # largest floor that fails at most budget * n calib trials
        allowed = int(budget * len(cv))
        floor = sorted(cv)[allowed]
        if allowed == 0:
            print(f"  ({rule}: {len(cv)} calib trials is too few for a {budget:.0%} budget -- floor = calib minimum)")
        fail_c = sum(v < floor for v in cv) / len(cv)
        fail_r = sum(v < floor for v in rv) / len(rv)
        cur_r = sum(v < CURRENT[rule] for v in rv) / len(rv)
        slip = fail_r / fail_c if fail_c > 0 else float("inf")
        print(f"  {rule:22s} suggested floor {floor:7.1f}  fail calib {fail_c:5.1%}  fail report {fail_r:5.1%}"
              f"  slippage x{slip:.2f}   | current {CURRENT[rule]} fails {cur_r:5.1%} of report")
    print("  Use a floor below the suggestion, not above it: the latency floor only has to sit clear of")
    print("  bots that react in < 1 frame; see section 5 for the margin.")

    # ---- 5. bots in the same log ----
    if sims:
        print("\n5. BOT-SIM TRIALS IN THE SAME LOG")
        bylvl = defaultdict(list)
        for r in sims:
            bylvl[r["sim"]].append(r["result"])
        for lvl in sorted(bylvl):
            rs = bylvl[lvl]
            passed = sum(x["decision"] == "HUMAN" for x in rs)
            on = [x["puzzle"]["onsetMs"] for x in rs if x["puzzle"]["onsetMs"] is not None]
            print(f"  L{lvl}: {passed}/{len(rs)} got HUMAN; net onsets "
                  + (", ".join(f"{v:.0f}" for v in sorted(on)) if on else "none timed"))
        hon = [x["onsetMs"] for x in pz if x["onsetMs"] is not None]
        if hon:
            print(f"  human net onset p01 = {pct(hon, 0.01):.0f} ms: the latency floor must sit between the"
                  " fastest bot that should fail (L3) and this")


if __name__ == "__main__":
    main()
