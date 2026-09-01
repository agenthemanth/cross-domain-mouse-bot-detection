"""Attack probe for the sequential-reveal CAPTCHA channel (CaptchaChallenge).

Drives the /challenge/* protocol end to end against a running DemoServer and
prints what verdict each attacker profile receives, so the rule engine's
behaviour is MEASURED rather than asserted.

    java -cp "target/classes;<weka>;<bounce>" DemoServer 8787
    python demo/attack_probe.py [port]

Every profile is handed the answer by /challenge/oracle -- i.e. a perfect OCR
attacker is granted for free, so the timing channel is the only thing under
test. That route is a demo affordance and must never ship.

The honest headline is the LAST row: a bot that sleeps a jittered, human-like
delay passes every rule here. The channel's value is not that it stops that bot,
it is that the bot had to be online and serialised to get there.

Stdlib only -- no dependencies, matching the rest of the project.
"""
import json
import math
import random
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8787"
BASE = "http://127.0.0.1:%s" % PORT

T0 = None


def post(path, obj=None):
    data = json.dumps(obj).encode() if obj is not None else b""
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def get(path):
    return json.load(urllib.request.urlopen(BASE + path))


def hop_path(frm, to, kind):
    """Walks one movement in REAL time, stamping off a wall clock.

    Spending the time matters: the server checks the submitted trace's span
    against the window it timed, and a trace can never legitimately span more
    time than the challenge was open. A sim that advanced a fake clock while
    consuming no real time would trip that check for a reason unrelated to the
    attack being modelled. Only the forgery profile rewrites timestamps, and it
    does so deliberately, after the fact.
    """
    global T0
    pts = []
    n = 6 if kind == "teleport" else 30
    step = 0.003 if kind == "teleport" else 0.012
    for i in range(1, n + 1):
        u = i / n
        e = u * u * u * (10 - 15 * u + 6 * u * u) if kind == "bezier" else u
        x = frm[0] + (to[0] - frm[0]) * e
        y = frm[1] + (to[1] - frm[1]) * e
        if kind == "bezier":
            y += math.sin(u * math.pi) * 34          # an arc, not a ruler
        time.sleep(step)
        now = time.perf_counter() * 1000.0
        if T0 is None:
            T0 = now
        pts.append([now - T0, x, y])
    return pts, [to[0], to[1]]


def run(name, delay_fn, kind, forge=None, wrong=False):
    """Plays one profile. `forge` multiplies every submitted timestamp;
    `wrong` clicks the wrong tile at hop 3."""
    global T0
    T0 = None
    ch = post("/challenge/new")
    tiles = {t["k"]: t for t in ch["tiles"]}
    half = ch["tile"] / 2
    trace, cur = [], [40.0, ch["boardH"] - 40.0]

    for h in range(ch["hops"]):
        o = get("/challenge/oracle?id=%s" % ch["id"])
        clicked = (o["k"] + 1) % len(tiles) if (wrong and h == 2) else o["k"]
        time.sleep(delay_fn() / 1000.0)
        target = [tiles[o["k"]]["x"] + half, tiles[o["k"]]["y"] + half]
        pts, cur = hop_path(cur, target, kind)
        trace += pts
        res = post("/challenge/arrive", {"id": ch["id"], "k": clicked})
        if res["state"] == "failed":
            print("%-20s REJECT   %s" % (name, res["reason"]))
            return
        if res["state"] == "complete":
            break

    sent = [[p[0] * forge, p[1], p[2]] for p in trace] if forge else trace
    v = post("/challenge/verify", {"id": ch["id"], "trace": sent})
    cover = "n/a" if v["traceCoverage"] is None else "%.0f%%" % (v["traceCoverage"] * 100)
    print("%-20s %-8s wall=%5dms  mad=%.4f  cover=%-5s pts=%3d  corr=%d"
          % (name, "PASS" if v["pass"] else "REJECT", v["totalWallMs"],
             v["hopWallMad"], cover, v["tracePoints"], v["corrections"]))
    # Per-hop times are printed because the regularity rule lives or dies on
    # them. They are what showed the original coefficient-of-variation rule was
    # useless: one ~350 ms scheduling hiccup in five hops took a bot sleeping
    # exactly 700 ms from CV 0.005 to 0.121, straight past the threshold. MAD
    # ignores that outlier. Keep an eye on hop 1 too -- its window opens at
    # issue(), so it carries board setup that hops 2-5 do not.
    print("%22s  hops: %s ms" % ("", " ".join("%d" % h for h in v["hopWallMs"])))
    for f in v["flags"]:
        print("%22s! %s" % ("", f))
    for n in v["notes"]:
        print("%22s. %s" % ("", n))


def jitter(mean, spread):
    return lambda: mean + (random.random() * 2 - 1) * spread


def main():
    random.seed(7)
    print("sequential-reveal CAPTCHA -- attack probe")
    print("server %s   (thresholds are UNCALIBRATED -- see CaptchaChallenge.Rules)" % BASE)
    print("=" * 96)
    print("%-20s %-8s %s" % ("profile", "verdict", "measurements"))
    print("-" * 96)

    run("human-paced",      jitter(820, 260), "bezier")
    run("bot instant",      lambda: 3,        "teleport")
    run("bot fixed-delay",  lambda: 700,      "bezier")
    run("bot forged-trace", jitter(760, 240), "bezier", forge=3.0)
    run("bot jittered",     jitter(700, 250), "bezier")
    run("wrong tile",       jitter(700, 250), "bezier", wrong=True)

    # a completed challenge must not be verifiable twice
    global T0
    T0 = None
    ch = post("/challenge/new")
    tiles = {t["k"]: t for t in ch["tiles"]}
    tr, cur = [], [40.0, 400.0]
    for _ in range(ch["hops"]):
        o = get("/challenge/oracle?id=%s" % ch["id"])
        time.sleep(0.6 + random.random() * 0.4)
        tgt = [tiles[o["k"]]["x"] + ch["tile"] / 2, tiles[o["k"]]["y"] + ch["tile"] / 2]
        p, cur = hop_path(cur, tgt, "bezier")
        tr += p
        post("/challenge/arrive", {"id": ch["id"], "k": o["k"]})
    first = post("/challenge/verify", {"id": ch["id"], "trace": tr})
    again = post("/challenge/verify", {"id": ch["id"], "trace": tr})
    print("-" * 96)
    print("%-20s %-8s second verify=%s  (%s)"
          % ("replay guard", "PASS" if first["pass"] else "REJECT",
             "PASS" if again["pass"] else "REJECT",
             again["flags"][0] if again["flags"] else ""))


if __name__ == "__main__":
    main()
