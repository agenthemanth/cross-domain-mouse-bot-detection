"""login_probe.py -- drives the moving puzzle + /auth/login end to end, in real
time, for each rung of the bot ladder and for human motor MODELS, under several
simulated network conditions, and prints the verdict each gets.

    java ... DemoServer 8791 --oracle        # the probe needs the oracle route
    python demo/login_probe.py 8791 > login_channel_results.txt

Stdlib only. The background (passive) channel is left EMPTY so the puzzle
channel is isolated -- every run reports passive "NO_EVIDENCE". The in-page
sims on login.html exercise both channels together, and find the slot by
vision instead of the oracle.

WHAT THE ORACLE DOES: the slot is only ever drawn into the board PNG. The probe
reads it from /puzzle/oracle instead of doing vision, which hands every bot a
perfect, free perceptual layer. That isolates the timing channel as the only
thing under test, the same as /challenge/oracle does for the CAPTCHA. The probe
only lets a bot use the MOVED slot after its client has actually received the
jump reply over the simulated link.

NETWORK: every report goes out on its own thread after a sampled uplink delay,
and its reply is only visible to the controller after a sampled downlink delay,
several in flight at once, like fetch() in the page. Delays are Gaussian
(base, sd), truncated at 0, independent per message, so reordering happens.
This is a model of a link, not a real one.
"""
import json
import math
import random
import sys
import threading
import time
import urllib.error
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
BASE = f"http://127.0.0.1:{PORT}"

# name, one-way base delay ms, one-way jitter sd ms  (RTT ~ 2 x base)
PROFILES = [("localhost", 0, 0), ("broadband", 20, 5), ("wan", 60, 15), ("mobile", 120, 35)]
MAX_INFLIGHT = 6
SEND_S = 0.016


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())


def min_jerk(u):
    return u * u * u * (10 - 15 * u + 6 * u * u)


class Client:
    """One puzzle session over a simulated link."""

    def __init__(self, link, rnd):
        self.base, self.sd = link
        self.rnd = rnd
        p = post("/puzzle/new", {})
        self.pid = p["id"]
        self.start = (p["startX"], p["startY"])
        o = get(f"/puzzle/oracle?id={self.pid}")
        self.old = (o["slotX"], o["slotY"])
        self.new = None
        self.seq = 0
        self.ack = None
        self.seen_at = None
        self.inflight = 0
        self.lock = threading.Lock()

    def delay(self):
        return max(0.0, self.rnd.gauss(self.base, self.sd)) / 1000.0

    def seen_slot(self):
        """What the controller can know: the old slot until the jump reply has arrived."""
        with self.lock:
            seen = self.ack is not None
        if not seen:
            return self.old
        if self.new is None:
            o = get(f"/puzzle/oracle?id={self.pid}")
            self.new = (o["slotX"], o["slotY"])
        return self.new

    def move(self, x, y):
        with self.lock:
            if self.inflight >= MAX_INFLIGHT:
                return
            self.seq += 1
            self.inflight += 1
            body = {"id": self.pid, "seq": self.seq, "x": round(x, 2), "y": round(y, 2)}
            if self.ack:
                body["ack"] = self.ack
        up, down = self.delay(), self.delay()

        def run():
            time.sleep(up)
            try:
                r = post("/puzzle/move", body)
            except Exception:
                r = {}
            time.sleep(down)
            with self.lock:
                self.inflight -= 1
                if r.get("state") == "jump" and self.ack is None:
                    self.ack = r["ack"]
                    self.seen_at = time.perf_counter()

        threading.Thread(target=run, daemon=True).start()

    def drop(self, x, y):
        while True:
            with self.lock:
                if self.inflight == 0:
                    break
            time.sleep(0.002)
        time.sleep(self.delay())
        body = {"id": self.pid, "x": round(x, 2), "y": round(y, 2)}
        if self.ack:
            body["ack"] = self.ack
        d = post("/puzzle/drop", body)
        time.sleep(self.delay())
        return d["state"]

    def login(self):
        return post("/auth/login", {"user": "demo", "pass": "letmein", "puzzle": self.pid, "bg": []})


def run_bot(level, link, seed):
    rnd = random.Random(seed)
    c = Client(link, rnd)
    sx, sy = c.start
    fixed = c.old
    seen = []                        # L4 delay line
    lat = 0.180 + rnd.random() * 0.080
    T = 0.65 if level == 1 else 0.80
    arc = (1 if rnd.random() < 0.5 else -1) * (12 + rnd.random() * 14)
    t0 = time.perf_counter()
    x = y = 0.0
    while True:
        now = time.perf_counter()
        u = min(1.0, (now - t0) / T)
        if level <= 2:
            tgt = fixed
        elif level == 3:
            tgt = c.seen_slot()
        else:
            seen.append((now, c.seen_slot()))
            tgt = seen[0][1]
            for t, s in seen:
                if t <= now - lat:
                    tgt = s
        e = u if level == 1 else min_jerk(u)
        dx, dy = tgt[0] - sx, tgt[1] - sy
        L = math.hypot(dx, dy) or 1
        w = math.sin(u * math.pi) * arc if level == 2 else 0
        x, y = sx + dx * e - dy / L * w, sy + dy * e + dx / L * w
        c.move(x, y)
        if u >= 1:
            if level <= 2:
                break
            # closed-loop bots keep going until they have seen the jump and (L4) waited
            if c.seen_at is not None and (level == 3 or now - c.seen_at >= lat) and tgt == c.seen_slot():
                break
            if now - t0 > 5:
                break
        time.sleep(SEND_S)
    time.sleep(0.04)
    return c.drop(x, y), c.login()


def run_human_model(kind, link, seed):
    """NOT a person -- a motor MODEL used to check the rules don't reject the
    shapes a person produces: curved, trembling drags and a miss-then-re-grab.

    arc:    min-jerk drag bowed +-30 (or 45) px, ~1.5 px tremor; notices the jump
            only once its reply has arrived, keeps going for 180-260 ms, then a
            smooth min-jerk re-aim from where it is.
    regrab: quick drag straight to the OLD slot, drops there (miss), pauses 3 s,
            re-grabs and drags to the moved slot.
    """
    rnd = random.Random(seed)
    c = Client(link, rnd)
    sx, sy = c.start
    delay = 0.180 + rnd.random() * 0.080
    arc = (1 if seed % 2 else -1) * (45 if kind == "arc45" else 30)

    def tremor():
        return rnd.gauss(0, 1.5)

    def stream(path_fn, T, stop_after_seen=False):
        t0 = time.perf_counter()
        while True:
            u = min(1.0, (time.perf_counter() - t0) / T)
            x, y = path_fn(u)
            c.move(x, y)
            if u >= 1 and not (stop_after_seen and c.seen_at is None):
                return x, y
            if stop_after_seen and c.seen_at and time.perf_counter() - c.seen_at >= delay:
                return x, y
            time.sleep(SEND_S)

    def mj_path(a, b, bow):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) or 1

        def f(u):
            e = min_jerk(u)
            w = math.sin(u * math.pi) * bow
            return (a[0] + dx * e - dy / L * w + tremor(), a[1] + dy * e + dx / L * w + tremor())
        return f

    if kind.startswith("arc"):
        x, y = stream(mj_path((sx, sy), c.old, arc), 1.0, stop_after_seen=True)
        new = c.seen_slot()
        x, y = stream(mj_path((x, y), new, 0), 0.45)
        x, y = new
    else:
        x, y = stream(mj_path((sx, sy), c.old, 0), 0.6)
        c.drop(x, y)
        time.sleep(3.0)
        new = c.seen_slot()
        x, y = stream(mj_path((x, y), new, 0), 0.6)
        x, y = new
    time.sleep(0.04)
    return c.drop(x, y), c.login()


def ms(v):
    return "-" if v is None else f"{v:.0f}"


def line(name, trial, drop, r):
    pz = r["puzzle"]
    raw_rule = ""
    if pz["rawOnsetMs"] is not None:
        raw_rule = "raw-rule PASS" if pz["rawOnsetMs"] >= 100 else "raw-rule FAIL"
    print(f"  {name:32s} t{trial}  drop={drop:9s} {r['decision']:5s}  "
          f"ack={ms(pz['ackMs']):>4s} raw={ms(pz['rawOnsetMs']):>4s} net={ms(pz['onsetMs']):>4s} "
          f"ack->drop={ms(pz['jumpToDropMs']):>5s} stale={pz['stale']:<2d} {raw_rule:13s} "
          f"{(pz['flags'] or ['no flag'])[0][:60]}")
    return r["decision"] == "HUMAN", pz


def main():
    names = {1: "L1 scripted, open-loop", 2: "L2 evasive, open-loop",
             3: "L3 closed-loop, instant", 4: "L4 closed-loop, 180-260 ms delay"}
    try:
        get("/puzzle/oracle?id=none")
    except urllib.error.HTTPError as e:
        if e.code != 404 or b"disabled" in e.read():
            sys.exit("the probe needs the oracle: start DemoServer with --oracle")
    print(f"probe against {BASE}  (passive channel empty -> puzzle isolated; slot read from the oracle)")
    print("columns: ack = jump sent -> ack received; raw = onset measured from the jump send (old rule);")
    print("         net = onset measured from the ack (the rule now used); raw-rule = what the old rule would say\n")

    summary = []
    for prof, base, sd in PROFILES:
        link = (base, sd)
        print(f"=== {prof}: one-way {base} ms +- {sd} ms  (RTT ~{2 * base} ms) ===")
        row = {"profile": prof}
        for level in (1, 2, 3, 4):
            passes, raw_pass = 0, 0
            for trial in range(3):
                drop, r = run_bot(level, link, seed=level * 100 + trial)
                ok, pz = line(names[level], trial, drop, r)
                passes += ok
                raw_pass += pz["rawOnsetMs"] is not None and pz["rawOnsetMs"] >= 100
            row[f"L{level}"] = passes
            if level == 3:
                row["L3raw"] = raw_pass
        hp, hn = 0, 0
        for kind, n, s0, label in (("arc", 6, 0, "model: arc +-30px, tremor, delay"),
                                   ("arc45", 4, 20, "model: arc +-45px, tremor, delay"),
                                   ("regrab", 2, 50, "model: miss, pause 3 s, re-grab")):
            for trial in range(n):
                drop, r = run_human_model(kind, link, seed=s0 + trial)
                ok, _ = line(label, trial, drop, r)
                hp += ok
                hn += 1
        row["human"] = (hp, hn)
        summary.append(row)
        print()

    c = Client((0, 0), random.Random(0))
    first = c.login()
    again = c.login()
    print(f"unsolved puzzle  -> {first['decision']}: {first['puzzle']['flags'][0]}")
    print(f"replayed puzzle  -> {again['decision']}: {again['puzzle']['flags'][0]}\n")

    print("SUMMARY (passes out of 3 bot trials; human models passed / run)")
    print(f"  {'profile':10s} {'L1':>3s} {'L2':>3s} {'L3':>3s} {'L3 if raw rule':>15s} {'L4':>3s}  human models")
    for r in summary:
        print(f"  {r['profile']:10s} {r['L1']:>3d} {r['L2']:>3d} {r['L3']:>3d} {r['L3raw']:>15d} {r['L4']:>3d}  "
              f"{r['human'][0]}/{r['human'][1]}")


if __name__ == "__main__":
    main()
