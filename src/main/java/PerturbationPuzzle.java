import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.GradientPaint;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Shape;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Path2D;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * PERTURBATION-RESPONSE PUZZLE -- a drag-to-slot challenge whose slot JUMPS
 * mid-drag, at a moment only the server knows.
 *
 * WHY
 * ---
 * Tier 4 was lost to an OPEN-LOOP bot: it plans a trajectory in advance whose
 * statistics look human. Passive observation cannot win that fight, because
 * statistics can be copied. So instead of watching, this challenge pokes: once
 * the piece has covered a secret, server-chosen fraction of the course, the
 * server answers a routine position report with a new board on which the slot
 * has moved. This is the double-step target paradigm of motor-control research
 * turned into a challenge. A person re-aims after a visuomotor delay; an
 * open-loop bot never re-aims, and a naive closed-loop bot re-aims with no delay.
 *
 * TRUST BOUNDARY
 * --------------
 * - The client streams the piece position to {@code /puzzle/move} with a
 *   sequence number; every sample is stamped on arrival by the SERVER clock.
 *   Out-of-order arrivals are dropped, never re-timed.
 * - The jump reply carries a random ACK token together with the new board. The
 *   client must echo it on every later report, and the server stamps the first
 *   echo it receives. Latency is measured from THAT stamp, not from when the jump
 *   was sent, so network round-trip time is not counted as "reaction": the ack
 *   and the re-aim both travel the same uplink. The asymmetry is deliberate. A
 *   client cannot echo before it has the token, and echoing late only makes its
 *   reaction look faster. There is no way to lengthen the measured latency except
 *   actually waiting.
 * - The slot position is never sent as a number. The board is a server-rendered
 *   PNG with same-style decoy slots, and the jump reply carries a new PNG in which
 *   the slot has moved and every decoy has shifted. A bot therefore needs a
 *   perception step (find the outline that matches the piece), not a JSON read.
 *   That is a speed bump, not the security property. A white-box vision bot finds
 *   the slot in a few ms, and the in-page L3/L4 sims do exactly that.
 * - Positions are still client-claimed. Only timing is server-owned.
 *
 * What still gets in: a closed-loop bot that locates the slot by vision and then
 * waits a human-like delay (ladder level L4). This raises the cost of an attack;
 * it does not make one impossible.
 *
 * NOT THE RANDOM FOREST
 * ---------------------
 * The drag trace is never fed to the forest (Tier 3: short goal-directed
 * segments are the worst input it has). This is its own channel with its own
 * rule-based scorer, combined with the passive channel by an explicit decision
 * table in {@link DemoServer}, never averaged.
 *
 * THRESHOLDS ARE UNCALIBRATED -- see {@link Rules}, and demo/analyze_trials.py
 * for how to calibrate them on a participant sample.
 */
public class PerturbationPuzzle {

    static final int BOARD_W = 520, BOARD_H = 240;
    static final int PIECE = 50;
    static final int SLOT = PIECE + 6;
    /** Drop within this many px of the current slot centre to place the piece. */
    static final int SNAP_PX = 18;
    static final int MAX_DROPS = 4;
    /** Board PNGs are rendered at this multiple of board units, for crisp display. */
    static final int SCALE = 2;
    static final int DECOYS = 3;
    private static final long TTL_MS = 5 * 60 * 1000L;

    /**
     * Hand-set, first-principles thresholds. No human sample exists for this
     * challenge yet; these are placeholders chosen to err toward passing a person.
     */
    static final class Rules {
        /**
         * Visuomotor correction to a target jump takes ~120-250 ms in the motor
         * control literature. Measured from the ACK (not the send), anything under
         * 100 ms is not a reaction to what was seen.
         */
        static final double MIN_ONSET_MS = 100;
        // No MAX onset rule, deliberately: a person who drops on the old slot and
        // re-grabs seconds later "re-aims" late, and must pass. Open-loop bots
        // already fail by never placing the piece on the moved slot.
        /** Ack -> accepted drop: reacting AND travelling. */
        static final double MIN_JUMP_TO_DROP_MS = 300;
        /** Server-clocked samples needed after the ack to judge anything. */
        static final int MIN_POST_JUMP_SAMPLES = 5;

        /**
         * Deviation along the jump axis that counts as a re-aim, and must persist.
         * 15 px, not smaller: a drag bowed 30 px drifts ~5 px off its pre-jump
         * momentum line within 100 ms from curvature alone, plus tremor. At 6 px
         * with a walk-back to baseline, 4 of 10 curved human-model drags were
         * falsely timed "too fast" (login_probe.py). The crossing time is
         * reported as-is, so latency is overstated -- errs toward passing a person.
         */
        static final double ONSET_PX = 15.0;
        static final double SUSTAIN_MS = 50;
        /** Momentum extrapolation window, before and after the ack. */
        static final double PRE_WINDOW_MS = 120;
        static final double EXTRAP_CAP_MS = 150;
    }

    enum Kind { CIRCLE, SQUARE, TRIANGLE, DIAMOND, STAR, CROSS }

    static final class Puzzle {
        final String id;
        final long createdAt = System.currentTimeMillis();
        final double startX, startY, slotX, slotY, jumpX, jumpY;
        /** Fraction of the start->slot course at which the slot jumps. Never sent. */
        final double trigger;
        final Kind kind;
        /** Decoys before and after the jump: {x, y, kindOrdinal}. */
        final List<double[]> decoysBefore = new ArrayList<>(), decoysAfter = new ArrayList<>();
        final long renderSeed;

        boolean jumped = false;
        long jumpNs = 0;
        /** Rendered at issue, so the jump reply costs no render time on the clock. */
        String jumpBoardUri;
        String ackToken;
        long ackNs = 0;
        String jumpReply;
        boolean placed = false;
        long placedNs = 0;
        boolean placedBeforeJump = false;
        int drops = 0;
        boolean consumed = false;
        long lastSeq = -1;
        int stale = 0;

        /** Server-clocked samples: {ns, x, y}. */
        final List<double[]> log = new ArrayList<>();

        Puzzle(String id, double sx, double sy, double tx, double ty, double jx, double jy,
               double trigger, Kind kind, long renderSeed) {
            this.id = id; startX = sx; startY = sy; slotX = tx; slotY = ty; jumpX = jx; jumpY = jy;
            this.trigger = trigger; this.kind = kind; this.renderSeed = renderSeed;
        }

        double curSlotX() { return jumped ? jumpX : slotX; }
        double curSlotY() { return jumped ? jumpY : slotY; }

        double progress(double x, double y) {
            double dx = slotX - startX, dy = slotY - startY;
            double len2 = dx * dx + dy * dy;
            return ((x - startX) * dx + (y - startY) * dy) / len2;
        }
    }

    private static final Map<String, Puzzle> LIVE = new ConcurrentHashMap<>();
    private static final AtomicLong COUNTER = new AtomicLong();
    private static final Random RNG = new Random();
    private static final SecureRandom TOKENS = new SecureRandom();

    static Puzzle issue() throws IOException {
        sweep();
        double half = PIECE / 2.0;
        double sx = 30 + half + RNG.nextInt(30);
        double sy = 55 + RNG.nextInt(BOARD_H - 110);
        double tx = BOARD_W - 40 - half - RNG.nextInt(60);
        double ty = 55 + RNG.nextInt(BOARD_H - 110);

        // jump roughly perpendicular to the course, to whichever side has room
        double dx = tx - sx, dy = ty - sy, len = Math.hypot(dx, dy);
        double px = -dy / len, py = dx / len;
        double mag = 60 + RNG.nextInt(26);
        double sign = RNG.nextBoolean() ? 1 : -1;
        double jx = 0, jy = 0;
        for (int attempt = 0; attempt < 4; attempt++) {
            jx = tx + sign * px * mag;
            jy = ty + sign * py * mag;
            if (inBounds(jx, jy, half)) break;
            sign = -sign;
            if (attempt == 1) mag *= 0.7;
        }
        jx = clamp(jx, half + 4, BOARD_W - half - 4);
        jy = clamp(jy, half + 4, BOARD_H - half - 4);

        double trigger = 0.20 + RNG.nextDouble() * 0.15;
        Kind[] kinds = Kind.values();
        Kind kind = kinds[RNG.nextInt(kinds.length)];
        String id = Long.toHexString(System.nanoTime()) + "-p" + COUNTER.incrementAndGet();
        Puzzle p = new Puzzle(id, sx, sy, tx, ty, jx, jy, trigger, kind, RNG.nextLong());
        placeDecoys(p);
        p.jumpBoardUri = dataUri(renderBoard(p, true));
        p.ackToken = Long.toHexString(TOKENS.nextLong()) + Long.toHexString(TOKENS.nextLong());
        LIVE.put(id, p);
        return p;
    }

    /**
     * Decoys keep clear of the start, the slot and where it will jump to. On the
     * jump each decoy shifts too, so diffing the two boards does not isolate the
     * slot: the attacker has to match the outline against the piece.
     */
    private static void placeDecoys(Puzzle p) {
        double r = SLOT / 2.0;
        List<Kind> others = new ArrayList<>();
        for (Kind k : Kind.values()) if (k != p.kind) others.add(k);
        java.util.Collections.shuffle(others, RNG);
        for (int d = 0; d < DECOYS && d < others.size(); d++) {
            for (int tries = 0; tries < 300; tries++) {
                double x = r + 6 + RNG.nextDouble() * (BOARD_W - 2 * r - 12);
                double y = r + 6 + RNG.nextDouble() * (BOARD_H - 2 * r - 12);
                if (Math.hypot(x - p.startX, y - p.startY) < 85) continue;
                if (Math.hypot(x - p.slotX, y - p.slotY) < 72) continue;
                if (Math.hypot(x - p.jumpX, y - p.jumpY) < 72) continue;
                if (tooClose(p.decoysBefore, x, y, 66)) continue;
                p.decoysBefore.add(new double[]{ x, y, others.get(d).ordinal() });
                break;
            }
        }
        for (double[] dcy : p.decoysBefore) {
            double[] moved = dcy;
            for (int tries = 0; tries < 60; tries++) {
                double a = RNG.nextDouble() * 2 * Math.PI, m = 15 + RNG.nextDouble() * 20;
                double x = dcy[0] + Math.cos(a) * m, y = dcy[1] + Math.sin(a) * m;
                if (!inBounds(x, y, r + 2)) continue;
                if (Math.hypot(x - p.jumpX, y - p.jumpY) < 72) continue;
                if (tooClose(p.decoysAfter, x, y, 66)) continue;
                moved = new double[]{ x, y, dcy[2] };
                break;
            }
            p.decoysAfter.add(moved);
        }
    }

    private static boolean tooClose(List<double[]> pts, double x, double y, double min) {
        for (double[] q : pts) if (Math.hypot(q[0] - x, q[1] - y) < min) return true;
        return false;
    }

    private static boolean inBounds(double x, double y, double half) {
        return x >= half + 4 && x <= BOARD_W - half - 4 && y >= half + 4 && y <= BOARD_H - half - 4;
    }
    private static double clamp(double v, double lo, double hi) { return Math.max(lo, Math.min(hi, v)); }

    static Puzzle get(String id) { return id == null ? null : LIVE.get(id); }

    private static void sweep() {
        long now = System.currentTimeMillis();
        for (Iterator<Puzzle> it = LIVE.values().iterator(); it.hasNext(); ) {
            if (now - it.next().createdAt > TTL_MS) it.remove();
        }
    }

    private static void noteAck(Puzzle p, String ack, long now) {
        if (p.jumped && p.ackNs == 0 && ack != null && ack.equals(p.ackToken)) p.ackNs = now;
    }

    /** A position report. May be the one whose reply carries the jump. */
    static String move(Puzzle p, double x, double y, long seq, String ack) throws IOException {
        long now = System.nanoTime();
        if (p.placed || p.consumed) return "{\"state\":\"placed\"}";
        noteAck(p, ack, now);
        if (seq >= 0) {
            if (seq <= p.lastSeq) { p.stale++; return pendingOrOk(p); }
            p.lastSeq = seq;
        }
        p.log.add(new double[]{ now, x, y });
        if (p.log.size() > 4000) p.log.remove(0);
        if (!p.jumped && p.progress(x, y) >= p.trigger) {
            p.jumped = true;
            p.jumpNs = now;
            p.jumpReply = "{\"state\":\"jump\",\"ack\":\"" + p.ackToken + "\",\"board\":\"" + p.jumpBoardUri + "\"}";
        }
        return pendingOrOk(p);
    }

    /** Until the client echoes the ack, every reply re-delivers the jump, so one lost reply cannot strand a person. */
    private static String pendingOrOk(Puzzle p) {
        return p.jumped && p.ackNs == 0 ? p.jumpReply : "{\"state\":\"ok\"}";
    }

    /** The piece was released at (x, y). Placed if on the CURRENT slot. */
    static String drop(Puzzle p, double x, double y, String ack) {
        long now = System.nanoTime();
        if (p.placed) return "{\"state\":\"placed\"}";
        if (p.consumed) return "{\"state\":\"expired\"}";
        noteAck(p, ack, now);
        p.log.add(new double[]{ now, x, y });
        p.drops++;
        if (Math.hypot(x - p.curSlotX(), y - p.curSlotY()) <= SNAP_PX) {
            p.placed = true;
            p.placedNs = now;
            p.placedBeforeJump = !p.jumped;
            return "{\"state\":\"placed\",\"slotX\":" + fmt(p.curSlotX()) + ",\"slotY\":" + fmt(p.curSlotY()) + "}";
        }
        if (p.drops >= MAX_DROPS) return "{\"state\":\"exhausted\"}";
        return "{\"state\":\"miss\",\"left\":" + (MAX_DROPS - p.drops) + "}";
    }

    // ---------------- verify ----------------

    static final class Verdict {
        boolean pass;
        final List<String> flags = new ArrayList<>();
        final List<String> notes = new ArrayList<>();
        boolean jumped, placed, acked;
        int drops, samples, postJumpSamples, stale;
        /** Re-aim onset measured from the ack's arrival: the number the rule uses. */
        double onsetMs = Double.NaN;
        /** Re-aim onset measured from when the jump was sent: includes the network round trip. */
        double rawOnsetMs = Double.NaN;
        /** Jump sent -> ack received: roughly one round trip plus one send interval. */
        double ackMs = Double.NaN;
        double jumpToDropMs = Double.NaN;
        /** Distance still travelled toward the OLD slot between the ack and the onset. */
        double carryPx = Double.NaN;
        double peakCorrSpeed = Double.NaN;
    }

    /** Single use: a second call reports a replay. */
    static Verdict verify(Puzzle p) {
        Verdict v = new Verdict();
        if (p == null) { v.flags.add("no puzzle was solved"); return v; }
        v.jumped = p.jumped; v.placed = p.placed; v.drops = p.drops; v.samples = p.log.size();
        v.stale = p.stale; v.acked = p.ackNs != 0;
        if (p.consumed) { v.flags.add("puzzle already used -- replay refused"); return v; }
        p.consumed = true;

        if (!p.placed) {
            v.flags.add(p.jumped
                    ? "piece never placed on the MOVED slot -- consistent with an open-loop path planned before the jump"
                    : "piece never placed");
            return v;
        }
        if (p.placedBeforeJump) {
            v.flags.add("placed without the slot ever moving -- the piece skipped the course, so the perturbation was never delivered");
            return v;
        }
        if (p.ackNs == 0) {
            v.flags.add("the jump was never acknowledged -- the client placed the piece without echoing the token that came with the moved board");
            return v;
        }

        double jMs = p.jumpNs / 1e6, aMs = p.ackNs / 1e6;
        v.ackMs = aMs - jMs;
        v.jumpToDropMs = (p.placedNs - p.ackNs) / 1e6;
        if (v.jumpToDropMs < Rules.MIN_JUMP_TO_DROP_MS) {
            v.flags.add(String.format("placed %.0f ms after receiving the moved board (server clock) -- below the %.0f ms floor for noticing it and moving there",
                    v.jumpToDropMs, Rules.MIN_JUMP_TO_DROP_MS));
        }

        // unit vector of the jump, and of the original course
        double ux = p.jumpX - p.slotX, uy = p.jumpY - p.slotY, ul = Math.hypot(ux, uy);
        ux /= ul; uy /= ul;
        double cx = p.slotX - p.startX, cy = p.slotY - p.startY, cl = Math.hypot(cx, cy);
        cx /= cl; cy /= cl;

        // Everything that arrived before the ack was sent before the client could
        // have seen the new board, so it counts as pre-reaction momentum. The
        // ack-carrying sample itself is POST: its position was written after the
        // board arrived, and treating it as baseline let a bot that re-aims in that
        // very message hide the re-aim inside the momentum fit.
        List<double[]> pre = new ArrayList<>(), post = new ArrayList<>();
        for (double[] s : p.log) {
            double t = s[0] / 1e6;
            if (t < aMs) { if (t >= aMs - Rules.PRE_WINDOW_MS) pre.add(new double[]{ t, s[1], s[2] }); }
            else post.add(new double[]{ t, s[1], s[2] });
        }
        v.postJumpSamples = post.size();
        if (post.size() < Rules.MIN_POST_JUMP_SAMPLES) {
            v.flags.add("only " + post.size() + " server-clocked samples after the ack (< "
                    + Rules.MIN_POST_JUMP_SAMPLES + ") -- the piece was not dragged there, it arrived");
            v.pass = false;
            return v;
        }

        // pre-ack velocity along the jump axis (least squares), for momentum
        double s0, t0, v0 = 0;
        if (pre.isEmpty()) {
            double[] f = post.get(0);
            s0 = f[1] * ux + f[2] * uy; t0 = aMs;
        } else {
            double[] last = pre.get(pre.size() - 1);
            s0 = last[1] * ux + last[2] * uy; t0 = last[0];
            if (pre.size() >= 3) {
                double mt = 0, ms = 0;
                for (double[] q : pre) { mt += q[0]; ms += q[1] * ux + q[2] * uy; }
                mt /= pre.size(); ms /= pre.size();
                double num = 0, den = 0;
                for (double[] q : pre) {
                    double dt = q[0] - mt;
                    num += dt * ((q[1] * ux + q[2] * uy) - ms);
                    den += dt * dt;
                }
                if (den > 0) v0 = num / den;
            }
        }

        int n = post.size();
        double[] dev = new double[n];
        for (int i = 0; i < n; i++) {
            double[] q = post.get(i);
            double pred = s0 + v0 * Math.min(q[0] - t0, Rules.EXTRAP_CAP_MS);
            dev[i] = (q[1] * ux + q[2] * uy) - pred;
        }

        int cross = -1;
        for (int i = 0; i < n && cross < 0; i++) {
            if (dev[i] <= Rules.ONSET_PX) continue;
            boolean sustained = true;
            boolean anyLater = false;
            for (int k = i + 1; k < n && post.get(k)[0] - post.get(i)[0] <= Rules.SUSTAIN_MS; k++) {
                anyLater = true;
                if (dev[k] <= Rules.ONSET_PX) { sustained = false; break; }
            }
            if (sustained && (anyLater || i == n - 1)) cross = i;
        }

        if (cross < 0) {
            // The piece IS on the moved slot, so it re-aimed; a strongly curved drag
            // just hides when. With no onset there is no too-fast claim to make.
            v.notes.add("re-aim could not be timed (curved drag) -- latency rule not applied");
        } else {
            int onset = cross;
            v.onsetMs = post.get(onset)[0] - aMs;
            v.rawOnsetMs = post.get(onset)[0] - jMs;

            double[] a = pre.isEmpty() ? post.get(0) : pre.get(pre.size() - 1);
            double[] b = onset > 0 ? post.get(onset - 1) : a;
            v.carryPx = (b[1] - a[1]) * cx + (b[2] - a[2]) * cy;

            double peak = 0;
            for (int i = Math.max(1, onset); i < n; i++) {
                double dt = Math.max(1.0, post.get(i)[0] - post.get(i - 1)[0]);
                peak = Math.max(peak, (dev[i] - dev[i - 1]) / dt);
            }
            v.peakCorrSpeed = peak;

            if (v.onsetMs < Rules.MIN_ONSET_MS) {
                v.flags.add(String.format("re-aimed %.0f ms after receiving the moved board (server clock) -- faster than the ~%.0f ms"
                        + " visuomotor floor; that is not a reaction to what was seen", v.onsetMs, Rules.MIN_ONSET_MS));
            }
        }

        if (p.drops > 1) v.notes.add((p.drops - 1) + " missed drop(s) before placing -- allowed");
        if (p.stale > 0) v.notes.add(p.stale + " out-of-order sample(s) dropped");
        v.notes.add("thresholds are hand-set, not calibrated on a human sample");
        v.pass = v.flags.isEmpty();
        return v;
    }

    // ---------------- rendering ----------------

    static Shape shape(Kind k, double cx, double cy, double size) {
        double r = size / 2;
        switch (k) {
            case CIRCLE: return new Ellipse2D.Double(cx - r, cy - r, size, size);
            case SQUARE: return new RoundRectangle2D.Double(cx - r * 0.88, cy - r * 0.88, r * 1.76, r * 1.76, r * 0.4, r * 0.4);
            case TRIANGLE: return polygon(cx, cy + r * 0.18, r * 1.05, 3, -Math.PI / 2, 1.0);
            case DIAMOND: return polygon(cx, cy, r, 4, -Math.PI / 2, 1.0);
            case STAR: return polygon(cx, cy, r * 1.05, 10, -Math.PI / 2, 0.48);
            case CROSS: {
                double a = r * 0.34;
                Path2D.Double path = new Path2D.Double();
                double[][] pts = { {-a, -r}, {a, -r}, {a, -a}, {r, -a}, {r, a}, {a, a},
                                   {a, r}, {-a, r}, {-a, a}, {-r, a}, {-r, -a}, {-a, -a} };
                path.moveTo(cx + pts[0][0], cy + pts[0][1]);
                for (int i = 1; i < pts.length; i++) path.lineTo(cx + pts[i][0], cy + pts[i][1]);
                path.closePath();
                return path;
            }
            default: throw new IllegalArgumentException(String.valueOf(k));
        }
    }

    /** n-gon; for n = 10 with inner < 1 it alternates outer/inner radius (a star). */
    private static Shape polygon(double cx, double cy, double r, int n, double rot, double inner) {
        Path2D.Double path = new Path2D.Double();
        for (int i = 0; i < n; i++) {
            double rr = (inner < 1 && i % 2 == 1) ? r * inner : r;
            double a = rot + 2 * Math.PI * i / n;
            if (i == 0) path.moveTo(cx + rr * Math.cos(a), cy + rr * Math.sin(a));
            else path.lineTo(cx + rr * Math.cos(a), cy + rr * Math.sin(a));
        }
        path.closePath();
        return path;
    }

    static byte[] renderBoard(Puzzle p, boolean afterJump) throws IOException {
        int w = BOARD_W * SCALE, h = BOARD_H * SCALE;
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);
        g.scale(SCALE, SCALE);
        g.setColor(new Color(0xFA, 0xFB, 0xFD));
        g.fillRect(0, 0, BOARD_W, BOARD_H);
        g.setColor(new Color(0xF0, 0xF2, 0xFA));
        g.setStroke(new BasicStroke(1f));
        for (int x = 0; x < BOARD_W; x += 26) g.drawLine(x, 0, x, BOARD_H);
        for (int y = 0; y < BOARD_H; y += 26) g.drawLine(0, y, BOARD_W, y);

        List<double[]> slots = new ArrayList<>(afterJump ? p.decoysAfter : p.decoysBefore);
        slots.add(new double[]{ afterJump ? p.jumpX : p.slotX, afterJump ? p.jumpY : p.slotY, p.kind.ordinal() });
        Color fill = new Color(47, 91, 234, 16), edge = new Color(0x8E, 0x9A, 0xC0);
        g.setStroke(new BasicStroke(2.5f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        for (double[] s : slots) {
            Shape sh = shape(Kind.values()[(int) s[2]], s[0], s[1], SLOT);
            g.setColor(fill);
            g.fill(sh);
            g.setColor(edge);
            g.draw(sh);
        }
        g.dispose();

        // light per-pixel noise, so two renders are never byte-identical
        Random rng = new Random(p.renderSeed ^ (afterJump ? 0x5DEECE66DL : 0L));
        for (int i = 0; i < w * h / 12; i++) {
            int x = rng.nextInt(w), y = rng.nextInt(h), c = img.getRGB(x, y), d = rng.nextInt(9) - 4;
            int r = clamp8(((c >> 16) & 255) + d), gg = clamp8(((c >> 8) & 255) + d), b = clamp8((c & 255) + d);
            img.setRGB(x, y, (r << 16) | (gg << 8) | b);
        }
        return png(img);
    }

    static byte[] renderPiece(Puzzle p) throws IOException {
        int s = PIECE * SCALE;
        BufferedImage img = new BufferedImage(s, s, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.scale(SCALE, SCALE);
        Shape sh = shape(p.kind, PIECE / 2.0, PIECE / 2.0, PIECE - 2);
        g.setPaint(new GradientPaint(0, 0, new Color(0x4D, 0x76, 0xFF), 0, PIECE, new Color(0x2F, 0x5B, 0xEA)));
        g.fill(sh);
        g.setColor(new Color(255, 255, 255, 90));
        g.setStroke(new BasicStroke(1.2f));
        g.draw(sh);
        g.dispose();
        return png(img);
    }

    private static int clamp8(int v) { return Math.max(0, Math.min(255, v)); }

    private static byte[] png(BufferedImage img) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ImageIO.write(img, "png", bos);
        return bos.toByteArray();
    }

    private static String dataUri(byte[] png) {
        return "data:image/png;base64," + Base64.getEncoder().encodeToString(png);
    }

    // ---------------- json ----------------

    /** Never contains the slot position: the board image is the only place it appears. */
    static String issueJson(Puzzle p) throws IOException {
        return "{\"id\":" + CaptchaChallenge.jsonStr(p.id)
                + ",\"boardW\":" + BOARD_W + ",\"boardH\":" + BOARD_H + ",\"piece\":" + PIECE
                + ",\"snap\":" + SNAP_PX
                + ",\"startX\":" + fmt(p.startX) + ",\"startY\":" + fmt(p.startY)
                + ",\"board\":\"" + dataUri(renderBoard(p, false)) + "\""
                + ",\"pieceImg\":\"" + dataUri(renderPiece(p)) + "\"}";
    }

    /** Simulation affordance only (see DemoServer): the current slot, so a probe can skip perception. */
    static String oracleJson(Puzzle p) {
        return "{\"slotX\":" + fmt(p.curSlotX()) + ",\"slotY\":" + fmt(p.curSlotY()) + ",\"jumped\":" + p.jumped + "}";
    }

    /**
     * The full server-side record, for the study log: geometry, the ack, and the
     * raw server-clocked sample log (ms relative to the jump, or to the first
     * sample if there was none), so the rules can be re-scored offline.
     */
    static String traceJson(Puzzle p) {
        if (p == null) return "null";
        double ref = p.jumped ? p.jumpNs : (p.log.isEmpty() ? 0 : p.log.get(0)[0]);
        StringBuilder sb = new StringBuilder("{");
        sb.append("\"kind\":\"").append(p.kind).append('"');
        sb.append(",\"start\":[").append(fmt(p.startX)).append(',').append(fmt(p.startY)).append(']');
        sb.append(",\"slot\":[").append(fmt(p.slotX)).append(',').append(fmt(p.slotY)).append(']');
        sb.append(",\"jumpTo\":[").append(fmt(p.jumpX)).append(',').append(fmt(p.jumpY)).append(']');
        sb.append(",\"trigger\":").append(fmt(p.trigger));
        sb.append(",\"jumped\":").append(p.jumped);
        sb.append(",\"ackMs\":").append(p.ackNs == 0 ? "null" : fmt((p.ackNs - ref) / 1e6));
        sb.append(",\"placedMs\":").append(p.placedNs == 0 ? "null" : fmt((p.placedNs - ref) / 1e6));
        sb.append(",\"stale\":").append(p.stale);
        sb.append(",\"log\":[");
        for (int i = 0; i < p.log.size(); i++) {
            double[] s = p.log.get(i);
            if (i > 0) sb.append(',');
            sb.append('[').append(fmt((s[0] - ref) / 1e6)).append(',').append(fmt(s[1])).append(',').append(fmt(s[2])).append(']');
        }
        return sb.append("]}").toString();
    }

    static String verdictJson(Verdict v) {
        StringBuilder sb = new StringBuilder("{");
        sb.append("\"pass\":").append(v.pass);
        sb.append(",\"jumped\":").append(v.jumped);
        sb.append(",\"acked\":").append(v.acked);
        sb.append(",\"placed\":").append(v.placed);
        sb.append(",\"drops\":").append(v.drops);
        sb.append(",\"samples\":").append(v.samples);
        sb.append(",\"postJumpSamples\":").append(v.postJumpSamples);
        sb.append(",\"stale\":").append(v.stale);
        sb.append(",\"onsetMs\":").append(num(v.onsetMs));
        sb.append(",\"rawOnsetMs\":").append(num(v.rawOnsetMs));
        sb.append(",\"ackMs\":").append(num(v.ackMs));
        sb.append(",\"jumpToDropMs\":").append(num(v.jumpToDropMs));
        sb.append(",\"carryPx\":").append(num(v.carryPx));
        sb.append(",\"peakCorrSpeed\":").append(num(v.peakCorrSpeed));
        sb.append(",\"flags\":").append(list(v.flags));
        sb.append(",\"notes\":").append(list(v.notes));
        return sb.append("}").toString();
    }

    private static String list(List<String> xs) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < xs.size(); i++) { if (i > 0) sb.append(','); sb.append(CaptchaChallenge.jsonStr(xs.get(i))); }
        return sb.append("]").toString();
    }
    private static String num(double d) { return Double.isNaN(d) ? "null" : fmt(d); }
    private static String fmt(double d) { return String.format(java.util.Locale.ROOT, "%.2f", d); }
}
