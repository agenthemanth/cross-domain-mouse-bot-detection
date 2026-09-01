import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import javax.imageio.ImageIO;

/**
 * SEQUENTIAL-REVEAL CAPTCHA -- the active challenge that the passive
 * mouse-dynamics model escalates to.
 *
 * WHY THIS EXISTS
 * ---------------
 * Tier 5 found the reject option is a free win -- deferring the uncertain score
 * band costs almost no recall and cuts human FPR -- but it is "only useful if
 * deferred traffic has somewhere to go". Nothing in tiers 1-7 filled that slot.
 * This is the destination: the RF handles the confident majority silently, and
 * only the deferred band is challenged.
 *
 * WHAT IS NEW ABOUT IT
 * --------------------
 * A classic CAPTCHA is one static puzzle: the attacker downloads it, solves it
 * offline with as much time and compute as it likes, and replays the answer.
 * Every trajectory attack in tiers 4-7 works the same way -- it needs the path
 * in advance, then matches its statistics (SHUFFLE reuses the victim chunk's own
 * (distance, dt) pairs; BALLISTIC reorders them onto a smooth envelope).
 *
 * Here the puzzle is revealed ONE HOP AT A TIME and each reveal is stamped by
 * the SERVER's clock:
 *
 *   1. the board shows 12 distorted glyph tiles, rendered server-side as PNGs --
 *      the letters are never in the DOM, so they have to be read, not scraped;
 *   2. the prompt ("go to the tile showing X") is also a server-rendered image,
 *      so the target has to be read too;
 *   3. only when the client ARRIVES at the current tile does the server reveal
 *      the next prompt -- and it records, on its own clock, when it did so.
 *
 * The whole path therefore cannot be precomputed, and it cannot be replayed: the
 * sequence is fresh per challenge and unknowable until you are already moving.
 *
 * WHAT THIS ACTUALLY BUYS -- stated honestly
 * ------------------------------------------
 * NOT "bots cannot forge this". A bot reads the board, OCRs the glyphs and
 * sleeps a plausible delay in a few lines of code. What the design buys is that
 * the attacker is forced ONLINE and SERIALISED: it must solve each hop live,
 * in order, inside a server-clocked window it cannot shrink. That is a
 * throughput tax, not a classification guarantee -- the same "defences are
 * attack-specific" conclusion the seven tiers reached, applied one level up.
 *
 * TRUST BOUNDARY -- the part that makes the timing worth anything
 * --------------------------------------------------------------
 * HARD signals are measured entirely on the server clock: it knows when it sent
 * each reveal and when the arrival for that hop came back. A client cannot make
 * those numbers smaller by lying; it can only actually be slow. SOFT signals
 * come from the client's submitted trace and are forgeable -- they are used only
 * as corroboration, and every one of them is cross-checked against the hard
 * window (see the consistency test in {@link #verify}).
 *
 * DELIBERATELY NOT THE RANDOM FOREST
 * ----------------------------------
 * The frozen AUGMENTED-18 model is NOT applied to these traces. Tier 3 measured
 * that exact mistake: short goal-directed segments (median 1.6 s / 11 pts) took
 * zero-shot human FPR from 9.85% to 32.8% and AUC from 0.53 to 0.36, because
 * they sit even further outside DELBOT's long circle-drawing distribution.
 * CAPTCHA hops are that shape. So this is a SECOND, INDEPENDENT channel with its
 * own scorer and its own thresholds; the two are never averaged into one number.
 *
 * THRESHOLDS ARE UNCALIBRATED
 * ---------------------------
 * Every constant in {@link Rules} is hand-set from first principles, not fitted
 * to data. There is no human sample for this challenge -- n=1 (the author's own
 * hand) is not calibration. They are conservative placeholders and are reported
 * as such in the verdict. Calibrating them on real users is open work.
 */
public class CaptchaChallenge {

    // ---------------- board geometry (server-authoritative) ----------------

    /** The client renders the board at exactly this logical size. */
    static final int BOARD_W = 760, BOARD_H = 440;
    static final int COLS = 4, ROWS = 3;
    static final int TILE = 96;
    static final int HOPS = 5;

    /** Ambiguous glyphs (0/O, 1/I/L, 5/S, 2/Z) are excluded, as in any usable CAPTCHA. */
    private static final String ALPHABET = "ACDEFGHJKMNPQRTUVWXY";

    private static final long TTL_MS = 5 * 60 * 1000L;

    // ---------------- uncalibrated decision rules ----------------

    /**
     * Hand-set thresholds. NOT fitted to any human sample -- see the class note.
     * Each is expressed as "what would have to be true for this to be a person",
     * chosen conservatively so the challenge errs toward passing.
     */
    static final class Rules {
        /**
         * A hop is: read a distorted prompt glyph, find the matching tile among
         * 12 distorted glyphs, then move to it. Under ~250 ms that is not a
         * visuomotor act, it is a lookup. Conservative -- an unhurried person
         * takes well over a second.
         */
        static final long MIN_HOP_WALL_MS = 250;

        /**
         * Regularity of the per-hop wall times, as median-absolute-deviation over
         * the median. A bot sleeping a constant delay lands near zero; people are
         * noisy across trials.
         *
         * MAD, not the coefficient of variation. CV was the first cut and it was
         * measured to be useless here: a single ~350 ms scheduling hiccup in one
         * of five hops took a bot sleeping exactly 700 ms from CV 0.005 to 0.121,
         * sailing past the threshold. Five samples is far too few for a
         * variance-based statistic. MAD ignores the outlier -- the same run gives
         * 0.003, against 0.03 for a jittered bot and 0.10 for a person.
         *
         * This is still the WEAKEST rule here, and it is secondary by design: five
         * samples cannot support a strong regularity claim, and an attacker who
         * adds deliberate jitter defeats it outright. The `bot jittered` row in
         * captcha_channel_results.txt is exactly that attacker, and it passes.
         * {@link #MIN_HOP_WALL_MS} is the load-bearing hard signal.
         */
        static final double MIN_HOP_WALL_MAD = 0.02;

        /**
         * The client's trace may not claim MORE elapsed time than the challenge
         * was open for. This is asymmetric on purpose, and the asymmetry is the
         * whole point:
         *
         *  - Shorter than the server window is NORMAL. An honest trace starts at
         *    the first cursor sample after a prompt appears and ends at the last
         *    one before the final click, so it is always missing the reaction
         *    time at the front and the click latency at the back.
         *  - Longer than the server window is IMPOSSIBLE. A client cannot have
         *    produced cursor samples spanning more time than the challenge
         *    existed. Clock drift over a few seconds is negligible and network
         *    latency only ever pushes this the other way.
         *
         * So an overrun is not "suspicious", it is proof the timestamps were
         * written rather than measured -- the cheapest attack on any
         * client-trusted clock, and the reason the hard signals are server-side.
         */
        static final double MAX_TRACE_OVERRUN = 0.10;

        /**
         * A trace covering almost none of the server window means the client
         * barely reported any movement. Reported as a note, never a rejection --
         * a sparse mouse event stream is a plausible browser, not an attack.
         */
        static final double MIN_TRACE_COVERAGE = 0.15;

        /** Below this many samples the client trace tells us nothing. */
        static final int MIN_TRACE_POINTS = 40;
    }

    // ---------------- challenge state ----------------

    static final class Tile {
        final int k;
        final char ch;
        final int x, y;          // top-left, board coordinates
        final long seed;
        Tile(int k, char ch, int x, int y, long seed) {
            this.k = k; this.ch = ch; this.x = x; this.y = y; this.seed = seed;
        }
        int cx() { return x + TILE / 2; }
        int cy() { return y + TILE / 2; }
    }

    static final class Challenge {
        final String id;
        final List<Tile> tiles;
        final int[] sequence;            // tile indices, in the order they must be visited
        final long createdAt = System.currentTimeMillis();

        int hop = 0;                     // how many hops have been completed
        boolean failed = false;
        String failReason = null;
        boolean consumed = false;        // verify() may only be called once

        /** Server clock. revealNs[i] = when the server released hop i's prompt. */
        final long[] revealNs = new long[HOPS];
        /** Server clock. arriveNs[i] = when hop i's arrival POST was accepted. */
        final long[] arriveNs = new long[HOPS];

        Challenge(String id, List<Tile> tiles, int[] sequence) {
            this.id = id; this.tiles = tiles; this.sequence = sequence;
            this.revealNs[0] = System.nanoTime();   // hop 0 is revealed at issue time
        }

        int currentTarget() { return sequence[hop]; }
        boolean done() { return hop >= HOPS; }
    }

    private static final Map<String, Challenge> LIVE = new ConcurrentHashMap<>();
    private static final AtomicLong COUNTER = new AtomicLong();
    private static final Random SEEDER = new Random();

    // ---------------- issue ----------------

    /** Builds a fresh board: 12 distinct letters on a jittered grid, plus a 5-hop route. */
    static Challenge issue() {
        sweepExpired();

        long seed = SEEDER.nextLong();
        Random rng = new Random(seed);

        // 12 distinct letters
        StringBuilder pool = new StringBuilder(ALPHABET);
        List<Tile> tiles = new ArrayList<>();
        int cellW = BOARD_W / COLS, cellH = BOARD_H / ROWS;
        int slack = 14;   // jitter so the grid does not read as a grid
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                int pick = rng.nextInt(pool.length());
                char ch = pool.charAt(pick);
                pool.deleteCharAt(pick);
                int x = c * cellW + (cellW - TILE) / 2 + rng.nextInt(2 * slack + 1) - slack;
                int y = r * cellH + (cellH - TILE) / 2 + rng.nextInt(2 * slack + 1) - slack;
                tiles.add(new Tile(tiles.size(), ch, x, y, rng.nextLong()));
            }
        }

        // A route of HOPS distinct tiles. Consecutive tiles are kept apart so every
        // hop is a real movement rather than a nudge.
        int[] seq = new int[HOPS];
        List<Integer> remaining = new ArrayList<>();
        for (int i = 0; i < tiles.size(); i++) remaining.add(i);
        int prev = -1;
        for (int h = 0; h < HOPS; h++) {
            int chosen = -1;
            for (int attempt = 0; attempt < 40 && chosen < 0; attempt++) {
                int cand = remaining.get(rng.nextInt(remaining.size()));
                if (prev < 0 || dist(tiles.get(prev), tiles.get(cand)) >= 180) chosen = cand;
            }
            if (chosen < 0) chosen = remaining.get(rng.nextInt(remaining.size()));
            remaining.remove(Integer.valueOf(chosen));
            seq[h] = chosen;
            prev = chosen;
        }

        String id = Long.toHexString(System.nanoTime()) + "-" + COUNTER.incrementAndGet();
        Challenge ch = new Challenge(id, tiles, seq);
        LIVE.put(id, ch);
        return ch;
    }

    private static double dist(Tile a, Tile b) {
        double dx = a.cx() - b.cx(), dy = a.cy() - b.cy();
        return Math.sqrt(dx * dx + dy * dy);
    }

    static Challenge get(String id) {
        sweepExpired();
        return id == null ? null : LIVE.get(id);
    }

    private static void sweepExpired() {
        long now = System.currentTimeMillis();
        for (Iterator<Map.Entry<String, Challenge>> it = LIVE.entrySet().iterator(); it.hasNext(); ) {
            if (now - it.next().getValue().createdAt > TTL_MS) it.remove();
        }
    }

    // ---------------- arrive ----------------

    /**
     * The client claims it has reached tile {@code k}. The server validates the
     * tile against its own route, stamps the arrival on its own clock, and -- if
     * more hops remain -- stamps the release of the next prompt.
     *
     * @return a JSON body describing what the client may now see.
     */
    static String arrive(Challenge ch, int k) {
        long now = System.nanoTime();
        if (ch.failed) return "{\"state\":\"failed\",\"reason\":" + jsonStr(ch.failReason) + "}";
        if (ch.done()) return "{\"state\":\"complete\",\"hops\":" + HOPS + "}";

        if (k != ch.currentTarget()) {
            ch.failed = true;
            ch.failReason = "wrong tile at hop " + (ch.hop + 1);
            return "{\"state\":\"failed\",\"reason\":" + jsonStr(ch.failReason) + "}";
        }

        ch.arriveNs[ch.hop] = now;
        ch.hop++;

        if (ch.done()) {
            return "{\"state\":\"complete\",\"hops\":" + HOPS + "}";
        }
        ch.revealNs[ch.hop] = System.nanoTime();     // the reveal instant, server clock
        return "{\"state\":\"next\",\"hop\":" + ch.hop + ",\"of\":" + HOPS + "}";
    }

    // ---------------- verify ----------------

    static final class Verdict {
        boolean pass;
        final List<String> flags = new ArrayList<>();
        final List<String> notes = new ArrayList<>();
        long[] hopWallMs = new long[0];
        /** median absolute deviation of the hop times, over their median. */
        double hopWallMad = Double.NaN;
        long totalWallMs = 0;
        /** client-claimed trace span / server-timed window. &gt;1 is impossible. */
        double traceCoverage = Double.NaN;
        int tracePoints = 0;
        int corrections = -1;
    }

    /**
     * Scores a completed challenge. Hard signals come from the server clock; soft
     * signals from the submitted trace, and are only trusted once the trace's own
     * span has been reconciled with the server-timed window.
     *
     * @param trace client samples as [t_ms, x, y] in board coordinates, may be null
     */
    static Verdict verify(Challenge ch, double[][] trace) {
        Verdict v = new Verdict();

        if (ch.failed) { v.pass = false; v.flags.add("challenge failed: " + ch.failReason); return v; }
        if (!ch.done()) { v.pass = false; v.flags.add("challenge incomplete (" + ch.hop + "/" + HOPS + " hops)"); return v; }
        if (ch.consumed) { v.pass = false; v.flags.add("challenge already verified -- replay refused"); return v; }
        ch.consumed = true;

        // ----- HARD: server clock only -----
        v.hopWallMs = new long[HOPS];
        long worst = Long.MAX_VALUE;
        for (int i = 0; i < HOPS; i++) {
            v.hopWallMs[i] = Math.round((ch.arriveNs[i] - ch.revealNs[i]) / 1e6);
            worst = Math.min(worst, v.hopWallMs[i]);
        }
        v.totalWallMs = Math.round((ch.arriveNs[HOPS - 1] - ch.revealNs[0]) / 1e6);

        v.hopWallMad = relativeMad(v.hopWallMs);

        if (worst < Rules.MIN_HOP_WALL_MS) {
            v.flags.add("hop completed in " + worst + " ms -- below the "
                    + Rules.MIN_HOP_WALL_MS + " ms floor for reading a glyph and moving to it");
        }
        if (v.hopWallMad < Rules.MIN_HOP_WALL_MAD) {
            v.flags.add(String.format("hop timing is near-constant (MAD %.4f < %.2f of the median)"
                    + " -- consistent with a fixed programmed delay", v.hopWallMad, Rules.MIN_HOP_WALL_MAD));
        }

        // ----- SOFT: client trace, only after reconciliation -----
        if (trace == null || trace.length < Rules.MIN_TRACE_POINTS) {
            v.tracePoints = trace == null ? 0 : trace.length;
            v.notes.add("trace too sparse to corroborate (" + v.tracePoints
                    + " < " + Rules.MIN_TRACE_POINTS + " samples) -- hard signals only");
        } else {
            v.tracePoints = trace.length;
            double claimed = trace[trace.length - 1][0] - trace[0][0];
            v.traceCoverage = v.totalWallMs > 0 ? claimed / (double) v.totalWallMs : 0;
            double overrun = v.traceCoverage - 1.0;

            if (overrun > Rules.MAX_TRACE_OVERRUN) {
                v.flags.add(String.format(
                        "submitted trace spans %.0f ms but the challenge was only open for %d ms"
                        + " -- a client cannot record more time than existed, so these"
                        + " timestamps were written, not measured", claimed, v.totalWallMs));
            } else {
                if (v.traceCoverage < Rules.MIN_TRACE_COVERAGE) {
                    v.notes.add(String.format("trace covers only %.0f%% of the window"
                            + " -- little movement reported, corroboration is weak",
                            v.traceCoverage * 100));
                }
                v.corrections = countCorrections(trace);
                if (v.corrections == 0) {
                    v.notes.add("no corrective submovements found -- weak signal, not decisive");
                }
            }
        }

        v.pass = v.flags.isEmpty();
        return v;
    }

    /**
     * Median absolute deviation of the hop times, divided by their median.
     * Robust: one outlying hop -- a scheduling hiccup, a moment's hesitation --
     * moves it barely at all, which is exactly why it replaced the coefficient of
     * variation here.
     */
    private static double relativeMad(long[] xs) {
        if (xs.length == 0) return Double.NaN;
        double med = median(xs);
        if (med <= 0) return 0;
        long[] dev = new long[xs.length];
        for (int i = 0; i < xs.length; i++) dev[i] = Math.abs(xs[i] - Math.round(med));
        return median(dev) / med;
    }

    private static double median(long[] xs) {
        long[] c = xs.clone();
        java.util.Arrays.sort(c);
        int n = c.length;
        return n % 2 == 1 ? c[n / 2] : (c[n / 2 - 1] + c[n / 2]) / 2.0;
    }

    /**
     * Counts direction reversals in the speed profile that look like corrective
     * submovements: a person acquiring a target decelerates, undershoots or
     * overshoots, then makes one or more small corrections. Reported for
     * inspection only -- it never fails a challenge on its own, because it is a
     * client-side signal and the hypothesis behind it is untested on this data
     * (Tier 6 is the cautionary tale for trusting an assumed motor mechanism).
     */
    private static int countCorrections(double[][] trace) {
        int n = trace.length;
        if (n < 6) return 0;
        double[] speed = new double[n - 1];
        for (int i = 1; i < n; i++) {
            double dt = Math.max(1.0, trace[i][0] - trace[i - 1][0]);
            double dx = trace[i][1] - trace[i - 1][1], dy = trace[i][2] - trace[i - 1][2];
            speed[i - 1] = Math.sqrt(dx * dx + dy * dy) / dt;
        }
        double peak = 0;
        for (double s : speed) peak = Math.max(peak, s);
        if (peak <= 0) return 0;

        // a "correction" = a local speed minimum near zero followed by a small
        // re-acceleration -- i.e. the cursor stopped, then moved again, without
        // that being a whole new hop
        int count = 0;
        double lowGate = peak * 0.08, riseGate = peak * 0.18;
        boolean settled = false;
        for (double s : speed) {
            if (s < lowGate) settled = true;
            else if (settled && s > riseGate) { count++; settled = false; }
        }
        return count;
    }

    // ---------------- server-side glyph rendering ----------------

    private static final String[] FACES = { Font.SANS_SERIF, Font.SERIF, Font.MONOSPACED };

    /** Each tile's face is fixed by its own seed. */
    private static int tileFace(Tile t) {
        return Math.floorMod((int) (t.seed >>> 17), FACES.length);
    }

    /** The tile glyph. Rendered here so the letter never reaches the client as text. */
    static byte[] renderTile(Challenge ch, int k) throws IOException {
        Tile t = ch.tiles.get(k);
        return renderGlyph(t.ch, t.seed, tileFace(t), TILE, TILE, false);
    }

    /**
     * The prompt glyph for hop {@code i}. Also an image -- the target is not
     * scrapable either.
     *
     * The prompt is rendered in a DIFFERENT typeface from its own tile, with an
     * independent warp. Without that, prompt and tile are two renders of the same
     * letter in the same face at the same weight, and an attacker skips reading
     * anything: fetch the prompt, fetch all twelve tiles, keep the nearest match.
     * A face change breaks that shortcut.
     *
     * It does not defeat a shape matcher or an OCR model, and it is not meant to.
     * The perceptual layer is a speed bump; the sequential reveal and the server
     * clock are the parts this design actually rests on.
     */
    static byte[] renderPrompt(Challenge ch, int i) throws IOException {
        Tile t = ch.tiles.get(ch.sequence[i]);
        int face = (tileFace(t) + 1 + Math.floorMod((int) t.seed, FACES.length - 1)) % FACES.length;
        return renderGlyph(t.ch, t.seed * 31 + 17, face, 72, 72, true);
    }

    /**
     * Draws one distorted character in the given typeface: random rotation and
     * shear, then a two-axis sine warp, speckle and a crossing stroke.
     *
     * This defeats naive template matching, and -- because a prompt is drawn in a
     * different face from its tile -- it also defeats matching the prompt image
     * straight against the twelve tile images. It does NOT defeat a shape matcher
     * or an OCR model, and is not intended to. The distortion is a speed bump
     * that makes each hop cost the attacker a real inference call inside the
     * server's timing window; the sequential reveal and the server clock are what
     * the design actually rests on.
     */
    private static byte[] renderGlyph(char ch, long seed, int faceIdx, int w, int h, boolean prompt)
            throws IOException {
        Random rng = new Random(seed);

        BufferedImage src = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = src.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);

        g.setColor(new Color(0, 0, 0, 0));
        g.fillRect(0, 0, w, h);

        // Always bold and large: the warp below eats thin strokes, and an
        // illegible CAPTCHA is a broken CAPTCHA.
        int size = (int) (h * (prompt ? 0.74 : 0.70)) + rng.nextInt(5);
        Font font = new Font(FACES[faceIdx], Font.BOLD, size);

        AffineTransform at = new AffineTransform();
        at.rotate((rng.nextDouble() - 0.5) * 0.42);
        at.shear((rng.nextDouble() - 0.5) * 0.24, (rng.nextDouble() - 0.5) * 0.10);
        g.setFont(font.deriveFont(at));

        java.awt.FontMetrics fm = g.getFontMetrics();
        String s = String.valueOf(ch);
        int tw = fm.stringWidth(s);
        int bx = (w - tw) / 2 + rng.nextInt(9) - 4;
        int by = (h + fm.getAscent() - fm.getDescent()) / 2 + rng.nextInt(7) - 3;

        g.setColor(prompt ? new Color(0xF4, 0xF8, 0xFF) : new Color(0xEC, 0xF2, 0xFC));
        g.drawString(s, bx, by);

        // one crossing stroke, thin enough not to be mistaken for the glyph
        g.setStroke(new BasicStroke(1.4f));
        g.setColor(new Color(255, 255, 255, 52));
        g.drawLine(rng.nextInt(w), rng.nextInt(h), rng.nextInt(w), rng.nextInt(h));
        g.dispose();

        // Two-axis sine warp. Long wavelength, small amplitude: enough to break
        // template matching and pixel-diffing between the prompt and its tile,
        // while leaving the letter obvious to a person.
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        double ax = 1.6 + rng.nextDouble() * 1.3, lx = 30 + rng.nextDouble() * 22, px = rng.nextDouble() * 6.28;
        double ay = 1.4 + rng.nextDouble() * 1.2, ly = 27 + rng.nextDouble() * 20, py = rng.nextDouble() * 6.28;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int sx = (int) Math.round(x + ax * Math.sin(2 * Math.PI * y / lx + px));
                int sy = (int) Math.round(y + ay * Math.sin(2 * Math.PI * x / ly + py));
                if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
                out.setRGB(x, y, src.getRGB(sx, sy));
            }
        }

        // light speckle
        Graphics2D go = out.createGraphics();
        for (int i = 0; i < w * h / 160; i++) {
            go.setColor(new Color(255, 255, 255, 24 + rng.nextInt(38)));
            go.fillRect(rng.nextInt(w), rng.nextInt(h), 1, 1);
        }
        go.dispose();

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ImageIO.write(out, "png", bos);
        return bos.toByteArray();
    }

    // ---------------- JSON (same hand-rolled style as DemoServer) ----------------

    static String boardJson(Challenge ch) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"id\":").append(jsonStr(ch.id))
          .append(",\"boardW\":").append(BOARD_W)
          .append(",\"boardH\":").append(BOARD_H)
          .append(",\"tile\":").append(TILE)
          .append(",\"hops\":").append(HOPS)
          .append(",\"tiles\":[");
        for (int i = 0; i < ch.tiles.size(); i++) {
            Tile t = ch.tiles.get(i);
            if (i > 0) sb.append(',');
            sb.append("{\"k\":").append(t.k).append(",\"x\":").append(t.x).append(",\"y\":").append(t.y).append('}');
        }
        sb.append("]}");
        return sb.toString();
    }

    static String verdictJson(Verdict v) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"pass\":").append(v.pass)
          .append(",\"totalWallMs\":").append(v.totalWallMs)
          .append(",\"hopWallMs\":[");
        for (int i = 0; i < v.hopWallMs.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(v.hopWallMs[i]);
        }
        sb.append("],\"hopWallMad\":").append(fmt(v.hopWallMad))
          .append(",\"traceCoverage\":").append(fmt(v.traceCoverage))
          .append(",\"tracePoints\":").append(v.tracePoints)
          .append(",\"corrections\":").append(v.corrections)
          .append(",\"flags\":[");
        for (int i = 0; i < v.flags.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append(jsonStr(v.flags.get(i)));
        }
        sb.append("],\"notes\":[");
        for (int i = 0; i < v.notes.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append(jsonStr(v.notes.get(i)));
        }
        sb.append("],\"calibrated\":false}");
        return sb.toString();
    }

    private static String fmt(double d) {
        return (Double.isNaN(d) || Double.isInfinite(d)) ? "null" : String.format("%.4f", d);
    }

    static String jsonStr(String s) {
        if (s == null) return "null";
        StringBuilder sb = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"':  sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
            }
        }
        return sb.append('"').toString();
    }
}
