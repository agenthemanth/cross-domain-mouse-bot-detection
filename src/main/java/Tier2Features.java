import weka.core.Attribute;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * TIER 2 feature extraction for cross-domain mouse-dynamics bot detection.
 *
 * WHY THIS EXISTS: Tier 1 established that the DELBOT-trained RandomForest does
 * not transfer to the Balabit domain at all (pooled bot ROC-AUC 0.53, TPR at
 * <=1% FPR = 0.00%, and it over-flags ~9.85% of real Balabit humans). The
 * diagnosis: the 7 Tier-1 features are dominated by ABSOLUTE velocity /
 * acceleration / jerk magnitude. DELBOT taught the model "fast + high-variance
 * = bot" (its bots are pynput / GAN tools); in the Balabit admin-work domain
 * that same fast, jerky signature belongs to the HUMANS. The absolute kinematic
 * scale is domain-specific and does not carry across.
 *
 * Tier 2 hypothesis: SCALE-FREE distribution-shape and path-geometry features
 * transfer where absolute magnitudes do not. This class computes three nested
 * feature sets, selected by {@link Mode}:
 *
 *   BASELINE  - the exact Tier-1 seven. Delegates to
 *               BalabitValidationPipeline.computeFeatures, so a BASELINE run of
 *               Tier2CrossDomainEval must reproduce BalabitCrossDomainEval's
 *               numbers bit-for-bit. This is the port sanity check.
 *
 *   AUGMENTED - the seven, plus 11 domain-bridging features (velocity-profile
 *               shape, turning geometry, pause structure, scale-free dispersion).
 *               Tests "does ADDING better features help while keeping the ones
 *               that already don't transfer?"
 *
 *   SCALEFREE - drops mean_velocity / std_velocity / mean_acceleration /
 *               mean_jerk (the absolute-magnitude features shown not to
 *               transfer) and keeps only dimensionless quantities:
 *               num_points, duration_ms, path_efficiency + the 11 Tier-2
 *               features. Tests the core hypothesis directly.
 *
 * All features are computed from raw (t_ms, x, y) samples by ONE code path for
 * both DELBOT training and Balabit evaluation, after the same tied-timestamp
 * collapse used everywhere else in the project.
 */
public final class Tier2Features {

    public enum Mode {
        BASELINE, AUGMENTED, SCALEFREE,
        /** AUGMENTED + 2 temporal-ordering features (Tier 6, targets the AdversarialBotSynthesizer). */
        AUGMENTED_SEQ,
        /** SCALEFREE + the same 2 temporal-ordering features. */
        SCALEFREE_SEQ
    }

    private Tier2Features() {}

    // ---- the 11 Tier-2 feature names, in order ----
    static final String[] TIER2_NAMES = {
            "velocity_cv",            // std_v / mean_v                 -- scale-free dispersion
            "accel_to_vel_ratio",     // mean|a| / mean_v               -- scale-free "jerkiness"
            "jerk_to_vel_ratio",      // mean|j| / mean_v
            "time_to_peak_vel_ratio", // argmax_i(v_i) / (#v - 1)  in [0,1]  -- ballistic profile
            "accel_fraction",         // share of steps with v still increasing
            "dir_reversal_rate",      // share of turns with turning angle > 90 deg
            "mean_turning_angle",     // radians, [0, pi]              -- path wobble
            "std_turning_angle",
            "mean_curvature",         // mean(turning_angle / step_dist)
            "pause_ratio",            // share of steps with v < 5% of mean_v
            "vel_p90_p50_ratio"       // 90th / 50th percentile of v   -- scale-free tail weight
    };

    /**
     * TIER 6 -- 2 temporal-ORDERING features. All 18 AUGMENTED features are
     * permutation-invariant on the per-step speed sequence, so the
     * {@link AdversarialBotSynthesizer} (which reuses the victim chunk's own
     * (step-distance, dt) pairs in SHUFFLED order) matches every one of them by
     * construction. These two read the ORDER the earlier features throw away:
     *
     *   velocity_lag1_autocorr  -- Pearson autocorrelation of the speed series at
     *       lag 1, over moving steps.
     *   velocity_step_roughness -- mean|v[i]-v[i-1]| / mean(v) over moving steps.
     *
     * Both dimensionless; < 4 moving steps -> (0, 0).
     *
     * MEASURED OUTCOME (Tier6FeatureProbe, tier6_feature_probe_results.txt): the
     * autocorrelation idea largely FAILS on Balabit gap-chunks -- human step-speed
     * autocorr is ~0 (median -0.014, i.e. the series is near-i.i.d. at this 13.6s /
     * 66-pt multi-action segmentation), statistically indistinguishable from the
     * shuffled bot. Single-feature human-vs-adversarial AUC: lag1_autocorr 0.45
     * (chance), step_roughness 0.57 (weak, correct direction). Net Tier-4 effect of
     * adding both to AUGMENTED: advAUC +0.0014 (real vs 0.0003 RF-seed spread),
     * ~+1.5pp recall@<=1%FPR, ~no human-FPR cost -- a marginal, mostly-step_roughness
     * gain, kept because it is free, NOT a fix. lag1_autocorr does cleanly separate
     * the SMOOTH naive Bezier/min-jerk bots (autocorr ~0.68), which are already caught.
     */
    static final String[] SEQ_NAMES = { "velocity_lag1_autocorr", "velocity_step_roughness" };

    /** Column names (excluding the class attribute) for the given mode. */
    static String[] featureNames(Mode mode) {
        String[] base = {"num_points", "duration_ms", "mean_velocity", "std_velocity",
                "mean_acceleration", "mean_jerk", "path_efficiency"};
        switch (mode) {
            case BASELINE:
                return base;
            case AUGMENTED: {
                String[] out = new String[base.length + TIER2_NAMES.length];
                System.arraycopy(base, 0, out, 0, base.length);
                System.arraycopy(TIER2_NAMES, 0, out, base.length, TIER2_NAMES.length);
                return out;
            }
            case SCALEFREE: {
                String[] keep = {"num_points", "duration_ms", "path_efficiency"};
                String[] out = new String[keep.length + TIER2_NAMES.length];
                System.arraycopy(keep, 0, out, 0, keep.length);
                System.arraycopy(TIER2_NAMES, 0, out, keep.length, TIER2_NAMES.length);
                return out;
            }
            case AUGMENTED_SEQ:
                return concat(featureNames(Mode.AUGMENTED), SEQ_NAMES);
            case SCALEFREE_SEQ:
                return concat(featureNames(Mode.SCALEFREE), SEQ_NAMES);
        }
        throw new IllegalStateException();
    }

    static ArrayList<Attribute> schema(Mode mode) {
        ArrayList<Attribute> attrs = new ArrayList<>();
        for (String n : featureNames(mode)) attrs.add(new Attribute(n));
        ArrayList<String> classVals = new ArrayList<>();
        classVals.add("0");
        classVals.add("1");
        attrs.add(new Attribute("is_bot", classVals));
        return attrs;
    }

    /**
     * Feature vector for one trajectory (order matches {@link #featureNames}).
     * {@code pointsMs} are raw (t_ms, x, y), ascending in t. Returns null for a
     * trajectory that cannot yield stable features (mirrors the Tier-1 contract).
     */
    static double[] compute(List<double[]> pointsMs, Mode mode) {
        // Tier-1 seven, from the exact Tier-1 code path (does its own collapse + NaN reject).
        double[] base = BalabitValidationPipeline.computeFeatures(pointsMs);
        if (base == null) return null;
        if (mode == Mode.BASELINE) return base;

        double[] t2 = tier2(pointsMs);
        if (t2 == null) return null;

        boolean seq = (mode == Mode.AUGMENTED_SEQ || mode == Mode.SCALEFREE_SEQ);
        double[] sq = seq ? seqFeatures(pointsMs) : null;
        if (seq && sq == null) return null;

        double[] head;
        if (mode == Mode.AUGMENTED || mode == Mode.AUGMENTED_SEQ) {
            head = new double[base.length + t2.length];
            System.arraycopy(base, 0, head, 0, base.length);
            System.arraycopy(t2, 0, head, base.length, t2.length);
        } else {
            // SCALEFREE / SCALEFREE_SEQ: num_points, duration_ms, path_efficiency (base[0,1,6]) + t2
            head = new double[3 + t2.length];
            head[0] = base[0];
            head[1] = base[1];
            head[2] = base[6];
            System.arraycopy(t2, 0, head, 3, t2.length);
        }
        if (!seq) return head;
        double[] out = Arrays.copyOf(head, head.length + sq.length);
        System.arraycopy(sq, 0, out, head.length, sq.length);
        return out;
    }

    private static String[] concat(String[] a, String[] b) {
        String[] out = new String[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /**
     * The 2 temporal-ordering features (see {@link #SEQ_NAMES}). Computed on the
     * per-step speed sequence over MOVING steps, mirroring {@link #velocityTailRatio}.
     */
    private static double[] seqFeatures(List<double[]> pointsMs) {
        List<double[]> p = BalabitValidationPipeline.collapseToDistinctTimestamps(pointsMs);
        int n = p.size();
        if (n < 4) return null;

        // Speed sequence over MOVING steps only (matches tier2() / velocityTailRatio).
        // Balabit chunks carry a median ~9% and up to ~58% zero-displacement steps; left
        // in, the lag-1 autocorrelation is dominated by the run pattern of zeros for BOTH
        // humans and the shuffled bot, which flattens the very discrimination this targets.
        double[] all = new double[n - 1];
        int moving = 0;
        for (int i = 1; i < n; i++) {
            double dt = Math.max(p.get(i)[0] - p.get(i - 1)[0], MIN_DT_MS);
            double ddx = p.get(i)[1] - p.get(i - 1)[1];
            double ddy = p.get(i)[2] - p.get(i - 1)[2];
            double s = Math.sqrt(ddx * ddx + ddy * ddy) / dt;
            all[i - 1] = s;
            if (s > 0.0) moving++;
        }
        if (moving < 4) return new double[]{0.0, 0.0};
        double[] v = new double[moving];
        int w = 0;
        for (double s : all) if (s > 0.0) v[w++] = s;

        double meanV = mean(v);
        if (meanV <= EPS) return new double[]{0.0, 0.0};

        double num = 0.0, den = 0.0;
        for (int i = 0; i < v.length; i++) {
            double d = v[i] - meanV;
            den += d * d;
            if (i > 0) num += d * (v[i - 1] - meanV);
        }
        double lag1 = den > EPS ? num / den : 0.0;

        double roughSum = 0.0;
        for (int i = 1; i < v.length; i++) roughSum += Math.abs(v[i] - v[i - 1]);
        double roughness = roughSum / ((v.length - 1) * meanV);

        if (Double.isNaN(lag1) || Double.isInfinite(lag1)
                || Double.isNaN(roughness) || Double.isInfinite(roughness)) return null;
        return new double[]{lag1, roughness};
    }

    // ---------------- the 11 Tier-2 features ----------------

    private static final double MIN_DT_MS = 1.0; // same physical floor as BalabitValidationPipeline
    private static final double EPS = 1e-9;

    private static double[] tier2(List<double[]> pointsMs) {
        List<double[]> p = BalabitValidationPipeline.collapseToDistinctTimestamps(pointsMs);
        int n = p.size();
        if (n < 4) return null; // need >=3 velocities for turning angles / accel structure

        int nSteps = n - 1;
        double[] v = new double[nSteps];
        double[] stepDist = new double[nSteps];
        double[] dx = new double[nSteps];
        double[] dy = new double[nSteps];
        double totalDist = 0.0;

        for (int i = 1; i < n; i++) {
            double dt = Math.max(p.get(i)[0] - p.get(i - 1)[0], MIN_DT_MS);
            double ddx = p.get(i)[1] - p.get(i - 1)[1];
            double ddy = p.get(i)[2] - p.get(i - 1)[2];
            double dist = Math.sqrt(ddx * ddx + ddy * ddy);
            dx[i - 1] = ddx;
            dy[i - 1] = ddy;
            stepDist[i - 1] = dist;
            v[i - 1] = dist / dt;
            totalDist += dist;
        }

        double meanV = mean(v);
        double stdV = std(v, meanV);

        // acceleration = first difference of velocity (Tier-1 convention); jerk = first diff of accel
        double[] a = diff(v);
        double[] j = diff(a);
        double meanAbsA = meanAbs(a);
        double meanAbsJ = meanAbs(j);

        // turning angle between consecutive displacement vectors (only where both steps moved)
        List<Double> turns = new ArrayList<>();
        List<Double> curvature = new ArrayList<>();
        int reversals = 0;
        for (int i = 1; i < nSteps; i++) {
            if (stepDist[i] < EPS || stepDist[i - 1] < EPS) continue;
            double dot = dx[i - 1] * dx[i] + dy[i - 1] * dy[i];
            double cross = dx[i - 1] * dy[i] - dy[i - 1] * dx[i];
            double ang = Math.atan2(Math.abs(cross), dot); // [0, pi]
            turns.add(ang);
            curvature.add(ang / (stepDist[i] + EPS));
            if (ang > Math.PI / 2.0) reversals++;
        }

        double velocityCv = stdV / (meanV + EPS);
        double accelToVel = meanAbsA / (meanV + EPS);
        double jerkToVel = meanAbsJ / (meanV + EPS);
        double timeToPeakRatio = (double) argmax(v) / (nSteps - 1);
        double accelFraction = countPositive(a) / (double) Math.max(a.length, 1);
        double dirReversalRate = turns.isEmpty() ? 0.0 : reversals / (double) turns.size();
        double meanTurn = turns.isEmpty() ? 0.0 : meanList(turns);
        double stdTurn = turns.isEmpty() ? 0.0 : stdList(turns, meanTurn);
        double meanCurv = curvature.isEmpty() ? 0.0 : meanList(curvature);
        double pauseRatio = countBelow(v, 0.05 * meanV) / (double) nSteps;
        double p90p50 = velocityTailRatio(v);

        double[] out = {
                velocityCv, accelToVel, jerkToVel, timeToPeakRatio, accelFraction,
                dirReversalRate, meanTurn, stdTurn, meanCurv, pauseRatio, p90p50
        };
        for (double x : out) if (Double.isNaN(x) || Double.isInfinite(x)) return null;
        return out;
    }

    /**
     * 90th/50th percentile of speed, computed over MOVING steps only.
     *
     * BUG FIXED 2026-08-29: the original computed percentile(v,.90)/(percentile(v,.50)+EPS)
     * over ALL steps. Balabit logs many consecutive events at the SAME position, so
     * 1.51% of chunks have >50% zero-displacement steps -> the median speed is exactly 0
     * -> the ratio became p90/1e-9, i.e. 1e8-1e9. That handful of chunks dragged the
     * feature's mean over real humans to ~2.3 MILLION while its median was a sane 11.7,
     * making the column effectively a "is this chunk mostly idle" indicator flag rather
     * than the intended scale-free speed-tail measure -- and duplicating pause_ratio.
     *
     * Restricting to steps that actually moved makes the denominator positive by
     * construction and keeps the semantics ("when the cursor IS moving, how heavy is the
     * speed tail?"). The idle share is already carried by pause_ratio, so no information
     * is lost. A fully stationary chunk has no tail: return 1.0.
     */
    private static double velocityTailRatio(double[] v) {
        int moving = 0;
        for (double s : v) if (s > 0.0) moving++;
        if (moving == 0) return 1.0;
        double[] m = new double[moving];
        int w = 0;
        for (double s : v) if (s > 0.0) m[w++] = s;
        double p50 = percentile(m, 0.50); // > 0 by construction
        double p90 = percentile(m, 0.90);
        return p90 / p50;
    }

    // ---------------- small numeric helpers ----------------

    private static double mean(double[] x) {
        double s = 0;
        for (double v : x) s += v;
        return x.length == 0 ? 0 : s / x.length;
    }

    private static double meanAbs(double[] x) {
        double s = 0;
        for (double v : x) s += Math.abs(v);
        return x.length == 0 ? 0 : s / x.length;
    }

    private static double std(double[] x, double m) {
        if (x.length == 0) return 0;
        double s = 0;
        for (double v : x) s += (v - m) * (v - m);
        return Math.sqrt(s / x.length);
    }

    private static double meanList(List<Double> x) {
        double s = 0;
        for (double v : x) s += v;
        return x.isEmpty() ? 0 : s / x.size();
    }

    private static double stdList(List<Double> x, double m) {
        if (x.isEmpty()) return 0;
        double s = 0;
        for (double v : x) s += (v - m) * (v - m);
        return Math.sqrt(s / x.size());
    }

    private static double[] diff(double[] x) {
        if (x.length < 2) return new double[0];
        double[] d = new double[x.length - 1];
        for (int i = 1; i < x.length; i++) d[i - 1] = x[i] - x[i - 1];
        return d;
    }

    private static int argmax(double[] x) {
        int idx = 0;
        for (int i = 1; i < x.length; i++) if (x[i] > x[idx]) idx = i;
        return idx;
    }

    private static int countPositive(double[] x) {
        int c = 0;
        for (double v : x) if (v > 0) c++;
        return c;
    }

    private static int countBelow(double[] x, double t) {
        int c = 0;
        for (double v : x) if (v < t) c++;
        return c;
    }

    /** Linear-interpolated percentile, q in [0,1]. */
    private static double percentile(double[] x, double q) {
        if (x.length == 0) return 0;
        double[] s = x.clone();
        java.util.Arrays.sort(s);
        if (s.length == 1) return s[0];
        double pos = q * (s.length - 1);
        int lo = (int) Math.floor(pos);
        int hi = (int) Math.ceil(pos);
        if (lo == hi) return s[lo];
        return s[lo] + (pos - lo) * (s[hi] - s[lo]);
    }
}
