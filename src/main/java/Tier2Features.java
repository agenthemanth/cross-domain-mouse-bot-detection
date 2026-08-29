import weka.core.Attribute;

import java.util.ArrayList;
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

    public enum Mode { BASELINE, AUGMENTED, SCALEFREE }

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

        if (mode == Mode.AUGMENTED) {
            double[] out = new double[base.length + t2.length];
            System.arraycopy(base, 0, out, 0, base.length);
            System.arraycopy(t2, 0, out, base.length, t2.length);
            return out;
        }
        // SCALEFREE: num_points, duration_ms, path_efficiency (base[0], base[1], base[6]) + t2
        double[] out = new double[3 + t2.length];
        out[0] = base[0];
        out[1] = base[1];
        out[2] = base[6];
        System.arraycopy(t2, 0, out, 3, t2.length);
        return out;
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
        double p90 = percentile(v, 0.90);
        double p50 = percentile(v, 0.50);
        double p90p50 = p90 / (p50 + EPS);

        double[] out = {
                velocityCv, accelToVel, jerkToVel, timeToPeakRatio, accelFraction,
                dirReversalRate, meanTurn, stdTurn, meanCurv, pauseRatio, p90p50
        };
        for (double x : out) if (Double.isNaN(x) || Double.isInfinite(x)) return null;
        return out;
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
