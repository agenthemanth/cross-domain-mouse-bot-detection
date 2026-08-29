import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generates synthetic BOT mouse trajectories inside the Balabit domain, by
 * replaying real Balabit human sessions as machine-generated movement.
 *
 * WHY THIS EXISTS: every number the project reported before this class was
 * one-sided -- false positive rate on all-human datasets. Cross-domain bot
 * recall (TPR) was never measured, which is precisely the side the literature
 * shows collapsing under domain shift (Iliou et al. 2021: web-dataset bot
 * recall 1.00 -> 0.45 against evasive bots; Acien et al. 2022: real-only
 * training drops to ~60% accuracy). Without a bot set in the Balabit domain
 * there is no TPR to report and no ROC/EER/AUC to compare against literature.
 *
 * MATCHED-PAIR DESIGN: each synthetic bot trajectory reuses its source human
 * chunk's start point, end point, point count, and total duration. Only the
 * PATH SHAPE and the TIMING DISTRIBUTION are machine-generated. This means
 * num_points and duration_ms are identical between a human chunk and its bot
 * twin, so the classifier cannot separate them on those two features at all
 * and is forced onto the genuine kinematic features (velocity, acceleration,
 * jerk, path efficiency). That makes this a deliberately HARDER and more
 * honest test than generating bots with free-running point counts, where a
 * Selenium-style step size would leak the label through num_points alone.
 *
 * BOT TYPES follow the sophistication ladder used in the literature -- Iliou
 * et al.'s "moderate" (browser fingerprint, no humanlike behaviour) vs
 * "advanced" (humanlike behaviour), and Acien et al.'s function-based
 * synthesis crossed with velocity profiles:
 *
 *   MODERATE_LINEAR            straight line, constant velocity, uniform
 *                              timing. Iliou's moderate bot ("step" of 1,
 *                              a continuous straight line).
 *   MODERATE_LINEAR_VP         straight line, but minimum-jerk velocity
 *                              profile (initial acceleration + final
 *                              deceleration). Acien et al. identify the
 *                              velocity profile as the single most
 *                              significant parameter when synthesising
 *                              trajectories that fool a detector.
 *   ADVANCED_BEZIER            cubic Bezier curved path (what humanlike
 *                              automation libraries such as ghost-cursor
 *                              actually emit), constant parameter velocity.
 *   ADVANCED_BEZIER_VP_JITTER  cubic Bezier + minimum-jerk velocity profile
 *                              + per-point pixel jitter + non-uniform event
 *                              timing. The evasive/humanlike case.
 *
 * LIMITATION TO STATE IN THE WRITE-UP: these are synthesised bots, not
 * captured real-world bot traffic. They are constructed from published
 * descriptions of bot movement models, so TPR against them is an upper bound
 * on what a real evasive adversary would allow -- notably it does NOT cover
 * GAN-generated trajectories, which are the strongest known evasion.
 */
public final class BalabitBotSynthesizer {

    public enum BotType {
        MODERATE_LINEAR,
        MODERATE_LINEAR_VP,
        ADVANCED_BEZIER,
        ADVANCED_BEZIER_VP_JITTER
    }

    /**
     * How the bot spends the source chunk's time budget.
     *
     * MATCHED_SPREAD - inter-event timing is (near-)uniform across the whole
     *   chunk duration. Tier-1 default. On Balabit this is unrealistically slow:
     *   gap-split chunks run a median ~13.6 s / 66 points, so the bot dribbles
     *   its move out over ~13 s and ends up looking like a very careful human.
     *
     * BURST - the bot completes ~85% of the motion inside a short burst
     *   (4-12% of the chunk duration) and the remaining points crawl to rest
     *   over the rest of the window. Produces the bimodal fast-then-idle
     *   velocity profile that a real click/drag automation actually emits,
     *   while still keeping point count, total duration and endpoints matched
     *   to the source human chunk (so those cannot leak the label).
     */
    public enum TimingModel { MATCHED_SPREAD, BURST }

    /** Perpendicular bow of the Bezier control points, as a fraction of end-to-end distance. */
    private static final double BEZIER_MIN_BOW = 0.08;
    private static final double BEZIER_MAX_BOW = 0.25;
    /** Std-dev of per-point coordinate jitter, in pixels. */
    private static final double JITTER_PX = 1.5;
    /** Relative std-dev applied to inter-event gaps for the jittered bot. */
    private static final double TIMING_JITTER = 0.30;

    private BalabitBotSynthesizer() {
    }

    /**
     * Builds a bot trajectory matched to {@code humanChunk}.
     *
     * @param humanChunk real Balabit points as (t_seconds, x, y), ascending in t
     * @param type       which bot movement model to emit
     * @param rng        seeded source of randomness, for reproducible runs
     * @return points as (t_seconds, x, y), same length / duration / endpoints as the input
     */
    public static List<double[]> synthesize(List<double[]> humanChunk, BotType type, Random rng) {
        return synthesize(humanChunk, type, TimingModel.MATCHED_SPREAD, rng);
    }

    /** As {@link #synthesize(List, BotType, Random)} but with an explicit {@link TimingModel}. */
    public static List<double[]> synthesize(List<double[]> humanChunk, BotType type,
                                            TimingModel timing, Random rng) {
        int n = humanChunk.size();
        if (n < 3) return null;

        double t0 = humanChunk.get(0)[0];
        double x0 = humanChunk.get(0)[1];
        double y0 = humanChunk.get(0)[2];
        double t1 = humanChunk.get(n - 1)[0];
        double x1 = humanChunk.get(n - 1)[1];
        double y1 = humanChunk.get(n - 1)[2];

        double duration = t1 - t0;
        if (duration <= 0) return null;

        boolean curved = (type == BotType.ADVANCED_BEZIER || type == BotType.ADVANCED_BEZIER_VP_JITTER);
        boolean velocityProfile = (type == BotType.MODERATE_LINEAR_VP || type == BotType.ADVANCED_BEZIER_VP_JITTER);
        boolean jitter = (type == BotType.ADVANCED_BEZIER_VP_JITTER);

        double dx = x1 - x0;
        double dy = y1 - y0;
        double dist = Math.hypot(dx, dy);

        // Control points for the curved variants. With a zero-length move there is no
        // perpendicular direction to bow along, so those degenerate to the straight path.
        double c1x = 0, c1y = 0, c2x = 0, c2y = 0;
        if (curved && dist > 1e-6) {
            double perpX = -dy / dist;
            double perpY = dx / dist;
            double bow1 = randomBow(rng) * dist;
            double bow2 = randomBow(rng) * dist;
            c1x = x0 + dx / 3.0 + perpX * bow1;
            c1y = y0 + dy / 3.0 + perpY * bow1;
            c2x = x0 + 2.0 * dx / 3.0 + perpX * bow2;
            c2y = y0 + 2.0 * dy / 3.0 + perpY * bow2;
        }

        double[] times;
        if (timing == TimingModel.BURST) {
            times = burstTimes(t0, duration, n, jitter, rng);
        } else {
            times = jitter ? jitteredTimes(t0, duration, n, rng) : uniformTimes(t0, duration, n);
        }

        List<double[]> bot = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            double u = (double) i / (n - 1);
            // Position advances along the path by s, while time advances uniformly by u.
            // Making s != u is exactly what creates an acceleration/deceleration profile.
            double s = velocityProfile ? minimumJerk(u) : u;

            double px;
            double py;
            if (curved && dist > 1e-6) {
                double mt = 1.0 - s;
                double a = mt * mt * mt;
                double b = 3.0 * mt * mt * s;
                double c = 3.0 * mt * s * s;
                double d = s * s * s;
                px = a * x0 + b * c1x + c * c2x + d * x1;
                py = a * y0 + b * c1y + c * c2y + d * y1;
            } else {
                px = x0 + dx * s;
                py = y0 + dy * s;
            }

            // Endpoints stay exact; jitter only perturbs the interior of the path.
            if (jitter && i > 0 && i < n - 1) {
                px += rng.nextGaussian() * JITTER_PX;
                py += rng.nextGaussian() * JITTER_PX;
            }

            // Real mouse hardware reports integer pixels. Emitting fractional coordinates
            // would hand the classifier a giveaway that no real bot would produce, and
            // would also understate the quantisation noise in the velocity signal.
            bot.add(new double[]{times[i], Math.rint(px), Math.rint(py)});
        }
        return bot;
    }

    /** Minimum-jerk position profile: smooth ease-in / ease-out, s(0)=0, s(1)=1. */
    private static double minimumJerk(double u) {
        return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u);
    }

    private static double randomBow(Random rng) {
        double magnitude = BEZIER_MIN_BOW + rng.nextDouble() * (BEZIER_MAX_BOW - BEZIER_MIN_BOW);
        return rng.nextBoolean() ? magnitude : -magnitude;
    }

    private static double[] uniformTimes(double t0, double duration, int n) {
        double[] times = new double[n];
        for (int i = 0; i < n; i++) {
            times[i] = t0 + duration * ((double) i / (n - 1));
        }
        return times;
    }

    /** Fraction of the trajectory's POINTS that land inside the burst window. */
    private static final double BURST_MOVE_FRACTION = 0.85;
    /** Burst window as a fraction of the source chunk's total duration (uniform in this range). */
    private static final double BURST_MIN_DURATION_FRAC = 0.04;
    private static final double BURST_MAX_DURATION_FRAC = 0.12;

    /**
     * Burst timing: the first ~85% of points are packed into a short window
     * ({@value BURST_MIN_DURATION_FRAC}-{@value BURST_MAX_DURATION_FRAC} of the
     * total duration) at the start of the chunk; the remaining points crawl to
     * rest across the rest of the window. times[0] == t0 and times[n-1] ==
     * t0 + duration exactly, so total duration stays matched. Strictly increasing.
     */
    private static double[] burstTimes(double t0, double duration, int n, boolean jitter, Random rng) {
        int kMove = Math.max(1, (int) Math.round(BURST_MOVE_FRACTION * (n - 1)));
        kMove = Math.min(kMove, n - 2 >= 1 ? n - 2 : n - 1); // leave at least one point for the tail when possible
        double burstFrac = BURST_MIN_DURATION_FRAC
                + rng.nextDouble() * (BURST_MAX_DURATION_FRAC - BURST_MIN_DURATION_FRAC);
        double burstLen = Math.max(duration * burstFrac, Math.min(duration, 1e-3));

        double[] times = new double[n];
        for (int i = 0; i < n; i++) {
            double tt;
            if (i <= kMove) {
                double f = kMove == 0 ? 0.0 : (double) i / kMove;
                if (jitter && i > 0 && i < kMove) f = clamp01(f + rng.nextGaussian() * (0.15 / kMove));
                tt = t0 + burstLen * f;
            } else {
                double f = (double) (i - kMove) / ((n - 1) - kMove);
                tt = t0 + burstLen + (duration - burstLen) * f;
            }
            times[i] = tt;
        }
        // enforce strict monotonicity against any jitter overshoot, then pin the endpoint
        for (int i = 1; i < n; i++) {
            if (times[i] <= times[i - 1]) times[i] = Math.nextUp(times[i - 1]);
        }
        times[0] = t0;
        times[n - 1] = t0 + duration;
        return times;
    }

    private static double clamp01(double v) {
        return v < 0 ? 0 : (v > 1 ? 1 : v);
    }

    /**
     * Non-uniform event timing, renormalised so the total duration still matches the
     * source human chunk exactly. Gaps are clamped strictly positive to keep times
     * monotonically increasing.
     */
    private static double[] jitteredTimes(double t0, double duration, int n, Random rng) {
        double[] gaps = new double[n - 1];
        double sum = 0.0;
        for (int i = 0; i < gaps.length; i++) {
            gaps[i] = Math.max(0.05, 1.0 + rng.nextGaussian() * TIMING_JITTER);
            sum += gaps[i];
        }
        double[] times = new double[n];
        times[0] = t0;
        double acc = 0.0;
        for (int i = 1; i < n; i++) {
            acc += gaps[i - 1];
            times[i] = t0 + duration * (acc / sum);
        }
        times[n - 1] = t0 + duration; // guard against floating-point drift at the endpoint
        return times;
    }
}
