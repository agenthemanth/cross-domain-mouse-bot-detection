import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * TIER 4 &amp; 7 -- an EVASIVE / feature-matched synthetic bot.
 *
 * The Tier-2 win (synthetic-bot training augmentation, cross-domain bot AUC
 * 0.53 -> 0.998) came with a stated caveat: the {@link BalabitBotSynthesizer}
 * bots are non-adversarial published movement models, so an augmented model
 * only proves it can beat naive bots. This class builds the hard case -- a bot
 * whose HAND-CRAFTED feature vector is, by construction, close to its source
 * human chunk's:
 *
 *   - velocity distribution: the synthetic trajectory reuses the source chunk's
 *     own per-step (distance, time-gap) pairs in a shuffled order, so the
 *     per-step speed sequence is a PERMUTATION of the source's. mean_velocity,
 *     std_velocity, velocity_cv, every percentile, pause_ratio and
 *     vel_p90_p50_ratio therefore match the source closely.
 *   - path_efficiency: the path is the straight start->end line plus a few
 *     random sine deviations whose amplitude is tuned by bisection so total
 *     path length (hence straight/path efficiency) matches the source.
 *   - endpoints, point count and total duration stay matched (as with every
 *     synthesiser here) so those cannot leak the label.
 *
 * What SHUFFLE does NOT reproduce is the sequential correlation structure of real
 * motion (a human accelerates smoothly; a shuffled speed sequence is
 * white-noise-like in its ordering) and whatever the sine-deviation path model
 * fails to capture. Whether an augmented RandomForest still separates these is
 * the Tier-4 question.  Deterministic given the supplied {@link Random}.
 *
 * TIER 7 -- {@link Ordering#BALLISTIC} closes that last gap. Instead of a random
 * permutation of the (distance, dt) pairs it arranges them into a single
 * rise-then-fall speed ramp (slow pairs at the ends, fast pairs in the middle),
 * so the per-step speed series is now smoothly autocorrelated and its successive
 * differences are small -- defeating the Tier-6 features velocity_lag1_autocorr
 * and velocity_step_roughness while STILL matching the speed multiset, endpoints,
 * count, duration and path length exactly. This is the Round-7 attacker: does the
 * AUGMENTED_SEQ gain survive a bot that also matches the ordering statistics?
 */
public final class AdversarialBotSynthesizer {

    private static final int DEVIATION_MODES = 7;
    private static final int LENGTH_FIT_ITERS = 30;

    /** Step-ordering strategy. SHUFFLE = Tier 4 (random). BALLISTIC = Tier 7 (smooth ramp). */
    public enum Ordering { SHUFFLE, BALLISTIC }

    private AdversarialBotSynthesizer() {}

    /** Tier-4 entry point: random-shuffle ordering. */
    public static List<double[]> synthesize(List<double[]> humanChunk, Random rng) {
        return synthesize(humanChunk, Ordering.SHUFFLE, rng);
    }

    /**
     * @param humanChunk source points (t_seconds, x, y), ascending in t
     * @param ordering   how to order the reused (distance, dt) pairs
     * @return matched evasive bot as (t_seconds, x, y), or null if unusable
     */
    public static List<double[]> synthesize(List<double[]> humanChunk, Ordering ordering, Random rng) {
        int n = humanChunk.size();
        if (n < 5) return null;

        double t0 = humanChunk.get(0)[0];
        double tEnd = humanChunk.get(n - 1)[0];
        double duration = tEnd - t0;
        if (duration <= 0) return null;

        double x0 = humanChunk.get(0)[1], y0 = humanChunk.get(0)[2];
        double x1 = humanChunk.get(n - 1)[1], y1 = humanChunk.get(n - 1)[2];

        double[] stepDist = new double[n - 1];
        double[] stepDt = new double[n - 1];
        double srcPathLen = 0.0;
        for (int i = 1; i < n; i++) {
            double dx = humanChunk.get(i)[1] - humanChunk.get(i - 1)[1];
            double dy = humanChunk.get(i)[2] - humanChunk.get(i - 1)[2];
            stepDist[i - 1] = Math.sqrt(dx * dx + dy * dy);
            stepDt[i - 1] = Math.max(humanChunk.get(i)[0] - humanChunk.get(i - 1)[0], 1e-4);
            srcPathLen += stepDist[i - 1];
        }
        if (srcPathLen < 1e-6) return null;

        // ---- reorder (distance, dt) pairs together -> speed multiset preserved ----
        int[] perm = orderPairs(stepDist, stepDt, ordering, rng);
        double[] arcFrac = new double[n];
        double[] times = new double[n];
        double accDist = 0, accTime = 0;
        for (int i = 1; i < n; i++) {
            accDist += stepDist[perm[i - 1]];
            accTime += stepDt[perm[i - 1]];
            arcFrac[i] = accDist;
            times[i] = accTime;
        }
        double arcTotal = arcFrac[n - 1] > 0 ? arcFrac[n - 1] : 1.0;
        double timeTotal = accTime > 0 ? accTime : 1.0;
        for (int i = 0; i < n; i++) {
            arcFrac[i] /= arcTotal;
            times[i] = t0 + duration * (times[i] / timeTotal);
        }
        times[0] = t0;
        times[n - 1] = tEnd;

        // ---- unit tangent / perpendicular of the start->end axis ----
        double axLen = Math.hypot(x1 - x0, y1 - y0);
        double tx, ty;
        if (axLen < 1e-6) { tx = 1; ty = 0; } else { tx = (x1 - x0) / axLen; ty = (y1 - y0) / axLen; }
        double perpX = -ty, perpY = tx;

        // ---- random deviation modes (unit amplitude) ----
        double[] freq = new double[DEVIATION_MODES];
        double[] phase = new double[DEVIATION_MODES];
        double[] ampPerp = new double[DEVIATION_MODES];
        double[] ampPar = new double[DEVIATION_MODES];
        for (int m = 0; m < DEVIATION_MODES; m++) {
            // higher, denser frequencies -> more direction changes per unit path length,
            // so the synthetic path's turning-angle / reversal-rate statistics get closer
            // to a real hand-moved cursor rather than a single smooth arc.
            freq[m] = (m + 1) * 1.7 * Math.PI + rng.nextDouble() * Math.PI;
            phase[m] = rng.nextDouble() * 2 * Math.PI;
            ampPerp[m] = Math.pow(m + 1, -0.7) * (0.5 + rng.nextDouble());
            ampPar[m] = 0.15 * (rng.nextDouble() - 0.5);
        }
        double devRef = Math.max(axLen, srcPathLen);

        // ---- bisection on deviation scale to hit srcPathLen ----
        double lo = 0.0, hi = 2.0, scale = 0.3;
        for (int it = 0; it < LENGTH_FIT_ITERS; it++) {
            scale = 0.5 * (lo + hi);
            double len = buildLength(n, arcFrac, x0, y0, x1, y1, tx, ty, perpX, perpY,
                    freq, phase, ampPerp, ampPar, scale * devRef, null);
            if (len > srcPathLen) hi = scale; else lo = scale;
        }

        // ---- emit ----
        List<double[]> bot = new ArrayList<>(n);
        buildLength(n, arcFrac, x0, y0, x1, y1, tx, ty, perpX, perpY,
                freq, phase, ampPerp, ampPar, scale * devRef, bot);
        for (int i = 0; i < n; i++) {
            bot.get(i)[0] = times[i];
            bot.get(i)[1] = Math.rint(bot.get(i)[1]);
            bot.get(i)[2] = Math.rint(bot.get(i)[2]);
        }
        bot.get(0)[1] = x0; bot.get(0)[2] = y0;
        bot.get(n - 1)[1] = x1; bot.get(n - 1)[2] = y1;
        return bot;
    }

    /**
     * Returns an ordering of the {@code n-1} step indices.
     *  SHUFFLE   -- uniform random permutation (Fisher-Yates). Tier 4.
     *  BALLISTIC -- sort steps by speed (dist/dt), then interleave so speed rises to a
     *               single mid-sequence peak and falls back: the slowest step first, then
     *               the next slowest LAST, next SECOND, ... i.e. odd ranks go on the rising
     *               limb, even ranks fill the falling limb from the end. The resulting speed
     *               series is monotone up then monotone down -> lag-1 autocorrelation near
     *               +1 and mean|dv| minimised, matching a real point-to-point movement,
     *               while the multiset of (dist, dt) pairs is unchanged. Tier 7.
     */
    private static int[] orderPairs(double[] stepDist, double[] stepDt, Ordering ordering, Random rng) {
        int m = stepDist.length;
        int[] perm = new int[m];
        for (int i = 0; i < m; i++) perm[i] = i;
        if (ordering == Ordering.SHUFFLE) {
            for (int i = m - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
            }
            return perm;
        }
        // BALLISTIC: rank steps slow -> fast, then place onto a rise-then-fall envelope.
        Integer[] bySpeed = new Integer[m];
        for (int i = 0; i < m; i++) bySpeed[i] = i;
        java.util.Arrays.sort(bySpeed, (a, b) -> Double.compare(
                stepDist[a] / Math.max(stepDt[a], 1e-4), stepDist[b] / Math.max(stepDt[b], 1e-4)));
        int lo = 0, hi = m - 1;
        for (int rank = 0; rank < m; rank++) {
            if (rank % 2 == 0) perm[lo++] = bySpeed[rank];   // rising limb, front
            else               perm[hi--] = bySpeed[rank];   // falling limb, back
        }
        return perm;
    }

    /**
     * Computes the deviated path and returns its total length. If {@code out} is non-null it
     * is cleared and filled with n rows {0, x, y} (timestamps filled by the caller).
     */
    private static double buildLength(int n, double[] arcFrac,
                                      double x0, double y0, double x1, double y1,
                                      double tx, double ty, double perpX, double perpY,
                                      double[] freq, double[] phase, double[] ampPerp, double[] ampPar,
                                      double devScale, List<double[]> out) {
        if (out != null) out.clear();
        double prevX = 0, prevY = 0, total = 0;
        for (int i = 0; i < n; i++) {
            double s = arcFrac[i];
            double env = Math.sin(Math.PI * s); // 0 at both ends
            double perp = 0, par = 0;
            for (int m = 0; m < freq.length; m++) {
                double w = Math.sin(freq[m] * s + phase[m]) * env;
                perp += ampPerp[m] * w;
                par += ampPar[m] * w;
            }
            double baseX = x0 + (x1 - x0) * s;
            double baseY = y0 + (y1 - y0) * s;
            double x = baseX + perpX * perp * devScale + tx * par * devScale;
            double y = baseY + perpY * perp * devScale + ty * par * devScale;
            if (i > 0) total += Math.hypot(x - prevX, y - prevY);
            prevX = x; prevY = y;
            if (out != null) out.add(new double[]{0, x, y});
        }
        return total;
    }
}
