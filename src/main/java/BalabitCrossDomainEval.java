import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

/**
 * TWO-SIDED cross-domain evaluation of the DELBOT-trained detector on the
 * Balabit domain.
 *
 * {@link BalabitValidationPipeline} answers only "how many real Balabit humans
 * do we wrongly flag as bots?" (false positive rate). It cannot answer "how
 * many Balabit-domain bots do we catch?" because there are no bots in the
 * Balabit dataset. That TPR side is exactly where the literature shows
 * cross-domain detection failing, so a one-sided number is not comparable to
 * any published result.
 *
 * This pipeline closes that gap:
 *   1. trains the SAME RandomForest on the SAME DELBOT features (code reused
 *      from BalabitValidationPipeline, not copied);
 *   2. scores every real Balabit human chunk  -> negative class;
 *   3. for each chunk, synthesises matched bot twins with
 *      {@link BalabitBotSynthesizer} (same endpoints, point count and
 *      duration) -> positive class;
 *   4. reports, per bot sophistication level and overall:
 *        - confusion matrix and balanced accuracy at the default 0.5 threshold
 *        - ROC-AUC (Mann-Whitney) and Equal Error Rate
 *        - TPR at a fixed 1% FPR and at a fixed 5% FPR operating point
 *          (the metric Iliou et al. highlight as the hard one -- a detector
 *          is only useful if it catches bots WITHOUT locking real users out).
 *
 * The 1% / 5% FPR thresholds are derived from the human score distribution
 * ONCE and then applied to every bot type, so the operating point is held
 * genuinely constant across the comparison.
 *
 * Run:  java ... BalabitCrossDomainEval
 */
public class BalabitCrossDomainEval {

    /** Base seed for synthetic-bot generation; combined with a per-chunk counter so
     *  every chunk's bot twins are reproducible regardless of iteration order. */
    private static final long RNG_SEED = 42L;
    private static final int RF_TREES = 100;

    private static final double[] TARGET_FPRS = {0.01, 0.05};

    public static void main(String[] args) throws Exception {
        System.out.println("--- STEP 1: TRAIN RANDOM FOREST ON DELBOT ONLY ---");
        Instances training = BalabitValidationPipeline.buildDelbotTrainingSet();
        System.out.println("DELBOT training instances: " + training.numInstances());

        RandomForest rf = new RandomForest();
        rf.setNumIterations(RF_TREES);
        rf.setSeed(1); // match BalabitValidationPipeline exactly (Weka default is 1;
                       // stated for reproducibility). Deterministic training-instance
                       // order is enforced in BalabitValidationPipeline.buildDelbotTrainingSet.
        rf.buildClassifier(training);
        System.out.println("Random Forest trained (" + RF_TREES + " trees).\n");

        int botClassIndex = training.classAttribute().indexOfValue("1");
        if (botClassIndex < 0) throw new IllegalStateException("training data has no '1' (bot) class value");

        System.out.println("--- STEP 2: SCORE REAL BALABIT HUMANS + SYNTHETIC BALABIT-DOMAIN BOTS ---");

        File dataDir = new File(BalabitValidationPipeline.BALABIT_DATA_DIR);
        if (!dataDir.exists()) {
            throw new RuntimeException("Balabit data not found at " + dataDir.getAbsolutePath());
        }
        List<File> sessionFiles = new ArrayList<>();
        BalabitValidationPipeline.findBalabitFiles(dataDir, sessionFiles);
        System.out.println("Found " + sessionFiles.size() + " Balabit raw session files.");

        DoubleBuf humanScores = new DoubleBuf();
        Map<BalabitBotSynthesizer.BotType, DoubleBuf> botScores =
                new EnumMap<>(BalabitBotSynthesizer.BotType.class);
        for (BalabitBotSynthesizer.BotType t : BalabitBotSynthesizer.BotType.values()) {
            botScores.put(t, new DoubleBuf());
        }
        // per user: [humanChunks, humanFalsePositivesAt0.5]
        Map<String, int[]> perUser = new TreeMap<>();

        int chunkCounter = 0;
        int filesProcessed = 0;
        for (File file : sessionFiles) {
            String userFolder = file.getParentFile().getName();
            List<double[]> rawPoints = BalabitValidationPipeline.parseBalabitFile(file); // (t_sec, x, y)

            for (List<double[]> chunkRaw :
                    BalabitValidationPipeline.splitByGap(rawPoints, BalabitValidationPipeline.GAP_THRESHOLD_SEC)) {
                // Collapse tied timestamps once; score the human AND synthesise its bot
                // twin from this same list so num_points / duration stay matched. Same
                // collapse-then-filter order as BalabitValidationPipeline.evaluateBalabit.
                List<double[]> chunk = BalabitValidationPipeline.collapseToDistinctTimestamps(chunkRaw);
                if (chunk.size() < BalabitValidationPipeline.MIN_POINTS_PER_SESSION) continue;

                double[] humanFeat = BalabitValidationPipeline.computeFeatures(toMillis(chunk));
                if (humanFeat == null) continue;

                double humanScore = botProbability(rf, training, humanFeat, botClassIndex);
                humanScores.add(humanScore);
                perUser.computeIfAbsent(userFolder, k -> new int[2]);
                perUser.get(userFolder)[0]++;
                if (humanScore > 0.5) perUser.get(userFolder)[1]++; // ties -> human, see countGt

                // Deterministic per-chunk seed: same chunk -> same bots on every run,
                // independent of iteration order or how many files precede it.
                long chunkSeed = RNG_SEED * 1_000_003L + (chunkCounter++);
                for (BalabitBotSynthesizer.BotType type : BalabitBotSynthesizer.BotType.values()) {
                    List<double[]> bot = BalabitBotSynthesizer.synthesize(
                            chunk, type, new Random(chunkSeed + type.ordinal()));
                    if (bot == null) continue;
                    double[] botFeat = BalabitValidationPipeline.computeFeatures(toMillis(bot));
                    if (botFeat == null) continue;
                    botScores.get(type).add(botProbability(rf, training, botFeat, botClassIndex));
                }
            }

            if (++filesProcessed % 200 == 0) {
                System.out.println("  ...processed " + filesProcessed + "/" + sessionFiles.size()
                        + " files, " + humanScores.size() + " human chunks scored");
            }
        }

        double[] humans = humanScores.toArray();
        System.out.println("\nReal Balabit human chunks scored: " + humans.length);
        for (BalabitBotSynthesizer.BotType t : BalabitBotSynthesizer.BotType.values()) {
            System.out.println("Synthetic " + t + " bot chunks scored: " + botScores.get(t).size());
        }

        report(humans, botScores, perUser);
    }

    // ---------------- reporting ----------------

    private static void report(double[] humans,
                               Map<BalabitBotSynthesizer.BotType, DoubleBuf> botScores,
                               Map<String, int[]> perUser) {
        int nHumans = humans.length;
        int humanFpAt50 = countGt(humans, 0.5); // ties -> human, see countGt
        double humanFprAt50 = 100.0 * humanFpAt50 / nHumans;

        System.out.println("\n========================================================================");
        System.out.println(" CROSS-DOMAIN (Balabit) TWO-SIDED EVALUATION");
        System.out.println("========================================================================");
        System.out.printf("Human false positive rate @0.5 threshold : %.2f%%  (%d / %d)%n",
                humanFprAt50, humanFpAt50, nHumans);
        System.out.println("(cross-check: identical to BalabitValidationPipeline's Balabit FPR --");
        System.out.println(" same DELBOT training, same forest, same tie-to-human 0.5 convention)");

        // Fixed operating points derived from the human score distribution.
        double[] thresholds = new double[TARGET_FPRS.length];
        System.out.println("\n--- FIXED OPERATING POINTS (threshold chosen on humans, applied to all bots) ---");
        for (int i = 0; i < TARGET_FPRS.length; i++) {
            thresholds[i] = thresholdForFpr(humans, TARGET_FPRS[i]);
            double achieved = 100.0 * countGe(humans, thresholds[i]) / nHumans;
            System.out.printf("  target FPR %.0f%%  ->  bot-probability threshold %.4f  (achieved human FPR %.2f%%)%n",
                    TARGET_FPRS[i] * 100, thresholds[i], achieved);
        }

        for (BalabitBotSynthesizer.BotType type : BalabitBotSynthesizer.BotType.values()) {
            double[] bots = botScores.get(type).toArray();
            if (bots.length == 0) continue;

            System.out.println("\n------------------------------------------------------------------------");
            System.out.println(" BOT TYPE: " + type);
            System.out.println("------------------------------------------------------------------------");

            int tp = countGt(bots, 0.5); // ties -> human, matching classifyInstance
            int fn = bots.length - tp;
            int fp = humanFpAt50;
            int tn = nHumans - fp;
            double tpr = 100.0 * tp / bots.length;
            double fprPct = 100.0 * fp / nHumans;
            double balancedAcc = 100.0 * 0.5 * ((double) tp / bots.length + (double) tn / nHumans);

            System.out.println("Confusion matrix @0.5   (rows = truth, cols = predicted)");
            System.out.printf("               pred BOT   pred HUMAN%n");
            System.out.printf("  actual BOT   %9d   %9d%n", tp, fn);
            System.out.printf("  actual HUMAN %9d   %9d%n", fp, tn);
            System.out.printf("TPR (bot recall) @0.5 : %.2f%%%n", tpr);
            System.out.printf("FPR               @0.5 : %.2f%%%n", fprPct);
            System.out.printf("Balanced accuracy @0.5 : %.2f%%%n", balancedAcc);

            double auc = rocAuc(bots, humans);
            double eer = equalErrorRate(bots, humans);
            System.out.printf("ROC-AUC               : %.4f%n", auc);
            System.out.printf("Equal Error Rate      : %.2f%%%n", eer * 100);

            for (int i = 0; i < TARGET_FPRS.length; i++) {
                double tprAt = 100.0 * countGe(bots, thresholds[i]) / bots.length;
                System.out.printf("TPR @ <=%.0f%% FPR       : %.2f%%%n", TARGET_FPRS[i] * 100, tprAt);
            }
        }

        // "All bots" pooled -- the realistic mixed-adversary case.
        DoubleBuf all = new DoubleBuf();
        for (DoubleBuf b : botScores.values()) all.addAll(b);
        double[] allBots = all.toArray();
        System.out.println("\n------------------------------------------------------------------------");
        System.out.println(" ALL BOT TYPES POOLED");
        System.out.println("------------------------------------------------------------------------");
        System.out.printf("ROC-AUC               : %.4f%n", rocAuc(allBots, humans));
        System.out.printf("Equal Error Rate      : %.2f%%%n", equalErrorRate(allBots, humans) * 100);
        System.out.printf("TPR @0.5 threshold    : %.2f%%%n", 100.0 * countGt(allBots, 0.5) / allBots.length);
        for (int i = 0; i < TARGET_FPRS.length; i++) {
            System.out.printf("TPR @ <=%.0f%% FPR       : %.2f%%%n",
                    TARGET_FPRS[i] * 100, 100.0 * countGe(allBots, thresholds[i]) / allBots.length);
        }

        System.out.println("\n--- HUMAN FALSE POSITIVE RATE BY USER (@0.5) ---");
        System.out.printf("%-10s %10s %10s %10s%n", "user", "chunks", "FPs", "FPR%");
        for (Map.Entry<String, int[]> e : perUser.entrySet()) {
            int[] c = e.getValue();
            double f = c[0] > 0 ? 100.0 * c[1] / c[0] : 0.0;
            System.out.printf("%-10s %10d %10d %9.2f%%%n", e.getKey(), c[0], c[1], f);
        }

        System.out.println("\nNOTE: bots here are SYNTHESISED from published movement models, not");
        System.out.println("captured bot traffic, and do not include GAN-generated trajectories.");
        System.out.println("TPR against them is an optimistic bound on real evasive adversaries.");
    }

    // ---------------- metrics ----------------

    /** Count of {@code arr} entries >= t. Used for FPR-target operating points, where
     *  {@link #thresholdForFpr} has already nudged the threshold above any tie group. */
    private static int countGe(double[] arr, double t) {
        int c = 0;
        for (double v : arr) if (v >= t) c++;
        return c;
    }

    /** Count of {@code arr} entries strictly > t. Used for the 0.5 operating point so
     *  that a 50/50 forest vote (P(bot) == 0.50 exactly) resolves to HUMAN -- matching
     *  Weka's Classifier.classifyInstance(), which BalabitValidationPipeline uses. With
     *  >= 0.5 those ~22 borderline chunks would flip to bot and the two pipelines'
     *  reported FPR would disagree by ~0.08 points despite an identical forest. */
    private static int countGt(double[] arr, double t) {
        int c = 0;
        for (double v : arr) if (v > t) c++;
        return c;
    }

    /**
     * ROC-AUC via the Mann-Whitney U statistic: the probability that a random
     * bot scores higher than a random human. Ties count as half. O(n log n).
     */
    private static double rocAuc(double[] pos, double[] neg) {
        int nPos = pos.length, nNeg = neg.length;
        if (nPos == 0 || nNeg == 0) return Double.NaN;

        double[] all = new double[nPos + nNeg];
        boolean[] isPos = new boolean[nPos + nNeg];
        for (int i = 0; i < nPos; i++) { all[i] = pos[i]; isPos[i] = true; }
        for (int i = 0; i < nNeg; i++) all[nPos + i] = neg[i];

        Integer[] order = new Integer[all.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(all[a], all[b]));

        // Average ranks over tie groups (ranks are 1-based).
        double[] rank = new double[all.length];
        int i = 0;
        while (i < order.length) {
            int j = i;
            while (j + 1 < order.length && all[order[j + 1]] == all[order[i]]) j++;
            double avgRank = (i + j) / 2.0 + 1.0;
            for (int k = i; k <= j; k++) rank[order[k]] = avgRank;
            i = j + 1;
        }

        double sumPosRanks = 0.0;
        for (int k = 0; k < all.length; k++) if (isPos[k]) sumPosRanks += rank[k];

        // (long) casts matter: with the pooled bot set nPos exceeds 100k, so
        // nPos*(nPos+1) and nPos*nNeg both overflow int and produce a garbage AUC.
        double u = sumPosRanks - (long) nPos * (nPos + 1L) / 2.0;
        return u / ((double) ((long) nPos * nNeg));
    }

    /**
     * Equal Error Rate: sweep every candidate threshold (each distinct score plus
     * one just above the maximum) and return the error level at the threshold
     * where |FPR - FNR| is smallest.
     */
    private static double equalErrorRate(double[] pos, double[] neg) {
        if (pos.length == 0 || neg.length == 0) return Double.NaN;
        double[] cand = distinctSorted(concat(pos, neg));
        double bestGap = Double.MAX_VALUE;
        double eerAtBest = 1.0;
        for (int i = 0; i <= cand.length; i++) {
            double t = (i < cand.length) ? cand[i] : Math.nextUp(cand[cand.length - 1]);
            double fpr = (double) countGe(neg, t) / neg.length;       // humans flagged as bot
            double fnr = 1.0 - (double) countGe(pos, t) / pos.length; // bots passed as human
            double gap = Math.abs(fpr - fnr);
            if (gap < bestGap) {
                bestGap = gap;
                eerAtBest = (fpr + fnr) / 2.0;
            }
        }
        return eerAtBest;
    }

    /**
     * Highest threshold whose human-FPR does not exceed {@code targetFpr}.
     * Returned threshold is then applied to bots to read off TPR at that FPR.
     */
    private static double thresholdForFpr(double[] humans, double targetFpr) {
        double[] sortedDesc = humans.clone();
        Arrays.sort(sortedDesc);
        // walk from the top; allow up to floor(targetFpr * n) humans above threshold
        int budget = (int) Math.floor(targetFpr * humans.length);
        int idxFromTop = Math.min(budget, humans.length - 1);
        // sortedDesc ascending -> element at (n-1-idxFromTop) is the (idxFromTop+1)-th largest
        int pos = humans.length - 1 - idxFromTop;
        double t = sortedDesc[pos];
        // nudge above ties so we don't accidentally exceed the FPR budget
        return Math.nextUp(t);
    }

    private static double[] distinctSorted(double[] a) {
        double[] s = a.clone();
        Arrays.sort(s);
        int w = 0;
        for (int i = 0; i < s.length; i++) {
            if (i == 0 || s[i] != s[i - 1]) s[w++] = s[i];
        }
        return Arrays.copyOf(s, w);
    }

    private static double[] concat(double[] a, double[] b) {
        double[] r = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }

    // ---------------- scoring helpers ----------------

    private static double botProbability(RandomForest rf, Instances schema, double[] feat, int botClassIndex)
            throws Exception {
        Instance inst = new DenseInstance(schema.numAttributes());
        inst.setDataset(schema);
        for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
        return rf.distributionForInstance(inst)[botClassIndex];
    }

    /** Balabit timestamps are seconds; DELBOT features are computed in ms. */
    private static List<double[]> toMillis(List<double[]> pointsSec) {
        List<double[]> ms = new ArrayList<>(pointsSec.size());
        for (double[] p : pointsSec) ms.add(new double[]{p[0] * 1000.0, p[1], p[2]});
        return ms;
    }

    // ---------------- growable primitive buffer ----------------

    private static final class DoubleBuf {
        private double[] data = new double[1024];
        private int size = 0;

        void add(double v) {
            if (size == data.length) data = Arrays.copyOf(data, data.length * 2);
            data[size++] = v;
        }

        void addAll(DoubleBuf other) {
            for (int i = 0; i < other.size; i++) add(other.data[i]);
        }

        int size() { return size; }

        double[] toArray() { return Arrays.copyOf(data, size); }
    }
}
