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
import java.util.TreeSet;

/**
 * TIER 2, step 2 -- TRAINING-DATA AUGMENTATION with synthetic in-domain bots.
 *
 * Tier-1 and Tier-2-features both showed a DELBOT-only model does not transfer
 * to Balabit (pooled bot ROC-AUC ~0.53, ~0.48 against burst-timed bots).
 * BeCAPTCHA-Mouse (Acien et al. 2022) reports the single biggest documented
 * jump comes from adding SYNTHETIC bots to the training set (real-only 60% ->
 * real+synthetic 96-98%). This evaluates that lever honestly:
 *
 *  - Balabit's 10 users are split into TRAIN and TEST halves (deterministic,
 *    alternating over the sorted user list). No user appears in both.
 *  - TRAIN-half human chunks + one synthetic bot twin each are added to the
 *    DELBOT training set. TEST-half humans + their bot twins are the eval set.
 *    Human chunks of held-out users are NEVER trained on.
 *  - Three training sets are compared, each in two feature modes
 *    (BASELINE = Tier-1 seven, SCALEFREE = dimensionless only):
 *      DELBOT_ONLY   - baseline, no augmentation (scored on the SAME test users
 *                      so the numbers are directly comparable)
 *      AUG_ALL       - + bot twins of all 4 BalabitBotSynthesizer types
 *      AUG_HELDOUT   - + bot twins of only MODERATE_LINEAR and ADVANCED_BEZIER;
 *                      the two "*_VP*" types are then UNSEEN at test time, so
 *                      their AUC measures generalisation beyond the exact bot
 *                      models seen in training (not generator memorisation).
 *
 * Training bots use BURST timing (the realistic case); test bots are reported
 * for MATCHED_SPREAD and BURST separately. Deterministic throughout
 * (RandomForest seed 1, single slot, sorted file order, per-chunk-seeded synth).
 */
public class Tier2AugmentedEval {

    private static final long RNG_SEED = 42L;
    private static final int RF_TREES = 100;
    private static final double[] TARGET_FPRS = {0.01, 0.05};

    private enum TrainSet { DELBOT_ONLY, AUG_ALL, AUG_HELDOUT }

    private static final BalabitBotSynthesizer.BotType[] HELDOUT_TRAIN_TYPES = {
            BalabitBotSynthesizer.BotType.MODERATE_LINEAR,
            BalabitBotSynthesizer.BotType.ADVANCED_BEZIER
    };

    public static void main(String[] args) throws Exception {
        File delbotDir = new File(BalabitValidationPipeline.DELBOT_DATA_DIR);
        File balabitDir = new File(BalabitValidationPipeline.BALABIT_DATA_DIR);
        if (!delbotDir.exists() || !balabitDir.exists()) throw new RuntimeException("data dirs missing");

        List<File> delbotFolders = new ArrayList<>();
        BalabitValidationPipeline.findSessionFolders(delbotDir, delbotFolders);
        List<File> balabitFiles = new ArrayList<>();
        BalabitValidationPipeline.findBalabitFiles(balabitDir, balabitFiles);

        // ---- gather Balabit human chunks (collapsed, seconds) bucketed by user ----
        Map<String, List<List<double[]>>> byUser = new TreeMap<>();
        for (File f : balabitFiles) {
            String user = f.getParentFile().getName();
            List<double[]> raw = BalabitValidationPipeline.parseBalabitFile(f);
            for (List<double[]> cr : BalabitValidationPipeline.splitByGap(
                    raw, BalabitValidationPipeline.GAP_THRESHOLD_SEC)) {
                List<double[]> c = BalabitValidationPipeline.collapseToDistinctTimestamps(cr);
                if (c.size() < BalabitValidationPipeline.MIN_POINTS_PER_SESSION) continue;
                byUser.computeIfAbsent(user, k -> new ArrayList<>()).add(c);
            }
        }

        List<String> users = new ArrayList<>(byUser.keySet()); // sorted (TreeMap)
        TreeSet<String> trainUsers = new TreeSet<>();
        TreeSet<String> testUsers = new TreeSet<>();
        for (int i = 0; i < users.size(); i++) (i % 2 == 0 ? trainUsers : testUsers).add(users.get(i));

        List<List<double[]>> trainChunks = new ArrayList<>();
        List<List<double[]>> testChunks = new ArrayList<>();
        List<String> testChunkUserList = new ArrayList<>();
        for (String u : trainUsers) trainChunks.addAll(byUser.get(u));
        for (String u : testUsers) {
            for (List<double[]> c : byUser.get(u)) { testChunks.add(c); testChunkUserList.add(u); }
        }

        System.out.println("Balabit users: " + users);
        System.out.println("  TRAIN users " + trainUsers + "  -> " + trainChunks.size() + " human chunks");
        System.out.println("  TEST  users " + testUsers + "  -> " + testChunks.size() + " human chunks");
        System.out.println("DELBOT session folders: " + delbotFolders.size());
        System.out.println("Training-bot timing: BURST | test-bot timing: reported per model\n");

        List<Row> table = new ArrayList<>();
        for (Tier2Features.Mode mode : new Tier2Features.Mode[]{
                Tier2Features.Mode.BASELINE, Tier2Features.Mode.SCALEFREE}) {

            // DELBOT instances for this feature mode, reused across the 3 training sets
            Instances delbotInstances = buildDelbot(mode, delbotFolders);

            for (TrainSet ts : TrainSet.values()) {
                Instances training = new Instances(delbotInstances, delbotInstances.numInstances() + 2 * trainChunks.size());
                training.addAll(delbotInstances);
                if (ts != TrainSet.DELBOT_ONLY) {
                    addAugmentation(training, mode, ts, trainChunks);
                }
                int botClassIndex = training.classAttribute().indexOfValue("1");

                RandomForest rf = new RandomForest();
                rf.setNumIterations(RF_TREES);
                rf.setSeed(1);
                rf.buildClassifier(training);

                for (BalabitBotSynthesizer.TimingModel tm : BalabitBotSynthesizer.TimingModel.values()) {
                    Row r = score(rf, training, botClassIndex, mode, ts, tm, testChunks, testChunkUserList);
                    table.add(r);
                    System.out.printf("[%-9s %-11s test=%-13s] trainN=%-7d humFPR=%5.2f%%  AUC=%.4f  EER=%5.2f%%  TPR@<=1%%=%.2f%%  TPR@<=5%%=%.2f%%%n",
                            mode, ts, tm, training.numInstances(), r.humanFpr, r.pooledAuc, r.pooledEer * 100,
                            r.tprAt1, r.tprAt5);
                    if (ts == TrainSet.AUG_HELDOUT) {
                        System.out.printf("            unseen-type AUC:  MODERATE_LINEAR_VP=%.4f  ADVANCED_BEZIER_VP_JITTER=%.4f%n",
                                r.perTypeAuc.getOrDefault(BalabitBotSynthesizer.BotType.MODERATE_LINEAR_VP, Double.NaN),
                                r.perTypeAuc.getOrDefault(BalabitBotSynthesizer.BotType.ADVANCED_BEZIER_VP_JITTER, Double.NaN));
                    }
                }
            }
            System.out.println();
        }

        printFinal(table);
    }

    // ---------------- training set construction ----------------

    private static Instances buildDelbot(Tier2Features.Mode mode, List<File> folders) throws Exception {
        Instances data = new Instances("delbot_" + mode, Tier2Features.schema(mode), 0);
        data.setClassIndex(data.numAttributes() - 1);
        for (File folder : folders) {
            int label = folder.getName().startsWith("circles_human") ? 0 : 1;
            File[] files = folder.listFiles((d, n) -> n.endsWith(".txt"));
            if (files == null) continue;
            Arrays.sort(files);
            for (File f : files) {
                double[] feat = Tier2Features.compute(BalabitValidationPipeline.parseDelbotPoints(f), mode);
                if (feat == null) continue;
                data.add(mkInstance(data, feat, label));
            }
        }
        return data;
    }

    /** Adds each TRAIN-user human chunk as a human + one synthetic bot twin as a bot. */
    private static void addAugmentation(Instances training, Tier2Features.Mode mode, TrainSet ts,
                                        List<List<double[]>> trainChunks) {
        BalabitBotSynthesizer.BotType[] pool = (ts == TrainSet.AUG_HELDOUT)
                ? HELDOUT_TRAIN_TYPES : BalabitBotSynthesizer.BotType.values();
        long seedBase = RNG_SEED * 7_000_003L;
        for (int ci = 0; ci < trainChunks.size(); ci++) {
            List<double[]> chunk = trainChunks.get(ci);

            double[] hf = Tier2Features.compute(toMillis(chunk), mode);
            if (hf != null) training.add(mkInstance(training, hf, 0));

            Random rng = new Random(seedBase + ci);
            BalabitBotSynthesizer.BotType type = pool[rng.nextInt(pool.length)];
            List<double[]> bot = BalabitBotSynthesizer.synthesize(
                    chunk, type, BalabitBotSynthesizer.TimingModel.BURST, rng);
            if (bot == null) continue;
            double[] bf = Tier2Features.compute(toMillis(bot), mode);
            if (bf != null) training.add(mkInstance(training, bf, 1));
        }
    }

    // ---------------- scoring one (mode, trainset, timing) cell ----------------

    private static Row score(RandomForest rf, Instances schema, int botClassIndex,
                             Tier2Features.Mode mode, TrainSet ts, BalabitBotSynthesizer.TimingModel tm,
                             List<List<double[]>> testChunks, List<String> testChunkUser) throws Exception {
        DoubleBuf humans = new DoubleBuf();
        Map<BalabitBotSynthesizer.BotType, DoubleBuf> bots = new EnumMap<>(BalabitBotSynthesizer.BotType.class);
        for (BalabitBotSynthesizer.BotType t : BalabitBotSynthesizer.BotType.values()) bots.put(t, new DoubleBuf());
        Map<String, int[]> perUser = new TreeMap<>();

        long seedBase = RNG_SEED * 9_000_011L + tm.ordinal() * 101L;
        for (int ci = 0; ci < testChunks.size(); ci++) {
            List<double[]> chunk = testChunks.get(ci);
            String user = testChunkUser.get(ci);

            double[] hf = Tier2Features.compute(toMillis(chunk), mode);
            if (hf == null) continue;
            double hs = prob(rf, schema, hf, botClassIndex);
            humans.add(hs);
            perUser.computeIfAbsent(user, k -> new int[2]);
            perUser.get(user)[0]++;
            if (hs > 0.5) perUser.get(user)[1]++;

            for (BalabitBotSynthesizer.BotType type : BalabitBotSynthesizer.BotType.values()) {
                Random rng = new Random(seedBase + (long) ci * 4 + type.ordinal());
                List<double[]> bot = BalabitBotSynthesizer.synthesize(chunk, type, tm, rng);
                if (bot == null) continue;
                double[] bf = Tier2Features.compute(toMillis(bot), mode);
                if (bf != null) bots.get(type).add(prob(rf, schema, bf, botClassIndex));
            }
        }

        double[] h = humans.toArray();
        DoubleBuf pooled = new DoubleBuf();
        Row r = new Row();
        r.mode = mode; r.trainSet = ts; r.timing = tm;
        r.nHumans = h.length;
        r.humanFp = countGt(h, 0.5);
        r.humanFpr = 100.0 * r.humanFp / h.length;
        for (Map.Entry<BalabitBotSynthesizer.BotType, DoubleBuf> e : bots.entrySet()) {
            double[] b = e.getValue().toArray();
            pooled.addAll(e.getValue());
            if (b.length > 0) r.perTypeAuc.put(e.getKey(), rocAuc(b, h));
        }
        double[] all = pooled.toArray();
        r.pooledAuc = rocAuc(all, h);
        r.pooledEer = equalErrorRate(all, h);
        double t1 = thresholdForFpr(h, TARGET_FPRS[0]);
        double t5 = thresholdForFpr(h, TARGET_FPRS[1]);
        r.tprAt1 = 100.0 * countGe(all, t1) / all.length;
        r.tprAt5 = 100.0 * countGe(all, t5) / all.length;
        r.perUser = perUser;
        return r;
    }

    private static void printFinal(List<Row> table) {
        System.out.println("========================================================================");
        System.out.println(" TIER 2 STEP 2 -- TRAINING AUGMENTATION  (held-out Balabit users)");
        System.out.println("========================================================================");
        System.out.printf("%-9s %-11s %-13s %8s %9s %9s %11s %11s%n",
                "feat", "trainset", "testBotTiming", "humFPR", "pooledAUC", "EER", "TPR@<=1%", "TPR@<=5%");
        for (Row r : table) {
            System.out.printf("%-9s %-11s %-13s %7.2f%% %9.4f %8.2f%% %10.2f%% %10.2f%%%n",
                    r.mode, r.trainSet, r.timing, r.humanFpr, r.pooledAuc, r.pooledEer * 100, r.tprAt1, r.tprAt5);
        }
        System.out.println("\nWIN CONDITION: pooledAUC materially above the DELBOT_ONLY row for the same");
        System.out.println("feat+timing, with humFPR held at or below the DELBOT_ONLY level. AUG_HELDOUT's");
        System.out.println("*_VP* types are unseen in training -- that row is the honest generalisation test.");
        System.out.println("Caveat: test bots share the BalabitBotSynthesizer generator; no GAN trajectories.");
    }

    // ---------------- helpers ----------------

    private static Instance mkInstance(Instances data, double[] feat, int label) {
        Instance inst = new DenseInstance(data.numAttributes());
        inst.setDataset(data);
        for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
        inst.setValue(data.numAttributes() - 1, String.valueOf(label));
        return inst;
    }

    private static double prob(RandomForest rf, Instances schema, double[] feat, int botClassIndex) throws Exception {
        Instance inst = new DenseInstance(schema.numAttributes());
        inst.setDataset(schema);
        for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
        return rf.distributionForInstance(inst)[botClassIndex];
    }

    private static List<double[]> toMillis(List<double[]> pointsSec) {
        List<double[]> ms = new ArrayList<>(pointsSec.size());
        for (double[] p : pointsSec) ms.add(new double[]{p[0] * 1000.0, p[1], p[2]});
        return ms;
    }

    private static final class Row {
        Tier2Features.Mode mode;
        TrainSet trainSet;
        BalabitBotSynthesizer.TimingModel timing;
        int nHumans, humanFp;
        double humanFpr, pooledAuc, pooledEer, tprAt1, tprAt5;
        Map<BalabitBotSynthesizer.BotType, Double> perTypeAuc = new EnumMap<>(BalabitBotSynthesizer.BotType.class);
        Map<String, int[]> perUser;
    }

    // ---- metrics (same pure-math helpers as BalabitCrossDomainEval / Tier2CrossDomainEval) ----

    private static int countGe(double[] a, double t) { int c = 0; for (double v : a) if (v >= t) c++; return c; }
    private static int countGt(double[] a, double t) { int c = 0; for (double v : a) if (v > t) c++; return c; }

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
        double[] rank = new double[all.length];
        int i = 0;
        while (i < order.length) {
            int j = i;
            while (j + 1 < order.length && all[order[j + 1]] == all[order[i]]) j++;
            double avg = (i + j) / 2.0 + 1.0;
            for (int k = i; k <= j; k++) rank[order[k]] = avg;
            i = j + 1;
        }
        double sp = 0.0;
        for (int k = 0; k < all.length; k++) if (isPos[k]) sp += rank[k];
        double u = sp - (long) nPos * (nPos + 1L) / 2.0;
        return u / ((double) ((long) nPos * nNeg));
    }

    private static double equalErrorRate(double[] pos, double[] neg) {
        if (pos.length == 0 || neg.length == 0) return Double.NaN;
        double[] cand = distinctSorted(concat(pos, neg));
        double bestGap = Double.MAX_VALUE, eer = 1.0;
        for (int i = 0; i <= cand.length; i++) {
            double t = (i < cand.length) ? cand[i] : Math.nextUp(cand[cand.length - 1]);
            double fpr = (double) countGe(neg, t) / neg.length;
            double fnr = 1.0 - (double) countGe(pos, t) / pos.length;
            double gap = Math.abs(fpr - fnr);
            if (gap < bestGap) { bestGap = gap; eer = (fpr + fnr) / 2.0; }
        }
        return eer;
    }

    private static double thresholdForFpr(double[] humans, double targetFpr) {
        double[] s = humans.clone();
        Arrays.sort(s);
        int budget = (int) Math.floor(targetFpr * humans.length);
        int idxFromTop = Math.min(budget, humans.length - 1);
        return Math.nextUp(s[humans.length - 1 - idxFromTop]);
    }

    private static double[] distinctSorted(double[] a) {
        double[] s = a.clone();
        Arrays.sort(s);
        int w = 0;
        for (int i = 0; i < s.length; i++) if (i == 0 || s[i] != s[i - 1]) s[w++] = s[i];
        return Arrays.copyOf(s, w);
    }

    private static double[] concat(double[] a, double[] b) {
        double[] r = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }

    private static final class DoubleBuf {
        private double[] data = new double[1024];
        private int size = 0;
        void add(double v) { if (size == data.length) data = Arrays.copyOf(data, data.length * 2); data[size++] = v; }
        void addAll(DoubleBuf o) { for (int i = 0; i < o.size; i++) add(o.data[i]); }
        double[] toArray() { return Arrays.copyOf(data, size); }
    }
}
