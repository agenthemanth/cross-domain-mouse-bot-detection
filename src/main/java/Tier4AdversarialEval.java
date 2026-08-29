import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * TIER 4 -- does the Tier-2 winner survive an EVASIVE bot?
 *
 * Tier 2 showed synthetic-bot training augmentation takes cross-domain bot
 * ROC-AUC 0.53 -> 0.998, but only against the non-adversarial
 * {@link BalabitBotSynthesizer} models. This evaluates the augmented model
 * against {@link AdversarialBotSynthesizer} -- a bot whose velocity
 * distribution and path efficiency are matched to its source human chunk by
 * construction.
 *
 * Held-out Balabit user split (same as Tier2AugmentedEval). Three training sets,
 * each in BASELINE and AUGMENTED feature modes:
 *   DELBOT_ONLY      - no augmentation
 *   AUG_NAIVE        - + naive bot twins of train-user chunks (the Tier-2 setup)
 *   AUG_NAIVE_PLUS   - + naive AND adversarial bot twins of train-user chunks
 *
 * Each is scored on the held-out users against BOTH a naive bot suite (BURST
 * timing, sanity check vs Tier 2) and the adversarial suite. The question:
 *   1. does AUG_NAIVE generalise to adversarial bots it never saw?
 *   2. does adding adversarial bots to training (AUG_NAIVE_PLUS) close the gap,
 *      and at what cost to human FPR / naive-bot recall?
 *
 * Deterministic (RF seed 1, single slot, sorted files, per-chunk-seeded synth).
 */
public class Tier4AdversarialEval {

    private static final long RNG_SEED = 42L;
    private static final int RF_TREES = 100;
    private static final double[] TARGET_FPRS = {0.01, 0.05};

    private enum TrainSet { DELBOT_ONLY, AUG_NAIVE, AUG_NAIVE_PLUS }

    public static void main(String[] args) throws Exception {
        File delbotDir = new File(BalabitValidationPipeline.DELBOT_DATA_DIR);
        File balabitDir = new File(BalabitValidationPipeline.BALABIT_DATA_DIR);
        if (!delbotDir.exists() || !balabitDir.exists()) throw new RuntimeException("data dirs missing");

        List<File> delbotFolders = new ArrayList<>();
        BalabitValidationPipeline.findSessionFolders(delbotDir, delbotFolders);
        List<File> balabitFiles = new ArrayList<>();
        BalabitValidationPipeline.findBalabitFiles(balabitDir, balabitFiles);

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
        List<String> users = new ArrayList<>(byUser.keySet());
        TreeSet<String> trainUsers = new TreeSet<>(), testUsers = new TreeSet<>();
        for (int i = 0; i < users.size(); i++) (i % 2 == 0 ? trainUsers : testUsers).add(users.get(i));
        List<List<double[]>> trainChunks = new ArrayList<>(), testChunks = new ArrayList<>();
        for (String u : trainUsers) trainChunks.addAll(byUser.get(u));
        for (String u : testUsers) testChunks.addAll(byUser.get(u));

        System.out.println("TRAIN users " + trainUsers + " (" + trainChunks.size() + " chunks)");
        System.out.println("TEST  users " + testUsers + " (" + testChunks.size() + " chunks)\n");

        List<Row> table = new ArrayList<>();
        for (Tier2Features.Mode mode : new Tier2Features.Mode[]{
                Tier2Features.Mode.BASELINE, Tier2Features.Mode.AUGMENTED}) {
            Instances delbot = buildDelbot(mode, delbotFolders);

            for (TrainSet ts : TrainSet.values()) {
                Instances training = new Instances(delbot, delbot.numInstances() + 4 * trainChunks.size());
                training.addAll(delbot);
                if (ts != TrainSet.DELBOT_ONLY) {
                    boolean withAdv = (ts == TrainSet.AUG_NAIVE_PLUS);
                    augment(training, mode, trainChunks, withAdv);
                }
                int botIdx = training.classAttribute().indexOfValue("1");

                RandomForest rf = new RandomForest();
                rf.setNumIterations(RF_TREES);
                rf.setSeed(1);
                rf.buildClassifier(training);

                Row r = new Row();
                r.mode = mode; r.trainSet = ts; r.trainN = training.numInstances();
                scoreTest(rf, training, botIdx, mode, testChunks, r);
                table.add(r);
                System.out.printf("[%-9s %-14s] trainN=%-7d humFPR=%5.2f%%  |  vs NAIVE: AUC=%.4f TPR@<=1%%=%.2f%%  |  vs ADVERSARIAL: AUC=%.4f TPR@<=1%%=%.2f%% EER=%.2f%%%n",
                        mode, ts, r.trainN, r.humanFpr, r.naiveAuc, r.naiveTpr1, r.advAuc, r.advTpr1, r.advEer * 100);
            }
            System.out.println();
        }

        System.out.println("========================================================================");
        System.out.println(" TIER 4 -- EVASIVE BOT  (held-out Balabit users)");
        System.out.println("========================================================================");
        System.out.printf("%-9s %-14s %8s | %9s %10s | %9s %10s %9s%n",
                "feat", "trainset", "humFPR", "naiveAUC", "naiveTPR@1%", "advAUC", "advTPR@1%", "advEER");
        for (Row r : table) {
            System.out.printf("%-9s %-14s %7.2f%% | %9.4f %9.2f%% | %9.4f %9.2f%% %8.2f%%%n",
                    r.mode, r.trainSet, r.humanFpr, r.naiveAuc, r.naiveTpr1, r.advAuc, r.advTpr1, r.advEer * 100);
        }
        System.out.println("\nAUG_NAIVE vs advAUC = does the Tier-2 winner generalise to an evasive bot it");
        System.out.println("never trained on. AUG_NAIVE_PLUS = adversarial bots added to training too.");
        System.out.println("Adversarial bot matches source velocity distribution + path efficiency by");
        System.out.println("construction; residual tells are accel/jerk ORDER and slightly-too-smooth curvature.");
    }

    // ---------------- training ----------------

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
                if (feat != null) data.add(mk(data, feat, label));
            }
        }
        return data;
    }

    private static void augment(Instances training, Tier2Features.Mode mode,
                                List<List<double[]>> trainChunks, boolean withAdv) {
        BalabitBotSynthesizer.BotType[] pool = BalabitBotSynthesizer.BotType.values();
        long base = RNG_SEED * 7_000_003L;
        for (int i = 0; i < trainChunks.size(); i++) {
            List<double[]> chunk = trainChunks.get(i);
            double[] hf = Tier2Features.compute(toMillis(chunk), mode);
            if (hf != null) training.add(mk(training, hf, 0));

            Random rng = new Random(base + i);
            BalabitBotSynthesizer.BotType type = pool[rng.nextInt(pool.length)];
            List<double[]> naive = BalabitBotSynthesizer.synthesize(
                    chunk, type, BalabitBotSynthesizer.TimingModel.BURST, rng);
            if (naive != null) {
                double[] bf = Tier2Features.compute(toMillis(naive), mode);
                if (bf != null) training.add(mk(training, bf, 1));
            }
            if (withAdv) {
                List<double[]> adv = AdversarialBotSynthesizer.synthesize(chunk, new Random(base + 5_000_011L + i));
                if (adv != null) {
                    double[] af = Tier2Features.compute(toMillis(adv), mode);
                    if (af != null) training.add(mk(training, af, 1));
                }
            }
        }
    }

    // ---------------- scoring ----------------

    private static void scoreTest(RandomForest rf, Instances schema, int botIdx, Tier2Features.Mode mode,
                                  List<List<double[]>> testChunks, Row r) throws Exception {
        DoubleBuf humans = new DoubleBuf(), naive = new DoubleBuf(), adv = new DoubleBuf();
        long base = RNG_SEED * 9_000_011L;
        for (int i = 0; i < testChunks.size(); i++) {
            List<double[]> chunk = testChunks.get(i);
            double[] hf = Tier2Features.compute(toMillis(chunk), mode);
            if (hf == null) continue;
            humans.add(prob(rf, schema, hf, botIdx));

            for (BalabitBotSynthesizer.BotType bt : BalabitBotSynthesizer.BotType.values()) {
                Random rng = new Random(base + (long) i * 8 + bt.ordinal());
                List<double[]> b = BalabitBotSynthesizer.synthesize(chunk, bt, BalabitBotSynthesizer.TimingModel.BURST, rng);
                if (b == null) continue;
                double[] bf = Tier2Features.compute(toMillis(b), mode);
                if (bf != null) naive.add(prob(rf, schema, bf, botIdx));
            }
            List<double[]> a = AdversarialBotSynthesizer.synthesize(chunk, new Random(base + 777_013L + i));
            if (a != null) {
                double[] af = Tier2Features.compute(toMillis(a), mode);
                if (af != null) adv.add(prob(rf, schema, af, botIdx));
            }
        }
        double[] h = humans.toArray(), nb = naive.toArray(), ab = adv.toArray();
        r.humanFpr = 100.0 * countGt(h, 0.5) / h.length;
        r.naiveAuc = rocAuc(nb, h);
        r.naiveTpr1 = 100.0 * countGe(nb, thresholdForFpr(h, TARGET_FPRS[0])) / nb.length;
        r.advAuc = rocAuc(ab, h);
        r.advEer = equalErrorRate(ab, h);
        r.advTpr1 = 100.0 * countGe(ab, thresholdForFpr(h, TARGET_FPRS[0])) / ab.length;
        r.advTpr5 = 100.0 * countGe(ab, thresholdForFpr(h, TARGET_FPRS[1])) / ab.length;
    }

    private static final class Row {
        Tier2Features.Mode mode;
        TrainSet trainSet;
        int trainN;
        double humanFpr, naiveAuc, naiveTpr1, advAuc, advEer, advTpr1, advTpr5;
    }

    // ---------------- helpers ----------------

    private static Instance mk(Instances data, double[] feat, int label) {
        Instance inst = new DenseInstance(data.numAttributes());
        inst.setDataset(data);
        for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
        inst.setValue(data.numAttributes() - 1, String.valueOf(label));
        return inst;
    }

    private static double prob(RandomForest rf, Instances schema, double[] feat, int botIdx) throws Exception {
        Instance inst = new DenseInstance(schema.numAttributes());
        inst.setDataset(schema);
        for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
        return rf.distributionForInstance(inst)[botIdx];
    }

    private static List<double[]> toMillis(List<double[]> pointsSec) {
        List<double[]> ms = new ArrayList<>(pointsSec.size());
        for (double[] p : pointsSec) ms.add(new double[]{p[0] * 1000.0, p[1], p[2]});
        return ms;
    }

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
        for (int k = 0; k < s.length; k++) if (k == 0 || s[k] != s[k - 1]) s[w++] = s[k];
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
        double[] toArray() { return Arrays.copyOf(data, size); }
    }
}
