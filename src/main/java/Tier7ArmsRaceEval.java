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
 * TIER 7 -- the arms race, round 7.
 *
 * Tier 6 added 2 temporal-ordering features (velocity_lag1_autocorr,
 * velocity_step_roughness) that gave a marginal gain against the Tier-4 evasive
 * bot, mostly via step_roughness (a random shuffle of the speed sequence is
 * slightly jerkier than real motion). Tier 7 asks the obvious follow-up: what if
 * the attacker also matches the ordering?
 *
 * {@link AdversarialBotSynthesizer.Ordering#BALLISTIC} arranges the SAME reused
 * (distance, dt) pairs into a single rise-then-fall speed ramp instead of a
 * random permutation. Speed multiset, endpoints, point count, duration and path
 * length stay matched (as in Tier 4); now lag-1 autocorrelation and successive-
 * difference are matched to real point-to-point motion too.
 *
 * Held-out Balabit user split (identical to Tier4AdversarialEval). Feature modes
 * AUGMENTED (18) and AUGMENTED_SEQ (20). Three training sets:
 *   AUG_SHUFFLE   - DELBOT + naive twins + SHUFFLE evasive twins (the Tier-4/5 best)
 *   AUG_BALLISTIC - DELBOT + naive twins + BALLISTIC evasive twins
 *   AUG_BOTH      - DELBOT + naive twins + SHUFFLE AND BALLISTIC evasive twins
 * Each scored against the naive suite, the SHUFFLE evasive bot and the BALLISTIC
 * evasive bot. The questions:
 *   1. does the BALLISTIC bot defeat a model trained only on SHUFFLE?
 *   2. is the AUGMENTED_SEQ edge gone once the attacker matches step_roughness?
 *   3. does training on BALLISTIC recover it, and at what human-FPR cost?
 *
 * Deterministic (RF seed 1, sorted files, per-chunk-seeded synthesis).
 */
public class Tier7ArmsRaceEval {

    private static final long RNG_SEED = 42L;
    private static final int RF_TREES = 100;
    private static final double TARGET_FPR = 0.01;

    private enum TrainSet { AUG_SHUFFLE, AUG_BALLISTIC, AUG_BOTH }

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

        // --- mechanistic check: does BALLISTIC actually match the Tier-6 features? ---
        int sb = Tier2Features.featureNames(Tier2Features.Mode.AUGMENTED).length; // seq cols start here
        double[] hA = new double[2], sA = new double[2], bA = new double[2];
        int hN = 0, sN = 0, bN = 0;
        long pb = RNG_SEED * 9_000_011L;
        for (int i = 0; i < testChunks.size(); i++) {
            List<double[]> chunk = testChunks.get(i);
            double[] hf = Tier2Features.compute(toMillis(chunk), Tier2Features.Mode.AUGMENTED_SEQ);
            if (hf != null) { hA[0] += hf[sb]; hA[1] += hf[sb + 1]; hN++; }
            List<double[]> s = AdversarialBotSynthesizer.synthesize(chunk,
                    AdversarialBotSynthesizer.Ordering.SHUFFLE, new Random(pb + 777_013L + i));
            if (s != null) { double[] f = Tier2Features.compute(toMillis(s), Tier2Features.Mode.AUGMENTED_SEQ);
                if (f != null) { sA[0] += f[sb]; sA[1] += f[sb + 1]; sN++; } }
            List<double[]> b = AdversarialBotSynthesizer.synthesize(chunk,
                    AdversarialBotSynthesizer.Ordering.BALLISTIC, new Random(pb + 888_019L + i));
            if (b != null) { double[] f = Tier2Features.compute(toMillis(b), Tier2Features.Mode.AUGMENTED_SEQ);
                if (f != null) { bA[0] += f[sb]; bA[1] += f[sb + 1]; bN++; } }
        }
        System.out.printf("mean SEQ features   lag1_autocorr   step_roughness%n");
        System.out.printf("  human       %14.4f %14.4f%n", hA[0] / hN, hA[1] / hN);
        System.out.printf("  SHUFFLE bot  %13.4f %14.4f%n", sA[0] / sN, sA[1] / sN);
        System.out.printf("  BALLISTIC bot%13.4f %14.4f%n%n", bA[0] / bN, bA[1] / bN);

        System.out.printf("%-14s %-14s %8s | %7s %7s %7s | %8s %8s %8s%n",
                "feat", "trainset", "humFPR",
                "nAUC", "shufA", "ballA", "shTPR@1", "baTPR@1", "baEER");
        List<Row> table = new ArrayList<>();
        for (Tier2Features.Mode mode : new Tier2Features.Mode[]{
                Tier2Features.Mode.AUGMENTED, Tier2Features.Mode.AUGMENTED_SEQ}) {
            Instances delbot = buildDelbot(mode, delbotFolders);
            for (TrainSet ts : TrainSet.values()) {
                Instances training = new Instances(delbot, delbot.numInstances() + 5 * trainChunks.size());
                training.addAll(delbot);
                augment(training, mode, trainChunks, ts);
                int botIdx = training.classAttribute().indexOfValue("1");

                RandomForest rf = new RandomForest();
                rf.setNumIterations(RF_TREES);
                rf.setSeed(1);
                rf.buildClassifier(training);

                Row r = score(rf, training, botIdx, mode, ts, testChunks);
                table.add(r);
                System.out.printf("%-14s %-14s %7.2f%% | %7.4f %7.4f %7.4f | %7.2f%% %7.2f%% %7.2f%%%n",
                        mode, ts, r.humFpr, r.naiveAuc, r.shufAuc, r.ballAuc,
                        r.shufTpr1, r.ballTpr1, r.ballEer * 100);
            }
            System.out.println();
        }

        System.out.println("========================================================================");
        System.out.println(" TIER 7 -- BALLISTIC evasive bot (matches speed ORDER as well as multiset)");
        System.out.println("========================================================================");
        System.out.println("nAUC=naive ROC-AUC  shufA/ballA=evasive ROC-AUC vs SHUFFLE / BALLISTIC bot");
        System.out.println("shTPR@1/baTPR@1=evasive recall at <=1% human FPR   baEER=BALLISTIC equal error rate");
        System.out.println();
        System.out.println("Read:");
        System.out.println(" * AUG_SHUFFLE x ballA  -- does a Tier-4/5 model meet a bot that also fixes ordering?");
        System.out.println(" * (AUGMENTED_SEQ ballA) vs (AUGMENTED ballA) -- is the Tier-6 edge gone?");
        System.out.println(" * AUG_BOTH -- does training on BALLISTIC recover it, and what does humFPR cost?");
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
                                List<List<double[]>> trainChunks, TrainSet ts) {
        BalabitBotSynthesizer.BotType[] pool = BalabitBotSynthesizer.BotType.values();
        long base = RNG_SEED * 7_000_003L;
        boolean wantShuf = (ts == TrainSet.AUG_SHUFFLE || ts == TrainSet.AUG_BOTH);
        boolean wantBall = (ts == TrainSet.AUG_BALLISTIC || ts == TrainSet.AUG_BOTH);
        for (int i = 0; i < trainChunks.size(); i++) {
            List<double[]> chunk = trainChunks.get(i);
            double[] hf = Tier2Features.compute(toMillis(chunk), mode);
            if (hf != null) training.add(mk(training, hf, 0));

            Random rng = new Random(base + i);
            List<double[]> naive = BalabitBotSynthesizer.synthesize(
                    chunk, pool[rng.nextInt(pool.length)], BalabitBotSynthesizer.TimingModel.BURST, rng);
            if (naive != null) {
                double[] bf = Tier2Features.compute(toMillis(naive), mode);
                if (bf != null) training.add(mk(training, bf, 1));
            }
            if (wantShuf) {
                List<double[]> s = AdversarialBotSynthesizer.synthesize(
                        chunk, AdversarialBotSynthesizer.Ordering.SHUFFLE, new Random(base + 5_000_011L + i));
                if (s != null) {
                    double[] af = Tier2Features.compute(toMillis(s), mode);
                    if (af != null) training.add(mk(training, af, 1));
                }
            }
            if (wantBall) {
                List<double[]> b = AdversarialBotSynthesizer.synthesize(
                        chunk, AdversarialBotSynthesizer.Ordering.BALLISTIC, new Random(base + 6_000_017L + i));
                if (b != null) {
                    double[] af = Tier2Features.compute(toMillis(b), mode);
                    if (af != null) training.add(mk(training, af, 1));
                }
            }
        }
    }

    // ---------------- scoring ----------------

    private static Row score(RandomForest rf, Instances schema, int botIdx, Tier2Features.Mode mode,
                             TrainSet ts, List<List<double[]>> testChunks) throws Exception {
        DoubleBuf humans = new DoubleBuf(), naive = new DoubleBuf(), shuf = new DoubleBuf(), ball = new DoubleBuf();
        long base = RNG_SEED * 9_000_011L;
        for (int i = 0; i < testChunks.size(); i++) {
            List<double[]> chunk = testChunks.get(i);
            double[] hf = Tier2Features.compute(toMillis(chunk), mode);
            if (hf == null) continue;
            humans.add(prob(rf, schema, hf, botIdx));

            for (BalabitBotSynthesizer.BotType bt : BalabitBotSynthesizer.BotType.values()) {
                List<double[]> b = BalabitBotSynthesizer.synthesize(chunk, bt,
                        BalabitBotSynthesizer.TimingModel.BURST, new Random(base + (long) i * 8 + bt.ordinal()));
                if (b == null) continue;
                double[] bf = Tier2Features.compute(toMillis(b), mode);
                if (bf != null) naive.add(prob(rf, schema, bf, botIdx));
            }
            List<double[]> s = AdversarialBotSynthesizer.synthesize(
                    chunk, AdversarialBotSynthesizer.Ordering.SHUFFLE, new Random(base + 777_013L + i));
            if (s != null) {
                double[] af = Tier2Features.compute(toMillis(s), mode);
                if (af != null) shuf.add(prob(rf, schema, af, botIdx));
            }
            List<double[]> b = AdversarialBotSynthesizer.synthesize(
                    chunk, AdversarialBotSynthesizer.Ordering.BALLISTIC, new Random(base + 888_019L + i));
            if (b != null) {
                double[] af = Tier2Features.compute(toMillis(b), mode);
                if (af != null) ball.add(prob(rf, schema, af, botIdx));
            }
        }
        double[] h = humans.toArray(), nb = naive.toArray(), sb = shuf.toArray(), bb = ball.toArray();
        Row r = new Row();
        r.mode = mode; r.ts = ts;
        r.humFpr = 100.0 * countGt(h, 0.5) / h.length;
        r.naiveAuc = auc(nb, h);
        r.shufAuc = auc(sb, h);
        r.ballAuc = auc(bb, h);
        double t = thresholdForFpr(h, TARGET_FPR);
        r.shufTpr1 = 100.0 * countGe(sb, t) / sb.length;
        r.ballTpr1 = 100.0 * countGe(bb, t) / bb.length;
        r.ballEer = eer(bb, h);
        return r;
    }

    private static final class Row {
        Tier2Features.Mode mode; TrainSet ts;
        double humFpr, naiveAuc, shufAuc, ballAuc, shufTpr1, ballTpr1, ballEer;
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

    private static List<double[]> toMillis(List<double[]> sec) {
        List<double[]> ms = new ArrayList<>(sec.size());
        for (double[] p : sec) ms.add(new double[]{p[0] * 1000.0, p[1], p[2]});
        return ms;
    }

    private static int countGe(double[] a, double t) { int c = 0; for (double v : a) if (v >= t) c++; return c; }
    private static int countGt(double[] a, double t) { int c = 0; for (double v : a) if (v > t) c++; return c; }

    private static double auc(double[] pos, double[] neg) {
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

    private static double eer(double[] pos, double[] neg) {
        if (pos.length == 0 || neg.length == 0) return Double.NaN;
        double[] cand = Arrays.copyOf(pos, pos.length + neg.length);
        System.arraycopy(neg, 0, cand, pos.length, neg.length);
        Arrays.sort(cand);
        double best = 1.0, gap = Double.MAX_VALUE;
        for (int i = 0; i <= cand.length; i++) {
            double t = (i < cand.length) ? cand[i] : Math.nextUp(cand[cand.length - 1]);
            double fpr = (double) countGe(neg, t) / neg.length;
            double fnr = 1.0 - (double) countGe(pos, t) / pos.length;
            if (Math.abs(fpr - fnr) < gap) { gap = Math.abs(fpr - fnr); best = (fpr + fnr) / 2.0; }
        }
        return best;
    }

    private static double thresholdForFpr(double[] humans, double targetFpr) {
        double[] s = humans.clone();
        Arrays.sort(s);
        int budget = (int) Math.floor(targetFpr * humans.length);
        int idxFromTop = Math.min(budget, humans.length - 1);
        return Math.nextUp(s[humans.length - 1 - idxFromTop]);
    }

    private static final class DoubleBuf {
        private double[] d = new double[1024];
        private int s = 0;
        void add(double v) { if (s == d.length) d = Arrays.copyOf(d, d.length * 2); d[s++] = v; }
        double[] toArray() { return Arrays.copyOf(d, s); }
    }
}
