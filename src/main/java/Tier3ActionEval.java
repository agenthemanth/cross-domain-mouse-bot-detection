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
 * TIER 3 -- action-level cross-domain evaluation.
 *
 * Same protocol as {@link Tier2AugmentedEval} (held-out Balabit user split,
 * DELBOT_ONLY vs augmented training, BASELINE vs SCALEFREE features, MATCHED
 * vs BURST test-bot timing) but the unit of classification is a single mouse
 * ACTION from {@link BalabitActionSegmenter} (median ~1.6 s / 11 points),
 * not a 3-second-gap chunk (median ~13.6 s / 66 points).
 *
 * Two questions:
 *   1. Does action-level segmentation ALONE improve zero-shot transfer over the
 *      Tier-1 gap-chunk baseline (pooled bot AUC 0.53)?  -> DELBOT_ONLY rows.
 *   2. Does it stack with synthetic-bot training augmentation, and is drag-drop
 *      the most discriminative action (Antal & Egyed-Zsigmond 2019)?  -> AUG_ALL
 *      rows + the per-action-type AUC breakdown.
 *
 * Deterministic: RF seed 1, single slot, sorted file order, per-action-seeded
 * bot synthesis, fixed per-(user,type) cap on action counts.
 */
public class Tier3ActionEval {

    private static final long RNG_SEED = 42L;
    private static final int RF_TREES = 100;
    private static final double[] TARGET_FPRS = {0.01, 0.05};
    /** Cap on actions kept per (user, action-type) -- keeps the run tractable and the
     *  action-type mix from being swamped by point-clicks. Taken in file order (deterministic). */
    private static final int PER_USER_TYPE_CAP = 1500;

    private enum TrainSet { DELBOT_ONLY, AUG_ALL }

    public static void main(String[] args) throws Exception {
        File delbotDir = new File(BalabitValidationPipeline.DELBOT_DATA_DIR);
        File balabitDir = new File(BalabitValidationPipeline.BALABIT_DATA_DIR);
        if (!delbotDir.exists() || !balabitDir.exists()) throw new RuntimeException("data dirs missing");

        List<File> delbotFolders = new ArrayList<>();
        BalabitValidationPipeline.findSessionFolders(delbotDir, delbotFolders);
        List<File> balabitFiles = new ArrayList<>();
        BalabitValidationPipeline.findBalabitFiles(balabitDir, balabitFiles);

        // ---- segment every session into typed actions, bucket by user ----
        Map<String, List<Act>> byUser = new TreeMap<>();
        for (File f : balabitFiles) {
            String user = f.getParentFile().getName();
            List<double[]> ev = BalabitValidationPipeline.parseBalabitEvents(f);
            List<Act> bucket = byUser.computeIfAbsent(user, k -> new ArrayList<>());
            for (BalabitActionSegmenter.Action a : BalabitActionSegmenter.segment(ev)) {
                bucket.add(new Act(a.type, a.points));
            }
        }
        // per-(user,type) cap, deterministic (segment() preserves file order; files are sorted)
        for (List<Act> bucket : byUser.values()) {
            EnumMap<BalabitActionSegmenter.ActionType, Integer> seen =
                    new EnumMap<>(BalabitActionSegmenter.ActionType.class);
            List<Act> kept = new ArrayList<>();
            for (Act a : bucket) {
                int c = seen.merge(a.type, 1, Integer::sum);
                if (c <= PER_USER_TYPE_CAP) kept.add(a);
            }
            bucket.clear();
            bucket.addAll(kept);
        }

        List<String> users = new ArrayList<>(byUser.keySet());
        TreeSet<String> trainUsers = new TreeSet<>(), testUsers = new TreeSet<>();
        for (int i = 0; i < users.size(); i++) (i % 2 == 0 ? trainUsers : testUsers).add(users.get(i));

        List<Act> trainActs = new ArrayList<>(), testActs = new ArrayList<>();
        for (String u : trainUsers) trainActs.addAll(byUser.get(u));
        for (String u : testUsers) testActs.addAll(byUser.get(u));

        System.out.println("Action-level cross-domain eval");
        System.out.println("  TRAIN users " + trainUsers + " -> " + trainActs.size() + " actions " + typeCounts(trainActs));
        System.out.println("  TEST  users " + testUsers + " -> " + testActs.size() + " actions " + typeCounts(testActs));
        System.out.println("  (per-(user,type) cap " + PER_USER_TYPE_CAP + ")\n");

        List<Row> table = new ArrayList<>();
        for (Tier2Features.Mode mode : new Tier2Features.Mode[]{
                Tier2Features.Mode.BASELINE, Tier2Features.Mode.SCALEFREE}) {

            Instances delbot = buildDelbot(mode, delbotFolders);

            for (TrainSet ts : TrainSet.values()) {
                Instances training = new Instances(delbot, delbot.numInstances() + 2 * trainActs.size());
                training.addAll(delbot);
                if (ts == TrainSet.AUG_ALL) addAugmentation(training, mode, trainActs);
                int botClassIndex = training.classAttribute().indexOfValue("1");

                RandomForest rf = new RandomForest();
                rf.setNumIterations(RF_TREES);
                rf.setSeed(1);
                rf.buildClassifier(training);

                for (BalabitBotSynthesizer.TimingModel tm : BalabitBotSynthesizer.TimingModel.values()) {
                    Row r = score(rf, training, botClassIndex, mode, ts, tm, testActs);
                    table.add(r);
                    System.out.printf("[%-9s %-11s test=%-13s] trainN=%-7d humFPR=%5.2f%%  AUC=%.4f  EER=%5.2f%%  TPR@<=1%%=%.2f%%  TPR@<=5%%=%.2f%%%n",
                            mode, ts, tm, training.numInstances(), r.humanFpr, r.pooledAuc, r.pooledEer * 100, r.tprAt1, r.tprAt5);
                    System.out.printf("            per-action-type AUC:  MM=%.4f  PC=%.4f  DD=%.4f%n",
                            r.typeAuc.getOrDefault(BalabitActionSegmenter.ActionType.MM, Double.NaN),
                            r.typeAuc.getOrDefault(BalabitActionSegmenter.ActionType.PC, Double.NaN),
                            r.typeAuc.getOrDefault(BalabitActionSegmenter.ActionType.DD, Double.NaN));
                }
            }
            System.out.println();
        }

        System.out.println("========================================================================");
        System.out.println(" TIER 3 -- ACTION-LEVEL  (held-out Balabit users)");
        System.out.println("========================================================================");
        System.out.printf("%-9s %-11s %-13s %8s %9s %9s %10s %10s%n",
                "feat", "trainset", "testBotTiming", "humFPR", "pooledAUC", "EER", "TPR@<=1%", "TPR@<=5%");
        for (Row r : table) {
            System.out.printf("%-9s %-11s %-13s %7.2f%% %9.4f %8.2f%% %9.2f%% %9.2f%%%n",
                    r.mode, r.trainSet, r.timing, r.humanFpr, r.pooledAuc, r.pooledEer * 100, r.tprAt1, r.tprAt5);
        }
        System.out.println("\nCompare DELBOT_ONLY here vs the Tier-1 gap-chunk baseline (pooled bot AUC 0.53,");
        System.out.println("human FPR 9.85%): a higher AUC = action segmentation helps zero-shot transfer.");
        System.out.println("Caveat: test bots share the BalabitBotSynthesizer family; no GAN trajectories.");
    }

    // ---------------- training set ----------------

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
                data.add(mk(data, feat, label));
            }
        }
        return data;
    }

    private static void addAugmentation(Instances training, Tier2Features.Mode mode, List<Act> trainActs) {
        BalabitBotSynthesizer.BotType[] pool = BalabitBotSynthesizer.BotType.values();
        long seedBase = RNG_SEED * 7_000_003L;
        for (int i = 0; i < trainActs.size(); i++) {
            Act a = trainActs.get(i);
            double[] hf = Tier2Features.compute(toMillis(a.points), mode);
            if (hf != null) training.add(mk(training, hf, 0));

            Random rng = new Random(seedBase + i);
            BalabitBotSynthesizer.BotType type = pool[rng.nextInt(pool.length)];
            List<double[]> bot = BalabitBotSynthesizer.synthesize(
                    a.points, type, BalabitBotSynthesizer.TimingModel.BURST, rng);
            if (bot == null) continue;
            double[] bf = Tier2Features.compute(toMillis(bot), mode);
            if (bf != null) training.add(mk(training, bf, 1));
        }
    }

    // ---------------- scoring ----------------

    private static Row score(RandomForest rf, Instances schema, int botClassIndex,
                             Tier2Features.Mode mode, TrainSet ts, BalabitBotSynthesizer.TimingModel tm,
                             List<Act> testActs) throws Exception {
        DoubleBuf humans = new DoubleBuf();
        DoubleBuf allBots = new DoubleBuf();
        Map<BalabitActionSegmenter.ActionType, DoubleBuf> typeHumans =
                new EnumMap<>(BalabitActionSegmenter.ActionType.class);
        Map<BalabitActionSegmenter.ActionType, DoubleBuf> typeBots =
                new EnumMap<>(BalabitActionSegmenter.ActionType.class);
        for (BalabitActionSegmenter.ActionType t : BalabitActionSegmenter.ActionType.values()) {
            typeHumans.put(t, new DoubleBuf());
            typeBots.put(t, new DoubleBuf());
        }

        long seedBase = RNG_SEED * 9_000_011L + tm.ordinal() * 101L;
        for (int i = 0; i < testActs.size(); i++) {
            Act a = testActs.get(i);
            double[] hf = Tier2Features.compute(toMillis(a.points), mode);
            if (hf == null) continue;
            double hs = prob(rf, schema, hf, botClassIndex);
            humans.add(hs);
            typeHumans.get(a.type).add(hs);

            for (BalabitBotSynthesizer.BotType bt : BalabitBotSynthesizer.BotType.values()) {
                Random rng = new Random(seedBase + (long) i * 4 + bt.ordinal());
                List<double[]> bot = BalabitBotSynthesizer.synthesize(a.points, bt, tm, rng);
                if (bot == null) continue;
                double[] bf = Tier2Features.compute(toMillis(bot), mode);
                if (bf == null) continue;
                double bs = prob(rf, schema, bf, botClassIndex);
                allBots.add(bs);
                typeBots.get(a.type).add(bs);
            }
        }

        double[] h = humans.toArray();
        double[] b = allBots.toArray();
        Row r = new Row();
        r.mode = mode; r.trainSet = ts; r.timing = tm;
        r.humanFpr = 100.0 * countGt(h, 0.5) / h.length;
        r.pooledAuc = rocAuc(b, h);
        r.pooledEer = equalErrorRate(b, h);
        r.tprAt1 = 100.0 * countGe(b, thresholdForFpr(h, TARGET_FPRS[0])) / b.length;
        r.tprAt5 = 100.0 * countGe(b, thresholdForFpr(h, TARGET_FPRS[1])) / b.length;
        for (BalabitActionSegmenter.ActionType t : BalabitActionSegmenter.ActionType.values()) {
            double[] th = typeHumans.get(t).toArray();
            double[] tb = typeBots.get(t).toArray();
            if (th.length > 0 && tb.length > 0) r.typeAuc.put(t, rocAuc(tb, th));
        }
        return r;
    }

    // ---------------- helpers ----------------

    private static final class Act {
        final BalabitActionSegmenter.ActionType type;
        final List<double[]> points;
        Act(BalabitActionSegmenter.ActionType type, List<double[]> points) { this.type = type; this.points = points; }
    }

    private static String typeCounts(List<Act> acts) {
        EnumMap<BalabitActionSegmenter.ActionType, Integer> m = new EnumMap<>(BalabitActionSegmenter.ActionType.class);
        for (Act a : acts) m.merge(a.type, 1, Integer::sum);
        return m.toString();
    }

    private static final class Row {
        Tier2Features.Mode mode;
        TrainSet trainSet;
        BalabitBotSynthesizer.TimingModel timing;
        double humanFpr, pooledAuc, pooledEer, tprAt1, tprAt5;
        Map<BalabitActionSegmenter.ActionType, Double> typeAuc =
                new EnumMap<>(BalabitActionSegmenter.ActionType.class);
    }

    private static Instance mk(Instances data, double[] feat, int label) {
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

    // ---- metrics (same pure-math helpers as the other evals) ----

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
        double[] toArray() { return Arrays.copyOf(data, size); }
    }
}
