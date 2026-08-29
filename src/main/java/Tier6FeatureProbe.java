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
 * TIER 6 -- is the temporal-ordering signal REAL, or is the ~1pp Tier-4 gain RF noise?
 *
 * Two dispositive checks, neither of which the Tier-4 sweep answers:
 *
 *  A. FEATURE DISTRIBUTIONS. For held-out Balabit test users, compute
 *     velocity_lag1_autocorr and velocity_step_roughness for (i) real human
 *     chunks, (ii) their AdversarialBotSynthesizer twins, (iii) their naive
 *     BalabitBotSynthesizer twins. Print mean / median / std per group and the
 *     single-feature ROC-AUC (human vs adversarial). If human autocorr sits well
 *     above adversarial, the feature captures what it was designed to; if both
 *     cluster near 0, it is broken and any downstream gain is noise.
 *     (Mirrors the vel_p90_p50_ratio distribution check from the Tier-2 notes.)
 *
 *  B. RF SEED VARIANCE. Rebuild the Tier-4 AUG_NAIVE_PLUS model for AUGMENTED and
 *     AUGMENTED_SEQ at RF seeds {1,2,3} and report the evasive-bot AUC / EER /
 *     TPR@<=1%FPR for each. If the AUGMENTED_SEQ - AUGMENTED gap is smaller than
 *     the across-seed spread, the feature adds nothing detectable at this sample
 *     size and the writeup must say so.
 *
 * Same held-out user split, synthesiser seeding and BURST timing as Tier4AdversarialEval.
 * Deterministic. Additive -- touches no Tier-1..5 code path.
 */
public class Tier6FeatureProbe {

    private static final long RNG_SEED = 42L;
    private static final int RF_TREES = 100;
    private static final int[] SEEDS = {1, 2, 3};

    public static void main(String[] args) throws Exception {
        File delbotDir = new File(BalabitValidationPipeline.DELBOT_DATA_DIR);
        File balabitDir = new File(BalabitValidationPipeline.BALABIT_DATA_DIR);
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

        // ---------------- A. feature distributions ----------------
        // SEQ feature indices in AUGMENTED_SEQ: the last two columns.
        int seqBase = Tier2Features.featureNames(Tier2Features.Mode.AUGMENTED).length; // 18
        Buf hAuto = new Buf(), hRough = new Buf(), aAuto = new Buf(), aRough = new Buf(), nAuto = new Buf(), nRough = new Buf();
        long base = RNG_SEED * 9_000_011L;
        for (int i = 0; i < testChunks.size(); i++) {
            List<double[]> chunk = testChunks.get(i);
            double[] hf = Tier2Features.compute(toMillis(chunk), Tier2Features.Mode.AUGMENTED_SEQ);
            if (hf != null) { hAuto.add(hf[seqBase]); hRough.add(hf[seqBase + 1]); }

            List<double[]> a = AdversarialBotSynthesizer.synthesize(chunk, new Random(base + 777_013L + i));
            if (a != null) {
                double[] af = Tier2Features.compute(toMillis(a), Tier2Features.Mode.AUGMENTED_SEQ);
                if (af != null) { aAuto.add(af[seqBase]); aRough.add(af[seqBase + 1]); }
            }
            Random rng = new Random(base + (long) i * 8);
            for (BalabitBotSynthesizer.BotType bt : BalabitBotSynthesizer.BotType.values()) {
                List<double[]> b = BalabitBotSynthesizer.synthesize(chunk, bt, BalabitBotSynthesizer.TimingModel.BURST,
                        new Random(base + (long) i * 8 + bt.ordinal()));
                if (b == null) continue;
                double[] bf = Tier2Features.compute(toMillis(b), Tier2Features.Mode.AUGMENTED_SEQ);
                if (bf != null) { nAuto.add(bf[seqBase]); nRough.add(bf[seqBase + 1]); }
            }
        }
        System.out.println("========================================================================");
        System.out.println(" A. SEQ FEATURE DISTRIBUTIONS  (held-out test users)");
        System.out.println("========================================================================");
        System.out.printf("%-26s %8s %8s %8s %8s%n", "group / feature", "n", "mean", "median", "std");
        row("human    velocity_lag1_autocorr", hAuto);
        row("adversar velocity_lag1_autocorr", aAuto);
        row("naive    velocity_lag1_autocorr", nAuto);
        row("human    velocity_step_roughness", hRough);
        row("adversar velocity_step_roughness", aRough);
        row("naive    velocity_step_roughness", nRough);
        System.out.printf("%nsingle-feature ROC-AUC (human vs adversarial):  lag1_autocorr=%.4f   step_roughness=%.4f%n",
                auc(aAuto.arr(), hAuto.arr()), auc(aRough.arr(), hRough.arr()));
        System.out.println("(AUC here = P(adversarial scores 'more bot' than human). For lag1_autocorr the bot");
        System.out.println(" is LOWER so its discriminating AUC is 1 - value; roughness the bot is higher.)");

        // ---------------- B. RF seed variance ----------------
        System.out.println("\n========================================================================");
        System.out.println(" B. RF SEED VARIANCE  -- Tier-4 AUG_NAIVE_PLUS, evasive bot");
        System.out.println("========================================================================");
        System.out.printf("%-16s %6s | %9s %9s %12s%n", "feat mode", "seed", "advAUC", "advEER", "advTPR@<=1%");
        for (Tier2Features.Mode mode : new Tier2Features.Mode[]{
                Tier2Features.Mode.AUGMENTED, Tier2Features.Mode.AUGMENTED_SEQ}) {
            Instances training = buildTraining(mode, delbotFolders, trainChunks);
            int botIdx = training.classAttribute().indexOfValue("1");
            double[] aucs = new double[SEEDS.length];
            for (int s = 0; s < SEEDS.length; s++) {
                RandomForest rf = new RandomForest();
                rf.setNumIterations(RF_TREES);
                rf.setSeed(SEEDS[s]);
                rf.buildClassifier(training);
                double[][] hb = scoreHumansAdv(rf, training, botIdx, mode, testChunks);
                double a = auc(hb[1], hb[0]);
                double eer = eer(hb[1], hb[0]);
                double tpr1 = tprAtFpr(hb[1], hb[0], 0.01);
                aucs[s] = a;
                System.out.printf("%-16s %6d | %9.4f %8.2f%% %11.2f%%%n", mode, SEEDS[s], a, eer * 100, tpr1 * 100);
            }
            System.out.printf("%-16s %6s | across-seed advAUC spread = %.4f  (min %.4f max %.4f)%n%n",
                    mode, "--", max(aucs) - min(aucs), min(aucs), max(aucs));
        }
        System.out.println("VERDICT RULE: if (AUGMENTED_SEQ mean advAUC - AUGMENTED mean advAUC) is not larger");
        System.out.println("than the bigger of the two across-seed spreads, the 2 features add nothing");
        System.out.println("detectable at this sample size -- report as a null result, not an improvement.");
    }

    // ---------------- training (mirrors Tier5 AUG_NAIVE_PLUS) ----------------

    private static Instances buildTraining(Tier2Features.Mode mode, List<File> folders,
                                           List<List<double[]>> trainChunks) throws Exception {
        Instances data = new Instances("t6_" + mode, Tier2Features.schema(mode), 0);
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
        BalabitBotSynthesizer.BotType[] pool = BalabitBotSynthesizer.BotType.values();
        long b = RNG_SEED * 7_000_003L;
        for (int i = 0; i < trainChunks.size(); i++) {
            List<double[]> chunk = trainChunks.get(i);
            double[] hf = Tier2Features.compute(toMillis(chunk), mode);
            if (hf != null) data.add(mk(data, hf, 0));
            Random rng = new Random(b + i);
            List<double[]> naive = BalabitBotSynthesizer.synthesize(
                    chunk, pool[rng.nextInt(pool.length)], BalabitBotSynthesizer.TimingModel.BURST, rng);
            if (naive != null) {
                double[] bf = Tier2Features.compute(toMillis(naive), mode);
                if (bf != null) data.add(mk(data, bf, 1));
            }
            List<double[]> adv = AdversarialBotSynthesizer.synthesize(chunk, new Random(b + 5_000_011L + i));
            if (adv != null) {
                double[] af = Tier2Features.compute(toMillis(adv), mode);
                if (af != null) data.add(mk(data, af, 1));
            }
        }
        return data;
    }

    private static double[][] scoreHumansAdv(RandomForest rf, Instances schema, int botIdx,
                                             Tier2Features.Mode mode, List<List<double[]>> testChunks) throws Exception {
        Buf h = new Buf(), a = new Buf();
        long base = RNG_SEED * 9_000_011L;
        for (int i = 0; i < testChunks.size(); i++) {
            List<double[]> chunk = testChunks.get(i);
            double[] hf = Tier2Features.compute(toMillis(chunk), mode);
            if (hf != null) h.add(prob(rf, schema, hf, botIdx));
            List<double[]> adv = AdversarialBotSynthesizer.synthesize(chunk, new Random(base + 777_013L + i));
            if (adv != null) {
                double[] af = Tier2Features.compute(toMillis(adv), mode);
                if (af != null) a.add(prob(rf, schema, af, botIdx));
            }
        }
        return new double[][]{h.arr(), a.arr()};
    }

    // ---------------- helpers ----------------

    private static void row(String label, Buf b) {
        double[] x = b.arr();
        System.out.printf("%-26s %8d %8.4f %8.4f %8.4f%n", label, x.length, mean(x), median(x), std(x, mean(x)));
    }

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

    private static double mean(double[] x) { double s = 0; for (double v : x) s += v; return x.length == 0 ? 0 : s / x.length; }
    private static double std(double[] x, double m) { if (x.length == 0) return 0; double s = 0; for (double v : x) s += (v - m) * (v - m); return Math.sqrt(s / x.length); }
    private static double median(double[] x) { if (x.length == 0) return 0; double[] s = x.clone(); Arrays.sort(s); return s.length % 2 == 1 ? s[s.length / 2] : 0.5 * (s[s.length / 2 - 1] + s[s.length / 2]); }
    private static double min(double[] x) { double m = Double.MAX_VALUE; for (double v : x) m = Math.min(m, v); return m; }
    private static double max(double[] x) { double m = -Double.MAX_VALUE; for (double v : x) m = Math.max(m, v); return m; }

    private static double auc(double[] pos, double[] neg) {
        int nPos = pos.length, nNeg = neg.length;
        if (nPos == 0 || nNeg == 0) return Double.NaN;
        double[] all = new double[nPos + nNeg];
        boolean[] isPos = new boolean[nPos + nNeg];
        for (int i = 0; i < nPos; i++) { all[i] = pos[i]; isPos[i] = true; }
        for (int i = 0; i < nNeg; i++) all[nPos + i] = neg[i];
        Integer[] order = new Integer[all.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (x, y) -> Double.compare(all[x], all[y]));
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
        double[] cand = concat(pos, neg);
        Arrays.sort(cand);
        double best = 1.0, gap = Double.MAX_VALUE;
        for (int i = 0; i <= cand.length; i++) {
            double t = (i < cand.length) ? cand[i] : Math.nextUp(cand[cand.length - 1]);
            double fpr = frac(neg, t), fnr = 1.0 - frac(pos, t);
            if (Math.abs(fpr - fnr) < gap) { gap = Math.abs(fpr - fnr); best = (fpr + fnr) / 2.0; }
        }
        return best;
    }

    private static double tprAtFpr(double[] pos, double[] neg, double targetFpr) {
        double[] s = neg.clone();
        Arrays.sort(s);
        int budget = (int) Math.floor(targetFpr * neg.length);
        int idxFromTop = Math.min(budget, neg.length - 1);
        double t = Math.nextUp(s[neg.length - 1 - idxFromTop]);
        return frac(pos, t);
    }

    private static double frac(double[] a, double t) { int c = 0; for (double v : a) if (v >= t) c++; return a.length == 0 ? 0 : c / (double) a.length; }

    private static double[] concat(double[] a, double[] b) {
        double[] r = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }

    private static final class Buf {
        private double[] d = new double[1024];
        private int s = 0;
        void add(double v) { if (s == d.length) d = Arrays.copyOf(d, d.length * 2); d[s++] = v; }
        double[] arr() { return Arrays.copyOf(d, s); }
    }
}
