import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * DEMO -- trains the ONE model the browser demo ships and freezes it to disk.
 *
 * Configuration (locked, do not substitute a different variant):
 *   features   : AUGMENTED (18)  -- Tier2Features.Mode.AUGMENTED
 *   training   : AUG_NAIVE_PLUS  -- DELBOT + one naive BalabitBotSynthesizer twin
 *                and one SHUFFLE AdversarialBotSynthesizer twin per training-user
 *                human chunk (identical to Tier4AdversarialEval / Tier5OperatingPointEval)
 *   held-out   : 5 Balabit users are never in training (same split as those tiers)
 *   RF         : 100 trees, seed 1
 *
 * Emits, under demo/model/ :
 *   rf.model            -- serialized weka.classifiers.trees.RandomForest
 *   schema.arff         -- the 18-attribute + class Instances header (empty)
 *   operating_point.txt -- the P(bot) threshold for a 1% human-FPR budget, plus
 *                          the metrics this exact build achieves on held-out users
 *   golden_vectors.json -- 25 real held-out Balabit chunks as {points, features}
 *                          for the JS<->Java feature-parity check
 *
 * Run:  java -cp "target/classes;<weka>;<bounce>" DemoModelTrainer
 */
public class DemoModelTrainer {

    private static final long RNG_SEED = 42L;
    private static final int RF_TREES = 100;
    private static final Tier2Features.Mode MODE = Tier2Features.Mode.AUGMENTED;
    private static final double FPR_BUDGET = 0.01;
    private static final int GOLDEN_N = 25;

    public static void main(String[] args) throws Exception {
        File out = new File("demo/model");
        out.mkdirs();

        List<File> delbotFolders = new ArrayList<>();
        BalabitValidationPipeline.findSessionFolders(new File(BalabitValidationPipeline.DELBOT_DATA_DIR), delbotFolders);
        List<File> balabitFiles = new ArrayList<>();
        BalabitValidationPipeline.findBalabitFiles(new File(BalabitValidationPipeline.BALABIT_DATA_DIR), balabitFiles);

        Map<String, List<List<double[]>>> byUser = new TreeMap<>();
        for (File f : balabitFiles) {
            String user = f.getParentFile().getName();
            List<double[]> raw = BalabitValidationPipeline.parseBalabitFile(f);
            for (List<double[]> cr : BalabitValidationPipeline.splitByGap(raw, BalabitValidationPipeline.GAP_THRESHOLD_SEC)) {
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
        System.out.println("TEST  users " + testUsers + " (" + testChunks.size() + " chunks)");

        // ---- training set: DELBOT + naive twin + SHUFFLE adversarial twin per train chunk ----
        Instances data = new Instances("demo_" + MODE, Tier2Features.schema(MODE), 0);
        data.setClassIndex(data.numAttributes() - 1);
        for (File folder : delbotFolders) {
            int label = folder.getName().startsWith("circles_human") ? 0 : 1;
            File[] files = folder.listFiles((d, n) -> n.endsWith(".txt"));
            if (files == null) continue;
            Arrays.sort(files);
            for (File f : files) {
                double[] feat = Tier2Features.compute(BalabitValidationPipeline.parseDelbotPoints(f), MODE);
                if (feat != null) data.add(mk(data, feat, label));
            }
        }
        BalabitBotSynthesizer.BotType[] pool = BalabitBotSynthesizer.BotType.values();
        long base = RNG_SEED * 7_000_003L;
        for (int i = 0; i < trainChunks.size(); i++) {
            List<double[]> chunk = trainChunks.get(i);
            double[] hf = Tier2Features.compute(toMillis(chunk), MODE);
            if (hf != null) data.add(mk(data, hf, 0));
            Random rng = new Random(base + i);
            List<double[]> naive = BalabitBotSynthesizer.synthesize(
                    chunk, pool[rng.nextInt(pool.length)], BalabitBotSynthesizer.TimingModel.BURST, rng);
            if (naive != null) {
                double[] bf = Tier2Features.compute(toMillis(naive), MODE);
                if (bf != null) data.add(mk(data, bf, 1));
            }
            List<double[]> adv = AdversarialBotSynthesizer.synthesize(
                    chunk, AdversarialBotSynthesizer.Ordering.SHUFFLE, new Random(base + 5_000_011L + i));
            if (adv != null) {
                double[] af = Tier2Features.compute(toMillis(adv), MODE);
                if (af != null) data.add(mk(data, af, 1));
            }
        }
        int nH = 0, nB = 0, botIdx = data.classAttribute().indexOfValue("1");
        for (int i = 0; i < data.numInstances(); i++)
            if ((int) data.instance(i).classValue() == botIdx) nB++; else nH++;
        System.out.printf("training : %d instances (%d human / %d bot)%n", data.numInstances(), nH, nB);

        RandomForest rf = new RandomForest();
        rf.setNumIterations(RF_TREES);
        rf.setSeed(1);
        rf.buildClassifier(data);

        // ---- held-out metrics + operating point ----
        double[] hScores = scoreAll(rf, data, botIdx, MODE, testChunks, null);
        long sbase = RNG_SEED * 9_000_011L;
        double[] naiveScores = scoreAll(rf, data, botIdx, MODE, testChunks, chunk -> {
            List<List<double[]>> all = new ArrayList<>();
            for (BalabitBotSynthesizer.BotType bt : pool) {
                List<double[]> b = BalabitBotSynthesizer.synthesize(chunk.chunk, bt,
                        BalabitBotSynthesizer.TimingModel.BURST, new Random(sbase + chunk.i * 8 + bt.ordinal()));
                if (b != null) all.add(b);
            }
            return all;
        });
        double[] advScores = scoreAll(rf, data, botIdx, MODE, testChunks, chunk -> {
            List<List<double[]>> l = new ArrayList<>();
            List<double[]> a = AdversarialBotSynthesizer.synthesize(chunk.chunk,
                    AdversarialBotSynthesizer.Ordering.SHUFFLE, new Random(sbase + 777_013L + chunk.i));
            if (a != null) l.add(a);
            return l;
        });

        double threshold = thresholdForFpr(hScores, FPR_BUDGET);
        double humFpr = 100.0 * countGe(hScores, threshold) / hScores.length;
        double humFprDefault = 100.0 * countGt(hScores, 0.5) / hScores.length;
        double naiveTpr = 100.0 * countGe(naiveScores, threshold) / naiveScores.length;
        double advTpr = 100.0 * countGe(advScores, threshold) / advScores.length;
        double naiveAuc = auc(naiveScores, hScores), advAuc = auc(advScores, hScores);

        // ---- serialize ----
        SerializationHelper.write(new File(out, "rf.model").getPath(), rf);
        Instances headerOnly = new Instances(data, 0);
        try (Writer w = new FileWriter(new File(out, "schema.arff"))) { w.write(headerOnly.toString()); }

        try (Writer w = new FileWriter(new File(out, "operating_point.txt"))) {
            w.write(String.format(
                "model            : AUGMENTED-18 / AUG_NAIVE_PLUS / RF 100 trees seed 1%n" +
                "P(bot) threshold : %.4f   (1%% human-FPR budget on held-out users)%n" +
                "                   tier5_operating_point_results.txt reject-option section quotes 0.8200%n" +
                "held-out users   : %s%n%n" +
                "AT THIS THRESHOLD (held-out Balabit users -- NOT a demo-page domain):%n" +
                "  human false-positive rate : %.2f%%%n" +
                "  naive-bot recall          : %.2f%%%n" +
                "  evasive-bot recall        : %.2f%%%n%n" +
                "RANKING (threshold-independent):%n" +
                "  naive-bot ROC-AUC   : %.4f%n" +
                "  evasive-bot ROC-AUC : %.4f%n%n" +
                "default 0.5 threshold human FPR : %.2f%%   (why the operating point is moved)%n",
                threshold, testUsers, humFpr, naiveTpr, advTpr, naiveAuc, advAuc, humFprDefault));
        }

        // ---- golden vectors for JS parity ----
        try (Writer w = new FileWriter(new File(out, "golden_vectors.json"))) {
            w.write("[\n");
            int written = 0;
            for (int i = 0; i < testChunks.size() && written < GOLDEN_N; i++) {
                List<double[]> ms = toMillis(testChunks.get(i));
                double[] feat = Tier2Features.compute(ms, MODE);
                if (feat == null) continue;
                if (written > 0) w.write(",\n");
                w.write("  {\"points\":[");
                for (int k = 0; k < ms.size(); k++) {
                    if (k > 0) w.write(",");
                    w.write(String.format("[%s,%s,%s]", trim(ms.get(k)[0]), trim(ms.get(k)[1]), trim(ms.get(k)[2])));
                }
                w.write("],\"features\":[");
                for (int k = 0; k < feat.length; k++) {
                    if (k > 0) w.write(",");
                    w.write(repr(feat[k]));
                }
                w.write("]}");
                written++;
            }
            w.write("\n]\n");
        }

        System.out.println("\n=== DEMO MODEL FROZEN -> demo/model/ ===");
        System.out.printf("threshold P(bot) >= %.4f  ->  BOT%n", threshold);
        System.out.printf("held-out: human FPR %.2f%%  |  naive recall %.2f%%  |  evasive recall %.2f%%%n",
                humFpr, naiveTpr, advTpr);
        System.out.printf("ROC-AUC: naive %.4f  evasive %.4f%n", naiveAuc, advAuc);
        System.out.printf("(default-0.5 human FPR would be %.2f%%)%n", humFprDefault);
    }

    // ---------------- helpers ----------------

    private interface BotGen { List<List<double[]>> gen(ChunkRef c); }
    private static final class ChunkRef { List<double[]> chunk; long i; }

    private static double[] scoreAll(RandomForest rf, Instances schema, int botIdx, Tier2Features.Mode mode,
                                     List<List<double[]>> testChunks, BotGen gen) throws Exception {
        List<Double> s = new ArrayList<>();
        ChunkRef ref = new ChunkRef();
        for (int i = 0; i < testChunks.size(); i++) {
            List<double[]> chunk = testChunks.get(i);
            if (gen == null) {
                double[] f = Tier2Features.compute(toMillis(chunk), mode);
                if (f != null) s.add(prob(rf, schema, f, botIdx));
            } else {
                ref.chunk = chunk; ref.i = i;
                for (List<double[]> bot : gen.gen(ref)) {
                    double[] f = Tier2Features.compute(toMillis(bot), mode);
                    if (f != null) s.add(prob(rf, schema, f, botIdx));
                }
            }
        }
        double[] arr = new double[s.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = s.get(i);
        return arr;
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

    private static int countGe(double[] a, double t) { int c = 0; for (double v : a) if (v >= t) c++; return c; }
    private static int countGt(double[] a, double t) { int c = 0; for (double v : a) if (v > t) c++; return c; }

    private static double thresholdForFpr(double[] humans, double targetFpr) {
        double[] s = humans.clone();
        Arrays.sort(s);
        int budget = (int) Math.floor(targetFpr * humans.length);
        int idxFromTop = Math.min(budget, humans.length - 1);
        return Math.nextUp(s[humans.length - 1 - idxFromTop]);
    }

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

    private static String trim(double v) {
        if (v == Math.rint(v) && !Double.isInfinite(v)) return Long.toString((long) v);
        return Double.toString(v);
    }
    private static String repr(double v) {
        if (Double.isNaN(v)) return "null";
        return Double.toString(v);
    }
}
