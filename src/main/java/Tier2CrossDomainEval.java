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
 * TIER 2 cross-domain evaluation. Same protocol as {@link BalabitCrossDomainEval}
 * (train RandomForest on DELBOT, score real Balabit human chunks as negatives +
 * matched synthetic {@link BalabitBotSynthesizer} bot twins as positives), run
 * three times with three feature sets from {@link Tier2Features}:
 *
 *   BASELINE  - the Tier-1 seven. MUST reproduce BalabitCrossDomainEval:
 *               pooled ROC-AUC 0.5326, human FPR 9.85%, TPR@0.5 0.39%.
 *   AUGMENTED - Tier-1 seven + 11 domain-bridging features.
 *   SCALEFREE - dimensionless features only (drops absolute velocity/accel/jerk).
 *
 * Prints a per-mode breakdown and a compact before/after comparison table.
 *
 * Everything is deterministic: RandomForest seed 1, single execution slot,
 * sorted DELBOT/Balabit file order, per-chunk-seeded bot synthesis.
 */
public class Tier2CrossDomainEval {

    private static final long RNG_SEED = 42L;
    private static final int RF_TREES = 100;
    private static final double[] TARGET_FPRS = {0.01, 0.05};

    private static BalabitBotSynthesizer.TimingModel timing = BalabitBotSynthesizer.TimingModel.MATCHED_SPREAD;

    public static void main(String[] args) throws Exception {
        if (args.length > 0) {
            timing = BalabitBotSynthesizer.TimingModel.valueOf(args[0].trim().toUpperCase());
        }
        System.out.println("Synthetic-bot timing model: " + timing
                + "  (pass MATCHED_SPREAD | BURST as arg 1)");

        File delbotDir = new File(BalabitValidationPipeline.DELBOT_DATA_DIR);
        File balabitDir = new File(BalabitValidationPipeline.BALABIT_DATA_DIR);
        if (!delbotDir.exists()) throw new RuntimeException("DELBOT data not found at " + delbotDir.getAbsolutePath());
        if (!balabitDir.exists()) throw new RuntimeException("Balabit data not found at " + balabitDir.getAbsolutePath());

        List<File> delbotFolders = new ArrayList<>();
        BalabitValidationPipeline.findSessionFolders(delbotDir, delbotFolders);
        List<File> balabitFiles = new ArrayList<>();
        BalabitValidationPipeline.findBalabitFiles(balabitDir, balabitFiles);
        System.out.println("DELBOT session folders: " + delbotFolders.size()
                + " | Balabit raw session files: " + balabitFiles.size());

        Tier2Features.Mode[] modes = Tier2Features.Mode.values();
        Map<Tier2Features.Mode, ModeResult> results = new EnumMap<>(Tier2Features.Mode.class);

        for (Tier2Features.Mode mode : modes) {
            System.out.println("\n########################################################################");
            System.out.println("# MODE: " + mode + "   features: "
                    + String.join(", ", Tier2Features.featureNames(mode)));
            System.out.println("########################################################################");
            results.put(mode, runMode(mode, delbotFolders, balabitFiles));
        }

        printComparison(results);
    }

    // ---------------- one mode: train + score + report ----------------

    private static ModeResult runMode(Tier2Features.Mode mode,
                                      List<File> delbotFolders,
                                      List<File> balabitFiles) throws Exception {
        Instances training = buildTraining(mode, delbotFolders);
        int botClassIndex = training.classAttribute().indexOfValue("1");
        int nHuman = 0, nBot = 0;
        for (int i = 0; i < training.numInstances(); i++) {
            if ((int) training.instance(i).classValue() == botClassIndex) nBot++; else nHuman++;
        }
        System.out.println("  DELBOT training: " + training.numInstances()
                + " instances (" + nHuman + " human / " + nBot + " bot), "
                + (training.numAttributes() - 1) + " features");

        RandomForest rf = new RandomForest();
        rf.setNumIterations(RF_TREES);
        rf.setSeed(1);
        rf.buildClassifier(training);

        DoubleBuf humanScores = new DoubleBuf();
        Map<BalabitBotSynthesizer.BotType, DoubleBuf> botScores =
                new EnumMap<>(BalabitBotSynthesizer.BotType.class);
        for (BalabitBotSynthesizer.BotType t : BalabitBotSynthesizer.BotType.values()) {
            botScores.put(t, new DoubleBuf());
        }
        Map<String, int[]> perUser = new TreeMap<>();

        int chunkCounter = 0, filesProcessed = 0;
        for (File file : balabitFiles) {
            String userFolder = file.getParentFile().getName();
            List<double[]> rawPoints = BalabitValidationPipeline.parseBalabitFile(file); // (t_sec, x, y)

            for (List<double[]> chunkRaw : BalabitValidationPipeline.splitByGap(
                    rawPoints, BalabitValidationPipeline.GAP_THRESHOLD_SEC)) {
                List<double[]> chunk = BalabitValidationPipeline.collapseToDistinctTimestamps(chunkRaw);
                if (chunk.size() < BalabitValidationPipeline.MIN_POINTS_PER_SESSION) continue;

                double[] humanFeat = Tier2Features.compute(toMillis(chunk), mode);
                if (humanFeat == null) continue;
                double humanScore = score(rf, training, humanFeat, botClassIndex);
                humanScores.add(humanScore);
                perUser.computeIfAbsent(userFolder, k -> new int[2]);
                perUser.get(userFolder)[0]++;
                if (humanScore > 0.5) perUser.get(userFolder)[1]++;

                long chunkSeed = RNG_SEED * 1_000_003L + (chunkCounter++);
                for (BalabitBotSynthesizer.BotType type : BalabitBotSynthesizer.BotType.values()) {
                    List<double[]> bot = BalabitBotSynthesizer.synthesize(
                            chunk, type, timing, new Random(chunkSeed + type.ordinal()));
                    if (bot == null) continue;
                    double[] botFeat = Tier2Features.compute(toMillis(bot), mode);
                    if (botFeat == null) continue;
                    botScores.get(type).add(score(rf, training, botFeat, botClassIndex));
                }
            }
            if (++filesProcessed % 400 == 0) {
                System.out.println("  ..." + filesProcessed + "/" + balabitFiles.size()
                        + " files, " + humanScores.size() + " human chunks");
            }
        }

        double[] humans = humanScores.toArray();
        Map<BalabitBotSynthesizer.BotType, double[]> bots = new EnumMap<>(BalabitBotSynthesizer.BotType.class);
        DoubleBuf pooled = new DoubleBuf();
        for (Map.Entry<BalabitBotSynthesizer.BotType, DoubleBuf> e : botScores.entrySet()) {
            bots.put(e.getKey(), e.getValue().toArray());
            pooled.addAll(e.getValue());
        }
        double[] allBots = pooled.toArray();

        ModeResult r = new ModeResult();
        r.mode = mode;
        r.numFeatures = training.numAttributes() - 1;
        r.nHumans = humans.length;
        r.humanFpAt50 = countGt(humans, 0.5);
        r.humanFprAt50 = 100.0 * r.humanFpAt50 / humans.length;
        r.pooledAuc = rocAuc(allBots, humans);
        r.pooledEer = equalErrorRate(allBots, humans);
        r.pooledTprAt50 = 100.0 * countGt(allBots, 0.5) / allBots.length;

        double[] thresholds = new double[TARGET_FPRS.length];
        for (int i = 0; i < TARGET_FPRS.length; i++) thresholds[i] = thresholdForFpr(humans, TARGET_FPRS[i]);
        r.pooledTprAt1pct = 100.0 * countGe(allBots, thresholds[0]) / allBots.length;
        r.pooledTprAt5pct = 100.0 * countGe(allBots, thresholds[1]) / allBots.length;

        // per-mode detail
        System.out.printf("  human FPR @0.5 : %.2f%%  (%d / %d)%n", r.humanFprAt50, r.humanFpAt50, r.nHumans);
        System.out.printf("  %-30s %8s %8s %8s %10s %10s%n",
                "bot type", "TPR@.5", "AUC", "EER", "TPR@<=1%", "TPR@<=5%");
        for (BalabitBotSynthesizer.BotType type : BalabitBotSynthesizer.BotType.values()) {
            double[] b = bots.get(type);
            if (b.length == 0) continue;
            double tpr50 = 100.0 * countGt(b, 0.5) / b.length;
            double auc = rocAuc(b, humans);
            double eer = equalErrorRate(b, humans) * 100;
            double t1 = 100.0 * countGe(b, thresholds[0]) / b.length;
            double t5 = 100.0 * countGe(b, thresholds[1]) / b.length;
            System.out.printf("  %-30s %7.2f%% %8.4f %7.2f%% %9.2f%% %9.2f%%%n",
                    type, tpr50, auc, eer, t1, t5);
            r.perTypeAuc.put(type, auc);
        }
        System.out.printf("  %-30s %7.2f%% %8.4f %7.2f%% %9.2f%% %9.2f%%%n",
                "POOLED", r.pooledTprAt50, r.pooledAuc, r.pooledEer * 100, r.pooledTprAt1pct, r.pooledTprAt5pct);
        r.perUser = perUser;
        return r;
    }

    private static Instances buildTraining(Tier2Features.Mode mode, List<File> delbotFolders) throws Exception {
        Instances data = new Instances("delbot_" + mode, Tier2Features.schema(mode), 0);
        data.setClassIndex(data.numAttributes() - 1);
        for (File folder : delbotFolders) {
            int label = folder.getName().startsWith("circles_human") ? 0 : 1;
            File[] files = folder.listFiles((d, n) -> n.endsWith(".txt"));
            if (files == null) continue;
            Arrays.sort(files);
            for (File f : files) {
                double[] feat = Tier2Features.compute(BalabitValidationPipeline.parseDelbotPoints(f), mode);
                if (feat == null) continue;
                Instance inst = new DenseInstance(data.numAttributes());
                inst.setDataset(data);
                for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
                inst.setValue(data.numAttributes() - 1, String.valueOf(label));
                data.add(inst);
            }
        }
        return data;
    }

    private static double score(RandomForest rf, Instances schema, double[] feat, int botClassIndex) throws Exception {
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

    // ---------------- comparison table ----------------

    private static void printComparison(Map<Tier2Features.Mode, ModeResult> results) {
        System.out.println("\n========================================================================");
        System.out.println(" TIER 2 BEFORE / AFTER  (cross-domain: DELBOT-trained -> Balabit)");
        System.out.println("========================================================================");
        System.out.printf("%-11s %6s %10s %10s %9s %11s %11s%n",
                "mode", "#feat", "humanFPR", "pooledAUC", "pooledEER", "TPR@<=1%FPR", "TPR@<=5%FPR");
        for (Tier2Features.Mode m : Tier2Features.Mode.values()) {
            ModeResult r = results.get(m);
            System.out.printf("%-11s %6d %9.2f%% %10.4f %8.2f%% %10.2f%% %10.2f%%%n",
                    m, r.numFeatures, r.humanFprAt50, r.pooledAuc, r.pooledEer * 100,
                    r.pooledTprAt1pct, r.pooledTprAt5pct);
        }
        System.out.println("\nPer-bot-type pooled ROC-AUC:");
        System.out.printf("%-30s", "bot type");
        for (Tier2Features.Mode m : Tier2Features.Mode.values()) System.out.printf(" %10s", m);
        System.out.println();
        for (BalabitBotSynthesizer.BotType t : BalabitBotSynthesizer.BotType.values()) {
            System.out.printf("%-30s", t);
            for (Tier2Features.Mode m : Tier2Features.Mode.values()) {
                Double a = results.get(m).perTypeAuc.get(t);
                System.out.printf(" %10.4f", a == null ? Double.NaN : a);
            }
            System.out.println();
        }
        System.out.println("\nInterpretation guide: AUC ~0.50 = no cross-domain discrimination (Tier-1 result");
        System.out.println("was 0.53). A material rise in SCALEFREE/AUGMENTED AUC with humanFPR held near or");
        System.out.println("below the 9.85% baseline is the Tier-2 win condition. Synthetic bots are an");
        System.out.println("optimistic bound (no GAN trajectories) -- see BalabitBotSynthesizer.");
    }

    // ---------------- result holder ----------------

    private static final class ModeResult {
        Tier2Features.Mode mode;
        int numFeatures;
        int nHumans;
        int humanFpAt50;
        double humanFprAt50;
        double pooledAuc, pooledEer, pooledTprAt50, pooledTprAt1pct, pooledTprAt5pct;
        Map<BalabitBotSynthesizer.BotType, Double> perTypeAuc =
                new EnumMap<>(BalabitBotSynthesizer.BotType.class);
        Map<String, int[]> perUser;
    }

    // ---------------- metrics (copied from BalabitCrossDomainEval; pure math, no feature logic) ----------------

    private static int countGe(double[] arr, double t) {
        int c = 0;
        for (double v : arr) if (v >= t) c++;
        return c;
    }

    private static int countGt(double[] arr, double t) {
        int c = 0;
        for (double v : arr) if (v > t) c++;
        return c;
    }

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
            double avgRank = (i + j) / 2.0 + 1.0;
            for (int k = i; k <= j; k++) rank[order[k]] = avgRank;
            i = j + 1;
        }
        double sumPosRanks = 0.0;
        for (int k = 0; k < all.length; k++) if (isPos[k]) sumPosRanks += rank[k];
        double u = sumPosRanks - (long) nPos * (nPos + 1L) / 2.0;
        return u / ((double) ((long) nPos * nNeg));
    }

    private static double equalErrorRate(double[] pos, double[] neg) {
        if (pos.length == 0 || neg.length == 0) return Double.NaN;
        double[] cand = distinctSorted(concat(pos, neg));
        double bestGap = Double.MAX_VALUE, eerAtBest = 1.0;
        for (int i = 0; i <= cand.length; i++) {
            double t = (i < cand.length) ? cand[i] : Math.nextUp(cand[cand.length - 1]);
            double fpr = (double) countGe(neg, t) / neg.length;
            double fnr = 1.0 - (double) countGe(pos, t) / pos.length;
            double gap = Math.abs(fpr - fnr);
            if (gap < bestGap) { bestGap = gap; eerAtBest = (fpr + fnr) / 2.0; }
        }
        return eerAtBest;
    }

    private static double thresholdForFpr(double[] humans, double targetFpr) {
        double[] sortedDesc = humans.clone();
        Arrays.sort(sortedDesc);
        int budget = (int) Math.floor(targetFpr * humans.length);
        int idxFromTop = Math.min(budget, humans.length - 1);
        int pos = humans.length - 1 - idxFromTop;
        return Math.nextUp(sortedDesc[pos]);
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
        void add(double v) {
            if (size == data.length) data = Arrays.copyOf(data, data.length * 2);
            data[size++] = v;
        }
        void addAll(DoubleBuf other) { for (int i = 0; i < other.size; i++) add(other.data[i]); }
        int size() { return size; }
        double[] toArray() { return Arrays.copyOf(data, size); }
    }
}
