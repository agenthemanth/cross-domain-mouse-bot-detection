import weka.classifiers.CostMatrix;
import weka.classifiers.Classifier;
import weka.classifiers.meta.CostSensitiveClassifier;
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
 * TIER 5 -- fixing the OPERATING POINT, the blocker left by Tier 4.
 *
 * Tier 4's best configuration (AUGMENTED 18 features, trained on DELBOT + naive
 * AND adversarial bot twins of held-out-user chunks) reaches ROC-AUC 0.966
 * against the evasive bot -- but at the DEFAULT 0.5 threshold it flags 12.25%
 * of real humans. A double-digit human false-positive rate is not shippable as
 * a primary gate, and it is the single number that blocks the "commercially
 * viable" claim.
 *
 * AUC 0.966 says the SCORES separate well, so the 12.25% is a threshold /
 * class-prior problem before it is a model problem. This evaluates four ways of
 * attacking it, all on the identical held-out user split:
 *
 *   PLAIN          - the Tier-4 model, unchanged (the reference row)
 *   CLASS_BALANCED - same RF, but human training instances up-weighted so the
 *                    effective human:bot prior is 1:1 (augmentation leaves ~2:1
 *                    bot-heavy, which biases the forest toward predicting bot)
 *   COST_REWEIGHT  - CostSensitiveClassifier, minimizeExpectedCost=false: the
 *                    cost matrix REWEIGHTS TRAINING DATA, so the trees actually
 *                    learn different splits
 *   COST_EXPECTED  - CostSensitiveClassifier, minimizeExpectedCost=true: the
 *                    cost matrix is applied to the predicted distribution at
 *                    inference. This is mathematically near-equivalent to moving
 *                    the threshold, and is included precisely to test whether the
 *                    fancier options beat plain threshold tuning at matched FPR.
 *
 * For every variant the report gives the default-0.5 point AND a threshold sweep
 * at fixed human-FPR budgets (0.5 / 1 / 2 / 3 / 5%), reporting bot recall at each
 * against BOTH the naive suite and the evasive bot. That is the table an operator
 * actually needs: "at the false-positive rate I can tolerate, what do I catch?"
 *
 * Finally a REJECT-OPTION analysis on the best variant: abstain inside a score
 * band, defer those sessions to another signal, and report coverage vs. error.
 *
 * Deterministic (RF seed 1, single execution slot, sorted file order,
 * per-chunk-seeded synthesis). Does not touch any Tier-1..4 code path.
 */
public class Tier5OperatingPointEval {

    private static final long RNG_SEED = 42L;
    private static final int RF_TREES = 100;
    /** Human-FPR budgets to report bot recall at. */
    private static final double[] FPR_BUDGETS = {0.005, 0.01, 0.02, 0.03, 0.05};
    /** False-positive:false-negative cost ratio for the cost-sensitive variants. */
    private static final double FP_COST = 10.0;

    private enum Variant { PLAIN, CLASS_BALANCED, COST_REWEIGHT, COST_EXPECTED }

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
        TreeSet<String> trainUsers = new TreeSet<>(), calibUsers = new TreeSet<>(), reportUsers = new TreeSet<>();
        // 3-way user split (was 2-way). Even index -> train. Of the held-out odd-index
        // users, every 2nd one -> CALIB (threshold selection), the rest -> REPORT
        // (the FPR/recall the paper quotes). This closes the Tier-4/5 optimism where the
        // operating point was chosen on the very humans it was then measured against.
        int held = 0;
        for (int i = 0; i < users.size(); i++) {
            if (i % 2 == 0) { trainUsers.add(users.get(i)); }
            else { (held++ % 2 == 0 ? reportUsers : calibUsers).add(users.get(i)); }
        }
        List<List<double[]>> trainChunks = new ArrayList<>(), calibChunks = new ArrayList<>(), reportChunks = new ArrayList<>();
        for (String u : trainUsers) trainChunks.addAll(byUser.get(u));
        for (String u : calibUsers) calibChunks.addAll(byUser.get(u));
        for (String u : reportUsers) reportChunks.addAll(byUser.get(u));

        // Default: the Tier-4 winner (AUGMENTED, 18 feats). Pass a Tier2Features.Mode
        // name as arg 1 to evaluate a different feature set at the same operating points
        // (e.g. AUGMENTED_SEQ adds the 2 temporal-ordering features).
        Tier2Features.Mode mode = (args.length > 0)
                ? Tier2Features.Mode.valueOf(args[0].trim().toUpperCase())
                : Tier2Features.Mode.AUGMENTED;
        System.out.println("TIER 5 -- operating-point tuning on the Tier-4 best model");
        System.out.println("features : " + mode + " (" + Tier2Features.featureNames(mode).length + ")");
        System.out.println("TRAIN  users " + trainUsers + " (" + trainChunks.size() + " chunks)");
        System.out.println("CALIB  users " + calibUsers + " (" + calibChunks.size() + " chunks) -- threshold selection");
        System.out.println("REPORT users " + reportUsers + " (" + reportChunks.size() + " chunks) -- quoted FPR / recall");

        Instances training = buildTraining(mode, delbotFolders, trainChunks);
        int botIdx = training.classAttribute().indexOfValue("1");
        int nH = 0, nB = 0;
        double wH = 0, wB = 0;
        for (int i = 0; i < training.numInstances(); i++) {
            Instance in = training.instance(i);
            if ((int) in.classValue() == botIdx) { nB++; wB += in.weight(); } else { nH++; wH += in.weight(); }
        }
        System.out.printf("training : %d instances (%d human / %d bot, %.2f:1 bot-heavy)%n%n",
                training.numInstances(), nH, nB, nB / (double) nH);

        List<Row> rows = new ArrayList<>();
        for (Variant v : Variant.values()) {
            Instances tr = new Instances(training);
            if (v == Variant.CLASS_BALANCED) {
                double up = nB / (double) nH; // up-weight humans to parity
                for (int i = 0; i < tr.numInstances(); i++) {
                    Instance in = tr.instance(i);
                    if ((int) in.classValue() != botIdx) in.setWeight(up);
                }
            }
            Classifier model = buildModel(v, tr, botIdx);
            Row r = score(model, tr, botIdx, mode, calibChunks, reportChunks, v);
            rows.add(r);
            report(r);
        }

        summary(rows);

        // reject option on the variant with the best evasive-bot recall at <=1% FPR
        Row best = rows.get(0);
        for (Row r : rows) if (r.advTprAt[1] > best.advTprAt[1]) best = r;
        rejectOption(best);
    }

    // ---------------- models ----------------

    private static Classifier buildModel(Variant v, Instances tr, int botIdx) throws Exception {
        RandomForest rf = new RandomForest();
        rf.setNumIterations(RF_TREES);
        rf.setSeed(1);
        if (v == Variant.PLAIN || v == Variant.CLASS_BALANCED) {
            rf.buildClassifier(tr);
            return rf;
        }
        // cost matrix: penalise predicting BOT when truth is HUMAN by FP_COST
        CostMatrix cm = new CostMatrix(2);
        int humanIdx = 1 - botIdx;
        cm.setElement(humanIdx, botIdx, FP_COST); // truth human, predicted bot
        cm.setElement(botIdx, humanIdx, 1.0);     // truth bot, predicted human
        CostSensitiveClassifier cs = new CostSensitiveClassifier();
        cs.setClassifier(rf);
        cs.setCostMatrix(cm);
        cs.setSeed(1);
        cs.setMinimizeExpectedCost(v == Variant.COST_EXPECTED);
        cs.buildClassifier(tr);
        return cs;
    }

    private static Instances buildTraining(Tier2Features.Mode mode, List<File> folders,
                                           List<List<double[]>> trainChunks) throws Exception {
        Instances data = new Instances("tier5_" + mode, Tier2Features.schema(mode), 0);
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
        // AUG_NAIVE_PLUS: human + one naive twin + one adversarial twin per train chunk
        BalabitBotSynthesizer.BotType[] pool = BalabitBotSynthesizer.BotType.values();
        long base = RNG_SEED * 7_000_003L;
        for (int i = 0; i < trainChunks.size(); i++) {
            List<double[]> chunk = trainChunks.get(i);
            double[] hf = Tier2Features.compute(toMillis(chunk), mode);
            if (hf != null) data.add(mk(data, hf, 0));

            Random rng = new Random(base + i);
            List<double[]> naive = BalabitBotSynthesizer.synthesize(
                    chunk, pool[rng.nextInt(pool.length)], BalabitBotSynthesizer.TimingModel.BURST, rng);
            if (naive != null) {
                double[] bf = Tier2Features.compute(toMillis(naive), mode);
                if (bf != null) data.add(mk(data, bf, 1));
            }
            List<double[]> adv = AdversarialBotSynthesizer.synthesize(chunk, new Random(base + 5_000_011L + i));
            if (adv != null) {
                double[] af = Tier2Features.compute(toMillis(adv), mode);
                if (af != null) data.add(mk(data, af, 1));
            }
        }
        return data;
    }

    // ---------------- scoring ----------------

    private static Row score(Classifier model, Instances schema, int botIdx, Tier2Features.Mode mode,
                             List<List<double[]>> calibChunks, List<List<double[]>> reportChunks,
                             Variant v) throws Exception {
        // CALIB humans only -- used to CHOOSE the threshold for each FPR budget.
        double[] calibHumans = scoreHumans(model, schema, botIdx, mode, calibChunks);

        DoubleBuf humans = new DoubleBuf(), naive = new DoubleBuf(), adv = new DoubleBuf();
        // The model's OWN default decision (classifyInstance). Using "score > 0.5" would be
        // wrong for COST_EXPECTED, whose scores are negated expected costs, not probabilities.
        int hN = 0, hBot = 0, nN = 0, nBot = 0, aN = 0, aBot = 0;
        long base = RNG_SEED * 9_000_011L;
        for (int i = 0; i < reportChunks.size(); i++) {
            List<double[]> chunk = reportChunks.get(i);
            double[] hf = Tier2Features.compute(toMillis(chunk), mode);
            if (hf == null) continue;
            humans.add(prob(model, schema, hf, botIdx));
            hN++; if (predictsBot(model, schema, hf, botIdx)) hBot++;
            for (BalabitBotSynthesizer.BotType bt : BalabitBotSynthesizer.BotType.values()) {
                Random rng = new Random(base + (long) i * 8 + bt.ordinal());
                List<double[]> b = BalabitBotSynthesizer.synthesize(
                        chunk, bt, BalabitBotSynthesizer.TimingModel.BURST, rng);
                if (b == null) continue;
                double[] bf = Tier2Features.compute(toMillis(b), mode);
                if (bf != null) {
                    naive.add(prob(model, schema, bf, botIdx));
                    nN++; if (predictsBot(model, schema, bf, botIdx)) nBot++;
                }
            }
            List<double[]> a = AdversarialBotSynthesizer.synthesize(chunk, new Random(base + 777_013L + i));
            if (a != null) {
                double[] af = Tier2Features.compute(toMillis(a), mode);
                if (af != null) {
                    adv.add(prob(model, schema, af, botIdx));
                    aN++; if (predictsBot(model, schema, af, botIdx)) aBot++;
                }
            }
        }

        Row r = new Row();
        r.variant = v;
        r.humans = humans.toArray();
        r.naive = naive.toArray();
        r.adv = adv.toArray();
        r.calibHumans = calibHumans;
        r.humanFprAt50 = 100.0 * hBot / hN;
        r.naiveTprAt50 = 100.0 * nBot / nN;
        r.advTprAt50 = 100.0 * aBot / aN;
        r.naiveAuc = rocAuc(r.naive, r.humans);
        r.advAuc = rocAuc(r.adv, r.humans);
        r.advEer = equalErrorRate(r.adv, r.humans);
        r.thresholds = new double[FPR_BUDGETS.length];
        r.achievedFpr = new double[FPR_BUDGETS.length];
        r.naiveTprAt = new double[FPR_BUDGETS.length];
        r.advTprAt = new double[FPR_BUDGETS.length];
        for (int k = 0; k < FPR_BUDGETS.length; k++) {
            // threshold chosen on CALIB humans, all rates measured on the disjoint REPORT set
            double t = thresholdForFpr(r.calibHumans, FPR_BUDGETS[k]);
            r.thresholds[k] = t;
            r.achievedFpr[k] = 100.0 * countGe(r.humans, t) / r.humans.length;
            r.naiveTprAt[k] = 100.0 * countGe(r.naive, t) / r.naive.length;
            r.advTprAt[k] = 100.0 * countGe(r.adv, t) / r.adv.length;
        }
        return r;
    }

    private static double[] scoreHumans(Classifier model, Instances schema, int botIdx,
                                        Tier2Features.Mode mode, List<List<double[]>> chunks) throws Exception {
        DoubleBuf b = new DoubleBuf();
        for (List<double[]> chunk : chunks) {
            double[] hf = Tier2Features.compute(toMillis(chunk), mode);
            if (hf != null) b.add(prob(model, schema, hf, botIdx));
        }
        return b.toArray();
    }

    private static void report(Row r) {
        System.out.println("------------------------------------------------------------------------");
        System.out.println(" VARIANT: " + r.variant);
        System.out.println("------------------------------------------------------------------------");
        System.out.printf("  own default  : humanFPR %6.2f%%   naiveTPR %6.2f%%   advTPR %6.2f%%%n",
                r.humanFprAt50, r.naiveTprAt50, r.advTprAt50);
        System.out.printf("  ranking      : naiveAUC %.4f   advAUC %.4f   advEER %.2f%%%n",
                r.naiveAuc, r.advAuc, r.advEer * 100);
        System.out.println("  (threshold picked on CALIB users; every rate below measured on the disjoint REPORT users)");
        System.out.printf("  %-14s %12s %12s %12s%n", "target FPR", "achieved FPR", "naive TPR", "evasive TPR");
        for (int k = 0; k < FPR_BUDGETS.length; k++) {
            System.out.printf("  %-14s %11.2f%% %11.2f%% %11.2f%%%n",
                    String.format("<= %.1f%%", FPR_BUDGETS[k] * 100),
                    r.achievedFpr[k], r.naiveTprAt[k], r.advTprAt[k]);
        }
        System.out.println();
    }

    private static void summary(List<Row> rows) {
        System.out.println("========================================================================");
        System.out.println(" TIER 5 SUMMARY -- evasive-bot recall at fixed human-FPR budgets");
        System.out.println("========================================================================");
        System.out.printf("%-15s %11s %10s |", "variant", "ownFPR", "advAUC");
        for (double b : FPR_BUDGETS) System.out.printf(" %9s", String.format("<=%.1f%%", b * 100));
        System.out.println();
        for (Row r : rows) {
            System.out.printf("%-15s %10.2f%% %10.4f |", r.variant, r.humanFprAt50, r.advAuc);
            for (double t : r.advTprAt) System.out.printf(" %8.2f%%", t);
            System.out.println();
        }
        System.out.println("\n(columns = evasive-bot recall on the REPORT users when the threshold is set on");
        System.out.println(" the DISJOINT CALIB users to hold their FPR at or below the stated budget. The");
        System.out.println(" Tier-4 headline was ~12% FPR at the model's DEFAULT decision; these rows are what");
        System.out.println(" the same models deliver once the operating point is chosen deliberately -- and");
        System.out.println(" unlike the earlier Tier-5 run the point is now chosen on held-out humans.)");
        System.out.println("\nEXPECTED: COST_EXPECTED's advAUC must equal PLAIN's -- applying a cost matrix");
        System.out.println("to the predicted distribution is a monotone rescoring, i.e. threshold tuning by");
        System.out.println("another name. It moves the DEFAULT decision, it adds no ranking information.");
        System.out.println("Only CLASS_BALANCED and COST_REWEIGHT change what the trees actually learn, so");
        System.out.println("only their AUC/recall columns can legitimately differ from PLAIN.");
    }

    /**
     * Reject option: abstain on scores inside a band around the decision threshold and
     * defer those sessions to another signal. Reports how much traffic must be deferred
     * to reach a given accuracy on what remains.
     */
    private static void rejectOption(Row r) {
        System.out.println("\n========================================================================");
        System.out.println(" REJECT-OPTION ANALYSIS -- variant " + r.variant);
        System.out.println("========================================================================");
        double t = r.thresholds[1]; // the <=1% FPR operating point
        // Band widths are expressed as a FRACTION of the observed score range, not absolute:
        // COST_EXPECTED's scores are negated costs on a different scale from probabilities,
        // so a fixed +/-0.05 would mean something different for each variant.
        double lo0 = Double.MAX_VALUE, hi0 = -Double.MAX_VALUE;
        for (double[] arr : new double[][]{r.humans, r.naive, r.adv})
            for (double s : arr) { lo0 = Math.min(lo0, s); hi0 = Math.max(hi0, s); }
        double range = Math.max(hi0 - lo0, 1e-12);
        System.out.printf("Decision threshold %.4f (the <=1%% human-FPR point); score range [%.3f, %.3f].%n",
                t, lo0, hi0);
        System.out.printf("%-12s %12s %14s %14s %14s%n",
                "band +/-", "coverage", "humanFPR|kept", "advTPR|kept", "naiveTPR|kept");
        for (double frac : new double[]{0.0, 0.05, 0.10, 0.15, 0.20, 0.30}) {
            double band = frac * range;
            double lo = t - band, hi = t + band;
            int hKept = 0, hFp = 0;
            for (double s : r.humans) { if (s <= lo || s >= hi) { hKept++; if (s >= t) hFp++; } }
            int aKept = 0, aTp = 0;
            for (double s : r.adv) { if (s <= lo || s >= hi) { aKept++; if (s >= t) aTp++; } }
            int nKept = 0, nTp = 0;
            for (double s : r.naive) { if (s <= lo || s >= hi) { nKept++; if (s >= t) nTp++; } }
            int totalKept = hKept + aKept + nKept;
            int totalAll = r.humans.length + r.adv.length + r.naive.length;
            // Once the band's upper edge passes the maximum attainable score, EVERY
            // above-threshold session is deferred: recall reads 0% not because the model
            // failed but because it was never allowed to decide. Flag those rows rather
            // than presenting them as a result.
            String flag = (hi > hi0) ? "  <- band exceeds score ceiling; row is vacuous" : "";
            System.out.printf("%-12s %11.2f%% %13.2f%% %13.2f%% %13.2f%%%s%n",
                    String.format("%.0f%% rng", frac * 100),
                    100.0 * totalKept / totalAll,
                    hKept == 0 ? Double.NaN : 100.0 * hFp / hKept,
                    aKept == 0 ? Double.NaN : 100.0 * aTp / aKept,
                    nKept == 0 ? Double.NaN : 100.0 * nTp / nKept,
                    flag);
        }
        System.out.println("\nA reject band only helps if the deferred traffic can go somewhere useful");
        System.out.println("(a second signal, a challenge, a human review queue). Coverage is the share");
        System.out.println("of sessions the detector still decides on by itself.");
    }

    // ---------------- helpers ----------------

    private static final class Row {
        Variant variant;
        double[] humans, naive, adv;
        double[] calibHumans;
        double humanFprAt50, naiveTprAt50, advTprAt50, naiveAuc, advAuc, advEer;
        double[] thresholds, achievedFpr, naiveTprAt, advTprAt;
    }

    private static Instance mk(Instances data, double[] feat, int label) {
        Instance inst = new DenseInstance(data.numAttributes());
        inst.setDataset(data);
        for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
        inst.setValue(data.numAttributes() - 1, String.valueOf(label));
        return inst;
    }

    /**
     * Bot-ness score, higher = more bot-like, comparable across variants.
     *
     * CAREFUL -- verified against Weka 3.8.6 behaviour, not assumed: for
     * CostSensitiveClassifier with minimizeExpectedCost=true,
     * {@code distributionForInstance} does NOT return probabilities OR expected costs.
     * It computes the expected costs, takes argmin, and returns a HARD ONE-HOT vector
     * (1.0 on the chosen class, 0.0 elsewhere). Scored naively that yields exactly two
     * distinct values, so ROC-AUC is meaningless and every fixed-FPR sweep degenerates
     * to 0% -- which is what a first run of this file produced.
     *
     * The meaningful continuous score for that variant is the UNDERLYING forest's
     * P(bot): the cost matrix is applied on top of it at decision time and changes only
     * where the default cut falls, never the ranking. Reading the base classifier makes
     * the variant's AUC come out identical to PLAIN by construction -- which is the
     * point of including it, and is the control that proves cost-sensitive-at-inference
     * is threshold tuning wearing a different hat.
     */
    private static double prob(Classifier model, Instances schema, double[] feat, int botIdx) throws Exception {
        Instance inst = new DenseInstance(schema.numAttributes());
        inst.setDataset(schema);
        for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
        Classifier scorer = model;
        if (model instanceof CostSensitiveClassifier
                && ((CostSensitiveClassifier) model).getMinimizeExpectedCost()) {
            scorer = ((CostSensitiveClassifier) model).getClassifier();
        }
        return scorer.distributionForInstance(inst)[botIdx];
    }

    /** The model's own default decision, whatever its score semantics. */
    private static boolean predictsBot(Classifier model, Instances schema, double[] feat, int botIdx)
            throws Exception {
        Instance inst = new DenseInstance(schema.numAttributes());
        inst.setDataset(schema);
        for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
        return ((int) model.classifyInstance(inst)) == botIdx;
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
