import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * ROUGH SANITY CHECK ONLY -- NOT the project's cross-dataset result.
 * Trains a Random Forest (100 trees) ONLY on DELBOT-Mouse (never on the human
 * reference set), then classifies every session in
 * human_mouse_reference_features.csv -- a held-out, known-all-human dataset --
 * as a quick smell test for gross over-flagging of real humans.
 *
 * This check is DEMOTED from the cross-dataset story (methodology decision,
 * 2026-08-28) because it is fundamentally confounded: mightymerge has no raw
 * x/y, so the reference CSV's velocity/acceleration/jerk columns cannot be
 * computed the way DELBOT training computes them (different formula AND ~1000x
 * scale gap, with garbage negatives). 3 of 7 features are on mismatched scales
 * train vs test; only num_points / duration_ms / path_efficiency are
 * comparable. --> BalabitValidationPipeline (real x/y/t, one feature-code path
 * end to end) is the sound cross-dataset evaluation; use its numbers.
 *
 * For the record: this check currently reports FPR 5.45% micro
 * (2479 / 45465 sessions) / 4.03% macro. The earlier "2.18% (J48) -> 0.71%
 * (RF)" numbers predate the training-side feature fix and are retracted.
 *
 * Random Forest is the final classifier, selected after a documented
 * head-to-head comparison against J48 (see project report). J48 remains
 * in the report as the interpretable model used to catch an earlier data-
 * leakage issue, but is no longer used in this pipeline going forward.
 *
 * The human reference data is NEVER used for training, and NEVER merged with
 * DELBOT. This deliberately isolates "does the model over-flag genuine humans
 * from a different data source" from "does the model separate human vs bot".
 *
 * Reporting is grouped by source_dataset and by person, not by random individual
 * rows -- a single prolific person or file should not be able to dominate the
 * headline false-positive number.
 *
 * Feature extraction is delegated to BalabitValidationPipeline.computeFeatures
 * so the DELBOT training set built here is bit-identical to the one used by
 * BalabitValidationPipeline and BalabitCrossDomainEval (same tied-timestamp
 * collapse, same 1 ms dt floor).
 */
public class FalsePositiveEvaluator {

    private static final String DELBOT_DATA_DIR = "delbot_data";
    private static final String HUMAN_REFERENCE_CSV = "human_mouse_reference_features.csv";

    public static void main(String[] args) throws Exception {
        System.out.println("NOTE: rough sanity check only -- NOT the project's cross-dataset result.");
        System.out.println("mightymerge has no raw x/y, so 3 of 7 features are computed differently");
        System.out.println("here than in DELBOT training. Use BalabitValidationPipeline for the");
        System.out.println("sound cross-dataset false-positive rate.\n");
        System.out.println("--- STEP 1: TRAIN RANDOM FOREST ON DELBOT ONLY ---");
        Instances trainingData = buildDelbotTrainingSet();
        System.out.println("DELBOT training instances: " + trainingData.numInstances());

        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);
        rf.setSeed(1); // explicit (Weka default is 1) -- reproducible across runs
        rf.buildClassifier(trainingData);
        System.out.println("Random Forest trained (100 trees).\n");

        System.out.println("--- STEP 2: EVALUATE ON HELD-OUT HUMAN REFERENCE SET (out-of-domain) ---");
        evaluateHumanReference(trainingData, rf);
    }

    // ---------------------------------------------------------------
    // Rebuild the exact DELBOT feature schema/training set (mirrors
    // DelbotValidationPipeline.java's feature engineering, duplicated
    // here rather than modifying that file).
    // ---------------------------------------------------------------
    private static Instances buildDelbotTrainingSet() throws Exception {
        File dataDir = new File(DELBOT_DATA_DIR);
        List<File> sessionFolders = new ArrayList<>();
        findSessionFolders(dataDir, sessionFolders);
        if (sessionFolders.isEmpty()) {
            throw new RuntimeException("No DELBOT session folders found under " + dataDir.getAbsolutePath()
                    + " -- run this from the same working directory as DelbotValidationPipeline.");
        }
        System.out.println("Found " + sessionFolders.size() + " DELBOT session folders.");

        ArrayList<Attribute> attrs = buildAttributeSchema();
        Instances data = new Instances("delbot_training", attrs, 0);
        data.setClassIndex(attrs.size() - 1);

        int foldersProcessed = 0;
        int filesSeen = 0;
        int filesParsed = 0;
        for (File folder : sessionFolders) {
            int label = folder.getName().startsWith("circles_human") ? 0 : 1;
            File[] files = folder.listFiles((d, n) -> n.endsWith(".txt"));
            if (files == null) continue;
            Arrays.sort(files); // deterministic training-instance order (File.listFiles has no guaranteed order)
            for (File f : files) {
                filesSeen++;
                double[] feat = parseDelbotSession(f);
                if (feat == null) continue;
                filesParsed++;
                Instance inst = new DenseInstance(attrs.size());
                inst.setDataset(data);
                for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
                inst.setValue(attrs.size() - 1, String.valueOf(label));
                data.add(inst);
            }
            foldersProcessed++;
            System.out.println("[" + foldersProcessed + "/" + sessionFolders.size() + "] " + folder.getName()
                    + " -> " + files.length + " files (parsed so far: " + filesParsed + "/" + filesSeen + ")");
        }
        return data;
    }

    private static ArrayList<Attribute> buildAttributeSchema() {
        ArrayList<Attribute> attrs = new ArrayList<>();
        attrs.add(new Attribute("num_points"));
        attrs.add(new Attribute("duration_ms"));
        attrs.add(new Attribute("mean_velocity"));
        attrs.add(new Attribute("std_velocity"));
        attrs.add(new Attribute("mean_acceleration"));
        attrs.add(new Attribute("mean_jerk"));
        attrs.add(new Attribute("path_efficiency"));
        ArrayList<String> classVals = new ArrayList<>(Arrays.asList("0", "1"));
        attrs.add(new Attribute("is_bot", classVals));
        return attrs;
    }

    private static void findSessionFolders(File dir, List<File> found) {
        File[] children = dir.listFiles(File::isDirectory);
        if (children == null) return;
        Arrays.sort(children);
        for (File child : children) {
            String name = child.getName();
            if (name.startsWith("circles_human") || name.startsWith("circles_bot")) {
                found.add(child);
            } else {
                findSessionFolders(child, found);
            }
        }
    }

    // Returns [num_points, duration_ms, mean_velocity, std_velocity, mean_acceleration, mean_jerk, path_efficiency]
    // BOTH parsing and feature math are delegated to BalabitValidationPipeline so the DELBOT
    // training set here is IDENTICAL to the one used by BalabitValidationPipeline and
    // BalabitCrossDomainEval -- tied-timestamp collapse, 1 ms dt floor, NaN rejection AND the
    // circles_human_fast normalised-coordinate rescale. (An earlier inline copy of the parse
    // loop missed that last one, which silently gave this pipeline different training data.)
    private static double[] parseDelbotSession(File file) {
        try {
            return BalabitValidationPipeline.computeFeatures(
                    BalabitValidationPipeline.parseDelbotPoints(file));
        } catch (Exception e) {
            return null;
        }
    }

    // ---------------------------------------------------------------
    // Evaluate the DELBOT-trained model against the human reference CSV.
    // Column indices below match HumanReferenceFeatureExtraction's header exactly:
    // 0 source_dataset, 1 person_id, 2 session_id, 3 is_bot, ...,
    // 23 num_points, 24 duration_ms_delbot, 25 mean_velocity_delbot,
    // 26 std_velocity_delbot, 27 mean_acceleration, 28 mean_jerk, 29 path_efficiency
    // ---------------------------------------------------------------
    private static void evaluateHumanReference(Instances trainingSchema, weka.classifiers.Classifier classifier) throws Exception {
        File csv = new File(HUMAN_REFERENCE_CSV);
        if (!csv.exists()) {
            throw new RuntimeException("Human reference CSV not found: " + csv.getAbsolutePath()
                    + " -- run HumanReferenceFeatureExtraction first.");
        }

        int totalRows = 0;
        int falsePositives = 0;

        Map<String, int[]> perSource = new TreeMap<>();  // [total, falsePositives]
        Map<String, int[]> perPerson = new TreeMap<>();  // key = source::person

        try (BufferedReader reader = new BufferedReader(new FileReader(csv))) {
            reader.readLine(); // skip header
            String line;
            while ((line = reader.readLine()) != null) {
                String[] cols = line.split(",", -1);
                if (cols.length < 30) continue;
                String source = cols[0];
                String person = cols[1];

                double numPoints = Double.parseDouble(cols[23]);
                double durationMs = Double.parseDouble(cols[24]);
                double meanVel = Double.parseDouble(cols[25]);
                double stdVel = Double.parseDouble(cols[26]);
                double meanAccel = Double.parseDouble(cols[27]);
                double meanJerk = Double.parseDouble(cols[28]);
                double pathEff = Double.parseDouble(cols[29]);

                Instance inst = new DenseInstance(8);
                inst.setDataset(trainingSchema);
                inst.setValue(0, numPoints);
                inst.setValue(1, durationMs);
                inst.setValue(2, meanVel);
                inst.setValue(3, stdVel);
                inst.setValue(4, meanAccel);
                inst.setValue(5, meanJerk);
                inst.setValue(6, pathEff);
                // class attribute left missing -- we are PREDICTING it, not
                // supplying ground truth to the classifier.

                double predicted = classifier.classifyInstance(inst);
                String predictedLabel = trainingSchema.classAttribute().value((int) predicted);
                boolean isFalsePositive = predictedLabel.equals("1"); // predicted bot, but source is 100% human

                totalRows++;
                if (isFalsePositive) falsePositives++;

                if (totalRows % 10000 == 0) {
                    System.out.println("  ...evaluated " + totalRows + " sessions so far");
                }

                perSource.computeIfAbsent(source, k -> new int[2]);
                perSource.get(source)[0]++;
                if (isFalsePositive) perSource.get(source)[1]++;

                String personKey = source + "::" + person;
                perPerson.computeIfAbsent(personKey, k -> new int[2]);
                perPerson.get(personKey)[0]++;
                if (isFalsePositive) perPerson.get(personKey)[1]++;
            }
        }

        System.out.println("Total human reference sessions evaluated: " + totalRows);
        System.out.println("False positives (predicted bot, truly human): " + falsePositives);

        double microFPR = totalRows > 0 ? (100.0 * falsePositives / totalRows) : 0.0;
        System.out.printf("Micro-average False Positive Rate (row-level): %.2f%%%n", microFPR);

        // Macro-average: mean of each PERSON's own FPR, so one prolific person's
        // sessions can't dominate the headline number the way a raw row-level
        // average would let them.
        double macroSum = 0;
        int personCount = 0;
        for (int[] counts : perPerson.values()) {
            if (counts[0] == 0) continue;
            macroSum += (100.0 * counts[1] / counts[0]);
            personCount++;
        }
        double macroFPR = personCount > 0 ? macroSum / personCount : 0.0;
        System.out.printf("Macro-average False Positive Rate (per-person): %.2f%% (across %d people)%n",
                macroFPR, personCount);

        System.out.println("\n--- FALSE POSITIVE RATE BY SOURCE FILE ---");
        System.out.printf("%-25s %10s %10s %10s%n", "source_dataset", "sessions", "FPs", "FPR%");
        for (Map.Entry<String, int[]> e : perSource.entrySet()) {
            int[] c = e.getValue();
            double fpr = c[0] > 0 ? (100.0 * c[1] / c[0]) : 0.0;
            System.out.printf("%-25s %10d %10d %9.2f%%%n", e.getKey(), c[0], c[1], fpr);
        }

        System.out.println("\n--- WORST 10 PEOPLE BY FALSE POSITIVE RATE (min 2 sessions) ---");
        perPerson.entrySet().stream()
                .filter(e -> e.getValue()[0] >= 2)
                .sorted((a, b) -> Double.compare(
                        (double) b.getValue()[1] / b.getValue()[0],
                        (double) a.getValue()[1] / a.getValue()[0]))
                .limit(10)
                .forEach(e -> {
                    int[] c = e.getValue();
                    double fpr = 100.0 * c[1] / c[0];
                    System.out.printf("%-40s %5d sessions, %5.1f%% FPR%n", e.getKey(), c[0], fpr);
                });
    }
}