import weka.classifiers.trees.J48;
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
 * Safe evaluation step: trains J48 ONLY on DELBOT-Mouse (never on the human
 * reference set), then classifies every session in human_mouse_reference_features.csv
 * -- a fully held-out, out-of-domain, known-all-human dataset -- to measure the
 * classifier's false positive rate on real human behavior it has never seen.
 *
 * The human reference data is NEVER used for training, and NEVER merged with
 * DELBOT. This deliberately isolates "does the model over-flag genuine humans
 * from a different data source" from "does the model separate human vs bot".
 *
 * Reporting is grouped by source_dataset and by person, not by random individual
 * rows -- a single prolific person or file should not be able to dominate the
 * headline false-positive number.
 *
 * This file duplicates the small amount of DELBOT-parsing logic it needs
 * internally rather than modifying DelbotValidationPipeline.java, so that file's
 * existing behavior is left untouched.
 */
public class FalsePositiveEvaluator {

    private static final String DELBOT_DATA_DIR = "delbot_data";
    private static final String HUMAN_REFERENCE_CSV = "human_mouse_reference_features.csv";
    private static final double JERK_CLAMP = 1e6;

    public static void main(String[] args) throws Exception {
        System.out.println("--- STEP 1: TRAIN J48 ON DELBOT ONLY ---");
        Instances trainingData = buildDelbotTrainingSet();
        System.out.println("DELBOT training instances: " + trainingData.numInstances());

        J48 tree = new J48();
        tree.setMinNumObj(20);
        tree.setBinarySplits(true);
        tree.buildClassifier(trainingData);
        System.out.println("Model trained on DELBOT only.\n");

        System.out.println("--- STEP 2: EVALUATE ON HELD-OUT HUMAN REFERENCE SET (out-of-domain) ---");
        evaluateHumanReference(trainingData, tree);
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
    private static double[] parseDelbotSession(File file) {
        try {
            List<double[]> points = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                String line;
                boolean first = true;
                while ((line = reader.readLine()) != null) {
                    if (first) { first = false; continue; }
                    String[] parts = line.split(",");
                    if (parts.length != 4) continue;
                    try {
                        double t = Double.parseDouble(parts[0]);
                        double x = Double.parseDouble(parts[2]);
                        double y = Double.parseDouble(parts[3]);
                        points.add(new double[]{t, x, y});
                    } catch (NumberFormatException ignored) {}
                }
            }
            if (points.size() < 3) return null;

            List<Double> velocities = new ArrayList<>();
            List<Double> accelerations = new ArrayList<>();
            List<Double> jerks = new ArrayList<>();
            double totalDist = 0.0;
            for (int i = 1; i < points.size(); i++) {
                double dt = Math.max(points.get(i)[0] - points.get(i - 1)[0], 1e-3);
                double dx = points.get(i)[1] - points.get(i - 1)[1];
                double dy = points.get(i)[2] - points.get(i - 1)[2];
                double dist = Math.sqrt(dx * dx + dy * dy);
                totalDist += dist;
                velocities.add(dist / dt);
            }
            for (int i = 1; i < velocities.size(); i++) accelerations.add(velocities.get(i) - velocities.get(i - 1));
            for (int i = 1; i < accelerations.size(); i++) {
                double j = accelerations.get(i) - accelerations.get(i - 1);
                if (Double.isInfinite(j) || Double.isNaN(j)) j = JERK_CLAMP;
                jerks.add(Math.min(Math.abs(j), JERK_CLAMP) * Math.signum(j));
            }

            double meanVel = velocities.stream().mapToDouble(v -> v).average().orElse(0);
            double variance = velocities.stream().mapToDouble(v -> (v - meanVel) * (v - meanVel)).average().orElse(0);
            double stdVel = Math.sqrt(variance);
            double meanAcc = accelerations.isEmpty() ? 0 : accelerations.stream().mapToDouble(a -> a).average().orElse(0);
            double meanJerk = jerks.isEmpty() ? 0 : jerks.stream().mapToDouble(j -> j).average().orElse(0);

            double straight = Math.hypot(
                    points.get(points.size() - 1)[1] - points.get(0)[1],
                    points.get(points.size() - 1)[2] - points.get(0)[2]);
            double pathEff = Math.min(straight / (totalDist + 1e-5), 1.0);
            double duration = points.get(points.size() - 1)[0] - points.get(0)[0];

            return new double[]{points.size(), duration, meanVel, stdVel, meanAcc, meanJerk, pathEff};
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
    private static void evaluateHumanReference(Instances trainingSchema, J48 tree) throws Exception {
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

                double predicted = tree.classifyInstance(inst);
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