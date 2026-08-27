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
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Validates the DELBOT-trained J48 model against the Balabit Mouse Dynamics
 * Challenge dataset (github.com/balabit/Mouse-Dynamics-Challenge) as a SECOND,
 * independent out-of-domain human-behavior dataset -- complementary to
 * FalsePositiveEvaluator's mightymerge check.
 *
 * Why this dataset matters: unlike mightymerge (which only gives per-event
 * velocity/angle/distance, requiring reconstructed pseudo-features), Balabit
 * provides REAL raw (x, y) coordinates and timestamps -- letting us compute
 * genuine, non-approximated acceleration/jerk/path-efficiency, the same way
 * DELBOT itself was computed. This is a stronger validation than mightymerge.
 *
 * IMPORTANT UNIT NOTE: Balabit's "record timestamp" is in SECONDS. DELBOT's
 * raw timestamps are in MILLISECONDS (verified: a ~3-second circle-drawing
 * task has a raw duration of ~3000 in DELBOT's files). This code converts
 * Balabit timestamps to milliseconds before computing any kinematic feature,
 * so the classifier's learned thresholds are applied on matching units.
 *
 * SEGMENTATION NOTE: Balabit's raw files are 40-80 minute continuous work
 * sessions (tens of thousands of points) -- wildly different scale from
 * DELBOT's few-second, ~100-200 point circle-drawing sessions. Sessions here
 * are split on natural inactivity gaps (3 seconds), matching the scale and
 * spirit of DELBOT's task-bounded sessions, rather than treating each huge
 * file as one session.
 *
 * SETUP: place the cloned Mouse-Dynamics-Challenge repo (or just its
 * training_files/ and test_files/ folders) at BALABIT_DATA_DIR below.
 */
public class BalabitValidationPipeline {

    private static final String DELBOT_DATA_DIR = "delbot_data";
    private static final String BALABIT_DATA_DIR = "balabit_data"; // update if placed elsewhere
    private static final double JERK_CLAMP = 1e6;
    private static final double GAP_THRESHOLD_SEC = 3.0;
    private static final int MIN_POINTS_PER_SESSION = 20;

    public static void main(String[] args) throws Exception {
        System.out.println("--- STEP 1: TRAIN J48 ON DELBOT ONLY ---");
        Instances trainingData = buildDelbotTrainingSet();
        System.out.println("DELBOT training instances: " + trainingData.numInstances());

        J48 tree = new J48();
        tree.setMinNumObj(20);
        tree.setBinarySplits(true);
        tree.buildClassifier(trainingData);
        System.out.println("Model trained on DELBOT only.\n");

        System.out.println("--- STEP 2: EVALUATE ON BALABIT (independent, real-coordinate human data) ---");
        evaluateBalabit(trainingData, tree);
    }

    // ---------------- DELBOT training set (same schema as FalsePositiveEvaluator) ----------------

    private static Instances buildDelbotTrainingSet() throws Exception {
        File dataDir = new File(DELBOT_DATA_DIR);
        List<File> sessionFolders = new ArrayList<>();
        findSessionFolders(dataDir, sessionFolders);
        if (sessionFolders.isEmpty()) {
            throw new RuntimeException("No DELBOT session folders found under " + dataDir.getAbsolutePath());
        }
        System.out.println("Found " + sessionFolders.size() + " DELBOT session folders.");

        ArrayList<Attribute> attrs = buildAttributeSchema();
        Instances data = new Instances("delbot_training", attrs, 0);
        data.setClassIndex(attrs.size() - 1);

        int foldersProcessed = 0;
        for (File folder : sessionFolders) {
            int label = folder.getName().startsWith("circles_human") ? 0 : 1;
            File[] files = folder.listFiles((d, n) -> n.endsWith(".txt"));
            if (files == null) continue;
            for (File f : files) {
                double[] feat = parseDelbotSession(f);
                if (feat == null) continue;
                Instance inst = new DenseInstance(attrs.size());
                inst.setDataset(data);
                for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
                inst.setValue(attrs.size() - 1, String.valueOf(label));
                data.add(inst);
            }
            foldersProcessed++;
            System.out.println("[" + foldersProcessed + "/" + sessionFolders.size() + "] " + folder.getName());
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
    // NOTE: explicitly rejects NaN/Infinite coordinates (DELBOT's "circleHide_*" files literally log the
    // text "nan" while the cursor is hidden -- Double.parseDouble only accepts exact-case "NaN", so those
    // rows correctly throw and get skipped here, same as the rest of malformed-line handling).
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
                    } catch (NumberFormatException ignored) {
                        // skips literal "nan" text and other malformed values
                    }
                }
            }
            return computeFeatures(points);
        } catch (Exception e) {
            return null;
        }
    }

    private static double[] computeFeatures(List<double[]> points) {
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

        double[] result = {points.size(), duration, meanVel, stdVel, meanAcc, meanJerk, pathEff};
        for (double v : result) {
            if (Double.isNaN(v) || Double.isInfinite(v)) return null; // reject any residual bad session
        }
        return result;
    }

    // ---------------- Balabit parsing + gap-based segmentation + evaluation ----------------

    private static void evaluateBalabit(Instances trainingSchema, J48 tree) throws Exception {
        File dataDir = new File(BALABIT_DATA_DIR);
        if (!dataDir.exists()) {
            throw new RuntimeException("Balabit data not found at " + dataDir.getAbsolutePath()
                    + " -- place training_files/ and test_files/ there.");
        }
        List<File> sessionFiles = new ArrayList<>();
        findBalabitFiles(dataDir, sessionFiles);
        System.out.println("Found " + sessionFiles.size() + " Balabit raw session files.");

        int totalRows = 0;
        int falsePositives = 0;
        Map<String, int[]> perUser = new TreeMap<>();

        int filesProcessed = 0;
        for (File file : sessionFiles) {
            String userFolder = file.getParentFile().getName(); // e.g. "user7"
            List<double[]> rawPoints = parseBalabitFile(file); // (t_sec, x, y)

            for (List<double[]> chunk : splitByGap(rawPoints, GAP_THRESHOLD_SEC)) {
                if (chunk.size() < MIN_POINTS_PER_SESSION) continue;

                // Convert seconds -> milliseconds to match DELBOT's units before feature computation
                List<double[]> chunkMs = new ArrayList<>();
                for (double[] p : chunk) chunkMs.add(new double[]{p[0] * 1000.0, p[1], p[2]});

                double[] feat = computeFeatures(chunkMs);
                if (feat == null) continue;

                Instance inst = new DenseInstance(8);
                inst.setDataset(trainingSchema);
                for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);

                double predicted = tree.classifyInstance(inst);
                String predictedLabel = trainingSchema.classAttribute().value((int) predicted);
                boolean isFalsePositive = predictedLabel.equals("1"); // predicted bot, but source is 100% human

                totalRows++;
                if (isFalsePositive) falsePositives++;

                perUser.computeIfAbsent(userFolder, k -> new int[2]);
                perUser.get(userFolder)[0]++;
                if (isFalsePositive) perUser.get(userFolder)[1]++;
            }

            filesProcessed++;
            if (filesProcessed % 200 == 0) {
                System.out.println("  ...processed " + filesProcessed + "/" + sessionFiles.size() + " raw files, "
                        + totalRows + " sessions evaluated so far");
            }
        }

        System.out.println("\nTotal Balabit sessions evaluated: " + totalRows);
        System.out.println("False positives (predicted bot, truly human): " + falsePositives);
        double fpr = totalRows > 0 ? (100.0 * falsePositives / totalRows) : 0.0;
        System.out.printf("Balabit False Positive Rate: %.2f%%%n", fpr);

        System.out.println("\n--- FALSE POSITIVE RATE BY USER ---");
        System.out.printf("%-10s %10s %10s %10s%n", "user", "sessions", "FPs", "FPR%");
        for (Map.Entry<String, int[]> e : perUser.entrySet()) {
            int[] c = e.getValue();
            double userFpr = c[0] > 0 ? (100.0 * c[1] / c[0]) : 0.0;
            System.out.printf("%-10s %10d %10d %9.2f%%%n", e.getKey(), c[0], c[1], userFpr);
        }

        System.out.println("\n--- COMPARISON TO MIGHTYMERGE (run FalsePositiveEvaluator for that number) ---");
        System.out.println("mightymerge.io FPR was 2.18% (browsing-style human behavior).");
        System.out.println("Balabit FPR above reflects continuous administrative/work-task mouse behavior --");
        System.out.println("a genuinely different behavioral context. A gap between these two numbers is a");
        System.out.println("real, reportable finding about generalization boundaries, not a bug.");
    }

    private static void findBalabitFiles(File dir, List<File> found) {
        File[] children = dir.listFiles();
        if (children == null) return;
        for (File child : children) {
            if (child.isDirectory()) {
                findBalabitFiles(child, found);
            } else if (child.getName().startsWith("session_")) {
                found.add(child);
            }
        }
    }

    // Returns list of (record_timestamp_seconds, x, y)
    private static List<double[]> parseBalabitFile(File file) {
        List<double[]> points = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            boolean first = true;
            while ((line = reader.readLine()) != null) {
                if (first) { first = false; continue; } // header: record timestamp,client timestamp,button,state,x,y
                String[] parts = line.split(",");
                if (parts.length < 6) continue;
                try {
                    double t = Double.parseDouble(parts[0].trim());
                    double x = Double.parseDouble(parts[4].trim());
                    double y = Double.parseDouble(parts[5].trim());
                    points.add(new double[]{t, x, y});
                } catch (NumberFormatException ignored) {
                }
            }
        } catch (Exception e) {
            // skip unreadable file
        }
        return points;
    }

    private static List<List<double[]>> splitByGap(List<double[]> points, double gapThresholdSec) {
        List<List<double[]>> sessions = new ArrayList<>();
        List<double[]> current = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            if (i > 0 && (points.get(i)[0] - points.get(i - 1)[0]) > gapThresholdSec) {
                if (!current.isEmpty()) sessions.add(current);
                current = new ArrayList<>();
            }
            current.add(points.get(i));
        }
        if (!current.isEmpty()) sessions.add(current);
        return sessions;
    }
}