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
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Validates the DELBOT-trained Random Forest model against the Balabit Mouse
 * Dynamics Challenge dataset (github.com/balabit/Mouse-Dynamics-Challenge) --
 * an independent, out-of-domain, all-human dataset. This is the project's
 * canonical cross-dataset false-positive evaluation: Balabit ships real raw
 * (x, y, t), so its features are computed by the exact same code as DELBOT
 * training. FalsePositiveEvaluator's mightymerge check is a rough sanity test
 * only (no raw x/y -> mismatched features) and is not part of this story.
 *
 * Random Forest is the final classifier, selected after a documented
 * head-to-head comparison against J48 (see project report). J48 remains in
 * the report as the interpretable model used to catch an earlier data-
 * leakage issue, but is no longer used in this pipeline going forward.
 *
 * CORRECTED RESULT (2026-08-28): Balabit human false-positive rate = 9.85%
 * (2381 / 24182 gap-split chunks). The earlier "16.69% (J48) -> 1.07% (RF)"
 * comparison was computed with a feature-pipeline bug (inter-sample dt was
 * clamped to 1e-3 ms, and Balabit logs ~16.5% of samples at tied timestamps,
 * inflating every kinematic feature by up to ~1e6x). Both of those numbers
 * are retracted. The fix -- collapseToDistinctTimestamps() plus a 1 ms dt
 * floor -- is below in computeFeatures().
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

    // Package-visible (not private) so BalabitCrossDomainEval reuses the IDENTICAL
    // feature-extraction and segmentation code rather than copying it. Two copies of
    // computeFeatures() would drift apart and silently make the FPR reported here
    // non-comparable to the cross-domain numbers reported there.
    static final String DELBOT_DATA_DIR = "delbot_data";
    static final String BALABIT_DATA_DIR = "balabit_data"; // update if placed elsewhere
    private static final double JERK_CLAMP = 1e6;
    static final double GAP_THRESHOLD_SEC = 3.0;
    static final int MIN_POINTS_PER_SESSION = 20;

    public static void main(String[] args) throws Exception {
        System.out.println("--- STEP 1: TRAIN RANDOM FOREST ON DELBOT ONLY ---");
        Instances trainingData = buildDelbotTrainingSet();
        System.out.println("DELBOT training instances: " + trainingData.numInstances());

        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);
        rf.setSeed(1); // Weka's default is already 1; stated explicitly so the run is
                       // provably reproducible and BalabitCrossDomainEval trains an
                       // identical forest (both also rely on the deterministic file
                       // ordering enforced in findSessionFolders / buildDelbotTrainingSet).
        rf.buildClassifier(trainingData);
        System.out.println("Random Forest trained (100 trees).\n");

        System.out.println("--- STEP 2: EVALUATE ON BALABIT (independent, real-coordinate human data) ---");
        evaluateBalabit(trainingData, rf);
    }

    // ---------------- DELBOT training set (same schema as FalsePositiveEvaluator) ----------------

    static Instances buildDelbotTrainingSet() throws Exception {
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
            // File.listFiles() gives no ordering guarantee, and the order training
            // instances are added changes RandomForest's bootstrap samples (order-
            // sensitive even at a fixed seed). Sort so the forest is identical run
            // to run and across pipelines.
            Arrays.sort(files);
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

    static ArrayList<Attribute> buildAttributeSchema() {
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

    static void findSessionFolders(File dir, List<File> found) {
        File[] children = dir.listFiles(File::isDirectory);
        if (children == null) return;
        Arrays.sort(children); // deterministic DFS order regardless of filesystem
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
    // NOTE on NaN handling: DELBOT's touch-input files (circles_human_tel, "*Touch" events) log the
    // coordinate text "NaN" on the trailing ReleasedTouch rows while the cursor is hidden. Java's
    // Double.parseDouble("NaN") does NOT throw -- it returns Double.NaN -- so those rows are parsed as
    // NaN points, propagate into the aggregate features, and the whole session is then rejected by the
    // isNaN/isInfinite guard at the end of computeFeatures(). Net effect: all 98 circles_human_tel
    // sessions are excluded from the DELBOT training set (955 human -> 857), which is acceptable here
    // (touch dynamics are a different modality from mouse) but IS a silent drop -- do not mistake it
    // for a parse-exception skip.
    private static double[] parseDelbotSession(File file) {
        try {
            return computeFeatures(parseDelbotPoints(file));
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Raw (timestamp_ms, x, y) samples from one DELBOT session file, ascending in t.
     * Package-visible so Tier2CrossDomainEval extracts its richer feature set from the
     * SAME parsed points as the Tier-1 pipeline. See the NaN note above parseDelbotSession.
     */
    static List<double[]> parseDelbotPoints(File file) throws Exception {
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
        return points;
    }

    /** Physical floor on the inter-sample gap (ms). 1000 Hz is about the fastest real
     *  mouse polling rate, so anything below 1 ms is measurement noise, not motion. The
     *  old floor was 1e-3 ms (1 microsecond), which turned tied timestamps into
     *  ~1e6x-inflated velocities. */
    private static final double MIN_DT_MS = 1.0;

    static double[] computeFeatures(List<double[]> pointsRaw) {
        // Balabit logs Move + Press/Drag/Release at the SAME record timestamp -- ~16.5%
        // of consecutive raw samples have dt == 0. Fed into dist/dt (with any small
        // floor) those produce velocities orders of magnitude outside the DELBOT
        // training range, making every kinematic feature meaningless cross-domain.
        // Collapsing each run of equal-timestamp samples to its final position puts
        // Balabit on the same "one position per distinct instant" basis DELBOT already
        // has (DELBOT has <0.1% tied samples, so this is a no-op there).
        List<double[]> points = collapseToDistinctTimestamps(pointsRaw);
        if (points.size() < 3) return null;

        List<Double> velocities = new ArrayList<>();
        List<Double> accelerations = new ArrayList<>();
        List<Double> jerks = new ArrayList<>();
        double totalDist = 0.0;

        for (int i = 1; i < points.size(); i++) {
            double dt = Math.max(points.get(i)[0] - points.get(i - 1)[0], MIN_DT_MS);
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

    /**
     * Keeps the last sample within each run of consecutive equal-timestamp points.
     * Input must be ascending in time (both DELBOT and Balabit parsers guarantee this).
     * Package-visible so BalabitCrossDomainEval scores the human chunk and synthesises
     * its bot twin from the SAME collapsed point list (keeps num_points matched).
     */
    static List<double[]> collapseToDistinctTimestamps(List<double[]> pts) {
        if (pts.size() < 2) return pts;
        List<double[]> out = new ArrayList<>(pts.size());
        for (int i = 0; i < pts.size(); i++) {
            if (i + 1 < pts.size() && pts.get(i + 1)[0] == pts.get(i)[0]) continue;
            out.add(pts.get(i));
        }
        return out;
    }

    // ---------------- Balabit parsing + gap-based segmentation + evaluation ----------------

    private static void evaluateBalabit(Instances trainingSchema, weka.classifiers.Classifier classifier) throws Exception {
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

            for (List<double[]> chunkRaw : splitByGap(rawPoints, GAP_THRESHOLD_SEC)) {
                // Collapse tied timestamps FIRST, then require MIN_POINTS distinct-time
                // samples -- a "session" of e.g. 5 real samples is not a usable trajectory
                // and its features fall outside the DELBOT training range.
                List<double[]> chunk = collapseToDistinctTimestamps(chunkRaw);
                if (chunk.size() < MIN_POINTS_PER_SESSION) continue;

                // Convert seconds -> milliseconds to match DELBOT's units before feature computation
                List<double[]> chunkMs = new ArrayList<>();
                for (double[] p : chunk) chunkMs.add(new double[]{p[0] * 1000.0, p[1], p[2]});

                double[] feat = computeFeatures(chunkMs);
                if (feat == null) continue;

                Instance inst = new DenseInstance(8);
                inst.setDataset(trainingSchema);
                for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);

                double predicted = classifier.classifyInstance(inst);
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

        System.out.println("\n--- THIS IS THE PROJECT'S CROSS-DATASET FALSE-POSITIVE RESULT ---");
        System.out.println("Balabit provides real raw (x, y, t), so every kinematic feature is computed");
        System.out.println("exactly the way DELBOT training computes it -- one feature-code path end to end.");
        System.out.println("The mightymerge check (FalsePositiveEvaluator) is a rough sanity test only:");
        System.out.println("it has no raw x/y, so 3 of 7 features are on mismatched scales train vs test,");
        System.out.println("and it is NOT part of the cross-dataset story (methodology decision, 2026-08-28).");
        System.out.println("The Balabit FPR reflects continuous administrative/work-task mouse behavior --");
        System.out.println("an out-of-domain context relative to DELBOT's few-second circle-drawing tasks.");
    }

    static void findBalabitFiles(File dir, List<File> found) {
        File[] children = dir.listFiles();
        if (children == null) return;
        // Order does not affect the per-chunk scores or the aggregate metrics, but
        // keeping it deterministic makes progress logs stable and guards any future
        // order-sensitive change.
        Arrays.sort(children);
        for (File child : children) {
            if (child.isDirectory()) {
                findBalabitFiles(child, found);
            } else if (child.getName().startsWith("session_")) {
                found.add(child);
            }
        }
    }

    // Returns list of (record_timestamp_seconds, x, y)
    static List<double[]> parseBalabitFile(File file) {
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

    /** State codes returned by {@link #parseBalabitEvents}. */
    static final int EV_MOVE = 0, EV_PRESSED = 1, EV_RELEASED = 2, EV_DRAG = 3, EV_OTHER = 4;

    /**
     * Full Balabit event stream as (record_timestamp_seconds, x, y, stateCode),
     * ascending in t. Unlike parseBalabitFile this keeps the press/release/drag
     * structure so BalabitActionSegmenter can cut the stream into typed mouse
     * actions (move / point-click / drag-drop) instead of blind 3s-gap chunks.
     */
    static List<double[]> parseBalabitEvents(File file) {
        List<double[]> events = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            boolean first = true;
            while ((line = reader.readLine()) != null) {
                if (first) { first = false; continue; } // header
                String[] parts = line.split(",");
                if (parts.length < 6) continue;
                try {
                    double t = Double.parseDouble(parts[0].trim());
                    String state = parts[3].trim();
                    double x = Double.parseDouble(parts[4].trim());
                    double y = Double.parseDouble(parts[5].trim());
                    int code;
                    switch (state) {
                        case "Move":     code = EV_MOVE; break;
                        case "Pressed":  code = EV_PRESSED; break;
                        case "Released": code = EV_RELEASED; break;
                        case "Drag":     code = EV_DRAG; break;
                        default:         code = EV_OTHER; break; // Scroll Up/Down etc.
                    }
                    events.add(new double[]{t, x, y, code});
                } catch (NumberFormatException ignored) {
                }
            }
        } catch (Exception e) {
            // skip unreadable file
        }
        return events;
    }

    static List<List<double[]>> splitByGap(List<double[]> points, double gapThresholdSec) {
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