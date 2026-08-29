import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.Table;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Validation pipeline for the DELBOT-Mouse dataset (github.com/chrisgdt/DELBOT-Mouse, MIT license).
 * Parses raw human/bot trajectory session files, aggregates each session into the same
 * style of kinematic features used in Phase2FeatureExtraction (velocity, acceleration,
 * jerk, path efficiency), then trains/evaluates a Random Forest (100 trees) and ranks
 * features by info gain -- exactly mirroring the click-fraud pipeline so results are
 * directly comparable. (J48 was the earlier classifier here; kept in the project report
 * as the interpretable baseline that caught a data-leakage issue.)
 *
 * No external JSON dependency needed: folder names already encode the label
 * (circles_human_* = human, circles_bot_* = bot), so we scan folders directly
 * instead of parsing sessions.json.
 *
 * SETUP: unzip delbot_data.zip so the circles_* folders sit in the same working
 * directory this program runs from (or update DATA_DIR below).
 */
public class DelbotValidationPipeline {

    private static final String DATA_DIR = "delbot_data"; // update if placed elsewhere
    private static final double JERK_CLAMP = 1e6;

    static class SessionFeatures {
        int numPoints;
        double durationMs;
        double meanVelocity;
        double stdVelocity;
        double meanAcceleration;
        double meanJerk;
        double pathEfficiency;
        int isBot; // 0 = human, 1 = bot
    }

    public static void main(String[] args) {
        try {
            System.out.println("--- DELBOT VALIDATION: SESSION PARSING & FEATURE ENGINEERING ---");

            File dataDir = new File(DATA_DIR);
            if (!dataDir.exists()) {
                throw new RuntimeException("DATA_DIR not found: " + dataDir.getAbsolutePath()
                        + " -- check the folder is actually at this path relative to your run working directory.");
            }

            // Recursively find every folder named circles_human* / circles_bot*,
            // regardless of nesting depth. This makes the pipeline resilient to
            // extraction tools that wrap the zip's own top-level folder in an
            // extra outer folder (a common Windows "Extract All" behavior).
            List<File> sessionFolders = new ArrayList<>();
            findSessionFolders(dataDir, sessionFolders);

            if (sessionFolders.isEmpty()) {
                throw new RuntimeException("No circles_human*/circles_bot* folders found anywhere under "
                        + dataDir.getAbsolutePath() + ". Check delbot_data wasn't double-extracted "
                        + "(e.g. delbot_data/delbot_data/circles_*) and that the folder was placed at the project root.");
            }

            List<SessionFeatures> allFeatures = new ArrayList<>();
            int skipped = 0;

            for (File folder : sessionFolders) {
                String folderName = folder.getName();
                int label = folderName.startsWith("circles_human") ? 0 : 1;

                File[] sessionFiles = folder.listFiles((d, name) -> name.endsWith(".txt"));
                if (sessionFiles == null) continue;
                Arrays.sort(sessionFiles); // deterministic instance order: RandomForest's
                                           // bootstrap sampling is order-sensitive even at a
                                           // fixed seed (matches BalabitValidationPipeline).

                for (File file : sessionFiles) {
                    SessionFeatures f = parseSession(file, label);
                    if (f != null) {
                        allFeatures.add(f);
                    } else {
                        skipped++;
                    }
                }
            }

            if (allFeatures.isEmpty()) {
                throw new RuntimeException("Found " + sessionFolders.size() + " session folders but parsed 0 usable "
                        + "sessions (skipped: " + skipped + "). Check .txt files inside those folders aren't empty "
                        + "or corrupted.");
            }

            System.out.println("Parsed sessions: " + allFeatures.size() + " (skipped: " + skipped + ")");
            long humanCount = allFeatures.stream().filter(f -> f.isBot == 0).count();
            long botCount = allFeatures.stream().filter(f -> f.isBot == 1).count();
            System.out.println("Human: " + humanCount + " | Bot: " + botCount);

            // --- Build Tablesaw table, mirroring Phase2's export style ---
            Table table = Table.create("delbot_features");
            IntColumn numPoints = IntColumn.create("num_points");
            DoubleColumn duration = DoubleColumn.create("duration_ms");
            DoubleColumn meanVel = DoubleColumn.create("mean_velocity");
            DoubleColumn stdVel = DoubleColumn.create("std_velocity");
            DoubleColumn meanAcc = DoubleColumn.create("mean_acceleration");
            DoubleColumn meanJerk = DoubleColumn.create("mean_jerk");
            DoubleColumn pathEff = DoubleColumn.create("path_efficiency");
            IntColumn isBot = IntColumn.create("is_bot");

            for (SessionFeatures f : allFeatures) {
                numPoints.append(f.numPoints);
                duration.append(f.durationMs);
                meanVel.append(f.meanVelocity);
                stdVel.append(f.stdVelocity);
                meanAcc.append(f.meanAcceleration);
                meanJerk.append(f.meanJerk);
                pathEff.append(f.pathEfficiency);
                isBot.append(f.isBot);
            }

            table.addColumns(numPoints, duration, meanVel, stdVel, meanAcc, meanJerk, pathEff, isBot);

            String outputCsv = "delbot_features_engineered.csv";
            table.write().csv(outputCsv);
            System.out.println("Saved engineered CSV: " + outputCsv + " (" + table.rowCount() + " rows)");

            // --- Load into Weka, mirroring Phase2's pipeline ---
            System.out.println("\n--- MODEL TRAINING & EVALUATION (DELBOT) ---");
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(outputCsv));
            Instances dataset = loader.getDataSet();
            System.out.println("Weka dataset loaded: " + dataset.numInstances() + " instances, "
                    + dataset.numAttributes() + " attributes.");

            int classIndex = dataset.attribute("is_bot").index();
            dataset.setClassIndex(classIndex);

            NumericToNominal convert = new NumericToNominal();
            convert.setAttributeIndices(String.valueOf(classIndex + 1));
            convert.setInputFormat(dataset);
            Instances nominalDataset = Filter.useFilter(dataset, convert);
            nominalDataset.setClassIndex(classIndex);

            System.out.println("Class attribute: " + nominalDataset.classAttribute().name());
            System.out.println("Number of classes: " + nominalDataset.numClasses());

            long trainStart = System.currentTimeMillis();
            RandomForest forest = new RandomForest();
            forest.setNumIterations(100);
            forest.buildClassifier(nominalDataset);
            System.out.println("\nRandom Forest (100 trees) built in " + (System.currentTimeMillis() - trainStart) + " ms.");
            // Note: no single readable tree to print for an ensemble -- see project report
            // for the earlier J48 comparison run and its readable tree output.

            Evaluation eval = new Evaluation(nominalDataset);
            eval.crossValidateModel(new RandomForest(), nominalDataset, 5, new java.util.Random(1));
            System.out.println("5-fold Cross-Validated Accuracy: " + String.format("%.2f", eval.pctCorrect()) + "%");

            Evaluation trainEval = new Evaluation(nominalDataset);
            trainEval.evaluateModel(forest, nominalDataset);
            System.out.println("Training Accuracy: " + String.format("%.2f", trainEval.pctCorrect()) + "%");

            // --- Info gain ranking, mirroring Phase2's diagnostic ---
            System.out.println("\n=== INFORMATION GAIN RANKING (DELBOT kinematic features vs is_bot) ===");
            InfoGainAttributeEval infoEval = new InfoGainAttributeEval();
            Ranker ranker = new Ranker();
            AttributeSelection selector = new AttributeSelection();
            selector.setEvaluator(infoEval);
            selector.setSearch(ranker);
            selector.SelectAttributes(nominalDataset);

            double[][] ranked = selector.rankedAttributes();
            System.out.printf("%-25s %s%n", "Feature", "Info Gain");
            System.out.println("-".repeat(40));
            for (double[] row : ranked) {
                int attrIndex = (int) row[0];
                double gain = row[1];
                String name = nominalDataset.attribute(attrIndex).name();
                if (name.equals("is_bot")) continue;
                System.out.printf("%-25s %.5f%n", name, gain);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void findSessionFolders(File dir, List<File> found) {
        File[] children = dir.listFiles(File::isDirectory);
        if (children == null) return;
        Arrays.sort(children); // deterministic DFS order regardless of filesystem
        for (File child : children) {
            String name = child.getName();
            if (name.startsWith("circles_human") || name.startsWith("circles_bot")) {
                found.add(child);
            } else {
                findSessionFolders(child, found); // recurse into wrapper/nested folders
            }
        }
    }

    private static SessionFeatures parseSession(File file, int label) {
        try {
            // Parsing is delegated to BalabitValidationPipeline.parseDelbotPoints so this
            // in-domain pipeline reads the SAME DELBOT rows as the cross-dataset ones --
            // including the circles_human_fast normalised-coordinate rescale, which an
            // inline parse loop here previously missed.
            List<double[]> points = BalabitValidationPipeline.parseDelbotPoints(file);
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
            for (int i = 1; i < velocities.size(); i++) {
                accelerations.add(velocities.get(i) - velocities.get(i - 1));
            }
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

            double straightLineDist = Math.sqrt(
                    Math.pow(points.get(points.size() - 1)[1] - points.get(0)[1], 2) +
                            Math.pow(points.get(points.size() - 1)[2] - points.get(0)[2], 2)
            );
            double pathEfficiency = Math.min(straightLineDist / (totalDist + 1e-5), 1.0);
            double durationMs = points.get(points.size() - 1)[0] - points.get(0)[0];

            // Reject any session with a NaN/Infinite feature. DELBOT's touch files
            // (circles_human_tel) log the coordinate text "NaN" on trailing ReleasedTouch
            // rows; Double.parseDouble("NaN") returns NaN without throwing, so those points
            // are parsed and poison the aggregates. This guard mirrors the one at the end of
            // BalabitValidationPipeline.computeFeatures, so this in-domain pipeline trains on
            // the SAME DELBOT rows (857 human / 2596 bot) as the cross-dataset pipelines --
            // previously the 98 circles_human_tel sessions leaked in here as missing-value rows.
            double[] feats = {meanVel, stdVel, meanAcc, meanJerk, pathEfficiency, durationMs};
            for (double v : feats) {
                if (Double.isNaN(v) || Double.isInfinite(v)) return null;
            }

            SessionFeatures f = new SessionFeatures();
            f.numPoints = points.size();
            f.durationMs = durationMs;
            f.meanVelocity = meanVel;
            f.stdVelocity = stdVel;
            f.meanAcceleration = meanAcc;
            f.meanJerk = meanJerk;
            f.pathEfficiency = pathEfficiency;
            f.isBot = label;
            return f;

        } catch (Exception e) {
            return null;
        }
    }
}