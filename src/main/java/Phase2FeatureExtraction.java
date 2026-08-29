import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.NumericColumn;
import tech.tablesaw.api.Table;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import java.io.File;

public class Phase2FeatureExtraction {
    public static void main(String[] args) {
        try {
            long startTime = System.currentTimeMillis();
            System.out.println("--- PHASE 2: FEATURE ENGINEERING ---");
            Table dataTable = Table.read().csv("click_fraud_dataset (1).csv");
            System.out.println("Loaded " + dataTable.rowCount() + " rows and " + dataTable.columnCount() + " columns in "
                    + (System.currentTimeMillis() - startTime) + " ms.");

            // --- ISSUE A / requirement: explicitly drop high-cardinality identifier
            // and metadata columns BEFORE writing the CSV. Weka's RemoveType(-T string)
            // does NOT catch these, because CSVLoader imports them as Nominal, not
            // String, so this has to happen here in Tablesaw, by name.
            String[] idColumnsToDrop = {
                    "click_id", "timestamp", "user_id", "ip_address", "referrer_url", "page_url"
            };
            for (String col : idColumnsToDrop) {
                if (dataTable.columnNames().contains(col)) {
                    dataTable.removeColumns(col);
                }
            }
            System.out.println("Dropped identifier/metadata columns. Remaining columns: " + dataTable.columnCount());

            NumericColumn<?> mouseMov = dataTable.nCol("mouse_movement");
            NumericColumn<?> clickDur = dataTable.nCol("click_duration");
            NumericColumn<?> scrollDepth = dataTable.nCol("scroll_depth");
            NumericColumn<?> timeSinceClick = dataTable.nCol("time_since_last_click");

            int rowCount = dataTable.rowCount();
            double[] velocity = new double[rowCount];
            double[] acceleration = new double[rowCount];
            double[] jerkEstimate = new double[rowCount];
            double[] pathEfficiency = new double[rowCount];
            double[] idleTimeRatio = new double[rowCount];

            int badJerkCount = 0;
            for (int i = 0; i < rowCount; i++) {
                double md = mouseMov.getDouble(i);
                double cd = Math.max(clickDur.getDouble(i), 1.0);
                double sd = scrollDepth.getDouble(i);
                double ts = timeSinceClick.getDouble(i);

                double v = md / cd;
                double a = v / cd;
                double j = a / cd;

                if (Double.isInfinite(j) || Double.isNaN(j) || j > 1e6) {
                    j = 1e6;
                    badJerkCount++;
                }

                velocity[i] = v;
                acceleration[i] = a;
                jerkEstimate[i] = j;
                pathEfficiency[i] = Math.min(md / (sd + 50.0), 1.0);
                idleTimeRatio[i] = ts / (cd * 1000.0 + 1.0);
            }
            System.out.println("Feature calculations complete. Clamped Jerk values: " + badJerkCount);

            dataTable.addColumns(
                    DoubleColumn.create("engineered_velocity", velocity),
                    DoubleColumn.create("engineered_acceleration", acceleration),
                    DoubleColumn.create("engineered_jerk", jerkEstimate),
                    DoubleColumn.create("path_efficiency", pathEfficiency),
                    DoubleColumn.create("idle_time_ratio", idleTimeRatio)
            );

            String outputCsv = "click_fraud_phase2_engineered.csv";
            long csvWriteStart = System.currentTimeMillis();
            dataTable.write().csv(outputCsv);
            System.out.println("Saved augmented CSV in " + (System.currentTimeMillis() - csvWriteStart) + " ms.");

            System.out.println("\n--- PHASE 3: MODEL TRAINING & EVALUATION ---");
            long wekaLoadStart = System.currentTimeMillis();
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(outputCsv));
            Instances rawDataset = loader.getDataSet();
            System.out.println("Weka dataset loaded: " + rawDataset.numInstances() + " instances, "
                    + rawDataset.numAttributes() + " attributes (" + (System.currentTimeMillis() - wekaLoadStart) + " ms).");

            // Defensive: still strip any true Weka String-type attributes if present.
            // Confirmed harmless no-op on this dataset (CSVLoader types everything
            // here as Numeric/Nominal), but kept as a safety net for other datasets.
            RemoveType removeStrings = new RemoveType();
            removeStrings.setOptions(new String[]{"-T", "string"});
            removeStrings.setInputFormat(rawDataset);
            Instances cleanDataset = Filter.useFilter(rawDataset, removeStrings);
            System.out.println("After string-attribute removal: " + cleanDataset.numAttributes() + " attributes.");

            // Exact target class setting: locate is_fraudulent BY NAME, convert
            // strictly that one column to nominal, leave every other numeric
            // predictor as continuous so the forest's trees use <= threshold
            // splits, not categorical equality matching.
            int classIndex = cleanDataset.attribute("is_fraudulent").index();
            cleanDataset.setClassIndex(classIndex);

            NumericToNominal convert = new NumericToNominal();
            convert.setAttributeIndices(String.valueOf(classIndex + 1)); // 1-indexed Range string
            convert.setInputFormat(cleanDataset);
            Instances nominalDataset = Filter.useFilter(cleanDataset, convert);
            nominalDataset.setClassIndex(classIndex);

            System.out.println("Class attribute: " + nominalDataset.classAttribute().name());
            System.out.println("Number of classes: " + nominalDataset.numClasses());

            // --- RUN A: all remaining features, including bot_likelihood_score ---
            runAndReport("RUN A (includes bot_likelihood_score - expect leakage / inflated accuracy)",
                    nominalDataset);

            // --- RUN B: drop bot_likelihood_score to check for label leakage ---
            int leakyIndex = nominalDataset.attribute("bot_likelihood_score").index();
            Remove dropLeaky = new Remove();
            dropLeaky.setAttributeIndices(String.valueOf(leakyIndex + 1));
            dropLeaky.setInputFormat(nominalDataset);
            Instances noLeakDataset = Filter.useFilter(nominalDataset, dropLeaky);
            // Removing an earlier attribute shifts the class index; class was last
            // remaining nominal target, so re-resolve it by name after the filter.
            noLeakDataset.setClassIndex(noLeakDataset.attribute("is_fraudulent").index());

            runAndReport("RUN B (bot_likelihood_score removed - genuine behavioral-feature accuracy)",
                    noLeakDataset);

            reportInfoGain(noLeakDataset);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Ranks each remaining predictor by information gain against is_fraudulent.
    // This tells us whether ANY behavioral feature carries real signal, or
    // whether Run B's near-baseline 75% accuracy reflects a genuinely weak
    // signal across the board rather than J48 underfitting one strong feature.
    private static void reportInfoGain(Instances dataset) throws Exception {
        System.out.println("\n=== INFORMATION GAIN RANKING (behavioral features vs is_fraudulent) ===");

        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();

        AttributeSelection selector = new AttributeSelection();
        selector.setEvaluator(eval);
        selector.setSearch(ranker);
        selector.SelectAttributes(dataset);

        double[][] ranked = selector.rankedAttributes();
        System.out.printf("%-30s %s%n", "Feature", "Info Gain");
        System.out.println("-".repeat(45));
        for (double[] row : ranked) {
            int attrIndex = (int) row[0];
            double gain = row[1];
            String name = dataset.attribute(attrIndex).name();
            if (name.equals("is_fraudulent")) continue; // skip the class itself
            System.out.printf("%-30s %.5f%n", name, gain);
        }
    }

    private static void runAndReport(String label, Instances dataset) throws Exception {
        System.out.println("\n=== " + label + " ===");
        System.out.println("Features used: " + (dataset.numAttributes() - 1));

        long trainStart = System.currentTimeMillis();
        RandomForest forest = new RandomForest();
        forest.setNumIterations(100);
        forest.buildClassifier(dataset);
        long trainMs = System.currentTimeMillis() - trainStart;
        System.out.println("Random Forest (100 trees) built in " + trainMs + " ms.");
        // Note: unlike J48, a Random Forest has no single readable tree structure to print --
        // it's an ensemble of 100 trees voting. See the project report for the earlier J48
        // comparison run, which was used specifically to catch the data-leakage issue via
        // its readable tree output before switching to this higher-performing ensemble.

        // IMPORTANT: Random Forest's individual trees are unconstrained here (no equivalent
        // of J48's minNumObj=20), so training accuracy alone can trivially hit ~100% by
        // memorizing rows even when there is no real signal in the data -- exactly what
        // happened on Run B before this fix. Cross-validated accuracy is the honest number;
        // training accuracy is kept only for reference/comparison, never as the headline result.
        Evaluation trainEval = new Evaluation(dataset);
        trainEval.evaluateModel(forest, dataset);
        System.out.println("Training Accuracy (reference only, expect inflated/overfit): "
                + String.format("%.2f", trainEval.pctCorrect()) + "%");

        Evaluation cvEval = new Evaluation(dataset);
        cvEval.crossValidateModel(new RandomForest(), dataset, 5, new java.util.Random(1));
        System.out.println("5-fold Cross-Validated Accuracy (the real, honest number): "
                + String.format("%.2f", cvEval.pctCorrect()) + "%");
    }
}