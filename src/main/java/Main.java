import tech.tablesaw.api.Table;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import java.io.File;

public class Main {
    public static void main(String[] args) {
        try {
            System.out.println("--- STARTING PHASE 1 PIPELINE ---");

            // 1. Load dataset using Tablesaw
            Table dataTable = Table.read().csv("click_fraud_dataset (1).csv");
            System.out.println("Dataset successfully loaded. Row count: " + dataTable.rowCount());

            // 2. Load dataset into Weka for classification
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("click_fraud_dataset (1).csv"));
            Instances dataset = loader.getDataSet();

            // Set the target class index to the last column ('is_fraudulent')
            dataset.setClassIndex(dataset.numAttributes() - 1);

            // 3. Convert numeric class (0 and 1) to Nominal (Classes) so J48 can classify it
            NumericToNominal convert = new NumericToNominal();
            String[] options = new String[2];
            options[0] = "-R";
            options[1] = "last"; // target the last column
            convert.setOptions(options);
            convert.setInputFormat(dataset);
            Instances nominalDataset = Filter.useFilter(dataset, convert);
            nominalDataset.setClassIndex(nominalDataset.numAttributes() - 1);

            // 4. Train Decision Tree (J48)
            J48 tree = new J48();
            tree.buildClassifier(nominalDataset);

            // 5. Evaluate using 10-fold Cross-Validation
            Evaluation eval = new Evaluation(nominalDataset);
            eval.crossValidateModel(tree, nominalDataset, 10, new java.util.Random(1));

            System.out.println("--- PHASE 1 RESULTS ---");
            System.out.println("Accuracy: " + String.format("%.2f", eval.pctCorrect()) + "%");
            System.out.println("ROC Area: " + String.format("%.4f", eval.weightedAreaUnderROC()));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}