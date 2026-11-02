import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Times the real-time path on the FROZEN demo model: 18-feature extraction
 * (Tier2Features AUGMENTED) plus one forest score, per chunk, on the 25 real
 * held-out Balabit chunks in demo/model/golden_vectors.json. The model is only
 * read, never changed.
 *
 *   java -cp ... LatencyBench [rounds]      -> latency_results.txt (redirect)
 *
 * Single thread, in-process, warm JIT: this is server-side compute only, not the
 * network or the browser's JS feature extraction.
 */
public class LatencyBench {

    public static void main(String[] args) throws Exception {
        int rounds = args.length > 0 ? Integer.parseInt(args[0]) : 400;
        File dir = new File("demo/model");
        RandomForest rf = (RandomForest) SerializationHelper.read(new File(dir, "rf.model").getPath());
        Instances schema = new Instances(new BufferedReader(new FileReader(new File(dir, "schema.arff"))));
        schema.setClassIndex(schema.numAttributes() - 1);
        int botIdx = schema.classAttribute().indexOfValue("1");

        List<List<double[]>> chunks = readChunks(new File(dir, "golden_vectors.json"));
        int pts = 0;
        for (List<double[]> c : chunks) pts += c.size();

        for (int w = 0; w < 50; w++) for (List<double[]> c : chunks) score(rf, schema, botIdx, c);   // JIT warm-up

        double[] feat = new double[rounds * chunks.size()], forest = new double[feat.length], total = new double[feat.length];
        int k = 0;
        double sink = 0;
        for (int r = 0; r < rounds; r++) {
            for (List<double[]> c : chunks) {
                long t0 = System.nanoTime();
                double[] f = Tier2Features.compute(c, Tier2Features.Mode.AUGMENTED);
                long t1 = System.nanoTime();
                Instance inst = new DenseInstance(schema.numAttributes());
                inst.setDataset(schema);
                for (int i = 0; i < f.length; i++) inst.setValue(i, f[i]);
                sink += rf.distributionForInstance(inst)[botIdx];
                long t2 = System.nanoTime();
                feat[k] = (t1 - t0) / 1e6; forest[k] = (t2 - t1) / 1e6; total[k] = (t2 - t0) / 1e6; k++;
            }
        }
        System.out.println("LatencyBench -- frozen demo model (" + rf.getNumIterations() + " trees, "
                + (schema.numAttributes() - 1) + " features), server-side compute only");
        System.out.println("java " + System.getProperty("java.version") + ", " + System.getProperty("os.name")
                + ", " + Runtime.getRuntime().availableProcessors() + " logical CPUs, single thread, warm JIT");
        System.out.printf(Locale.ROOT, "%d real held-out Balabit chunks (%d points, mean %.0f pts/chunk) x %d rounds = %d scorings%n%n",
                chunks.size(), pts, (double) pts / chunks.size(), rounds, k);
        System.out.println("ms per chunk        p50      p95      p99      max");
        row("feature extraction", feat);
        row("forest score", forest);
        row("total", total);
        System.out.printf(Locale.ROOT, "%n(checksum %.3f)%n", sink);
    }

    private static double score(RandomForest rf, Instances schema, int botIdx, List<double[]> c) throws Exception {
        double[] f = Tier2Features.compute(c, Tier2Features.Mode.AUGMENTED);
        Instance inst = new DenseInstance(schema.numAttributes());
        inst.setDataset(schema);
        for (int i = 0; i < f.length; i++) inst.setValue(i, f[i]);
        return rf.distributionForInstance(inst)[botIdx];
    }

    private static void row(String name, double[] xs) {
        double[] s = xs.clone();
        Arrays.sort(s);
        System.out.printf(Locale.ROOT, "%-18s %8.3f %8.3f %8.3f %8.3f%n", name,
                q(s, 0.5), q(s, 0.95), q(s, 0.99), s[s.length - 1]);
    }

    private static double q(double[] sorted, double p) { return sorted[(int) Math.min(sorted.length - 1, Math.round(p * (sorted.length - 1)))]; }

    /** Each golden entry starts with "points":[[t,x,y],...]; take that array only. */
    private static List<List<double[]>> readChunks(File f) throws Exception {
        String txt = Files.readString(f.toPath());
        List<List<double[]>> out = new ArrayList<>();
        Matcher m = Pattern.compile("\"points\"\\s*:\\s*\\[(.*?)\\]\\s*\\]").matcher(txt);
        Pattern trip = Pattern.compile("\\[\\s*(-?[0-9.eE+]+)\\s*,\\s*(-?[0-9.eE+]+)\\s*,\\s*(-?[0-9.eE+]+)\\s*\\]");
        while (m.find()) {
            List<double[]> c = new ArrayList<>();
            Matcher t = trip.matcher(m.group(1) + "]");
            while (t.find()) c.add(new double[]{ Double.parseDouble(t.group(1)), Double.parseDouble(t.group(2)), Double.parseDouble(t.group(3)) });
            out.add(c);
        }
        return out;
    }
}
