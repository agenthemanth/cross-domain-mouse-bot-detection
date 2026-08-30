import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * DEMO server -- loads the frozen model from demo/model/ and scores one feature
 * vector at a time. Also serves the static demo files, so the whole thing runs
 * from a single command with no web framework.
 *
 *   java -cp "target/classes;<weka>;<bounce>" DemoServer         (port 8787)
 *   java -cp "..." DemoServer 9000                               (custom port)
 *
 * GET  /                       -> demo/index.html  (Meander landing page)
 * GET  /lab.html               -> the live detector
 * GET  /features.js , /demo.js , /model/golden_vectors.json , ...
 * POST /score   body {"features":[<18 numbers>]}
 *              -> {"score":0.93,"label":"BOT","threshold":0.87}
 *
 * The 18 features must be in Tier2Features.featureNames(AUGMENTED) order -- the
 * browser computes them with demo/features.js, which demo/parity.js proves is
 * bit-parity with the Java pipeline.
 */
public class DemoServer {

    private static final File DEMO_DIR = new File("demo");
    private static final File MODEL_DIR = new File(DEMO_DIR, "model");

    private static RandomForest rf;
    private static Instances schema;
    private static int botIdx;
    private static double threshold;

    public static void main(String[] args) throws Exception {
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 8787;

        rf = (RandomForest) SerializationHelper.read(new File(MODEL_DIR, "rf.model").getPath());
        schema = new Instances(new BufferedReader(new FileReader(new File(MODEL_DIR, "schema.arff"))));
        schema.setClassIndex(schema.numAttributes() - 1);
        botIdx = schema.classAttribute().indexOfValue("1");
        threshold = readThreshold();

        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        server.createContext("/score", DemoServer::handleScore);
        server.createContext("/", DemoServer::handleStatic);
        server.setExecutor(null);
        server.start();

        System.out.println("demo model  : " + schema.numAttributes() + " attrs, threshold P(bot) >= " + threshold);
        System.out.println("serving      : http://127.0.0.1:" + port + "/");
        System.out.println("(Ctrl+C to stop)");
    }

    // ---------------- /score ----------------

    private static void handleScore(HttpExchange ex) throws IOException {
        cors(ex);
        if ("OPTIONS".equals(ex.getRequestMethod())) { ex.sendResponseHeaders(204, -1); ex.close(); return; }
        if (!"POST".equals(ex.getRequestMethod())) { send(ex, 405, "text/plain", "POST only"); return; }

        String body;
        try (InputStream in = ex.getRequestBody()) {
            body = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
        double[] feat = parseFeatures(body);
        if (feat == null || feat.length != schema.numAttributes() - 1) {
            send(ex, 400, "application/json",
                    "{\"error\":\"expected 'features' array of " + (schema.numAttributes() - 1) + " numbers\"}");
            return;
        }
        try {
            Instance inst = new DenseInstance(schema.numAttributes());
            inst.setDataset(schema);
            for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
            double score = rf.distributionForInstance(inst)[botIdx];
            String label = score >= threshold ? "BOT" : "HUMAN";
            send(ex, 200, "application/json", String.format(
                    "{\"score\":%.6f,\"label\":\"%s\",\"threshold\":%.4f}", score, label, threshold));
        } catch (Exception e) {
            send(ex, 500, "application/json", "{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    /** Pulls the numbers out of {@code {"features":[ ... ]}} without a JSON library. */
    private static double[] parseFeatures(String body) {
        Matcher m = Pattern.compile("\"features\"\\s*:\\s*\\[([^\\]]*)\\]").matcher(body);
        if (!m.find()) return null;
        String inner = m.group(1).trim();
        if (inner.isEmpty()) return new double[0];
        String[] parts = inner.split("\\s*,\\s*");
        double[] out = new double[parts.length];
        try {
            for (int i = 0; i < parts.length; i++) out[i] = Double.parseDouble(parts[i]);
        } catch (NumberFormatException e) {
            return null;
        }
        return out;
    }

    // ---------------- static files ----------------

    private static void handleStatic(HttpExchange ex) throws IOException {
        cors(ex);
        String p = ex.getRequestURI().getPath();
        if (p.equals("/") || p.isEmpty()) p = "/index.html";
        File f = new File(DEMO_DIR, p.replace("/", File.separator)).getCanonicalFile();
        if (!f.getPath().startsWith(DEMO_DIR.getCanonicalFile().getPath()) || !f.isFile()) {
            send(ex, 404, "text/plain", "not found: " + p);
            return;
        }
        byte[] bytes = Files.readAllBytes(f.toPath());
        ex.getResponseHeaders().set("Content-Type", contentType(f.getName()));
        ex.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    private static String contentType(String name) {
        if (name.endsWith(".html")) return "text/html; charset=utf-8";
        if (name.endsWith(".js")) return "text/javascript; charset=utf-8";
        if (name.endsWith(".css")) return "text/css; charset=utf-8";
        if (name.endsWith(".json")) return "application/json; charset=utf-8";
        return "application/octet-stream";
    }

    // ---------------- helpers ----------------

    private static double readThreshold() throws IOException {
        String txt = Files.readString(new File(MODEL_DIR, "operating_point.txt").toPath());
        Matcher m = Pattern.compile("threshold\\s*:\\s*([0-9.]+)").matcher(txt);
        if (m.find()) return Double.parseDouble(m.group(1));
        return 0.5;
    }

    private static void cors(HttpExchange ex) {
        ex.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        ex.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
    }

    private static void send(HttpExchange ex, int code, String type, String body) throws IOException {
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", type);
        ex.sendResponseHeaders(code, b.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(b); }
    }
}
