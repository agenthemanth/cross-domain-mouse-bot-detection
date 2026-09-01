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
import java.util.HashMap;
import java.util.Map;
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
 * GET  /                       -> demo/index.html  (the live detector -- only page)
 * GET  /challenge.html         -> the sequential-reveal CAPTCHA
 * GET  /features.js , /demo.js , /model/golden_vectors.json , ...
 * POST /score   body {"features":[<18 numbers>]}
 *              -> {"score":0.93,"label":"BOT","threshold":0.87}
 *
 * CAPTCHA channel (CaptchaChallenge) -- deliberately NOT scored by the forest:
 * POST /challenge/new                 -> board layout + id
 * GET  /challenge/tile?id=..&k=..     -> distorted glyph PNG (letters never sent as text)
 * GET  /challenge/prompt?id=..        -> the current target glyph, also a PNG
 * POST /challenge/arrive  {id,k}      -> validates the hop, stamps the SERVER clock,
 *                                        releases the next prompt
 * POST /challenge/verify  {id,trace}  -> verdict from server-clocked timing,
 *                                        corroborated by the client trace
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
        // the CAPTCHA renders glyphs with AWT; never needs a display
        System.setProperty("java.awt.headless", "true");
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 8787;

        rf = (RandomForest) SerializationHelper.read(new File(MODEL_DIR, "rf.model").getPath());
        schema = new Instances(new BufferedReader(new FileReader(new File(MODEL_DIR, "schema.arff"))));
        schema.setClassIndex(schema.numAttributes() - 1);
        botIdx = schema.classAttribute().indexOfValue("1");
        threshold = readThreshold();

        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        server.createContext("/score", DemoServer::handleScore);
        server.createContext("/challenge/", DemoServer::handleChallenge);
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

    // ---------------- /challenge/* (the CAPTCHA channel) ----------------

    private static void handleChallenge(HttpExchange ex) throws IOException {
        cors(ex);
        if ("OPTIONS".equals(ex.getRequestMethod())) { ex.sendResponseHeaders(204, -1); ex.close(); return; }

        String path = ex.getRequestURI().getPath();
        Map<String, String> q = query(ex.getRequestURI().getRawQuery());

        try {
            switch (path) {
                case "/challenge/new": {
                    CaptchaChallenge.Challenge ch = CaptchaChallenge.issue();
                    send(ex, 200, "application/json", CaptchaChallenge.boardJson(ch));
                    return;
                }
                case "/challenge/tile": {
                    CaptchaChallenge.Challenge ch = CaptchaChallenge.get(q.get("id"));
                    if (ch == null) { send(ex, 404, "application/json", "{\"error\":\"unknown challenge\"}"); return; }
                    int k = Integer.parseInt(q.getOrDefault("k", "-1"));
                    if (k < 0 || k >= ch.tiles.size()) { send(ex, 400, "application/json", "{\"error\":\"bad tile\"}"); return; }
                    sendBytes(ex, CaptchaChallenge.renderTile(ch, k));
                    return;
                }
                case "/challenge/prompt": {
                    CaptchaChallenge.Challenge ch = CaptchaChallenge.get(q.get("id"));
                    if (ch == null) { send(ex, 404, "application/json", "{\"error\":\"unknown challenge\"}"); return; }
                    if (ch.failed || ch.done()) { send(ex, 409, "application/json", "{\"error\":\"no active hop\"}"); return; }
                    // NOTE: only the CURRENT hop is renderable. Asking for a later
                    // one is exactly the precomputation this design refuses.
                    sendBytes(ex, CaptchaChallenge.renderPrompt(ch, ch.hop));
                    return;
                }
                case "/challenge/oracle": {
                    // SIMULATION AFFORDANCE, NOT PART OF THE SCHEME. Hands the
                    // caller the current answer so a browser bot-sim can play
                    // without an OCR model. It grants the attacker a PERFECT
                    // perceptual layer for free, which is exactly the point:
                    // it isolates the timing channel as the only thing under
                    // test. A real deployment must not expose this route.
                    CaptchaChallenge.Challenge ch = CaptchaChallenge.get(q.get("id"));
                    if (ch == null) { send(ex, 404, "application/json", "{\"error\":\"unknown challenge\"}"); return; }
                    if (ch.failed || ch.done()) { send(ex, 409, "application/json", "{\"error\":\"no active hop\"}"); return; }
                    send(ex, 200, "application/json",
                            "{\"k\":" + ch.currentTarget() + ",\"hop\":" + ch.hop + "}");
                    return;
                }
                case "/challenge/arrive": {
                    if (!"POST".equals(ex.getRequestMethod())) { send(ex, 405, "text/plain", "POST only"); return; }
                    String body = readBody(ex);
                    CaptchaChallenge.Challenge ch = CaptchaChallenge.get(jsonString(body, "id"));
                    if (ch == null) { send(ex, 404, "application/json", "{\"error\":\"unknown challenge\"}"); return; }
                    int k = (int) jsonNumber(body, "k", -1);
                    send(ex, 200, "application/json", CaptchaChallenge.arrive(ch, k));
                    return;
                }
                case "/challenge/verify": {
                    if (!"POST".equals(ex.getRequestMethod())) { send(ex, 405, "text/plain", "POST only"); return; }
                    String body = readBody(ex);
                    CaptchaChallenge.Challenge ch = CaptchaChallenge.get(jsonString(body, "id"));
                    if (ch == null) { send(ex, 404, "application/json", "{\"error\":\"unknown challenge\"}"); return; }
                    send(ex, 200, "application/json",
                            CaptchaChallenge.verdictJson(CaptchaChallenge.verify(ch, parseTrace(body))));
                    return;
                }
                default:
                    send(ex, 404, "application/json", "{\"error\":\"no such endpoint\"}");
            }
        } catch (Exception e) {
            send(ex, 500, "application/json", "{\"error\":" + CaptchaChallenge.jsonStr(String.valueOf(e)) + "}");
        }
    }

    private static String readBody(HttpExchange ex) throws IOException {
        try (InputStream in = ex.getRequestBody()) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static Map<String, String> query(String raw) {
        Map<String, String> out = new HashMap<>();
        if (raw == null) return out;
        for (String pair : raw.split("&")) {
            int eq = pair.indexOf('=');
            if (eq > 0) out.put(pair.substring(0, eq), pair.substring(eq + 1));
        }
        return out;
    }

    private static String jsonString(String body, String key) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"").matcher(body);
        return m.find() ? m.group(1) : null;
    }

    private static double jsonNumber(String body, String key, double fallback) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*(-?[0-9.]+)").matcher(body);
        return m.find() ? Double.parseDouble(m.group(1)) : fallback;
    }

    /** Pulls {@code "trace":[[t,x,y],...]} out of the body. Returns null when absent. */
    private static double[][] parseTrace(String body) {
        Matcher outer = Pattern.compile("\"trace\"\\s*:\\s*\\[(.*)\\]\\s*\\}?\\s*$",
                Pattern.DOTALL).matcher(body);
        if (!outer.find()) return null;
        Matcher triple = Pattern.compile("\\[\\s*(-?[0-9.eE+]+)\\s*,\\s*(-?[0-9.eE+]+)\\s*,\\s*(-?[0-9.eE+]+)\\s*\\]")
                .matcher(outer.group(1));
        java.util.List<double[]> pts = new java.util.ArrayList<>();
        while (triple.find()) {
            try {
                pts.add(new double[]{ Double.parseDouble(triple.group(1)),
                        Double.parseDouble(triple.group(2)), Double.parseDouble(triple.group(3)) });
            } catch (NumberFormatException ignored) { /* skip malformed sample */ }
        }
        return pts.isEmpty() ? null : pts.toArray(new double[0][]);
    }

    private static void sendBytes(HttpExchange ex, byte[] png) throws IOException {
        ex.getResponseHeaders().set("Content-Type", "image/png");
        ex.getResponseHeaders().set("Cache-Control", "no-store");
        ex.sendResponseHeaders(200, png.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(png); }
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
