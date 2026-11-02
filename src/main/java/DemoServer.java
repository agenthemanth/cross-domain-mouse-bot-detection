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
 * Test login page (demo/login.html) -- moving puzzle (PerturbationPuzzle) + passive forest:
 * POST /puzzle/new                   -> start, board PNG + piece PNG (never the slot as a number)
 * POST /puzzle/move  {id,seq,x,y,ack} -> server-clock stamped position; replies carry the jump
 *                                        (ack token + moved board PNG) until the ack is echoed
 * POST /puzzle/drop  {id,x,y,ack}     -> placed | miss | exhausted
 * GET  /puzzle/oracle?id=             -> current slot; only with --oracle (probe affordance)
 * POST /auth/login   {user,pass,puzzle,bg,participant?,sim?}
 *                                     -> both channels judged server-side, decision table, sign-in;
 *                                        with --log-trials, one line appended to demo/study_data/trials.jsonl
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
    private static boolean ORACLE = false;
    private static boolean LOG_TRIALS = false;
    private static final File TRIALS_FILE = new File(new File(DEMO_DIR, "study_data"), "trials.jsonl");

    /** Largest request body accepted. The biggest legitimate body is a sign-in's background trace. */
    static final int MAX_BODY_BYTES = 1 << 20;   // 1 MiB

    /** Static files the server will hand out; everything else under demo/ stays private. */
    private static final java.util.Set<String> STATIC_EXT =
            java.util.Set.of(".html", ".js", ".css", ".json", ".png", ".svg", ".ico");

    private static final String USAGE =
            "usage: DemoServer [port] [--host <addr>] [--oracle] [--log-trials]\n"
          + "  port          TCP port (default 8787)\n"
          + "  --host        bind address (default 127.0.0.1, loopback only)\n"
          + "  --oracle      enable /challenge/oracle and /puzzle/oracle (bot sims and probes;\n"
          + "                they hand out answers -- never enable on a real deployment)\n"
          + "  --log-trials  append every sign-in to demo/study_data/trials.jsonl (participant study)";

    public static void main(String[] args) throws Exception {
        // the CAPTCHA renders glyphs with AWT; never needs a display
        System.setProperty("java.awt.headless", "true");
        int port = 8787;
        String host = "127.0.0.1";
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--oracle")) ORACLE = true;
            else if (a.equals("--log-trials")) LOG_TRIALS = true;
            else if (a.equals("--host") && i + 1 < args.length) host = args[++i];
            else if (a.equals("-h") || a.equals("--help")) { System.out.println(USAGE); return; }
            else {
                try {
                    port = Integer.parseInt(a);
                    if (port < 1 || port > 65535) throw new NumberFormatException();
                } catch (NumberFormatException e) {
                    System.err.println("unrecognised argument: " + a + "\n" + USAGE);
                    System.exit(2);
                }
            }
        }

        File modelFile = new File(MODEL_DIR, "rf.model");
        if (!modelFile.isFile()) {
            System.err.println("model not found: " + modelFile.getPath()
                    + "\nRun from the repo root, after freezing the model with DemoModelTrainer (see demo/README.md).");
            System.exit(1);
        }
        rf = (RandomForest) SerializationHelper.read(modelFile.getPath());
        try (BufferedReader r = new BufferedReader(new FileReader(new File(MODEL_DIR, "schema.arff")))) {
            schema = new Instances(r);
        }
        schema.setClassIndex(schema.numAttributes() - 1);
        botIdx = schema.classAttribute().indexOfValue("1");
        threshold = readThreshold();

        HttpServer server = HttpServer.create(new InetSocketAddress(host, port), 0);
        server.createContext("/score", guarded(DemoServer::handleScore));
        server.createContext("/challenge/", guarded(DemoServer::handleChallenge));
        server.createContext("/puzzle/", guarded(DemoServer::handlePuzzle));
        server.createContext("/auth/", guarded(DemoServer::handleAuth));
        server.createContext("/healthz", guarded(ex -> send(ex, 200, "application/json", "{\"status\":\"ok\"}")));
        server.createContext("/", guarded(DemoServer::handleStatic));
        // Deliberately single-threaded: per-puzzle / per-challenge state is not
        // synchronised, and the server-clock timestamps are the measurement.
        server.setExecutor(null);
        server.start();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> server.stop(1)));

        System.out.println("demo model   : " + schema.numAttributes() + " attrs, threshold P(bot) >= " + threshold);
        System.out.println("serving      : http://" + host + ":" + port + "/");
        if (ORACLE) System.out.println("WARNING      : oracle routes are ON (they hand out answers) -- demos and probes only, never a real deployment");
        if (LOG_TRIALS) System.out.println("study log    : appending every sign-in to " + TRIALS_FILE.getPath());
        System.out.println("(Ctrl+C to stop)");
    }

    /** Thrown by {@link #readBody} when a request body exceeds {@link #MAX_BODY_BYTES}. */
    static final class PayloadTooLarge extends RuntimeException {
        PayloadTooLarge() { super("request body too large", null, false, false); }
    }

    /**
     * Wraps every route: security headers, an early 413 on an oversized
     * Content-Length, and a last-resort handler so no exception text or stack
     * trace ever reaches the client.
     */
    private static com.sun.net.httpserver.HttpHandler guarded(com.sun.net.httpserver.HttpHandler h) {
        return ex -> {
            try {
                securityHeaders(ex);
                String len = ex.getRequestHeaders().getFirst("Content-Length");
                if (len != null) {
                    long n;
                    try { n = Long.parseLong(len.trim()); } catch (NumberFormatException e) { n = -1; }
                    if (n < 0 || n > MAX_BODY_BYTES) {
                        send(ex, n < 0 ? 400 : 413, "application/json",
                                n < 0 ? "{\"error\":\"bad Content-Length\"}" : "{\"error\":\"request body too large\"}");
                        return;
                    }
                }
                h.handle(ex);
            } catch (Throwable t) {
                fail(ex, t);
            } finally {
                ex.close();
            }
        };
    }

    /** Maps an exception to a response without leaking its message; logs the detail server-side. */
    private static void fail(HttpExchange ex, Throwable t) {
        try {
            if (t instanceof PayloadTooLarge) {
                send(ex, 413, "application/json", "{\"error\":\"request body too large\"}");
            } else {
                System.err.println("[" + ex.getRequestMethod() + " " + ex.getRequestURI().getPath() + "] " + t);
                send(ex, 500, "application/json", "{\"error\":\"internal error\"}");
            }
        } catch (IOException | RuntimeException alreadyCommitted) {
            // response headers were already sent; nothing more can be reported to this client
        }
    }

    // ---------------- /score ----------------

    private static void handleScore(HttpExchange ex) throws IOException {
        if (!"POST".equals(ex.getRequestMethod())) { send(ex, 405, "text/plain", "POST only"); return; }

        String body = readBody(ex);
        double[] feat = parseFeatures(body);
        if (feat == null || feat.length != schema.numAttributes() - 1) {
            send(ex, 400, "application/json",
                    "{\"error\":\"expected 'features' array of " + (schema.numAttributes() - 1) + " numbers\"}");
            return;
        }
        for (double v : feat) {
            if (!Double.isFinite(v)) { send(ex, 400, "application/json", "{\"error\":\"features must be finite numbers\"}"); return; }
        }
        try {
            double score = forestScore(feat);
            String label = score >= threshold ? "BOT" : "HUMAN";
            send(ex, 200, "application/json", String.format(java.util.Locale.ROOT,
                    "{\"score\":%.6f,\"label\":\"%s\",\"threshold\":%.4f}", score, label, threshold));
        } catch (Exception e) {
            fail(ex, e);
        }
    }

    private static double forestScore(double[] feat) throws Exception {
        Instance inst = new DenseInstance(schema.numAttributes());
        inst.setDataset(schema);
        for (int i = 0; i < feat.length; i++) inst.setValue(i, feat[i]);
        return rf.distributionForInstance(inst)[botIdx];
    }

    // ---------------- /puzzle/* (perturbation-response channel) ----------------

    private static void handlePuzzle(HttpExchange ex) throws IOException {
        String path = ex.getRequestURI().getPath();
        try {
            if (path.equals("/puzzle/oracle")) {
                // SIMULATION AFFORDANCE, OFF unless the server was started with --oracle.
                // Hands a probe the current slot so it can skip perception: it isolates
                // the timing channel, exactly like /challenge/oracle. Never ship it.
                if (!ORACLE) { send(ex, 404, "application/json", "{\"error\":\"oracle disabled (start DemoServer with --oracle)\"}"); return; }
                PerturbationPuzzle.Puzzle p = PerturbationPuzzle.get(query(ex.getRequestURI().getRawQuery()).get("id"));
                if (p == null) { send(ex, 404, "application/json", "{\"error\":\"unknown puzzle\"}"); return; }
                send(ex, 200, "application/json", PerturbationPuzzle.oracleJson(p));
                return;
            }
            if (!"POST".equals(ex.getRequestMethod())) { send(ex, 405, "text/plain", "POST only"); return; }
            String body = readBody(ex);
            if (path.equals("/puzzle/new")) {
                send(ex, 200, "application/json", PerturbationPuzzle.issueJson(PerturbationPuzzle.issue()));
                return;
            }
            PerturbationPuzzle.Puzzle p = PerturbationPuzzle.get(jsonString(body, "id"));
            if (p == null) { send(ex, 404, "application/json", "{\"error\":\"unknown puzzle\"}"); return; }
            double x = jsonNumber(body, "x", Double.NaN), y = jsonNumber(body, "y", Double.NaN);
            String ack = jsonString(body, "ack");
            switch (path) {
                case "/puzzle/move":
                    if (Double.isNaN(x) || Double.isNaN(y)) { send(ex, 400, "application/json", "{\"error\":\"x,y required\"}"); return; }
                    send(ex, 200, "application/json", PerturbationPuzzle.move(p, x, y, (long) jsonNumber(body, "seq", -1), ack));
                    return;
                case "/puzzle/drop":
                    if (Double.isNaN(x) || Double.isNaN(y)) { send(ex, 400, "application/json", "{\"error\":\"x,y required\"}"); return; }
                    send(ex, 200, "application/json", PerturbationPuzzle.drop(p, x, y, ack));
                    return;
                default:
                    send(ex, 404, "application/json", "{\"error\":\"no such endpoint\"}");
            }
        } catch (Exception e) {
            fail(ex, e);
        }
    }

    // ---------------- /auth/login (the test website's sign-in) ----------------

    static final String DEMO_USER = "demo", DEMO_PASS = "letmein";
    private static final int MAX_PASSIVE_CHUNKS = 6;

    /**
     * Body: {"user","pass","puzzle":"<id>","bg":[[t_ms,x,y,seg],...]}.
     * bg is the page-wide background mouse trace, with drag samples excluded and
     * a new seg id at every drag boundary.
     *
     * Both channels are judged here, on the server, and combined by a decision
     * table -- never averaged:
     *   puzzle FAIL                         -> BOT
     *   puzzle PASS, passive >= threshold   -> BOT   (either channel can veto)
     *   puzzle PASS, passive <  threshold   -> HUMAN
     *   puzzle PASS, passive not scorable   -> HUMAN (passive: no evidence)
     */
    private static void handleAuth(HttpExchange ex) throws IOException {
        if (!"/auth/login".equals(ex.getRequestURI().getPath())) { send(ex, 404, "application/json", "{\"error\":\"no such endpoint\"}"); return; }
        if (!"POST".equals(ex.getRequestMethod())) { send(ex, 405, "text/plain", "POST only"); return; }
        String body = readBody(ex);
        try {
            PerturbationPuzzle.Puzzle puzzle = PerturbationPuzzle.get(jsonString(body, "puzzle"));
            PerturbationPuzzle.Verdict pv = PerturbationPuzzle.verify(puzzle);

            // passive channel: split by segment id, then by the 3 s gap rule
            java.util.TreeMap<Integer, java.util.List<double[]>> bySeg = new java.util.TreeMap<>();
            Matcher q = Pattern.compile("\\[\\s*(-?[0-9.eE+]+)\\s*,\\s*(-?[0-9.eE+]+)\\s*,\\s*(-?[0-9.eE+]+)\\s*,\\s*(-?[0-9]+)\\s*\\]")
                    .matcher(bgArray(body));
            while (q.find()) {
                bySeg.computeIfAbsent(Integer.parseInt(q.group(4)), k -> new java.util.ArrayList<>())
                        .add(new double[]{ Double.parseDouble(q.group(1)), Double.parseDouble(q.group(2)), Double.parseDouble(q.group(3)) });
            }
            java.util.List<double[]> scores = new java.util.ArrayList<>();   // {score, distinctPoints, durationMs}
            for (java.util.List<double[]> seg : bySeg.values()) {
                for (java.util.List<double[]> chunk : BalabitValidationPipeline.splitByGap(seg, 3000.0)) {
                    int distinct = BalabitValidationPipeline.collapseToDistinctTimestamps(chunk).size();
                    if (distinct < BalabitValidationPipeline.MIN_POINTS_PER_SESSION) continue;
                    double[] f = Tier2Features.compute(chunk, Tier2Features.Mode.AUGMENTED);
                    if (f == null) continue;
                    scores.add(new double[]{ forestScore(f), distinct, chunk.get(chunk.size() - 1)[0] - chunk.get(0)[0] });
                }
            }
            while (scores.size() > MAX_PASSIVE_CHUNKS) scores.remove(0);
            double passive = Double.NaN;
            if (!scores.isEmpty()) {
                double[] s = new double[scores.size()];
                for (int i = 0; i < s.length; i++) s[i] = scores.get(i)[0];
                java.util.Arrays.sort(s);
                passive = s.length % 2 == 1 ? s[s.length / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
            }
            String passiveVerdict = Double.isNaN(passive) ? "NO_EVIDENCE" : passive >= threshold ? "BOT" : "HUMAN";

            String decision, reason;
            if (!pv.pass) { decision = "BOT"; reason = "moving puzzle failed"; }
            else if ("BOT".equals(passiveVerdict)) { decision = "BOT"; reason = "background mouse movement vetoed"; }
            else { decision = "HUMAN"; reason = "NO_EVIDENCE".equals(passiveVerdict)
                    ? "puzzle passed; too little background movement to score" : "both channels passed"; }

            boolean credsOk = DEMO_USER.equals(jsonString(body, "user")) && DEMO_PASS.equals(jsonString(body, "pass"));
            boolean loggedIn = "HUMAN".equals(decision) && credsOk;

            StringBuilder sb = new StringBuilder("{");
            sb.append("\"decision\":\"").append(decision).append('"');
            sb.append(",\"reason\":").append(CaptchaChallenge.jsonStr(reason));
            sb.append(",\"credentialsOk\":").append(credsOk);
            sb.append(",\"loggedIn\":").append(loggedIn);
            sb.append(",\"puzzle\":").append(PerturbationPuzzle.verdictJson(pv));
            sb.append(",\"passive\":{\"verdict\":\"").append(passiveVerdict).append('"');
            sb.append(",\"median\":").append(Double.isNaN(passive) ? "null" : String.format(java.util.Locale.ROOT, "%.4f", passive));
            sb.append(",\"threshold\":").append(String.format(java.util.Locale.ROOT, "%.4f", threshold));
            sb.append(",\"samples\":").append(countSamples(bySeg));
            sb.append(",\"chunks\":[");
            for (int i = 0; i < scores.size(); i++) {
                double[] c = scores.get(i);
                if (i > 0) sb.append(',');
                sb.append(String.format(java.util.Locale.ROOT, "{\"score\":%.4f,\"points\":%d,\"ms\":%.0f}", c[0], (int) c[1], c[2]));
            }
            sb.append("]}}");
            String response = sb.toString();
            if (LOG_TRIALS) logTrial(body, response, puzzle);
            send(ex, 200, "application/json", response);
        } catch (Exception e) {
            fail(ex, e);
        }
    }

    /**
     * One JSON line per sign-in, for the participant study (demo/analyze_trials.py).
     * Stores the participant code, whether a bot sim produced it, both channels'
     * verdicts and the puzzle's raw server-clocked trace. Never the username,
     * password, or the background trace itself.
     */
    private static synchronized void logTrial(String body, String response, PerturbationPuzzle.Puzzle puzzle) {
        String participant = jsonString(body, "participant");
        if (participant == null || !participant.matches("[A-Za-z0-9_-]{1,32}")) participant = "";
        int sim = (int) jsonNumber(body, "sim", 0);
        String line = "{\"ts\":" + System.currentTimeMillis()
                + ",\"participant\":\"" + participant + "\""
                + ",\"sim\":" + sim
                + ",\"result\":" + response
                + ",\"trace\":" + PerturbationPuzzle.traceJson(puzzle) + "}\n";
        try {
            TRIALS_FILE.getParentFile().mkdirs();
            Files.writeString(TRIALS_FILE.toPath(), line, StandardCharsets.UTF_8,
                    java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
        } catch (IOException e) {
            System.err.println("study log write failed: " + e);
        }
    }

    /** The text of the {@code "bg":[...]} array only, so no other field can be misread as a sample. */
    private static String bgArray(String body) {
        int k = body.indexOf("\"bg\"");
        if (k < 0) return "";
        int open = body.indexOf('[', k);
        if (open < 0) return "";
        int depth = 0;
        for (int i = open; i < body.length(); i++) {
            char c = body.charAt(i);
            if (c == '[') depth++;
            else if (c == ']' && --depth == 0) return body.substring(open, i + 1);
        }
        return "";
    }

    private static int countSamples(Map<Integer, java.util.List<double[]>> bySeg) {
        int n = 0;
        for (java.util.List<double[]> s : bySeg.values()) n += s.size();
        return n;
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

        String path = ex.getRequestURI().getPath();
        Map<String, String> q = query(ex.getRequestURI().getRawQuery());

        try {
            switch (path) {
                case "/challenge/new": {
                    if (!"POST".equals(ex.getRequestMethod())) { send(ex, 405, "text/plain", "POST only"); return; }
                    CaptchaChallenge.Challenge ch = CaptchaChallenge.issue();
                    send(ex, 200, "application/json", CaptchaChallenge.boardJson(ch));
                    return;
                }
                case "/challenge/tile": {
                    CaptchaChallenge.Challenge ch = CaptchaChallenge.get(q.get("id"));
                    if (ch == null) { send(ex, 404, "application/json", "{\"error\":\"unknown challenge\"}"); return; }
                    int k;
                    try { k = Integer.parseInt(q.getOrDefault("k", "-1")); } catch (NumberFormatException e) { k = -1; }
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
                    // test. A real deployment must not expose this route, so it
                    // is OFF unless the server was started with --oracle.
                    if (!ORACLE) { send(ex, 404, "application/json", "{\"error\":\"oracle disabled (start DemoServer with --oracle)\"}"); return; }
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
            fail(ex, e);
        }
    }

    /** Reads the body, refusing anything over {@link #MAX_BODY_BYTES} (also catches chunked bodies with no Content-Length). */
    private static String readBody(HttpExchange ex) throws IOException {
        try (InputStream in = ex.getRequestBody()) {
            byte[] b = in.readNBytes(MAX_BODY_BYTES + 1);
            if (b.length > MAX_BODY_BYTES) throw new PayloadTooLarge();
            return new String(b, StandardCharsets.UTF_8);
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
        if (!m.find()) return fallback;
        try { return Double.parseDouble(m.group(1)); } catch (NumberFormatException e) { return fallback; }
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

    /**
     * Serves the page assets only. demo/ also holds the model, study logs,
     * server logs and Python probes, so this is an allow-list: a known web
     * extension, no hidden/underscore files, and never anything under study_data/.
     */
    private static void handleStatic(HttpExchange ex) throws IOException {
        String method = ex.getRequestMethod();
        if (!"GET".equals(method) && !"HEAD".equals(method)) { send(ex, 405, "text/plain", "GET only"); return; }
        String p = ex.getRequestURI().getPath();
        if (p.equals("/") || p.isEmpty()) p = "/index.html";
        File root = DEMO_DIR.getCanonicalFile();
        File f = new File(DEMO_DIR, p.replace("/", File.separator)).getCanonicalFile();
        String name = f.getName();
        int dot = name.lastIndexOf('.');
        String ext = dot < 0 ? "" : name.substring(dot).toLowerCase(java.util.Locale.ROOT);
        boolean allowed = f.toPath().startsWith(root.toPath())
                && !f.toPath().startsWith(new File(root, "study_data").toPath())
                && !name.startsWith(".") && !name.startsWith("_")
                && STATIC_EXT.contains(ext)
                && f.isFile();
        if (!allowed) {
            send(ex, 404, "text/plain", "not found");
            return;
        }
        byte[] bytes = Files.readAllBytes(f.toPath());
        ex.getResponseHeaders().set("Content-Type", contentType(ext));
        // pages and scripts change between runs of a demo; always revalidate
        ex.getResponseHeaders().set("Cache-Control", "no-cache");
        if ("HEAD".equals(method)) { ex.sendResponseHeaders(200, -1); return; }
        ex.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    private static String contentType(String ext) {
        switch (ext) {
            case ".html": return "text/html; charset=utf-8";
            case ".js":   return "text/javascript; charset=utf-8";
            case ".css":  return "text/css; charset=utf-8";
            case ".json": return "application/json; charset=utf-8";
            case ".png":  return "image/png";
            case ".svg":  return "image/svg+xml";
            case ".ico":  return "image/x-icon";
            default:      return "application/octet-stream";
        }
    }

    // ---------------- helpers ----------------

    private static double readThreshold() throws IOException {
        String txt = Files.readString(new File(MODEL_DIR, "operating_point.txt").toPath());
        Matcher m = Pattern.compile("threshold\\s*:\\s*([0-9.]+)").matcher(txt);
        if (m.find()) return Double.parseDouble(m.group(1));
        return 0.5;
    }

    /**
     * Same-origin only: the pages are served by this server, so no CORS headers
     * are sent (the old wildcard let any website drive the API from a visitor's
     * browser). Pages may not be framed, which also blocks clickjacking the puzzle.
     */
    private static void securityHeaders(HttpExchange ex) {
        var h = ex.getResponseHeaders();
        h.set("X-Content-Type-Options", "nosniff");
        h.set("X-Frame-Options", "DENY");
        h.set("Referrer-Policy", "no-referrer");
        h.set("Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
              + "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
    }

    private static void send(HttpExchange ex, int code, String type, String body) throws IOException {
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", type);
        ex.sendResponseHeaders(code, b.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(b); }
    }
}
