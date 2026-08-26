import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Streams the mightymerge human-behavior reference dataset (57 dataset_*.txt files,
 * ~1.59GB total, semicolon-delimited: idman;duration;angle;distance;velocity;local_date;local_time)
 * and turns it into one session-level feature row per (source_file, idman, session).
 *
 * Memory model: files are processed ONE AT A TIME. Only the current file's events
 * (grouped by idman) are held in memory; each file's map is discarded before the
 * next file is read. The full 1.59GB is never loaded at once.
 *
 * idman/local_date/local_time/filename/source are NEVER used as ML features --
 * they exist only as metadata columns (source_dataset, person_id, session_id) for
 * grouping and auditing. is_bot=0 is written for every row since this entire
 * dataset is human-only.
 *
 * This file does NOT merge into DELBOT or any training set. See
 * FalsePositiveEvaluator.java for the safe, held-out evaluation step.
 */
public class HumanReferenceFeatureExtraction {

    // Kept OUTSIDE the project directory intentionally. Add as many source
    // directories here as you have -- every folder is scanned independently
    // for dataset_*.txt files and merged into one output CSV. Files in each
    // folder still only ever hold that ONE file's events in memory at a time.
    private static final String[] DEFAULT_SOURCE_DIRS = {
            "C:/Users/espar/Downloads/mightymerge.io__z0m3ceve/mightymerge.io__z0m3ceve"
            // add the new folder's path here, e.g.:
            // , "C:/Users/espar/Downloads/mightymerge_extra/mightymerge_extra"
    };

    private static final String OUTPUT_CSV = "human_mouse_reference_features.csv";

    private static final long SESSION_GAP_MS = 5 * 60 * 1000L; // new session after 5 min idle
    private static final long PAUSE_THRESHOLD_MS = 500L;
    private static final int MIN_EVENTS_PER_SESSION = 3;

    static class Event {
        LocalDateTime timestamp;
        Double duration;  // nullable -- present on duration/click-type events
        Double angle;     // nullable -- present on movement-type events
        Double distance;  // nullable -- present on movement-type events
        Double velocity;  // nullable -- present on movement-type events

        boolean isMovement() {
            return angle != null && distance != null && velocity != null;
        }
    }

    public static void main(String[] args) throws Exception {
        String[] sourceDirs = args.length > 0 ? args : DEFAULT_SOURCE_DIRS;

        List<File> allFiles = new ArrayList<>();
        for (String sourceDir : sourceDirs) {
            File dir = new File(sourceDir);
            if (!dir.exists()) {
                throw new RuntimeException("Source directory not found: " + dir.getAbsolutePath());
            }
            File[] files = dir.listFiles((d, name) -> name.startsWith("dataset_") && name.endsWith(".txt"));
            if (files == null || files.length == 0) {
                System.out.println("WARNING: no dataset_*.txt files found in " + dir.getAbsolutePath());
                continue;
            }
            allFiles.addAll(Arrays.asList(files));
            System.out.println(dir.getAbsolutePath() + " -> " + files.length + " files");
        }

        if (allFiles.isEmpty()) {
            throw new RuntimeException("No dataset_*.txt files found across any of the " + sourceDirs.length
                    + " source directories provided.");
        }

        allFiles.sort(Comparator.comparing(File::getName));
        File[] files = allFiles.toArray(new File[0]);
        System.out.println("Total dataset files across all sources: " + files.length + ". Streaming one at a time...");

        int totalSessions = 0;
        int totalPeople = 0;
        int totalSkippedLines = 0;
        Map<String, Integer> aggregateSkipReasons = new java.util.TreeMap<>();
        List<String> globalSampleSkippedLines = new ArrayList<>();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_CSV))) {
            writer.write(String.join(",",
                    "source_dataset", "person_id", "session_id", "is_bot",
                    "event_count", "session_duration_ms",
                    "mean_velocity", "std_velocity",
                    "mean_distance", "std_distance",
                    "mean_gap_ms", "std_gap_ms",
                    "pause_count_500ms",
                    "mean_angle", "std_angle",
                    "mean_angle_change", "std_angle_change",
                    "reversal_count",
                    "speed_entropy", "direction_entropy",
                    "nonnull_duration_count", "mean_duration", "std_duration",
                    // DELBOT-compatible subset, reconstructed from velocity/angle/distance,
                    // for safe cross-dataset evaluation against the DELBOT-trained classifier.
                    "num_points", "duration_ms_delbot", "mean_velocity_delbot", "std_velocity_delbot",
                    "mean_acceleration", "mean_jerk", "path_efficiency",
                    // Diagnostic columns appended at the end -- safe to ignore, existing
                    // column indices 0-29 above are unchanged.
                    "movement_event_count", "duration_event_count"
            ));
            writer.newLine();

            for (int fi = 0; fi < files.length; fi++) {
                File file = files[fi];
                // Qualify with parent folder name so filenames that happen to repeat
                // across different source directories don't collide in the output
                // (e.g. two folders both containing "dataset_1.txt").
                String parentName = file.getParentFile() != null ? file.getParentFile().getName() : "root";
                String sourceName = parentName + "__" + file.getName().replace(".txt", "");

                ParseResult result = parseFile(file); // holds only THIS file's events
                totalSkippedLines += result.skippedLines;
                for (Map.Entry<String, Integer> e : result.skipReasonCounts.entrySet()) {
                    aggregateSkipReasons.merge(e.getKey(), e.getValue(), Integer::sum);
                }
                if (globalSampleSkippedLines.size() < 10) {
                    for (String s : result.sampleSkippedLines) {
                        if (globalSampleSkippedLines.size() < 10) globalSampleSkippedLines.add(s);
                    }
                }

                int sessionsInFile = 0;
                for (Map.Entry<String, List<Event>> entry : result.byPerson.entrySet()) {
                    String personId = entry.getKey();
                    List<Event> events = entry.getValue();
                    events.sort(Comparator.comparing(e -> e.timestamp));

                    List<List<Event>> sessions = splitIntoSessions(events);
                    int sessionIndex = 0;
                    for (List<Event> session : sessions) {
                        if (session.size() < MIN_EVENTS_PER_SESSION) continue;
                        String sessionId = sourceName + "_" + personId + "_s" + sessionIndex;
                        writeFeatureRow(writer, sourceName, personId, sessionId, session);
                        sessionIndex++;
                        sessionsInFile++;
                        totalSessions++;
                    }
                }
                totalPeople += result.byPerson.size();
                // result goes out of scope here -- GC-eligible before next file is read.

                System.out.println("[" + (fi + 1) + "/" + files.length + "] " + file.getName()
                        + " -> " + result.byPerson.size() + " people, " + sessionsInFile + " sessions"
                        + " (skipped lines: " + result.skippedLines + ")");
            }
        }

        System.out.println("\nDone.");
        System.out.println("Total sessions written: " + totalSessions);
        System.out.println("Total people (person-file pairs): " + totalPeople);
        System.out.println("Total malformed lines skipped: " + totalSkippedLines);
        System.out.println("Output: " + new File(OUTPUT_CSV).getAbsolutePath());

        System.out.println("\n--- SKIP REASON BREAKDOWN (why lines were discarded) ---");
        for (Map.Entry<String, Integer> e : aggregateSkipReasons.entrySet()) {
            System.out.printf("%-60s %d%n", e.getKey(), e.getValue());
        }

        System.out.println("\n--- SAMPLE SKIPPED LINES (raw text, first 10) ---");
        for (String s : globalSampleSkippedLines) {
            System.out.println("  " + s);
        }
    }

    private static class ParseResult {
        Map<String, List<Event>> byPerson = new HashMap<>();
        int skippedLines = 0;
        Map<String, Integer> skipReasonCounts = new java.util.TreeMap<>();
        List<String> sampleSkippedLines = new ArrayList<>();
    }

    private static ParseResult parseFile(File file) throws IOException {
        ParseResult result = new ParseResult();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            boolean first = true;
            while ((line = reader.readLine()) != null) {
                if (first) {
                    first = false;
                    if (line.toLowerCase().startsWith("idman")) continue; // header row, skip
                }
                String[] parts = line.split(";", -1);
                if (parts.length < 7) {
                    result.skippedLines++;
                    result.skipReasonCounts.merge("wrong_column_count(" + parts.length + ")", 1, Integer::sum);
                    if (result.sampleSkippedLines.size() < 5) result.sampleSkippedLines.add(line);
                    continue;
                }
                try {
                    String idman = parts[0].trim();
                    Double duration = parseNullableDouble(parts[1]);
                    Double angle = parseNullableDouble(parts[2]);
                    Double distance = parseNullableDouble(parts[3]);
                    Double velocity = parseNullableDouble(parts[4]);
                    LocalDate date = LocalDate.parse(parts[5].trim());
                    LocalTime time = LocalTime.parse(parts[6].trim());

                    Event e = new Event();
                    e.timestamp = LocalDateTime.of(date, time);
                    e.duration = duration;
                    e.angle = angle;
                    e.distance = distance;
                    e.velocity = velocity;

                    result.byPerson.computeIfAbsent(idman, k -> new ArrayList<>()).add(e);
                } catch (Exception parseEx) {
                    result.skippedLines++;
                    String reason = parseEx.getClass().getSimpleName() + ": " + parseEx.getMessage();
                    result.skipReasonCounts.merge(reason, 1, Integer::sum);
                    if (result.sampleSkippedLines.size() < 5) result.sampleSkippedLines.add(line);
                }
            }
        }
        return result;
    }

    private static Double parseNullableDouble(String raw) {
        String s = raw.trim();
        if (s.isEmpty() || s.equalsIgnoreCase("NULL")) return null;
        return Double.parseDouble(s);
    }

    private static List<List<Event>> splitIntoSessions(List<Event> events) {
        List<List<Event>> sessions = new ArrayList<>();
        List<Event> current = new ArrayList<>();
        for (int i = 0; i < events.size(); i++) {
            if (i > 0) {
                long gap = Duration.between(events.get(i - 1).timestamp, events.get(i).timestamp).toMillis();
                if (gap > SESSION_GAP_MS) {
                    sessions.add(current);
                    current = new ArrayList<>();
                }
            }
            current.add(events.get(i));
        }
        if (!current.isEmpty()) sessions.add(current);
        return sessions;
    }

    private static void writeFeatureRow(BufferedWriter writer, String source, String person, String sessionId,
                                        List<Event> session) throws IOException {
        int n = session.size();

        // Split into the two event types this dataset actually contains.
        List<Event> movementEvents = new ArrayList<>();
        List<Event> durationEvents = new ArrayList<>();
        for (Event e : session) {
            if (e.isMovement()) movementEvents.add(e);
            if (e.duration != null) durationEvents.add(e);
        }

        // --- Overall session timing (uses ALL events, both types -- timestamp is always present) ---
        long[] allGaps = new long[Math.max(n - 1, 0)];
        for (int i = 1; i < n; i++) {
            allGaps[i - 1] = Duration.between(session.get(i - 1).timestamp, session.get(i).timestamp).toMillis();
        }
        double sessionDurationMs = Duration.between(session.get(0).timestamp, session.get(n - 1).timestamp).toMillis();
        double meanGap = meanLong(allGaps);
        double stdGap = stdLong(allGaps, meanGap);
        long pauseCount = Arrays.stream(allGaps).filter(g -> g > PAUSE_THRESHOLD_MS).count();

        // --- Movement-based kinematic features (only from movement events) ---
        int m = movementEvents.size();
        double[] velocities = new double[m];
        double[] distances = new double[m];
        double[] angles = new double[m];
        for (int i = 0; i < m; i++) {
            velocities[i] = movementEvents.get(i).velocity;
            distances[i] = movementEvents.get(i).distance;
            angles[i] = movementEvents.get(i).angle;
        }

        double meanVel = mean(velocities);
        double stdVel = std(velocities, meanVel);
        double meanDist = mean(distances);
        double stdDist = std(distances, meanDist);
        double meanAngle = mean(angles);
        double stdAngle = std(angles, meanAngle);

        double[] angleChanges = new double[Math.max(m - 1, 0)];
        int reversalCount = 0;
        for (int i = 1; i < m; i++) {
            double diff = circularDiff(angles[i - 1], angles[i]);
            angleChanges[i - 1] = diff;
            if (Math.abs(diff) > 90.0) reversalCount++;
        }
        double meanAngleChange = mean(angleChanges);
        double stdAngleChange = std(angleChanges, meanAngleChange);

        double speedEntropy = entropy(velocities, 10);
        double directionEntropy = entropy(angles, 16);

        // --- Duration/click-event statistics (only from duration events) ---
        double[] durArr = new double[durationEvents.size()];
        for (int i = 0; i < durationEvents.size(); i++) durArr[i] = durationEvents.get(i).duration;
        double meanDur = mean(durArr);
        double stdDur = std(durArr, meanDur);

        // --- DELBOT-compatible subset: acceleration/jerk derived from consecutive
        // movement events' own timestamps (not the full session's gap sequence,
        // since duration events are interleaved and carry no velocity value). ---
        long[] moveGaps = new long[Math.max(m - 1, 0)];
        for (int i = 1; i < m; i++) {
            moveGaps[i - 1] = Duration.between(movementEvents.get(i - 1).timestamp, movementEvents.get(i).timestamp).toMillis();
        }
        double[] accelerations = new double[Math.max(m - 1, 0)];
        for (int i = 1; i < m; i++) {
            double dt = Math.max(moveGaps[i - 1] / 1000.0, 1e-3);
            accelerations[i - 1] = (velocities[i] - velocities[i - 1]) / dt;
        }
        double meanAccel = mean(accelerations);

        double[] jerks = new double[Math.max(accelerations.length - 1, 0)];
        for (int i = 1; i < accelerations.length; i++) {
            double dt = Math.max(moveGaps[i] / 1000.0, 1e-3);
            jerks[i - 1] = (accelerations[i] - accelerations[i - 1]) / dt;
        }
        double meanJerk = mean(jerks);

        // Reconstruct a pseudo-path from distance+angle vectors (movement events only)
        // to compute path efficiency, since raw x/y coordinates aren't available.
        double sumDx = 0, sumDy = 0, totalPathLen = 0;
        for (int i = 0; i < m; i++) {
            double rad = Math.toRadians(angles[i]);
            sumDx += distances[i] * Math.cos(rad);
            sumDy += distances[i] * Math.sin(rad);
            totalPathLen += distances[i];
        }
        double netDisplacement = Math.hypot(sumDx, sumDy);
        double pathEfficiency = totalPathLen > 0 ? Math.min(netDisplacement / totalPathLen, 1.0) : 0.0;

        String row = String.join(",",
                source, person, sessionId, "0",
                String.valueOf(n), fmt(sessionDurationMs),
                fmt(meanVel), fmt(stdVel),
                fmt(meanDist), fmt(stdDist),
                fmt(meanGap), fmt(stdGap),
                String.valueOf(pauseCount),
                fmt(meanAngle), fmt(stdAngle),
                fmt(meanAngleChange), fmt(stdAngleChange),
                String.valueOf(reversalCount),
                fmt(speedEntropy), fmt(directionEntropy),
                String.valueOf(durationEvents.size()), fmt(meanDur), fmt(stdDur),
                // DELBOT-compatible subset -- indices 23-29, unchanged position,
                // so FalsePositiveEvaluator's column indices still line up.
                String.valueOf(m), fmt(sessionDurationMs), fmt(meanVel), fmt(stdVel),
                fmt(meanAccel), fmt(meanJerk), fmt(pathEfficiency),
                // Extra diagnostic columns appended at the END so nothing upstream shifts.
                String.valueOf(m), String.valueOf(durationEvents.size())
        );
        writer.write(row);
        writer.newLine();
    }

    private static double circularDiff(double a1, double a2) {
        double diff = (a2 - a1) % 360.0;
        if (diff > 180.0) diff -= 360.0;
        if (diff < -180.0) diff += 360.0;
        return diff;
    }

    private static double entropy(double[] values, int bins) {
        if (values.length == 0) return 0.0;
        double min = Arrays.stream(values).min().orElse(0);
        double max = Arrays.stream(values).max().orElse(0);
        if (max - min < 1e-9) return 0.0;
        int[] counts = new int[bins];
        for (double v : values) {
            int bin = (int) (((v - min) / (max - min)) * (bins - 1));
            bin = Math.max(0, Math.min(bins - 1, bin));
            counts[bin]++;
        }
        double h = 0;
        for (int c : counts) {
            if (c == 0) continue;
            double p = (double) c / values.length;
            h -= p * (Math.log(p) / Math.log(2));
        }
        return h;
    }

    private static double mean(double[] arr) {
        return arr.length == 0 ? 0.0 : Arrays.stream(arr).average().orElse(0.0);
    }

    private static double std(double[] arr, double mean) {
        if (arr.length == 0) return 0.0;
        double variance = Arrays.stream(arr).map(v -> (v - mean) * (v - mean)).average().orElse(0.0);
        return Math.sqrt(variance);
    }

    private static double meanLong(long[] arr) {
        return arr.length == 0 ? 0.0 : Arrays.stream(arr).average().orElse(0.0);
    }

    private static double stdLong(long[] arr, double mean) {
        if (arr.length == 0) return 0.0;
        double variance = Arrays.stream(arr).mapToDouble(v -> (v - mean) * (v - mean)).average().orElse(0.0);
        return Math.sqrt(variance);
    }

    private static String fmt(double d) {
        if (Double.isNaN(d) || Double.isInfinite(d)) return "0.0";
        return String.valueOf(d);
    }
}