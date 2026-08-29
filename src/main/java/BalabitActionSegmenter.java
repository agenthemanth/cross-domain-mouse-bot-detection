import java.util.ArrayList;
import java.util.List;

/**
 * TIER 3 -- cuts a Balabit event stream into individual mouse ACTIONS instead of
 * blind 3-second-gap chunks.
 *
 * Tier 2, step 1 showed that per-chunk aggregate features hide the signal: a
 * gap-split chunk runs a median ~13.6 s / 66 points and mixes several distinct
 * movements, so a bot's burst-then-idle profile averages out to "slow careful
 * human". Antal & Egyed-Zsigmond (2019, IET Biometrics -- Balabit auth, set-of-
 * actions AUC 0.92, drag-and-drop the most discriminative) segment to the action
 * level instead. This class does the same segmentation:
 *
 *   MM  mouse-move  : a run of Move events, no click, split on a >1 s pause
 *   PC  point-click : a Move run that ends in Pressed then Released at ~the same
 *                     spot (no drag) -- the approach movement plus the click
 *   DD  drag-drop   : Pressed -> Drag... -> Released with real displacement
 *
 * Each action is returned as its trajectory points (t_seconds, x, y), ascending
 * in t, tied timestamps already collapsed. Actions that are too short, too
 * small, or too long (a "long action" is really several -- keeping it would
 * re-introduce the aggregation problem) are dropped.
 */
public final class BalabitActionSegmenter {

    public enum ActionType { MM, PC, DD }

    public static final class Action {
        public final ActionType type;
        public final List<double[]> points; // (t_seconds, x, y)
        Action(ActionType type, List<double[]> points) {
            this.type = type;
            this.points = points;
        }
        public double durationSec() { return points.get(points.size() - 1)[0] - points.get(0)[0]; }
    }

    private static final double MM_SPLIT_PAUSE_SEC = 1.0;
    private static final int MIN_ACTION_POINTS = 6;      // after tied-timestamp collapse
    private static final double MIN_PATH_PX = 20.0;      // drop micro-movements / no-ops
    private static final double MAX_ACTION_SEC = 8.0;    // longer => not a single action
    private static final double DD_MIN_DISPLACEMENT_PX = 15.0;

    private BalabitActionSegmenter() {}

    /** @param events output of BalabitValidationPipeline.parseBalabitEvents (t, x, y, stateCode) */
    public static List<Action> segment(List<double[]> events) {
        List<Action> out = new ArrayList<>();
        List<double[]> cur = new ArrayList<>();      // current movement buffer (t, x, y)
        boolean pressActive = false;
        boolean dragSeen = false;
        int pressIdx = -1;

        for (double[] e : events) {
            double t = e[0], x = e[1], y = e[2];
            int code = (int) e[3];

            switch (code) {
                case BalabitValidationPipeline.EV_MOVE:
                    if (!cur.isEmpty() && t - cur.get(cur.size() - 1)[0] > MM_SPLIT_PAUSE_SEC && !pressActive) {
                        flushMove(cur, out);
                        cur = new ArrayList<>();
                    }
                    cur.add(new double[]{t, x, y});
                    break;

                case BalabitValidationPipeline.EV_PRESSED:
                    cur.add(new double[]{t, x, y});
                    pressActive = true;
                    dragSeen = false;
                    pressIdx = cur.size() - 1;
                    break;

                case BalabitValidationPipeline.EV_DRAG:
                    cur.add(new double[]{t, x, y});
                    if (pressActive) dragSeen = true;
                    break;

                case BalabitValidationPipeline.EV_RELEASED:
                    cur.add(new double[]{t, x, y});
                    if (pressActive && dragSeen && pressIdx >= 0) {
                        double[] p0 = cur.get(pressIdx);
                        double disp = Math.hypot(x - p0[1], y - p0[2]);
                        if (disp >= DD_MIN_DISPLACEMENT_PX) {
                            emit(ActionType.DD, new ArrayList<>(cur.subList(pressIdx, cur.size())), out);
                        }
                        cur = new ArrayList<>();
                    } else if (pressActive) {
                        emit(ActionType.PC, cur, out); // approach + click
                        cur = new ArrayList<>();
                    }
                    pressActive = false;
                    dragSeen = false;
                    pressIdx = -1;
                    break;

                default: // scroll / other -> hard boundary
                    flushMove(cur, out);
                    cur = new ArrayList<>();
                    pressActive = false;
                    dragSeen = false;
                    pressIdx = -1;
            }
        }
        flushMove(cur, out);
        return out;
    }

    private static void flushMove(List<double[]> cur, List<Action> out) {
        if (cur.size() >= MIN_ACTION_POINTS) emit(ActionType.MM, cur, out);
    }

    private static void emit(ActionType type, List<double[]> raw, List<Action> out) {
        List<double[]> pts = BalabitValidationPipeline.collapseToDistinctTimestamps(raw);
        if (pts.size() < MIN_ACTION_POINTS) return;

        double dur = pts.get(pts.size() - 1)[0] - pts.get(0)[0];
        if (dur <= 0 || dur > MAX_ACTION_SEC) return;

        double pathLen = 0.0;
        for (int i = 1; i < pts.size(); i++) {
            pathLen += Math.hypot(pts.get(i)[1] - pts.get(i - 1)[1], pts.get(i)[2] - pts.get(i - 1)[2]);
        }
        if (pathLen < MIN_PATH_PX) return;

        out.add(new Action(type, pts));
    }
}
