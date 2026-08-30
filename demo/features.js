/* features.js -- browser port of the Java feature pipeline.
 *
 * MUST stay bit-parity with:
 *   BalabitValidationPipeline.computeFeatures  (the 7 base features)
 *   Tier2Features.tier2 / .compute(AUGMENTED)  (the 11 domain-bridging features)
 *   BalabitValidationPipeline.collapseToDistinctTimestamps
 *   BalabitValidationPipeline.splitByGap
 *
 * Points are [t_ms, x, y] triples, ascending in t. The Java pipeline computes
 * features in MILLISECONDS, so capture ms timestamps and pass them straight in.
 *
 * parity.html loads demo/model/golden_vectors.json (real held-out Balabit chunks
 * with their Java-computed feature vectors) and asserts extractAugmented() matches.
 */
(function (global) {
  'use strict';

  var MIN_DT_MS = 1.0;          // BalabitValidationPipeline.MIN_DT_MS
  var JERK_CLAMP = 1e6;         // BalabitValidationPipeline.JERK_CLAMP
  var EPS = 1e-9;               // Tier2Features.EPS
  var GAP_THRESHOLD_MS = 3000;  // GAP_THRESHOLD_SEC * 1000
  var MIN_POINTS = 20;          // MIN_POINTS_PER_SESSION

  // --- BalabitValidationPipeline.collapseToDistinctTimestamps ---
  // keep the LAST sample of each run of equal timestamps
  function collapse(pts) {
    if (pts.length < 2) return pts.slice();
    var out = [];
    for (var i = 0; i < pts.length; i++) {
      if (i + 1 < pts.length && pts[i + 1][0] === pts[i][0]) continue;
      out.push(pts[i]);
    }
    return out;
  }

  // --- BalabitValidationPipeline.splitByGap (gap in ms here) ---
  function splitByGap(points, gapMs) {
    var sessions = [], current = [];
    for (var i = 0; i < points.length; i++) {
      if (i > 0 && (points[i][0] - points[i - 1][0]) > gapMs) {
        if (current.length) sessions.push(current);
        current = [];
      }
      current.push(points[i]);
    }
    if (current.length) sessions.push(current);
    return sessions;
  }

  function mean(a) {
    if (a.length === 0) return 0;
    var s = 0;
    for (var i = 0; i < a.length; i++) s += a[i];
    return s / a.length;
  }
  function meanAbs(a) {
    if (a.length === 0) return 0;
    var s = 0;
    for (var i = 0; i < a.length; i++) s += Math.abs(a[i]);
    return s / a.length;
  }
  function std(a, m) {
    if (a.length === 0) return 0;
    var s = 0;
    for (var i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
    return Math.sqrt(s / a.length);            // population, matches Java
  }
  function diff(x) {
    if (x.length < 2) return [];
    var d = new Array(x.length - 1);
    for (var i = 1; i < x.length; i++) d[i - 1] = x[i] - x[i - 1];
    return d;
  }
  function argmax(x) {
    var idx = 0;
    for (var i = 1; i < x.length; i++) if (x[i] > x[idx]) idx = i;   // strict: first max wins
    return idx;
  }
  function countPositive(x) { var c = 0; for (var i = 0; i < x.length; i++) if (x[i] > 0) c++; return c; }
  function countBelow(x, t) { var c = 0; for (var i = 0; i < x.length; i++) if (x[i] < t) c++; return c; }

  // Tier2Features.percentile -- linear-interpolated, q in [0,1]
  function percentile(x, q) {
    if (x.length === 0) return 0;
    var s = x.slice().sort(function (a, b) { return a - b; });
    if (s.length === 1) return s[0];
    var pos = q * (s.length - 1);
    var lo = Math.floor(pos), hi = Math.ceil(pos);
    if (lo === hi) return s[lo];
    return s[lo] + (pos - lo) * (s[hi] - s[lo]);
  }

  // Tier2Features.velocityTailRatio -- p90/p50 over MOVING steps
  function velocityTailRatio(v) {
    var moving = 0, i;
    for (i = 0; i < v.length; i++) if (v[i] > 0.0) moving++;
    if (moving === 0) return 1.0;
    var m = new Array(moving), w = 0;
    for (i = 0; i < v.length; i++) if (v[i] > 0.0) m[w++] = v[i];
    var p50 = percentile(m, 0.50);
    var p90 = percentile(m, 0.90);
    return p90 / p50;
  }

  function finite(v) { return !(Number.isNaN(v) || !Number.isFinite(v)); }

  // --- BalabitValidationPipeline.computeFeatures -> [num_points, duration_ms,
  //     mean_velocity, std_velocity, mean_acceleration, mean_jerk, path_efficiency] ---
  function computeBase(pointsRaw) {
    var points = collapse(pointsRaw);
    if (points.length < 3) return null;

    var velocities = [], i, dt, dx, dy, dist, totalDist = 0;
    for (i = 1; i < points.length; i++) {
      dt = Math.max(points[i][0] - points[i - 1][0], MIN_DT_MS);
      dx = points[i][1] - points[i - 1][1];
      dy = points[i][2] - points[i - 1][2];
      dist = Math.sqrt(dx * dx + dy * dy);
      totalDist += dist;
      velocities.push(dist / dt);
    }
    var accelerations = [];
    for (i = 1; i < velocities.length; i++) accelerations.push(velocities[i] - velocities[i - 1]);
    var jerks = [];
    for (i = 1; i < accelerations.length; i++) {
      var j = accelerations[i] - accelerations[i - 1];
      if (!Number.isFinite(j) || Number.isNaN(j)) j = JERK_CLAMP;
      jerks.push(Math.min(Math.abs(j), JERK_CLAMP) * Math.sign(j));
    }

    var meanVel = mean(velocities);
    var variance = 0;
    for (i = 0; i < velocities.length; i++) variance += (velocities[i] - meanVel) * (velocities[i] - meanVel);
    variance = velocities.length ? variance / velocities.length : 0;
    var stdVel = Math.sqrt(variance);
    var meanAcc = accelerations.length === 0 ? 0 : mean(accelerations);
    var meanJerk = jerks.length === 0 ? 0 : mean(jerks);

    var straight = Math.hypot(
      points[points.length - 1][1] - points[0][1],
      points[points.length - 1][2] - points[0][2]);
    var pathEff = Math.min(straight / (totalDist + 1e-5), 1.0);
    var duration = points[points.length - 1][0] - points[0][0];

    var result = [points.length, duration, meanVel, stdVel, meanAcc, meanJerk, pathEff];
    for (i = 0; i < result.length; i++) if (!finite(result[i])) return null;
    return result;
  }

  // --- Tier2Features.tier2 -> the 11 domain-bridging features ---
  function computeTier2(pointsMs) {
    var p = collapse(pointsMs);
    var n = p.length;
    if (n < 4) return null;

    var nSteps = n - 1;
    var v = new Array(nSteps), stepDist = new Array(nSteps),
        dx = new Array(nSteps), dy = new Array(nSteps);
    var totalDist = 0, i;
    for (i = 1; i < n; i++) {
      var dt = Math.max(p[i][0] - p[i - 1][0], MIN_DT_MS);
      var ddx = p[i][1] - p[i - 1][1];
      var ddy = p[i][2] - p[i - 1][2];
      var d = Math.sqrt(ddx * ddx + ddy * ddy);
      dx[i - 1] = ddx; dy[i - 1] = ddy; stepDist[i - 1] = d;
      v[i - 1] = d / dt;
      totalDist += d;
    }

    var meanV = mean(v);
    var stdV = std(v, meanV);
    var a = diff(v), j = diff(a);
    var meanAbsA = meanAbs(a), meanAbsJ = meanAbs(j);

    var turns = [], curvature = [], reversals = 0;
    for (i = 1; i < nSteps; i++) {
      if (stepDist[i] < EPS || stepDist[i - 1] < EPS) continue;
      var dot = dx[i - 1] * dx[i] + dy[i - 1] * dy[i];
      var cross = dx[i - 1] * dy[i] - dy[i - 1] * dx[i];
      var ang = Math.atan2(Math.abs(cross), dot);        // [0, pi]
      turns.push(ang);
      curvature.push(ang / (stepDist[i] + EPS));
      if (ang > Math.PI / 2.0) reversals++;
    }

    var velocityCv = stdV / (meanV + EPS);
    var accelToVel = meanAbsA / (meanV + EPS);
    var jerkToVel = meanAbsJ / (meanV + EPS);
    var timeToPeakRatio = argmax(v) / (nSteps - 1);
    var accelFraction = countPositive(a) / Math.max(a.length, 1);
    var dirReversalRate = turns.length === 0 ? 0.0 : reversals / turns.length;
    var meanTurn = turns.length === 0 ? 0.0 : mean(turns);
    var stdTurn = turns.length === 0 ? 0.0 : std(turns, meanTurn);
    var meanCurv = curvature.length === 0 ? 0.0 : mean(curvature);
    var pauseRatio = countBelow(v, 0.05 * meanV) / nSteps;
    var p90p50 = velocityTailRatio(v);

    var out = [velocityCv, accelToVel, jerkToVel, timeToPeakRatio, accelFraction,
      dirReversalRate, meanTurn, stdTurn, meanCurv, pauseRatio, p90p50];
    for (i = 0; i < out.length; i++) if (!finite(out[i])) return null;
    return out;
  }

  // --- Tier2Features.compute(pointsMs, AUGMENTED) -> 18-vector or null ---
  function extractAugmented(pointsMs) {
    var base = computeBase(pointsMs);
    if (base === null) return null;
    var t2 = computeTier2(pointsMs);
    if (t2 === null) return null;
    return base.concat(t2);
  }

  var api = {
    collapse: collapse,
    splitByGap: splitByGap,
    computeBase: computeBase,
    computeTier2: computeTier2,
    extractAugmented: extractAugmented,
    GAP_THRESHOLD_MS: GAP_THRESHOLD_MS,
    MIN_POINTS: MIN_POINTS
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.MouseFeatures = api;
})(typeof window !== 'undefined' ? window : globalThis);
