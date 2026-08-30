/* parity.js -- asserts the JS feature port matches the Java pipeline exactly.
 *
 *   node demo/parity.js
 *
 * Loads demo/model/golden_vectors.json (real held-out Balabit chunks, each with
 * the 18-feature vector Java's Tier2Features.compute(AUGMENTED) produced) and
 * recomputes each vector with features.js. Any element off by more than TOL fails
 * the run with a non-zero exit code.
 *
 * This is the check that must pass before the live demo is trusted: if the JS and
 * Java feature math drift, the browser score is wrong even with a perfect model.
 */
'use strict';
var fs = require('fs');
var path = require('path');
var MF = require('./features.js');

var TOL = 1e-6;   // relative-or-absolute tolerance per element

var golden = JSON.parse(fs.readFileSync(path.join(__dirname, 'model', 'golden_vectors.json'), 'utf8'));
var NAMES = ['num_points', 'duration_ms', 'mean_velocity', 'std_velocity', 'mean_acceleration',
  'mean_jerk', 'path_efficiency', 'velocity_cv', 'accel_to_vel_ratio', 'jerk_to_vel_ratio',
  'time_to_peak_vel_ratio', 'accel_fraction', 'dir_reversal_rate', 'mean_turning_angle',
  'std_turning_angle', 'mean_curvature', 'pause_ratio', 'vel_p90_p50_ratio'];

function close(a, b) {
  if (a === b) return true;
  var d = Math.abs(a - b);
  return d <= TOL || d <= TOL * Math.max(Math.abs(a), Math.abs(b));
}

var failures = 0, worst = 0, worstAt = '';
golden.forEach(function (g, ci) {
  var got = MF.extractAugmented(g.points);
  if (got === null) {
    console.error('chunk ' + ci + ': JS returned null, Java did not');
    failures++;
    return;
  }
  for (var k = 0; k < 18; k++) {
    var e = g.features[k], a = got[k];
    var rel = Math.abs(a - e) / (Math.max(Math.abs(a), Math.abs(e)) || 1);
    if (rel > worst) { worst = rel; worstAt = 'chunk ' + ci + ' / ' + NAMES[k]; }
    if (!close(a, e)) {
      failures++;
      console.error('chunk ' + ci + '  ' + NAMES[k].padEnd(24) +
        '  java=' + e + '  js=' + a + '  rel=' + rel.toExponential(3));
    }
  }
});

console.log('chunks checked : ' + golden.length + ' x 18 features');
console.log('worst relative error : ' + worst.toExponential(3) + '   (' + worstAt + ')');
if (failures) {
  console.log('\nPARITY FAIL -- ' + failures + ' element(s) outside tolerance ' + TOL);
  process.exit(1);
}
console.log('\nPARITY PASS -- JS features match Java within ' + TOL);
