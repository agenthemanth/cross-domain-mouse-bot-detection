/* demo.js -- capture loop, bot simulators, live rendering.
 * Uses window.MouseFeatures (features.js) and POST /score (DemoServer). */
(function () {
  'use strict';
  var MF = window.MouseFeatures;

  var WINDOW_MS = 4000;      // rolling capture window
  var TICK_MS = 400;         // re-score cadence
  var SERVER_THRESH = 0.87;  // filled from the first /score response
  var UI_THRESH = 0.87;      // what the slider currently shows

  var arena = document.getElementById('arena');
  var canvas = document.getElementById('trail');
  var ctx = canvas.getContext('2d');
  var el = {
    verdict: document.getElementById('verdict'),
    prob: document.getElementById('prob'),
    fill: document.getElementById('fill'),
    thr: document.getElementById('thr'),
    thrNote: document.getElementById('thrNote'),
    thrHint: document.getElementById('thrHint'),
    thrSlider: document.getElementById('thrSlider'),
    catchN: document.getElementById('catchN'),
    sN: document.getElementById('sN'), sDur: document.getElementById('sDur'),
    sVel: document.getElementById('sVel'), sEff: document.getElementById('sEff'),
    sPause: document.getElementById('sPause'),
    rows: document.getElementById('rows')
  };

  var buffer = [];
  var lastSource = 'you';
  var busy = false;
  var lastScore = null;            // {score, source, feat, seg} of the most recent chunk
  var botRuns = [];                // {score} for each bot-sim run, for the "caught" counter

  var CSS = getComputedStyle(document.documentElement);
  var COL_HUMAN = CSS.getPropertyValue('--human').trim();
  var COL_BOT = CSS.getPropertyValue('--bot').trim();

  // ---------- canvas ----------
  function resize() {
    var r = canvas.getBoundingClientRect();
    canvas.width = r.width * devicePixelRatio;
    canvas.height = r.height * devicePixelRatio;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }
  window.addEventListener('resize', resize);
  resize();

  function draw(points, colour) {
    var rect = arena.getBoundingClientRect();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!points || points.length < 2) return;
    for (var i = 1; i < points.length; i++) {
      var a = points[i - 1], b = points[i];
      ctx.strokeStyle = colour;
      ctx.globalAlpha = Math.max(0.08, i / points.length);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(a[1] - rect.left, a[2] - rect.top);
      ctx.lineTo(b[1] - rect.left, b[2] - rect.top);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    var last = points[points.length - 1];
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(last[1] - rect.left, last[2] - rect.top, 3.5, 0, 7);
    ctx.fill();
  }

  // ---------- threshold slider ----------
  function labelFor(score) { return score >= UI_THRESH ? 'BOT' : 'HUMAN'; }

  function applyThreshold() {
    el.thr.style.left = (UI_THRESH * 100) + '%';
    el.thrNote.textContent = 'decision line P(bot) ≥ ' + UI_THRESH.toFixed(2);
    var off = Math.abs(UI_THRESH - SERVER_THRESH) > 0.001;
    el.thrHint.textContent = off
      ? 'moved from the 0.' + Math.round(SERVER_THRESH * 100) + ' operating point — click to reset'
      : 'Tier 5 · 1% human-FPR budget · reset';
    if (lastScore) render(lastScore.score, lastScore.source);
    var caught = botRuns.filter(function (b) { return b.score >= UI_THRESH; }).length;
    el.catchN.textContent = caught + ' / ' + botRuns.length;
  }
  el.thrSlider.addEventListener('input', function () {
    UI_THRESH = parseFloat(el.thrSlider.value);
    applyThreshold();
  });
  el.thrHint.addEventListener('click', function () {
    UI_THRESH = SERVER_THRESH;
    el.thrSlider.value = String(SERVER_THRESH);
    applyThreshold();
  });

  // ---------- render ----------
  function render(score, source) {
    var label = labelFor(score);
    el.verdict.textContent = label;
    el.verdict.className = 'label ' + (label === 'BOT' ? 'bot' : 'human');
    el.prob.textContent = 'P(bot) = ' + score.toFixed(3) + '  ·  ' + source;
    el.fill.style.width = (score * 100).toFixed(1) + '%';
    el.fill.style.background = label === 'BOT' ? 'var(--bot)' : 'var(--human)';
  }
  function setIdle(msg) {
    el.verdict.textContent = '—';
    el.verdict.className = 'label idle';
    el.prob.textContent = msg || 'P(bot) = ·';
    el.fill.style.width = '0%';
  }
  function logRow(source, score) {
    var label = labelFor(score);
    var row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = '<span>' + new Date().toLocaleTimeString().slice(0, 8) + '</span>' +
      '<span>' + source + '</span>' +
      '<span class="v ' + (label === 'BOT' ? 'bot' : 'human') + '">' + label + '</span>' +
      '<span>' + score.toFixed(3) + '</span>';
    el.rows.insertBefore(row, el.rows.firstChild);
    while (el.rows.children.length > 40) el.rows.removeChild(el.rows.lastChild);
  }
  function showFeatures(feat, seg) {
    el.sN.textContent = MF.collapse(seg).length;
    el.sDur.textContent = Math.round(feat[1]) + ' ms';
    el.sVel.textContent = feat[2].toFixed(3) + ' px/ms';
    el.sEff.textContent = feat[6].toFixed(3);
    el.sPause.textContent = feat[16].toFixed(3);
  }

  // ---------- scoring ----------
  function score(feat, seg, source, isBot) {
    return fetch('/score', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: feat })
    }).then(function (r) { return r.json(); }).then(function (res) {
      if (res.error) { setIdle(res.error); return; }
      if (SERVER_THRESH !== res.threshold && botRuns.length === 0 &&
          Math.abs(UI_THRESH - SERVER_THRESH) < 0.001) {
        SERVER_THRESH = res.threshold; UI_THRESH = res.threshold;
        el.thrSlider.value = String(res.threshold);
      }
      SERVER_THRESH = res.threshold;
      lastScore = { score: res.score, source: source, feat: feat, seg: seg };
      render(res.score, source);
      showFeatures(feat, seg);
      logRow(source, res.score);
      if (isBot) { botRuns.push({ score: res.score }); }
      applyThreshold();
    }).catch(function () { setIdle('scorer offline — is DemoServer running?'); });
  }

  // ---------- live capture ----------
  arena.addEventListener('mousemove', function (e) {
    arena.classList.add('active');
    buffer.push([performance.now(), e.clientX, e.clientY]);
    lastSource = 'you';
    var cut = performance.now() - WINDOW_MS;
    while (buffer.length && buffer[0][0] < cut) buffer.shift();
    draw(buffer, COL_HUMAN);
  });

  function evaluateLive() {
    if (busy || lastSource !== 'you') return;
    var segs = MF.splitByGap(buffer, MF.GAP_THRESHOLD_MS);
    var seg = segs.length ? segs[segs.length - 1] : [];
    var nc = MF.collapse(seg).length;
    if (nc < MF.MIN_POINTS) { setIdle('keep moving — ' + nc + '/' + MF.MIN_POINTS + ' samples'); return; }
    var feat = MF.extractAugmented(seg);
    if (!feat) { setIdle('chunk not scorable yet'); return; }
    busy = true;
    score(feat, seg, 'you', false).finally(function () { busy = false; });
  }
  setInterval(evaluateLive, TICK_MS);

  // ---------- bot simulators ----------
  function box() {
    var r = arena.getBoundingClientRect();
    return { x0: r.left + 60, y0: r.top + 60, x1: r.left + r.width - 60, y1: r.top + r.height - 60 };
  }
  function runSim(source, points) {
    lastSource = source;
    var i = 0;
    (function step() {
      i += 4;
      draw(points.slice(0, Math.min(i, points.length)), COL_BOT);
      if (i < points.length) requestAnimationFrame(step);
    })();
    var feat = MF.extractAugmented(points);
    if (!feat) { setIdle('sim produced no scorable chunk'); return; }
    score(feat, points, source, true);
  }

  function bezierPath() {                       // smooth min-jerk -- NaturalMouseMotion / GAN
    var b = box(), n = 110, out = [];
    var ax = b.x0 + 10, ay = b.y1 - 10, bx = b.x1 - 10, by = b.y0 + 10;
    var cx = (ax + bx) / 2 + (Math.random() * 120 - 60), cy = (ay + by) / 2 + (Math.random() * 120 - 60);
    for (var i = 0; i < n; i++) {
      var u = i / (n - 1), e = u * u * u * (10 - 15 * u + 6 * u * u);   // quintic min-jerk
      var x = (1 - e) * (1 - e) * ax + 2 * (1 - e) * e * cx + e * e * bx;
      var y = (1 - e) * (1 - e) * ay + 2 * (1 - e) * e * cy + e * e * by;
      out.push([i * 18, Math.round(x), Math.round(y)]);
    }
    return out;
  }
  function linearPath() {                       // constant velocity, fixed 8 ms step
    var b = box(), n = 150, out = [];
    var ax = b.x0 + Math.random() * 40, ay = b.y1 - Math.random() * 40;
    var bx = b.x1 - Math.random() * 40, by = b.y0 + Math.random() * 40;
    for (var i = 0; i < n; i++) {
      var s = i / (n - 1);
      out.push([i * 8, Math.round(ax + (bx - ax) * s), Math.round(ay + (by - ay) * s)]);
    }
    return out;
  }
  function stepPath() {                         // teleport + micro-settle
    var b = box(), out = [], t = 0;
    var pts = [[b.x0, b.y1], [b.x0 + (b.x1 - b.x0) * 0.55, b.y1 - (b.y1 - b.y0) * 0.4],
      [b.x1 - 30, b.y0 + 40], [b.x1 - 20, b.y0 + 30]];
    for (var k = 1; k < pts.length; k++) {
      var seg = 6 + Math.floor(Math.random() * 4);
      for (var i = 0; i < seg; i++) {
        var s = i / seg; t += 4 + Math.random() * 3;
        out.push([t, Math.round(pts[k - 1][0] + (pts[k][0] - pts[k - 1][0]) * s),
          Math.round(pts[k - 1][1] + (pts[k][1] - pts[k - 1][1]) * s)]);
      }
      t += 90 + Math.random() * 60;
    }
    for (var j = 0; j < 20; j++) { t += 10; out.push([t, pts[3][0] + (Math.random() * 3 - 1.5), pts[3][1] + (Math.random() * 3 - 1.5)]); }
    return out;
  }
  function evasivePath() {                      // min-jerk speed profile, shuffled per step
    var b = box(), n = 120;
    var ax = b.x0 + 20, ay = b.y1 - 20, bx = b.x1 - 20, by = b.y0 + 20;
    var steps = [];
    for (var i = 0; i < n; i++) { var u = (i + 0.5) / n; steps.push(Math.pow(u * (1 - u), 1.2) + 0.02); }
    for (var k = steps.length - 1; k > 0; k--) {
      var m = Math.floor(Math.random() * (k + 1)); var tmp = steps[k]; steps[k] = steps[m]; steps[m] = tmp;
    }
    var sum = steps.reduce(function (a, c) { return a + c; }, 0);
    var out = [], acc = 0, t = 0;
    for (var q = 0; q <= n; q++) {
      var frac = acc / sum;
      var px = ax + (bx - ax) * frac;
      var py = ay + (by - ay) * frac + Math.sin(frac * Math.PI) * 26 * Math.sin(frac * 9);
      t += 8 + Math.random() * 8;
      out.push([t, Math.round(px), Math.round(py)]);
      if (q < n) acc += steps[q];
    }
    return out;
  }

  document.getElementById('btnBezier').onclick = function () { runSim('bot·bezier', bezierPath()); };
  document.getElementById('btnLinear').onclick = function () { runSim('bot·linear', linearPath()); };
  document.getElementById('btnStep').onclick = function () { runSim('bot·step', stepPath()); };
  document.getElementById('btnEvasive').onclick = function () { runSim('bot·evasive', evasivePath()); };

  applyThreshold();
  setIdle('move your mouse in the box');
})();
