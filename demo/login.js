/* login.js -- the test login page.
 *
 * Two channels run at the same time and are judged together by the server at
 * sign-in (POST /auth/login), never averaged:
 *   1. background mouse movement anywhere on the page, drag samples excluded,
 *      scored by the frozen forest;
 *   2. the moving puzzle: while you drag, the piece position is streamed to
 *      /puzzle/move (sequence-numbered, several in flight) and stamped on the
 *      SERVER clock. At a secret point a reply delivers a new board image on which
 *      the slot has moved, plus an ack token that every later report echoes. The
 *      server times the re-aim from the ack's arrival, so network round trip is
 *      not counted as reaction time.
 *
 * The slot position is never sent as a number, only drawn into the board PNG
 * among decoys. The L1-L4 bot sims below therefore find it by looking at pixels.
 *
 * Study mode: /login.html?study=1&p=<code>. Every sign-in is logged server-side
 * under the participant code (DemoServer --log-trials), the verdict panel is
 * hidden so people don't adapt to it, and a fresh puzzle follows each trial. */
(function () {
  'use strict';
  var post = App.postJson, esc = App.esc, sleep = App.sleep, minJerk = App.minJerk;
  var MF = window.MouseFeatures;

  var $ = function (id) { return document.getElementById(id); };
  var board = $('pboard'), pieceEl = $('piece'), ghost = $('ghost');
  var form = $('form'), userEl = $('user'), passEl = $('pass');

  var params = new URLSearchParams(location.search);
  var STUDY = params.get('study') === '1';
  var PARTICIPANT = (params.get('p') || '').replace(/[^A-Za-z0-9_-]/g, '').slice(0, 32);
  var trialCount = 0;

  // ---------- state ----------
  var bg = [];            // [t_ms, clientX, clientY, seg]
  var seg = 0;            // bumped at every drag boundary so chunks never span a drag
  var simRunning = false, simLevel = 0;

  var pz = null;          // issued puzzle
  var boardUri = null;
  var pos = { x: 0, y: 0 };
  var dragging = false, placed = false, jumped = false;
  var grab = { dx: 0, dy: 0 };

  var MAX_INFLIGHT = 6, SEND_MS = 16;
  var seq = 0, inflightN = 0, ackToken = null, pumpTimer = null;

  // ---------- background capture ----------
  document.addEventListener('mousemove', function (e) {
    if (simRunning || dragging) return;
    pushBg(performance.now(), e.clientX, e.clientY);
  });
  function pushBg(t, x, y) {
    bg.push([t, x, y, seg]);
    if (bg.length > 8000) bg.shift();
    $('pvSamples').textContent = bg.length;
  }

  // live preview of the passive channel: the current segment's latest chunk
  if (!STUDY) setInterval(function () {
    var cur = bg.filter(function (p) { return p[3] === seg; })
      .map(function (p) { return [p[0], p[1], p[2]]; });
    var chunks = MF.splitByGap(cur, MF.GAP_THRESHOLD_MS);
    var last = chunks.length ? chunks[chunks.length - 1] : [];
    if (MF.collapse(last).length < MF.MIN_POINTS) return;
    var f = MF.extractAugmented(last);
    if (!f) return;
    post('/score', { features: f })
      .then(function (r) {
        if (r.error) return;
        $('pvLive').textContent = r.score.toFixed(3);
        $('pvThr').textContent = r.threshold.toFixed(2);
      }).catch(function () {});
  }, 1000);

  // ---------- puzzle geometry ----------
  function scale() { return board.getBoundingClientRect().width / pz.boardW; }
  function toBoard(cx, cy) {
    var r = board.getBoundingClientRect(), s = scale();
    return { x: (cx - r.left) / s, y: (cy - r.top) / s };
  }
  function toClient(x, y) {
    var r = board.getBoundingClientRect(), s = scale();
    return { x: r.left + x * s, y: r.top + y * s };
  }
  function pct(v, total) { return (v / total * 100) + '%'; }
  function place(el, x, y, size) {
    el.style.left = pct(x, pz.boardW); el.style.top = pct(y, pz.boardH);
    el.style.width = pct(size, pz.boardW); el.style.height = pct(size, pz.boardH);
  }
  function render() { place(pieceEl, pos.x, pos.y, pz.piece); }
  function setBoard(uri) {
    boardUri = uri;
    board.style.backgroundImage = 'url("' + uri + '")';
  }
  function clampPos() {
    var h = pz.piece / 2;
    pos.x = Math.max(h, Math.min(pz.boardW - h, pos.x));
    pos.y = Math.max(h, Math.min(pz.boardH - h, pos.y));
  }
  function status(t) { $('pstatus').textContent = t; }

  function newPuzzle() {
    placed = false; dragging = false; jumped = false;
    seq = 0; ackToken = null;
    board.classList.remove('placed');
    status('loading puzzle…');
    return post('/puzzle/new', {}).then(function (res) {
      pz = res;
      pos = { x: res.startX, y: res.startY };
      setBoard(res.board);
      pieceEl.style.backgroundImage = 'url("' + res.pieceImg + '")';
      render();
      $('ptries').textContent = '';
      status('drag the blue piece onto the outline of the same shape');
    }).catch(function () { status('server offline — is DemoServer running?'); });
  }

  // ---------- streaming to the server clock ----------
  function sendMove() {
    if (!dragging || placed || inflightN >= MAX_INFLIGHT) return;
    inflightN++;
    var body = { id: pz.id, seq: ++seq, x: +pos.x.toFixed(2), y: +pos.y.toFixed(2) };
    if (ackToken) body.ack = ackToken;
    post('/puzzle/move', body).then(function (res) {
      inflightN--;
      if (res.state === 'jump' && !jumped) {
        jumped = true;
        ackToken = res.ack;
        setBoard(res.board);
      }
    }).catch(function () { inflightN--; });
  }
  function idle() {
    return new Promise(function (done) {
      (function wait() { if (inflightN === 0) done(); else setTimeout(wait, 3); })();
    });
  }

  function beginDrag() {
    dragging = true; seg++;
    pieceEl.classList.add('dragging');
    sendMove();
    pumpTimer = setInterval(sendMove, SEND_MS);
  }
  function endDrag() {
    dragging = false; seg++;
    clearInterval(pumpTimer);
    pieceEl.classList.remove('dragging');
    return idle().then(function () {
      var body = { id: pz.id, x: +pos.x.toFixed(2), y: +pos.y.toFixed(2) };
      if (ackToken) body.ack = ackToken;
      return post('/puzzle/drop', body);
    }).then(function (res) {
      if (res.state === 'placed') {
        placed = true;
        if (res.slotX != null) { pos = { x: res.slotX, y: res.slotY }; render(); }
        board.classList.add('placed');
        status('placed — you can sign in');
      } else if (res.state === 'miss') {
        board.classList.remove('shake'); void board.offsetWidth; board.classList.add('shake');
        $('ptries').textContent = res.left + ' tries left';
        status('not quite — drop it onto the matching outline');
      } else if (res.state === 'exhausted') {
        status('too many misses — here is a fresh puzzle');
        setTimeout(newPuzzle, 900);
      }
      return res.state;
    });
  }

  // human pointer input
  pieceEl.addEventListener('pointerdown', function (e) {
    if (!pz || placed || simRunning) return;
    e.preventDefault();
    pieceEl.setPointerCapture(e.pointerId);
    var b = toBoard(e.clientX, e.clientY);
    grab = { dx: b.x - pos.x, dy: b.y - pos.y };
    beginDrag();
  });
  pieceEl.addEventListener('pointermove', function (e) {
    if (!dragging || simRunning) return;
    var b = toBoard(e.clientX, e.clientY);
    pos = { x: b.x - grab.dx, y: b.y - grab.dy };
    clampPos(); render();
  });
  pieceEl.addEventListener('pointerup', function () {
    if (dragging && !simRunning) endDrag();
  });
  window.addEventListener('resize', function () { if (pz) render(); });

  // ---------- sign in ----------
  form.addEventListener('submit', function (e) {
    e.preventDefault();
    if (!placed) { showResult('warn', 'Finish the puzzle first', 'drag the piece onto the matching outline'); return; }
    signIn();
  });

  function signIn() {
    $('signin').disabled = true;
    var body = {
      user: userEl.value, pass: passEl.value, puzzle: pz ? pz.id : '',
      bg: bg.map(function (p) { return [+p[0].toFixed(2), +p[1].toFixed(1), +p[2].toFixed(1), p[3]]; })
    };
    if (PARTICIPANT) body.participant = PARTICIPANT;
    if (simLevel) body.sim = simLevel;
    return post('/auth/login', body).then(function (r) {
      $('signin').disabled = false;
      if (r.error) { showResult('bad', 'Server error', r.error); return r; }
      if (STUDY) {
        trialCount++;
        $('studyTrial').textContent = trialCount;
        showResult('ok', 'Trial ' + trialCount + ' recorded', 'a new puzzle is loaded — do it again, at your normal pace');
      } else {
        showChannels(r);
        if (r.loggedIn) showResult('ok', 'Welcome, ' + userEl.value + ' — signed in', 'both checks judged you human');
        else if (r.decision === 'BOT') showResult('bad', 'Blocked — this looks automated', r.reason);
        else showResult('warn', 'Wrong username or password', 'you passed the bot checks; try demo / letmein');
      }
      bg = []; seg++; $('pvSamples').textContent = '0'; $('pvLive').textContent = '·';
      return newPuzzle().then(function () { return r; });
    }).catch(function () {
      $('signin').disabled = false;
      showResult('bad', 'Server offline', 'is DemoServer running?');
    });
  }

  function showResult(kind, title, sub) {
    var el = $('result');
    el.className = 'result on ' + kind;
    el.innerHTML = '<b>' + esc(title) + '</b>' + esc(sub || '');
  }

  function showChannels(r) {
    var pv = r.passive, p = r.puzzle;
    setPill('pvPill', pv.verdict === 'BOT' ? 'bot' : pv.verdict === 'HUMAN' ? 'human' : 'none',
      pv.verdict === 'NO_EVIDENCE' ? 'no evidence' : pv.verdict);
    $('pvMedian').textContent = pv.median == null ? '·' : pv.median.toFixed(3);
    $('pvChunks').textContent = pv.chunks.length
      ? pv.chunks.map(function (c) { return c.score.toFixed(2); }).join(' · ') : 'none ≥20 pts';
    $('pvThr').textContent = pv.threshold.toFixed(2);

    setPill('pzPill', p.pass ? 'human' : 'bot', p.pass ? 'PASS' : 'FAIL');
    $('pzJump').textContent = !p.jumped ? 'no' : !p.acked ? 'yes, never acked'
      : p.ackMs == null ? 'yes' : 'yes (ack ' + Math.round(p.ackMs) + ' ms)';
    $('pzOnset').textContent = p.onsetMs == null ? '·' : Math.round(p.onsetMs) + ' ms';
    $('pzDrop').textContent = p.jumpToDropMs == null ? '·' : Math.round(p.jumpToDropMs) + ' ms';
    $('pzCarry').textContent = p.carryPx == null ? '·' : Math.round(p.carryPx) + ' px';
    $('pzSamples').textContent = p.samples + ' (' + p.postJumpSamples + ' after ack)';
    var ul = $('pzFlags'); ul.innerHTML = '';
    p.flags.forEach(function (f) { var li = document.createElement('li'); li.textContent = f; ul.appendChild(li); });
    p.notes.forEach(function (f) { var li = document.createElement('li'); li.className = 'note'; li.textContent = f; ul.appendChild(li); });

    var d = $('decision');
    d.className = 'decision ' + (r.decision === 'BOT' ? 'bot' : 'human');
    d.innerHTML = '<b>' + r.decision + '</b>' + esc(r.reason);
  }
  function setPill(id, cls, text) { var el = $(id); el.className = 'pill ' + cls; el.textContent = text; }

  // ---------- helpers ----------
  function centreOf(el) { var r = el.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; }
  function loadImage(uri) {
    return new Promise(function (done, fail) {
      var im = new Image(); im.onload = function () { done(im); }; im.onerror = fail; im.src = uri;
    });
  }

  // ================= vision (what the bot sims see) =================
  // A white-box attacker: it knows the board is outlines on a light background
  // and that the slot has the piece's shape. It thresholds the board image, takes
  // connected components, and picks the outline that best matches the piece's
  // boundary. This is the perception step the image forces, and the per-analysis
  // cost is measured and shown.

  var SLOT = 56, PIECE_SHAPE = 48;          // must match PerturbationPuzzle.SLOT and PIECE - 2
  var vis = { tmpl: null, uri: null, busy: null, loc: null, ms: [] };

  function buildTemplate() {
    return loadImage(pz.pieceImg).then(function (im) {
      var k = SLOT / PIECE_SHAPE, n = Math.ceil(pz.piece * k) + 6;
      var c = document.createElement('canvas'); c.width = c.height = n;
      var g = c.getContext('2d');
      var s = pz.piece * k;
      g.drawImage(im, (n - s) / 2, (n - s) / 2, s, s);
      var a = g.getImageData(0, 0, n, n).data, inside = function (x, y) {
        return x >= 0 && y >= 0 && x < n && y < n && a[(y * n + x) * 4 + 3] > 128;
      };
      var pts = [], minX = n, maxX = 0, minY = n, maxY = 0;
      for (var y = 0; y < n; y++) for (var x = 0; x < n; x++) {
        if (!inside(x, y)) continue;
        if (inside(x - 1, y) && inside(x + 1, y) && inside(x, y - 1) && inside(x, y + 1)) continue;
        pts.push([x, y]);
        minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
      var bx = (minX + maxX) / 2, by = (minY + maxY) / 2;
      vis.tmpl = {
        pts: pts.map(function (p) { return [p[0] - bx, p[1] - by]; }),
        anchor: [n / 2 - bx, n / 2 - by]              // shape anchor relative to its bbox centre
      };
    });
  }

  function analyse(im) {
    var t0 = performance.now();
    var W = pz.boardW, H = pz.boardH;
    var c = document.createElement('canvas'); c.width = W; c.height = H;
    var g = c.getContext('2d');
    g.drawImage(im, 0, 0, W, H);
    var d = g.getImageData(0, 0, W, H).data;
    var dark = new Uint8Array(W * H);
    for (var i = 0; i < W * H; i++) {
      var l = 0.3 * d[i * 4] + 0.59 * d[i * 4 + 1] + 0.11 * d[i * 4 + 2];
      dark[i] = l < 215 ? 1 : 0;
    }
    var label = new Int32Array(W * H), comps = [], stack = [];
    for (var s0 = 0; s0 < W * H; s0++) {
      if (!dark[s0] || label[s0]) continue;
      var id = comps.length + 1, px = [];
      label[s0] = id; stack.push(s0);
      while (stack.length) {
        var q = stack.pop(), qx = q % W, qy = (q - qx) / W;
        px.push(q);
        if (qx > 0 && dark[q - 1] && !label[q - 1]) { label[q - 1] = id; stack.push(q - 1); }
        if (qx < W - 1 && dark[q + 1] && !label[q + 1]) { label[q + 1] = id; stack.push(q + 1); }
        if (qy > 0 && dark[q - W] && !label[q - W]) { label[q - W] = id; stack.push(q - W); }
        if (qy < H - 1 && dark[q + W] && !label[q + W]) { label[q + W] = id; stack.push(q + W); }
      }
      comps.push(px);
    }
    var best = null, bestScore = -1, R = 2;
    comps.forEach(function (px, ci) {
      if (px.length < 30) return;
      var minX = W, maxX = 0, minY = H, maxY = 0;
      px.forEach(function (q) { var x = q % W, y = (q - x) / W;
        minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); });
      var cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
      var hit = 0;
      vis.tmpl.pts.forEach(function (p) {
        var tx = Math.round(cx + p[0]), ty = Math.round(cy + p[1]), ok = false;
        for (var dy = -R; dy <= R && !ok; dy++) for (var dx = -R; dx <= R && !ok; dx++) {
          var xx = tx + dx, yy = ty + dy;
          if (xx >= 0 && yy >= 0 && xx < W && yy < H && label[yy * W + xx] === ci + 1) ok = true;
        }
        if (ok) hit++;
      });
      var recall = hit / vis.tmpl.pts.length;
      var size = Math.max(maxX - minX, maxY - minY);
      var fit = recall * Math.min(1, 1 - Math.abs(size - SLOT) / SLOT);
      if (fit > bestScore) { bestScore = fit; best = { x: cx + vis.tmpl.anchor[0], y: cy + vis.tmpl.anchor[1] }; }
    });
    vis.ms.push(performance.now() - t0);
    return best;
  }

  /** Re-analyse whenever the board image changes; meanwhile the bot keeps its last reading. */
  function refreshVision() {
    if (vis.uri === boardUri || vis.busy) return vis.busy || Promise.resolve(vis.loc);
    var uri = boardUri;
    vis.busy = loadImage(uri).then(function (im) {
      var loc = analyse(im);
      vis.uri = uri; vis.busy = null;
      if (loc) vis.loc = loc;
      return vis.loc;
    });
    return vis.busy;
  }

  // ================= bot ladder =================
  // Bots cannot fire trusted mouse events, so they write straight into the same
  // buffers the event handlers feed -- which is also what a real browser-driving
  // bot's synthetic events would end up as. Everything runs in real time.

  function setGhost(x, y) { ghost.style.transform = 'translate(' + x + 'px,' + y + 'px)'; }

  /** Background leg in viewport coords, as [relative_ms, x, y]. */
  function legLinear(a, b) {                      // pyautogui-style: straight line, fixed 8 ms step
    var n = 150, out = [];
    for (var i = 0; i < n; i++) {
      var s = i / (n - 1);
      out.push([i * 8, Math.round(a.x + (b.x - a.x) * s), Math.round(a.y + (b.y - a.y) * s)]);
    }
    return out;
  }
  function legEvasive(a, b) {                     // shuffled min-jerk speeds (Tier 4 style)
    var n = 70, steps = [];
    for (var i = 0; i < n; i++) { var u = (i + 0.5) / n; steps.push(Math.pow(u * (1 - u), 1.2) + 0.02); }
    for (var k = n - 1; k > 0; k--) { var m = Math.floor(Math.random() * (k + 1)); var tmp = steps[k]; steps[k] = steps[m]; steps[m] = tmp; }
    var sum = steps.reduce(function (s, c) { return s + c; }, 0);
    var dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1;
    var out = [], acc = 0, t = 0;
    for (var q = 0; q <= n; q++) {
      var f = acc / sum, w = Math.sin(f * Math.PI) * 20 * Math.sin(f * 9);
      out.push([t, a.x + dx * f - dy / L * w, a.y + dy * f + dx / L * w]);
      t += 8 + Math.random() * 8;
      if (q < n) acc += steps[q];
    }
    return out;
  }
  function playLeg(pts) {
    var base = performance.now(), i = 0;
    return new Promise(function (done) {
      (function tick() {
        var now = performance.now() - base;
        while (i < pts.length && pts[i][0] <= now) {
          pushBg(base + pts[i][0], pts[i][1], pts[i][2]);
          setGhost(pts[i][1], pts[i][2]);
          i++;
        }
        if (i < pts.length) setTimeout(tick, 4); else done(pts[pts.length - 1]);
      })();
    });
  }
  function typeInto(el, text) {
    el.value = '';
    var i = 0;
    return new Promise(function (done) {
      (function next() {
        if (i >= text.length) return done();
        el.value += text[i++];
        setTimeout(next, 70 + Math.random() * 60);
      })();
    });
  }

  function drivePuzzle(level) {
    vis.uri = null; vis.loc = null; vis.busy = null; vis.ms = [];
    return buildTemplate().then(refreshVision).then(function (first) {
      var from = { x: pos.x, y: pos.y };
      var fixed = { x: first.x, y: first.y };             // open-loop bots plan once, here
      var T = level === 1 ? 650 : 800;
      var arc = (Math.random() < 0.5 ? -1 : 1) * (12 + Math.random() * 14);
      var lat = 180 + Math.random() * 80;
      var seen = [];                                      // L4's delay line of slot readings
      var t0 = performance.now();
      beginDrag();
      return new Promise(function (done) {
        (function step() {
          var now = performance.now(), u = Math.min(1, (now - t0) / T);
          if (level >= 3) refreshVision();
          var target;
          if (level <= 2) target = fixed;
          else if (level === 3) target = vis.loc;
          else {
            seen.push({ t: now, s: { x: vis.loc.x, y: vis.loc.y } });
            target = seen[0].s;
            for (var k = 0; k < seen.length && seen[k].t <= now - lat; k++) target = seen[k].s;
          }
          var e = level === 1 ? u : minJerk(u);
          var dx = target.x - from.x, dy = target.y - from.y, L = Math.hypot(dx, dy) || 1;
          var w = level === 2 ? Math.sin(u * Math.PI) * arc : 0;
          pos = { x: from.x + dx * e - dy / L * w, y: from.y + dy * e + dx / L * w };
          render();
          var c = toClient(pos.x, pos.y); setGhost(c.x, c.y);
          if (u < 1) setTimeout(step, 12);
          else if (level >= 3 && vis.busy) setTimeout(step, 12);
          else if (level === 4 && seen[seen.length - 1].t - lat < lastChange(seen)) setTimeout(step, 12);
          else done();
        })();
      });
    }).then(function () { return sleep(40); }).then(endDrag);
  }
  function lastChange(seen) {
    var t = 0;
    for (var i = 1; i < seen.length; i++) {
      if (Math.hypot(seen[i].s.x - seen[i - 1].s.x, seen[i].s.y - seen[i - 1].s.y) > 3) t = seen[i].t;
    }
    return t;
  }

  function runBot(level) {
    var buttons = document.querySelectorAll('.sim');
    Array.prototype.forEach.call(buttons, function (b) { b.disabled = true; });
    $('result').className = 'result';
    simRunning = true; simLevel = level;
    bg = []; seg++;
    userEl.value = ''; passEl.value = '';
    ghost.classList.add('on');
    var leg = level === 1 ? legLinear : legEvasive;
    var r0 = form.getBoundingClientRect();
    var cur = { x: r0.left + r0.width * 0.8, y: r0.top + 20 };
    setGhost(cur.x, cur.y);

    return newPuzzle()
      .then(function () { return playLeg(leg(cur, centreOf(userEl))); })
      .then(function () { return typeInto(userEl, 'demo'); })
      .then(function () { return playLeg(leg(centreOf(userEl), centreOf(passEl))); })
      .then(function () { return typeInto(passEl, 'letmein'); })
      .then(function () { return playLeg(leg(centreOf(passEl), centreOf(pieceEl))); })
      .then(function () { return sleep(120); })
      .then(function () { return drivePuzzle(level); })
      .then(function () {
        var ms = vis.ms.slice().sort(function (a, b) { return a - b; });
        $('visCost').textContent = ms.length
          ? ms.length + ' board read(s), median ' + ms[ms.length >> 1].toFixed(1) + ' ms each' : '·';
        return sleep(250);
      })
      .then(signIn)                                       // bots post the form whether or not the puzzle is done
      .then(function () {
        simRunning = false; simLevel = 0;
        ghost.classList.remove('on');
        Array.prototype.forEach.call(buttons, function (b) { b.disabled = false; });
      });
  }
  Array.prototype.forEach.call(document.querySelectorAll('.sim'), function (b) {
    b.addEventListener('click', function () { runBot(+b.dataset.level); });
  });

  if (STUDY) {
    document.body.classList.add('study');
    $('studyCode').textContent = PARTICIPANT || '(none — add &p=<code> to the URL)';
    userEl.value = 'demo'; passEl.value = 'letmein';
  }
  newPuzzle();
})();
