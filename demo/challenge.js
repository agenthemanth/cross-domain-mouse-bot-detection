/* challenge.js -- sequential-reveal CAPTCHA client.
 *
 * The client is deliberately dumb about timing: it never tells the server when a
 * prompt appeared or when it reacted. It reports where the cursor went; the
 * server times everything that matters on its own clock. The submitted trace is
 * corroboration only, and the server reconciles its span against that clock
 * before trusting any of it. */
(function () {
  'use strict';

  var board = document.getElementById('board');
  var canvas = document.getElementById('trail');
  var ctx = canvas.getContext('2d');
  var el = {
    prompt: document.getElementById('prompt'),
    pips: document.getElementById('pips'),
    veil: document.getElementById('veil'),
    hint: document.getElementById('hint'),
    badge: document.getElementById('badge'),
    who: document.getElementById('who'),
    hops: document.getElementById('hops'),
    flags: document.getElementById('flags'),
    sTotal: document.getElementById('sTotal'), sCv: document.getElementById('sCv'),
    sMis: document.getElementById('sMis'), sPts: document.getElementById('sPts'),
    sCorr: document.getElementById('sCorr')
  };

  var ch = null;         // {id, tiles, hops, boardW, boardH, tile}
  var hop = 0;
  var trace = [];        // [t_ms, x, y] in board coordinates
  var t0 = 0;
  var live = false;      // accepting input
  var actor = 'you';

  // ---------- geometry ----------
  // The board is authored at the server's logical size and may be scaled down by
  // CSS. Everything reported to the server is in LOGICAL board coordinates, so a
  // narrow window cannot change what the trace looks like.
  function scale() {
    if (!ch) return 1;
    return board.getBoundingClientRect().width / ch.boardW;
  }
  function toBoard(clientX, clientY) {
    var r = board.getBoundingClientRect(), s = scale();
    return [(clientX - r.left) / s, (clientY - r.top) / s];
  }

  function resizeCanvas() {
    var r = board.getBoundingClientRect();
    canvas.width = r.width * devicePixelRatio;
    canvas.height = r.height * devicePixelRatio;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }
  window.addEventListener('resize', function () { resizeCanvas(); redraw(); });

  function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (trace.length < 2) return;
    var s = scale();
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = 2;
    var col = actor === 'you' ? '#5fe9b6' : '#ff7d6c';
    ctx.strokeStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 7;
    for (var i = 1; i < trace.length; i++) {
      ctx.globalAlpha = Math.max(0.07, i / trace.length);
      ctx.beginPath();
      ctx.moveTo(trace[i - 1][1] * s, trace[i - 1][2] * s);
      ctx.lineTo(trace[i][1] * s, trace[i][2] * s);
      ctx.stroke();
    }
    ctx.globalAlpha = 1; ctx.shadowBlur = 0;
  }

  // ---------- challenge lifecycle ----------
  function newChallenge() {
    setVeil(false);
    board.classList.remove('locked');
    resetPanel();
    el.hint.textContent = 'loading challenge…';
    actor = 'you';
    return fetch('/challenge/new', { method: 'POST' })
      .then(function (r) { return r.json(); })
      .then(function (res) {
        ch = res; hop = 0; trace = []; t0 = 0; live = true;
        buildBoard();
        refreshPrompt();
        buildPips();
        resizeCanvas(); redraw();
        el.hint.textContent = 'click the tile showing the prompt glyph — ' +
          ch.hops + ' hops, one revealed at a time';
      })
      .catch(function () { el.hint.textContent = 'server offline — is DemoServer running?'; });
  }

  function buildBoard() {
    Array.prototype.slice.call(board.querySelectorAll('.tile'))
      .forEach(function (t) { t.remove(); });
    var pct = function (v, total) { return (v / total * 100) + '%'; };
    ch.tiles.forEach(function (t) {
      var d = document.createElement('div');
      d.className = 'tile';
      d.style.left = pct(t.x, ch.boardW);
      d.style.top = pct(t.y, ch.boardH);
      d.style.width = pct(ch.tile, ch.boardW);
      d.style.height = pct(ch.tile, ch.boardH);
      var img = document.createElement('img');
      img.src = '/challenge/tile?id=' + encodeURIComponent(ch.id) + '&k=' + t.k;
      img.alt = '';
      d.appendChild(img);
      d.addEventListener('click', function () { if (live) clickTile(t.k); });
      board.appendChild(d);
    });
  }

  function refreshPrompt() {
    // cache-bust per hop -- the prompt is a fresh render each time
    el.prompt.src = '/challenge/prompt?id=' + encodeURIComponent(ch.id) + '&n=' + hop;
  }

  function buildPips() {
    el.pips.innerHTML = '';
    for (var i = 0; i < ch.hops; i++) {
      var p = document.createElement('span');
      p.className = 'pip' + (i < hop ? ' done' : i === hop ? ' now' : '');
      el.pips.appendChild(p);
    }
  }

  function tileEl(k) { return board.querySelectorAll('.tile')[k]; }

  function flash(k, cls) {
    var t = tileEl(k);
    if (!t) return;
    t.classList.add(cls);
    setTimeout(function () { t.classList.remove(cls); }, 420);
  }

  function clickTile(k) {
    if (!live) return;
    return fetch('/challenge/arrive', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: ch.id, k: k })
    }).then(function (r) { return r.json(); }).then(function (res) {
      if (res.state === 'next') {
        flash(k, 'hit');
        hop = res.hop; buildPips(); refreshPrompt();
        el.hint.textContent = 'hop ' + (hop + 1) + ' of ' + ch.hops + ' — next glyph released';
        return 'next';
      }
      if (res.state === 'complete') {
        flash(k, 'hit');
        hop = ch.hops; buildPips();
        live = false; board.classList.add('locked');
        el.hint.textContent = 'route complete — verifying';
        return submit().then(function () { return 'complete'; });
      }
      // failed
      flash(k, 'miss');
      live = false; board.classList.add('locked');
      setVeil(true, 'Wrong tile', res.reason || 'that is not the glyph you were shown');
      showVerdict({ pass: false, flags: [res.reason || 'wrong tile'], notes: [],
        hopWallMs: [], hopWallMad: null, traceCoverage: null,
        tracePoints: trace.length, corrections: -1, totalWallMs: 0 });
      return 'failed';
    });
  }

  function submit(overrideTrace) {
    return fetch('/challenge/verify', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: ch.id, trace: overrideTrace || trace })
    }).then(function (r) { return r.json(); }).then(function (v) {
      showVerdict(v);
      setVeil(true, v.pass ? 'Passed' : 'Rejected',
        v.pass ? 'no rule tripped — see the panel for the timings that decided it'
               : (v.flags[0] || 'a rule tripped — see the panel'));
      el.hint.textContent = 'done — start a new challenge to try again';
    }).catch(function () { el.hint.textContent = 'verify failed — server offline?'; });
  }

  function setVeil(on, title, sub) {
    el.veil.classList.toggle('on', !!on);
    if (on) el.veil.innerHTML = '<div>' + esc(title) + '<small>' + esc(sub || '') + '</small></div>';
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  // ---------- verdict panel ----------
  function resetPanel() {
    el.badge.className = 'badge idle'; el.badge.textContent = '—';
    el.who.textContent = 'no attempt yet';
    el.hops.innerHTML = ''; el.flags.innerHTML = '';
    ['sTotal', 'sCv', 'sMis', 'sPts', 'sCorr'].forEach(function (k) { el[k].textContent = '·'; });
  }

  function showVerdict(v) {
    el.badge.className = 'badge ' + (v.pass ? 'pass' : 'fail');
    el.badge.textContent = v.pass ? 'PASS' : 'REJECT';
    el.who.textContent = actor + ' · uncalibrated rules';

    el.hops.innerHTML = '';
    var max = Math.max.apply(null, (v.hopWallMs || []).concat([1200]));
    (v.hopWallMs || []).forEach(function (ms, i) {
      var row = document.createElement('div');
      row.className = 'hoprow' + (ms < 250 ? ' fast' : '');
      row.innerHTML = '<span>hop ' + (i + 1) + '</span>' +
        '<span class="track"><i style="width:' + Math.min(100, ms / max * 100) + '%"></i></span>' +
        '<b>' + ms + ' ms</b>';
      el.hops.appendChild(row);
    });

    el.sTotal.textContent = v.totalWallMs ? v.totalWallMs + ' ms' : '·';
    el.sCv.textContent = v.hopWallMad == null ? '·' : Number(v.hopWallMad).toFixed(4);
    el.sMis.textContent = v.traceCoverage == null ? '·'
      : (Number(v.traceCoverage) * 100).toFixed(0) + '% of window';
    el.sPts.textContent = v.tracePoints;
    el.sCorr.textContent = v.corrections < 0 ? '·' : v.corrections;

    el.flags.innerHTML = '';
    (v.flags || []).forEach(function (f) { addFlag(f, false); });
    (v.notes || []).forEach(function (n) { addFlag(n, true); });
    if (!(v.flags || []).length && !(v.notes || []).length) addFlag('no rule tripped', true);
  }
  function addFlag(text, isNote) {
    var li = document.createElement('li');
    if (isNote) li.className = 'note';
    li.textContent = text;
    el.flags.appendChild(li);
  }

  // ---------- human capture ----------
  board.addEventListener('mousemove', function (e) {
    if (!live || actor !== 'you') return;
    var now = performance.now();
    if (!t0) t0 = now;
    var p = toBoard(e.clientX, e.clientY);
    trace.push([now - t0, p[0], p[1]]);
    if (trace.length > 6000) trace.shift();
    redraw();
  });

  document.getElementById('btnNew').onclick = function () { newChallenge(); };

  // ---------- attacks ----------
  // Every sim is handed the answer by /challenge/oracle, i.e. we grant a perfect
  // OCR attacker for free and test only the timing channel. That route exists for
  // this demo alone and must never ship.

  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  function oracle() {
    return fetch('/challenge/oracle?id=' + encodeURIComponent(ch.id))
      .then(function (r) { return r.json(); });
  }

  function centre(k) {
    var t = ch.tiles[k];
    return [t.x + ch.tile / 2, t.y + ch.tile / 2];
  }

  /**
   * Walks a synthetic hop path in REAL time, stamping each sample off the same
   * wall clock a person's mousemove events would use.
   *
   * Moving in real time is not cosmetic. The server compares the trace's span
   * against the window it timed, and a trace can never legitimately span more
   * time than the challenge was open. A sim that advanced a fake clock while
   * consuming no real time would fail that test for a reason unrelated to the
   * attack being modelled -- so the honest sims spend the time, and only the
   * forgery attack rewrites the timestamps afterwards.
   */
  function movePath(from, to, kind) {
    var n = kind === 'teleport' ? 6 : 30;
    var stepMs = kind === 'teleport' ? 3 : 12;
    var i = 0;
    return new Promise(function (done) {
      (function tick() {
        i++;
        var u = i / n;
        var e = kind === 'bezier' ? u * u * u * (10 - 15 * u + 6 * u * u) : u;  // quintic min-jerk
        var x = from[0] + (to[0] - from[0]) * e;
        var y = from[1] + (to[1] - from[1]) * e;
        if (kind === 'bezier') y += Math.sin(u * Math.PI) * 34;                 // an arc, not a ruler
        var now = performance.now();
        if (!t0) t0 = now;
        trace.push([now - t0, x, y]);
        redraw();
        if (i < n) setTimeout(tick, stepMs);
        else done([x, y]);
      })();
    });
  }

  /**
   * Drives one full attack run.
   * @param delayFn returns the milliseconds to wait after a prompt is revealed
   * @param kind    path style for the synthetic movement
   * @param forge   if set, the trace submitted at the end is rewritten to this
   *                fraction of its real duration (the cheap forgery)
   */
  function runAttack(name, delayFn, kind, forge) {
    return newChallenge().then(function () {
      actor = name;
      live = false;                       // sim drives the hops itself
      trace = []; t0 = 0;
      var cursor = [40, ch.boardH - 40];
      var step = function () {
        return oracle().then(function (o) {
          return sleep(delayFn()).then(function () {
            return movePath(cursor, centre(o.k), kind);
          }).then(function (at) {
            cursor = at;
            return fetch('/challenge/arrive', {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ id: ch.id, k: o.k })
            }).then(function (r) { return r.json(); }).then(function (res) {
              flash(o.k, 'hit');
              if (res.state === 'next') {
                hop = res.hop; buildPips(); refreshPrompt();
                el.hint.textContent = name + ' — hop ' + (hop + 1) + ' of ' + ch.hops;
                return step();
              }
              hop = ch.hops; buildPips();
              board.classList.add('locked');
              var sent = trace;
              if (forge) {
                // Rewrite every timestamp so the trace CLAIMS a longer span than
                // the challenge was open. The cheapest attack on any
                // client-trusted clock -- and the reason the hard signals are
                // measured on the server's.
                sent = trace.map(function (p) { return [p[0] * forge, p[1], p[2]]; });
              }
              return submit(sent);
            });
          });
        });
      };
      el.hint.textContent = name + ' running…';
      return step();
    });
  }

  function jitter(mean, spread) {
    return function () { return mean + (Math.random() * 2 - 1) * spread; };
  }

  document.getElementById('atkInstant').onclick = function () {
    runAttack('bot·instant', function () { return 3; }, 'teleport');
  };
  document.getElementById('atkFixed').onclick = function () {
    runAttack('bot·fixed-delay', function () { return 700; }, 'bezier');
  };
  document.getElementById('atkForge').onclick = function () {
    // human-plausible wall times, but the submitted trace claims it took 3x longer
    runAttack('bot·forged-trace', jitter(760, 240), 'bezier', 3.0);
  };
  document.getElementById('atkJitter').onclick = function () {
    runAttack('bot·jittered', jitter(700, 250), 'bezier');
  };

  newChallenge();
})();
