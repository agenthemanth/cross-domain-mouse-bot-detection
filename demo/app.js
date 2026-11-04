/* app.js -- shared by every page, loaded before the page script.
 * 1. window.App: small helpers the page scripts share.
 * 2. The server-status pill in the top nav: pings GET /healthz now and every
 *    15 s, so a stopped server is visible at a glance instead of surfacing as a
 *    failed score later. */
window.App = {
  /** POST a JSON body and parse the JSON reply. */
  postJson: function (url, body) {
    return fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body) }).then(function (r) { return r.json(); });
  },
  esc: function (s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  },
  sleep: function (ms) { return new Promise(function (r) { setTimeout(r, ms); }); },
  /** A custom property from the stylesheet, read now so it follows light/dark changes. */
  cssVar: function (name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  },
  /** Quintic min-jerk easing, u in [0,1]. */
  minJerk: function (u) { return u * u * u * (10 - 15 * u + 6 * u * u); }
};

(function () {
  'use strict';
  var pill = document.getElementById('serverStatus');
  if (!pill) return;

  function set(state, text) {
    pill.className = 'status ' + state;
    pill.textContent = text;
  }

  function check() {
    fetch('/healthz', { cache: 'no-store' })
      .then(function (r) { if (!r.ok) throw new Error(r.status); set('ok', 'Scorer online'); })
      .catch(function () { set('down', 'Scorer offline'); });
  }

  check();
  setInterval(function () { if (document.visibilityState === 'visible') check(); }, 15000);
  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState === 'visible') check();
  });
})();
