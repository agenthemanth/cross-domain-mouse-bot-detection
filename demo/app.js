/* app.js -- shared page chrome: the server-status pill in the top nav.
 * Pings GET /healthz now and every 15 s, so a stopped server is visible at a
 * glance instead of surfacing as a failed score later. Touches nothing else. */
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
