/* script.js
   - Wires UI to data.js
   - Handles loading, user selection, generating recs, and metrics display
*/

let MODEL = null;

const $ = sel => document.querySelector(sel);
const $$ = sel => document.querySelectorAll(sel);

function showOverlay(show, text){
  const ov = $('#overlay');
  const txt = $('#overlayText');
  if (text) txt.textContent = text;
  ov.classList.toggle('hidden', !show);
}

function setBar(id, value) {
  const el = $(id);
  const pct = Math.round((value || 0) * 100);
  el.style.setProperty('--w', `${Math.max(0, Math.min(100, pct))}%`);
}
function setText(id, value) {
  $(id).textContent = (value !== undefined && value !== null)
    ? (value * 100).toFixed(1) + '%'
    : '–';
}

function renderUserInfo(user){
  if (!user) { $('#userInfo').innerHTML = ''; return; }
  const ageB = (age)=>{
    if (age < 18) return "<18";
    if (age <= 24) return "18–24";
    if (age <= 34) return "25–34";
    if (age <= 44) return "35–44";
    if (age <= 49) return "45–49";
    if (age <= 55) return "50–55";
    return "56+";
  };
  $('#userInfo').innerHTML = `
    <div><strong>User:</strong> ${user.id}</div>
    <div><strong>Age:</strong> ${user.age} (${ageB(user.age)})</div>
    <div><strong>Gender:</strong> ${user.gender}</div>
    <div><strong>Occupation:</strong> ${user.occupation}</div>
  `;
}

function renderTable(tbody, rows, movies) {
  tbody.innerHTML = '';
  rows.forEach((r, idx) => {
    const m = movies.get(r.movieId);
    const tr = document.createElement('tr');
    const genres = m?.genres?.join(', ') || '—';
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td>${m ? m.title : r.movieId}</td>
      <td>${(r.score).toFixed(3)}</td>
      <td>${genres}</td>
    `;
    tbody.appendChild(tr);
  });
}

function highlightBetterBars() {
  // Add "best" class to the higher score in each pair (already styled to green)
  const pairs = [
    ['precBasic','precDeep'],
    ['recBasic','recDeep'],
    ['accBasic','accDeep'],
  ];
  pairs.forEach(([a, b]) => {
    const va = parseFloat($('#' + a + 'Val').textContent) || 0;
    const vb = parseFloat($('#' + b + 'Val').textContent) || 0;
    const ea = $('#' + a);
    const eb = $('#' + b);
    ea.classList.toggle('best', va >= vb);
    eb.classList.toggle('best', vb >= va);
  });
}

async function loadData(basePath) {
  showOverlay(true, 'Loading data files & training models…');
  try {
    MODEL = await DataLoader.loadAll(basePath || '');
    // Populate users
    const userSelect = $('#userSelect');
    userSelect.innerHTML = '';
    // For UX speed, present users sorted by id
    const ids = Array.from(MODEL.users.keys()).sort((a,b)=> a-b);
    for (const uid of ids) {
      const u = MODEL.users.get(uid);
      const opt = document.createElement('option');
      opt.value = String(uid);
      opt.textContent = `${uid} — ${u.gender}, ${u.age}, ${u.occupation}`;
      userSelect.appendChild(opt);
    }
    userSelect.disabled = false;
    $('#runBtn').disabled = false;

    // Initial user info
    const first = MODEL.users.get(ids[0]);
    renderUserInfo(first);

    // Render metrics
    const mb = MODEL.metrics.baseline;
    const md = MODEL.metrics.deep;

    setBar('#precBasic', mb.precision);
    setText('#precBasicVal', mb.precision);

    setBar('#precDeep', md.precision);
    setText('#precDeepVal', md.precision);

    setBar('#recBasic', mb.recall);
    setText('#recBasicVal', mb.recall);

    setBar('#recDeep', md.recall);
    setText('#recDeepVal', md.recall);

    setBar('#accBasic', mb.accuracy);
    setText('#accBasicVal', mb.accuracy);

    setBar('#accDeep', md.accuracy);
    setText('#accDeepVal', md.accuracy);

    highlightBetterBars();
  } catch (e) {
    console.error(e);
    alert(`Error: ${e.message}. Make sure the files "u.data", "u.item" and "u.user" are served by a local web server and the path is correct.`);
  } finally {
    showOverlay(false);
  }
}

function runForSelectedUser() {
  const sel = $('#userSelect');
  if (!MODEL || !sel.value) return;
  const uid = parseInt(sel.value, 10);
  const user = MODEL.users.get(uid);
  renderUserInfo(user);

  showOverlay(true, 'Scoring movies for this user…');

  // Small timeout to let overlay paint
  setTimeout(() => {
    try {
      const topBasic = MODEL.recommend.baselineTopK(uid, 10);
      const topDeep = MODEL.recommend.deepTopK(uid, 10);

      renderTable($('#basicTable tbody'), topBasic, MODEL.movies);
      renderTable($('#deepTable tbody'), topDeep, MODEL.movies);
    } finally {
      showOverlay(false);
    }
  }, 30);
}

document.addEventListener('DOMContentLoaded', () => {
  // Wire controls
  $('#reloadBtn').addEventListener('click', () => {
    const path = $('#basePath').value.trim();
    loadData(path);
  });
  $('#userSelect').addEventListener('change', () => {
    const uid = parseInt($('#userSelect').value, 10);
    const u = MODEL?.users?.get(uid);
    renderUserInfo(u);
  });
  $('#runBtn').addEventListener('click', runForSelectedUser);

  // Auto-load once on startup
  loadData('');
});
