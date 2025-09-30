/* data.js — GitHub Pages friendly (expects u.item & u.data in the repo root)
   Exposes: loadData(), parseItemData(), parseRatingData(), and live counts.
*/

const MOVIELENS_BASE = "."; // same folder as index.html

let movies = new Map();     // movieId -> { id, title, year }
let ratings = [];           // { userId, movieId, rating }
let numUsers = 0;
let numMovies = 0;

/* Parse u.item (pipe-separated) */
function parseItemData(text) {
  const map = new Map();
  const lines = text.split(/\r?\n/).filter(Boolean);
  for (const line of lines) {
    const parts = line.split("|");
    const id = parseInt(parts[0], 10);
    const rawTitle = parts[1] || `Movie ${id}`;
    let year = null;
    const m = rawTitle.match(/\((\d{4})\)$/);
    if (m) year = m[1];
    const title = rawTitle.replace(/\s*\(\d{4}\)\s*$/, "").trim();
    map.set(id, { id, title: title || `Movie ${id}`, year });
  }
  return map;
}

/* Parse u.data (tab-separated): userId \t itemId \t rating \t timestamp */
function parseRatingData(text) {
  const arr = [];
  const lines = text.split(/\r?\n/).filter(Boolean);
  let maxU = 0, maxI = 0;

  for (const line of lines) {
    const [u, i, r] = line.split(/\s+/);
    const userId = parseInt(u, 10);
    const movieId = parseInt(i, 10);
    const rating = parseFloat(r);
    if (Number.isFinite(userId) && Number.isFinite(movieId) && Number.isFinite(rating)) {
      arr.push({ userId, movieId, rating });
      if (userId > maxU) maxU = userId;
      if (movieId > maxI) maxI = movieId;
    }
  }
  numUsers = maxU;
  numMovies = maxI;
  return arr;
}

/* Fetch and populate globals (same-origin on GH Pages) */
async function loadData() {
  const [itemRes, dataRes] = await Promise.all([
    fetch(`${MOVIELENS_BASE}/u.item`),
    fetch(`${MOVIELENS_BASE}/u.data`)
  ]);

  if (!itemRes.ok || !dataRes.ok) {
    throw new Error("Failed to fetch u.item/u.data. Make sure both files are in the repo root.");
  }

  const [itemText, dataText] = await Promise.all([itemRes.text(), dataRes.text()]);
  movies = parseItemData(itemText);
  ratings = parseRatingData(dataText);
  if (movies.size > numMovies) numMovies = movies.size;
}

/* Small debug helper & global exposure */
window.__ml100k__ = {
  get movies(){ return movies; },
  get ratings(){ return ratings; },
  get numUsers(){ return numUsers; },
  get numMovies(){ return numMovies; }
};

window.loadData = loadData;
window.parseItemData = parseItemData;
window.parseRatingData = parseRatingData;
