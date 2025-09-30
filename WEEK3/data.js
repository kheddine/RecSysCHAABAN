/* data.js
   Loads MovieLens files (u.item, u.data) from the same folder as index.html.
   Works fine on GitHub Pages because files are same-origin.
*/

const MOVIELENS_BASE = ".";  // current directory

let movies = new Map();     // movieId -> { id, title, year }
let ratings = [];           // { userId, movieId, rating }
let numUsers = 0;
let numMovies = 0;

/**
 * Parse u.item (pipe '|' separated)
 */
function parseItemData(text) {
  const result = new Map();
  const lines = text.split(/\r?\n/).filter(Boolean);

  for (const line of lines) {
    const parts = line.split("|");
    const id = parseInt(parts[0], 10);
    const rawTitle = parts[1] || `Movie ${id}`;
    let year = null;
    const m = rawTitle.match(/\((\d{4})\)$/);
    if (m) year = m[1];
    const title = rawTitle.replace(/\s*\(\d{4}\)\s*$/, "").trim();
    result.set(id, { id, title: title || `Movie ${id}`, year });
  }
  return result;
}

/**
 * Parse u.data (tab separated)
 */
function parseRatingData(text) {
  const result = [];
  const lines = text.split(/\r?\n/).filter(Boolean);

  let maxUser = 0;
  let maxItem = 0;

  for (const line of lines) {
    const [u, i, r] = line.split(/\s+/);
    const userId = parseInt(u, 10);
    const movieId = parseInt(i, 10);
    const rating = parseFloat(r);
    if (Number.isFinite(userId) && Number.isFinite(movieId) && Number.isFinite(rating)) {
      result.push({ userId, movieId, rating });
      if (userId > maxUser) maxUser = userId;
      if (movieId > maxItem) maxItem = movieId;
    }
  }

  numUsers = maxUser;
  numMovies = maxItem;

  return result;
}

/**
 * Load both files from the repo root
 */
async function loadData() {
  const [itemRes, dataRes] = await Promise.all([
    fetch(`${MOVIELENS_BASE}/u.item`),
    fetch(`${MOVIELENS_BASE}/u.data`)
  ]);

  if (!itemRes.ok || !dataRes.ok) {
    throw new Error(`Failed to fetch u.item or u.data (check they exist in repo root).`);
  }

  const [itemText, dataText] = await Promise.all([
    itemRes.text(),
    dataRes.text()
  ]);

  movies = parseItemData(itemText);
  ratings = parseRatingData(dataText);

  if (movies.size > numMovies) numMovies = movies.size;
}

// For console debugging
window.__ml100k__ = {
  get movies() { return movies; },
  get ratings() { return ratings; },
  get numUsers() { return numUsers; },
  get numMovies() { return numMovies; }
};

// Export functions globally
window.loadData = loadData;
window.parseItemData = parseItemData;
window.parseRatingData = parseRatingData;
