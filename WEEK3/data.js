/* data.js
   - Loads MovieLens 100K files: u.item, u.data
   - Exposes: loadData(), parseItemData(), parseRatingData()
   - Globals: movies (Map by movieId), ratings (Array), numUsers, numMovies
*/

const MOVIELENS_BASE =
  "https://files.grouplens.org/datasets/movielens/ml-100k";

let movies = new Map();     // movieId (1-based) -> { id, title, year }
let ratings = [];           // { userId (1-based), movieId (1-based), rating }
let numUsers = 0;
let numMovies = 0;

/**
 * Parse u.item (pipe '|' separated)
 * line format (we only need first 2-3 fields):
 *  movie id | movie title | release date | ...
 */
function parseItemData(text) {
  const result = new Map();
  const lines = text.split(/\r?\n/).filter(Boolean);

  for (const line of lines) {
    const parts = line.split("|");
    const id = parseInt(parts[0], 10);
    const rawTitle = parts[1] || `Movie ${id}`;
    // Try to extract year from title "(1995)" or from release date column
    let year = null;
    const m = rawTitle.match(/\((\d{4})\)$/);
    if (m) year = m[1];
    const title = rawTitle.replace(/\s*\(\d{4}\)\s*$/, "").trim();
    result.set(id, { id, title: title || `Movie ${id}`, year });
  }
  return result;
}

/**
 * Parse u.data (tab separated): user id \t item id \t rating \t timestamp
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

  // Update globals for counts (MovieLens IDs are dense and 1-based)
  numUsers = maxUser;
  numMovies = maxItem;

  return result;
}

/**
 * Fetch and load both files, then populate globals.
 */
async function loadData() {
  const [itemRes, dataRes] = await Promise.all([
    fetch(`${MOVIELENS_BASE}/u.item`, { mode: "cors" }),
    fetch(`${MOVIELENS_BASE}/u.data`, { mode: "cors" })
  ]);

  if (!itemRes.ok || !dataRes.ok) {
    throw new Error(`Failed to fetch MovieLens data (HTTP ${itemRes.status}/${dataRes.status}).`);
  }

  const [itemText, dataText] = await Promise.all([itemRes.text(), dataRes.text()]);

  movies = parseItemData(itemText);
  ratings = parseRatingData(dataText);

  // Safety: if items file had more movies than seen in ratings, respect that as numMovies
  if (movies.size > numMovies) numMovies = movies.size;
}

// For dev console visibility
window.__ml100k__ = { movies, ratings, get numUsers(){return numUsers;}, get numMovies(){return numMovies;} };

/* Export names for clarity in other scripts (browser global scope) */
window.loadData = loadData;
window.parseItemData = parseItemData;
window.parseRatingData = parseRatingData;
window.movies = movies;
window.ratings = ratings;
window.numUsers = numUsers;
window.numMovies = numMovies;
