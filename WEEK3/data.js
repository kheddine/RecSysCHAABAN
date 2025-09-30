// data.js
// ------- Data loading & parsing for MovieLens 100K -------

// Globals required by the spec
let numUsers;
let numMovies;

// Additional shared globals
let movies = []; // [{id0: number, title: string}]
let ratingsData = {
  userIds: [],   // zero-based
  movieIds: [],  // zero-based
  ratings: []    // 1..5
};
let userIdList = []; // distinct zero-based ids present in ratings

/**
 * Fetch and parse MovieLens 100K u.item and u.data.
 * Populates: movies[], ratingsData, userIdList, numUsers, numMovies
 */
async function loadData() {
  const itemURL = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.item';
  const dataURL = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.data';

  const [itemResp, dataResp] = await Promise.all([
    fetch(itemURL),
    fetch(dataURL)
  ]);

  if (!itemResp.ok || !dataResp.ok) {
    throw new Error('Failed to fetch MovieLens files. Please check your network/CORS.');
  }

  const [itemText, dataText] = await Promise.all([
    itemResp.text(),
    dataResp.text()
  ]);

  parseItemData(itemText);
  parseRatingData(dataText);

  // Derive counts
  const maxUser = Math.max(...ratingsData.userIds) + 1;   // zero-based -> +1
  const maxMovie = Math.max(...ratingsData.movieIds) + 1; // zero-based -> +1

  numUsers = maxUser;
  numMovies = Math.max(maxMovie, movies.length); // safety if a movie id exists without title

  // Build distinct user id list for populating the dropdown
  userIdList = Array.from(new Set(ratingsData.userIds)).sort((a,b)=>a-b);
}

/**
 * Parse u.item (pipe-delimited). Keeps id and title.
 * @param {string} text
 */
function parseItemData(text) {
  movies = [];
  const lines = text.split('\n').filter(Boolean);
  for (const line of lines) {
    // Format: movie id | movie title | release date | video release date | IMDb URL | ...genre flags
    const parts = line.split('|');
    if (parts.length >= 2) {
      const rawId = Number(parts[0]);
      const id0 = rawId - 1; // zero-based for embeddings
      const title = parts[1].trim();
      if (!Number.isNaN(id0) && title) {
        movies[id0] = { id0, title }; // sparse-safe assignment
      }
    }
  }
  // Fill any missing titles as "Movie #<id>"
  for (let i=0; i<movies.length; i++){
    if (!movies[i]) movies[i] = { id0: i, title: `Movie #${i+1}` };
  }
}

/**
 * Parse u.data (tab/whitespace-delimited)
 * @param {string} text
 */
function parseRatingData(text) {
  ratingsData = { userIds: [], movieIds: [], ratings: [] };
  const lines = text.split('\n').filter(Boolean);
  for (const line of lines) {
    // Format: user id \t item id \t rating \t timestamp
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 3) {
      const u = Number(parts[0]) - 1; // zero-based
      const m = Number(parts[1]) - 1; // zero-based
      const r = Number(parts[2]);
      if ([u,m,r].every(x => Number.isFinite(x))) {
        ratingsData.userIds.push(u);
        ratingsData.movieIds.push(m);
        ratingsData.ratings.push(r);
      }
    }
  }
}
