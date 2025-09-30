// Global dataset storage
let movies = [];        // movie titles
let ratings = [];       // ratings matrix [user][movie]
let numUsers = 0;
let numMovies = 0;

/**
 * Load MovieLens data files and parse them.
 * Expects u.item and u.data to be placed in the same folder.
 * @param {Function} callback - runs after data is fully loaded
 */
function loadData(callback) {
  Promise.all([
    fetch("u.item").then(res => res.text()),
    fetch("u.data").then(res => res.text())
  ])
  .then(([itemText, ratingText]) => {
    parseItemData(itemText);
    parseRatingData(ratingText);
    if (callback) callback();
  })
  .catch(err => console.error("Error loading data:", err));
}

/**
 * Parse u.item into movies[]
 */
function parseItemData(text) {
  const lines = text.trim().split("\n");
  movies = lines.map(line => line.split("|")[1]); // title is second field
  numMovies = movies.length;
}

/**
 * Parse u.data into ratings[][] and numUsers
 */
function parseRatingData(text) {
  const lines = text.trim().split("\n");
  let maxUser = 0;

  // Find maximum user ID
  lines.forEach(line => {
    const [userId] = line.split("\t").map(Number);
    if (userId > maxUser) maxUser = userId;
  });
  numUsers = maxUser;

  // Initialize matrix with zeros
  ratings = Array.from({ length: numUsers }, () => Array(numMovies).fill(0));

  // Fill with ratings
  lines.forEach(line => {
    const [userId, movieId, rating] = line.split("\t").map(Number);
    ratings[userId - 1][movieId - 1] = rating;
  });
}
