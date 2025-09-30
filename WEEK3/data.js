// Global storage
let movies = [];        // array of movie titles
let ratings = [];       // 2D ratings matrix [user][movie]
let numUsers = 0;
let numMovies = 0;

/**
 * Load both datasets asynchronously and then parse them
 * @param {Function} callback - called after both files are loaded and parsed
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
 * Parse u.item file content into movies[]
 * @param {string} text - raw file text
 */
function parseItemData(text) {
  const lines = text.trim().split("\n");
  movies = lines.map(line => {
    const parts = line.split("|");
    return parts[1]; // movie title
  });
  numMovies = movies.length;
}

/**
 * Parse u.data file content into ratings[] and set numUsers
 * @param {string} text - raw file text
 */
function parseRatingData(text) {
  const lines = text.trim().split("\n");
  let userIds = new Set();

  // First pass: get unique users
  lines.forEach(line => {
    const [userId] = line.split("\t").map(Number);
    userIds.add(userId);
  });

  numUsers = Math.max(...userIds); // user IDs are sequential in MovieLens

  // Initialize ratings matrix with zeros
  ratings = Array.from({ length: numUsers }, () => Array(numMovies).fill(0));

  // Fill matrix
  lines.forEach(line => {
    const [userId, movieId, rating] = line.split("\t").map(Number);
    ratings[userId - 1][movieId - 1] = rating;
  });
}
