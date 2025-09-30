let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

/**
 * Load data files and parse them
 */
async function loadData() {
  const [itemText, ratingText] = await Promise.all([
    fetch("u.item").then(r => r.text()),
    fetch("u.data").then(r => r.text())
  ]);

  parseItemData(itemText);
  parseRatingData(ratingText);
}

/**
 * Parse movie metadata (u.item)
 */
function parseItemData(text) {
  const lines = text.trim().split("\n");
  movies = lines.map(line => {
    const parts = line.split("|");
    return { id: parseInt(parts[0]), title: parts[1] };
  });
  numMovies = movies.length;
}

/**
 * Parse ratings (u.data)
 */
function parseRatingData(text) {
  const lines = text.trim().split("\n");
  let maxUserId = 0;

  // Get max user ID
  lines.forEach(line => {
    const [userId] = line.split("\t").map(Number);
    if (userId > maxUserId) maxUserId = userId;
  });
  numUsers = maxUserId;

  // Initialize matrix
  ratings = Array.from({ length: numUsers }, () => Array(numMovies).fill(0));

  // Fill with ratings
  lines.forEach(line => {
    const [userId, movieId, rating] = line.split("\t").map(Number);
    ratings[userId - 1][movieId - 1] = rating;
  });
}
