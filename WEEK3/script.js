// Parameters for Matrix Factorization
const K = 2; // latent factors
const steps = 500; // training steps
const alpha = 0.002; // learning rate
const beta = 0.02; // regularization factor

// Matrix Factorization using Gradient Descent
function matrixFactorization(R, K, steps, alpha, beta) {
  const m = R.length; // users
  const n = R[0].length; // movies

  // Initialize user and item latent feature matrices randomly
  let P = Array.from({ length: m }, () => Array.from({ length: K }, () => Math.random()));
  let Q = Array.from({ length: n }, () => Array.from({ length: K }, () => Math.random()));

  // Gradient descent
  for (let step = 0; step < steps; step++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (R[i][j] > 0) {
          let prediction = dotProduct(P[i], Q[j]);
          let eij = R[i][j] - prediction;

          for (let k = 0; k < K; k++) {
            P[i][k] += alpha * (2 * eij * Q[j][k] - beta * P[i][k]);
            Q[j][k] += alpha * (2 * eij * P[i][k] - beta * Q[j][k]);
          }
        }
      }
    }
  }

  return multiply(P, Q);
}

// Dot product helper
function dotProduct(a, b) {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

// Multiply P and Q^T to get full predicted matrix
function multiply(P, Q) {
  const m = P.length;
  const n = Q.length;
  const result = Array.from({ length: m }, () => Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      result[i][j] = dotProduct(P[i], Q[j]);
    }
  }
  return result;
}

// Train prediction matrix
const predictedRatings = matrixFactorization(ratings, K, steps, alpha, beta);

// Populate user select
const userSelect = document.getElementById("userSelect");
users.forEach((u, idx) => {
  let option = document.createElement("option");
  option.value = idx;
  option.textContent = u;
  userSelect.appendChild(option);
});

// Show recommendations
document.getElementById("recommendBtn").addEventListener("click", () => {
  const userIdx = parseInt(userSelect.value);
  const userRatings = ratings[userIdx];
  const userPredictions = predictedRatings[userIdx];

  // Recommend movies not rated yet
  const recs = movies
    .map((m, idx) => ({ movie: m, score: userPredictions[idx], rated: userRatings[idx] }))
    .filter(item => item.rated === 0) // only unseen
    .sort((a, b) => b.score - a.score)
    .slice(0, 5); // top 5

  const recList = document.getElementById("recommendations");
  recList.innerHTML = "";
  recs.forEach(r => {
    let li = document.createElement("li");
    li.textContent = `${r.movie} (Predicted rating: ${r.score.toFixed(2)})`;
    recList.appendChild(li);
  });
});
