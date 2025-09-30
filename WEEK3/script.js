// MF Parameters
const K = 20;
const steps = 50;
const alpha = 0.002;
const beta = 0.02;

// Dot product
function dotProduct(a, b) {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

// Multiply P and Q^T
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

// Matrix Factorization (SGD)
function matrixFactorization(R, K, steps, alpha, beta) {
  const m = R.length;
  const n = R[0].length;
  let P = Array.from({ length: m }, () => Array.from({ length: K }, () => Math.random()));
  let Q = Array.from({ length: n }, () => Array.from({ length: K }, () => Math.random()));

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

// Main logic (wait until dataset is loaded)
loadData(() => {
  console.log("Loaded dataset:", numUsers, "users,", numMovies, "movies");

  // Train MF
  const predictedRatings = matrixFactorization(ratings, K, steps, alpha, beta);

  // Populate dropdown
  const userSelect = document.getElementById("userSelect");
  for (let u = 1; u <= numUsers; u++) {
    let opt = document.createElement("option");
    opt.value = u - 1;
    opt.textContent = "User " + u;
    userSelect.appendChild(opt);
  }

  // Handle button click
  document.getElementById("recommendBtn").addEventListener("click", () => {
    const userIdx = parseInt(userSelect.value);
    const userRatings = ratings[userIdx];
    const userPreds = predictedRatings[userIdx];

    // Recommend unseen movies
    const recs = movies
      .map((title, idx) => ({
        title,
        score: userPreds[idx],
        rated: userRatings[idx]
      }))
      .filter(m => m.rated === 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);

    // Render
    const recList = document.getElementById("recommendations");
    recList.innerHTML = "";
    recs.forEach(r => {
      let li = document.createElement("li");
      li.textContent = `${r.title} (Predicted: ${r.score.toFixed(2)})`;
      recList.appendChild(li);
    });
  });
});
