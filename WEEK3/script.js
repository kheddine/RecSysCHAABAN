/* script.js
   - Defines, trains, and serves predictions from a Matrix Factorization model in TF.js
   - Relies on data.js for: loadData(), movies, ratings, numUsers, numMovies
*/

let model = null; // global trained model

const userSelect = () => document.getElementById("user-select");
const movieSelect = () => document.getElementById("movie-select");
const resultEl = () => document.getElementById("result");
const predictBtn = () => document.getElementById("predict-btn");

function setStatus(html, { ok = false, warn = false } = {}) {
  const el = resultEl();
  el.classList.remove("result-ok", "result-warn");
  if (ok) el.classList.add("result-ok");
  if (warn) el.classList.add("result-warn");
  el.innerHTML = html;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Populate dropdowns after data load
 */
function populateDropdowns() {
  // Users: IDs are 1..numUsers
  const uSel = userSelect();
  uSel.innerHTML = "";
  for (let uid = 1; uid <= window.__ml100k__.numUsers; uid++) {
    const opt = document.createElement("option");
    opt.value = String(uid);
    opt.textContent = `User ${uid}`;
    uSel.appendChild(opt);
  }
  uSel.disabled = false;

  // Movies: sort by title for easier browsing
  const mSel = movieSelect();
  mSel.innerHTML = "";
  const movieArr = Array.from(window.__ml100k__.movies.values())
    .sort((a, b) => a.title.localeCompare(b.title));
  for (const m of movieArr) {
    const opt = document.createElement("option");
    opt.value = String(m.id);
    opt.textContent = m.year ? `${m.title} (${m.year})` : m.title;
    mSel.appendChild(opt);
  }
  mSel.disabled = false;

  predictBtn().disabled = true; // will be enabled after training
}

/**
 * Matrix Factorization model with embeddings + user/movie bias
 * userInput: integer ID (0-based index)
 * movieInput: integer ID (0-based index)
 */
function createModel(nUsers, nMovies, latentDim = 32) {
  // Inputs are scalar indices [batch, 1]
  const userInput = tf.input({ shape: [1], dtype: "int32", name: "user" });
  const movieInput = tf.input({ shape: [1], dtype: "int32", name: "movie" });

  // Embeddings
  const userEmbedding = tf.layers.embedding({
    inputDim: nUsers,
    outputDim: latentDim,
    inputLength: 1,
    embeddingsInitializer: "glorotUniform",
    name: "userEmbedding"
  }).apply(userInput);

  const movieEmbedding = tf.layers.embedding({
    inputDim: nMovies,
    outputDim: latentDim,
    inputLength: 1,
    embeddingsInitializer: "glorotUniform",
    name: "movieEmbedding"
  }).apply(movieInput);

  // Bias terms (scalars)
  const userBias = tf.layers.embedding({
    inputDim: nUsers,
    outputDim: 1,
    inputLength: 1,
    embeddingsInitializer: "zeros",
    name: "userBias"
  }).apply(userInput);

  const movieBias = tf.layers.embedding({
    inputDim: nMovies,
    outputDim: 1,
    inputLength: 1,
    embeddingsInitializer: "zeros",
    name: "movieBias"
  }).apply(movieInput);

  // Flatten to vectors/scalars
  const uVec = tf.layers.flatten().apply(userEmbedding);
  const mVec = tf.layers.flatten().apply(movieEmbedding);
  const uB = tf.layers.flatten().apply(userBias);
  const mB = tf.layers.flatten().apply(movieBias);

  // Dot product for interaction
  const dot = tf.layers.dot({ axes: 1, name: "dotUserMovie" }).apply([uVec, mVec]);

  // Add biases
  const addBias = tf.layers.add().apply([dot, uB, mB]);

  // Optional: learn a global offset
  const pred = tf.layers.add().apply([addBias, tf.layers.dense({
    units: 1, useBias: true, activation: null, name: "globalOffset"
  }).apply(tf.layers.onesLike().apply(addBias))]);

  // Model
  const m = tf.model({
    inputs: [userInput, movieInput],
    outputs: pred,
    name: "mfRecommender"
  });

  return m;
}

/**
 * Train the model on loaded ratings
 */
async function trainModel() {
  const nUsers = window.__ml100k__.numUsers;
  const nMovies = window.__ml100k__.numMovies;

  model = createModel(nUsers, nMovies, 32);
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "meanSquaredError"
  });

  // Prepare tensors (convert 1-based IDs to 0-based indices for embeddings)
  const N = ratings.length;
  const userIdx = new Int32Array(N);
  const movieIdx = new Int32Array(N);
  const target = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    const r = ratings[i];
    userIdx[i] = r.userId - 1;
    movieIdx[i] = r.movieId - 1;
    target[i] = r.rating; // 1..5
  }

  const userTensor = tf.tensor2d(userIdx, [N, 1], "int32");
  const movieTensor = tf.tensor2d(movieIdx, [N, 1], "int32");
  const ratingTensor = tf.tensor2d(target, [N, 1], "float32");

  // Train
  const batchSize = 1024; // keeps browser happier
  const epochs = 7;

  setStatus(`
    <div class="dot-pulse" aria-hidden="true"></div>
    <div><strong>Training:</strong> Matrix Factorization on ${N.toLocaleString()} ratings (epochs: ${epochs}, batch: ${batchSize})…</div>
  `);

  await model.fit([userTensor, movieTensor], ratingTensor, {
    batchSize,
    epochs,
    shuffle: true,
    verbose: 0,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        setStatus(`
          <div class="dot-pulse" aria-hidden="true"></div>
          <div><strong>Training:</strong> epoch ${epoch + 1}/${epochs} — loss: ${logs.loss.toFixed(4)}</div>
        `);
        await tf.nextFrame(); // yield to UI
      }
    }
  });

  userTensor.dispose();
  movieTensor.dispose();
  ratingTensor.dispose();

  predictBtn().disabled = false;
  setStatus(
    `<div>✅ <strong>Model ready!</strong> Choose a user and a movie, then click <em>Predict Rating</em>.</div>`,
    { ok: true }
  );
}

/**
 * Predict a rating for the selected user & movie
 */
async function predictRating() {
  if (!model) return;

  const uid = parseInt(userSelect().value, 10);
  const mid = parseInt(movieSelect().value, 10);

  // Create 0-based tensors for the model
  const uT = tf.tensor2d([uid - 1], [1, 1], "int32");
  const mT = tf.tensor2d([mid - 1], [1, 1], "int32");

  const predTensor = model.predict([uT, mT]);
  const val = (await predTensor.data())[0];

  uT.dispose(); mT.dispose();
  if (Array.isArray(predTensor)) predTensor.forEach(t => t.dispose());
  else predTensor.dispose();

  const clipped = clamp(val, 1, 5);
  const movieObj = window.__ml100k__.movies.get(mid);
  const title = movieObj ? (movieObj.year ? `${movieObj.title} (${movieObj.year})` : movieObj.title) : `Movie ${mid}`;

  setStatus(
    `<div>📈 <strong>Predicted rating</strong> for <em>User ${uid}</em> → <em>${title}</em>: <strong>${clipped.toFixed(2)}</strong> (raw: ${val.toFixed(2)})</div>`,
    { ok: true }
  );
}

/**
 * Initialize on page load:
 *  - load data
 *  - populate dropdowns
 *  - train the model
 */
window.addEventListener("load", async () => {
  try {
    setStatus(`
      <div class="dot-pulse" aria-hidden="true"></div>
      <div><strong>Status:</strong> Loading MovieLens data…</div>
    `);

    await loadData();
    // After loadData, globals in __ml100k__ are updated
    window.__ml100k__.movies = movies;
    window.__ml100k__.ratings = ratings;

    populateDropdowns();

    setStatus(`
      <div class="dot-pulse" aria-hidden="true"></div>
      <div><strong>Status:</strong> Starting training…</div>
    `);

    await tf.nextFrame();
    await trainModel();
  } catch (err) {
    console.error(err);
    setStatus(
      `<div>❌ <strong>Error:</strong> ${err.message}. If this page is loaded from a local file, try using a local server to avoid CORS/file protocol issues.</div>`,
      { warn: true }
    );
  }
});
