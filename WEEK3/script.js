/* script.js — Matrix Factorization in TF.js, iOS-style UI updates */

let model = null;

const $ = (id) => document.getElementById(id);
const userSel  = () => $("user-select");
const movieSel = () => $("movie-select");
const resultEl = () => $("result");
const predictBtn = () => $("predict-btn");

function setStatus(html, { ok=false, warn=false } = {}) {
  const card = $("status");
  card.classList.remove("result-ok", "result-warn");
  if (ok) card.classList.add("result-ok");
  if (warn) card.classList.add("result-warn");
  resultEl().innerHTML = html;
}

function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

/* Populate dropdowns */
function populateDropdowns() {
  // Users
  userSel().innerHTML = "";
  for (let uid = 1; uid <= window.__ml100k__.numUsers; uid++) {
    const opt = document.createElement("option");
    opt.value = String(uid);
    opt.textContent = `User ${uid}`;
    userSel().appendChild(opt);
  }
  userSel().disabled = false;

  // Movies (sorted by title)
  const arr = Array.from(window.__ml100k__.movies.values())
    .sort((a,b) => a.title.localeCompare(b.title));
  movieSel().innerHTML = "";
  for (const m of arr) {
    const opt = document.createElement("option");
    opt.value = String(m.id);
    opt.textContent = m.year ? `${m.title} (${m.year})` : m.title;
    movieSel().appendChild(opt);
  }
  movieSel().disabled = false;

  predictBtn().disabled = true; // enabled after training
}

/* Create MF model: user/movie embeddings + biases, dot product */
function createModel(nUsers, nMovies, latentDim = 32) {
  const userInput = tf.input({ shape:[1], dtype:"int32", name:"user" });
  const movieInput = tf.input({ shape:[1], dtype:"int32", name:"movie" });

  const userEmb = tf.layers.embedding({
    inputDim: nUsers, outputDim: latentDim, inputLength:1,
    embeddingsInitializer: "glorotUniform", name: "userEmb"
  }).apply(userInput);
  const movieEmb = tf.layers.embedding({
    inputDim: nMovies, outputDim: latentDim, inputLength:1,
    embeddingsInitializer: "glorotUniform", name: "movieEmb"
  }).apply(movieInput);

  const userBias = tf.layers.embedding({
    inputDim: nUsers, outputDim: 1, inputLength:1, name:"userBias",
    embeddingsInitializer: "zeros"
  }).apply(userInput);
  const movieBias = tf.layers.embedding({
    inputDim: nMovies, outputDim: 1, inputLength:1, name:"movieBias",
    embeddingsInitializer: "zeros"
  }).apply(movieInput);

  const u = tf.layers.flatten().apply(userEmb);
  const m = tf.layers.flatten().apply(movieEmb);
  const ub = tf.layers.flatten().apply(userBias);
  const mb = tf.layers.flatten().apply(movieBias);

  const dot = tf.layers.dot({ axes:1 }).apply([u, m]);
  const pred = tf.layers.add().apply([dot, ub, mb]); // simple & fast

  return tf.model({ inputs:[userInput, movieInput], outputs: pred, name: "MF" });
}

/* Train on full ratings (1-based IDs -> 0-based indices) */
async function trainModel() {
  const nUsers = window.__ml100k__.numUsers;
  const nMovies = window.__ml100k__.numMovies;

  model = createModel(nUsers, nMovies, 24);
  model.compile({ optimizer: tf.train.adam(0.001), loss: "meanSquaredError" });

  const N = window.__ml100k__.ratings.length;
  const uIdx = new Int32Array(N);
  const iIdx = new Int32Array(N);
  const y    = new Float32Array(N);

  for (let k = 0; k < N; k++) {
    const r = window.__ml100k__.ratings[k];
    uIdx[k] = r.userId - 1;
    iIdx[k] = r.movieId - 1;
    y[k]    = r.rating;
  }

  const uT = tf.tensor2d(uIdx, [N,1], "int32");
  const iT = tf.tensor2d(iIdx, [N,1], "int32");
  const yT = tf.tensor2d(y,    [N,1], "float32");

  const epochs = 6, batch = 1024;

  setStatus(`
    <div class="spinner" aria-hidden="true"></div>
    <div><strong>Training</strong> on ${N.toLocaleString()} ratings…</div>
  `);

  await model.fit([uT, iT], yT, {
    epochs, batchSize: batch, shuffle: true, verbose: 0,
    callbacks: {
      onEpochEnd: async (e, logs) => {
        setStatus(`
          <div class="spinner" aria-hidden="true"></div>
          <div><strong>Epoch ${e+1}/${epochs}</strong> · loss ${logs.loss.toFixed(4)}</div>
        `);
        await tf.nextFrame();
      }
    }
  });

  uT.dispose(); iT.dispose(); yT.dispose();

  predictBtn().disabled = false;
  setStatus(`✅ <strong>Model ready.</strong> Choose a user and a movie, then tap <em>Predict Rating</em>.`, { ok:true });
}

/* Predict for chosen user+movie */
async function predictRating() {
  if (!model) return;
  const uid = parseInt(userSel().value, 10);
  const mid = parseInt(movieSel().value, 10);

  const uT = tf.tensor2d([uid-1], [1,1], "int32");
  const mT = tf.tensor2d([mid-1], [1,1], "int32");
  const out = model.predict([uT, mT]);
  const val = (await out.data())[0];
  uT.dispose(); mT.dispose(); out.dispose();

  const clipped = clamp(val, 1, 5);
  const mov = window.__ml100k__.movies.get(mid);
  const title = mov ? (mov.year ? `${mov.title} (${mov.year})` : mov.title) : `Movie ${mid}`;

  setStatus(`📈 Predicted for <strong>User ${uid}</strong> → <em>${title}</em>: <strong>${clipped.toFixed(2)}</strong> (raw ${val.toFixed(2)})`, { ok:true });
}

/* Init */
window.addEventListener("load", async () => {
  try {
    setStatus(`<div class="spinner"></div><div><strong>Status:</strong> Loading data…</div>`);
    await loadData();
    populateDropdowns();
    setStatus(`<div class="spinner"></div><div><strong>Status:</strong> Starting training…</div>`);
    await tf.nextFrame();
    await trainModel();
  } catch (e) {
    console.error(e);
    setStatus(`❌ <strong>Error:</strong> ${e.message}`, { warn:true });
  }
});
