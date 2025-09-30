// script.js
// ------- Model definition, training, and prediction (TensorFlow.js) -------

let model; // global model variable as required

// Utility: status updates
function setStatus(msg){
  const el = document.getElementById('result');
  if (el) el.textContent = msg;
}

// Populate UI dropdowns after data is loaded
function populateUserSelect() {
  const select = document.getElementById('user-select');
  select.innerHTML = '';
  for (const uid0 of userIdList) {
    const opt = document.createElement('option');
    opt.value = String(uid0);
    opt.textContent = `User ${uid0 + 1}`;
    select.appendChild(opt);
  }
}

function populateMovieSelect() {
  const select = document.getElementById('movie-select');
  select.innerHTML = '';
  for (let i = 0; i < movies.length; i++) {
    const { id0, title } = movies[i];
    const opt = document.createElement('option');
    opt.value = String(id0);
    opt.textContent = title || `Movie ${id0 + 1}`;
    select.appendChild(opt);
  }
}

// Create matrix factorization model with embeddings & dot product
function createModel(numUsers, numMovies, latentDim = 16) {
  const userInput = tf.input({shape: [1], dtype: 'int32', name: 'userInput'});
  const movieInput = tf.input({shape: [1], dtype: 'int32', name: 'movieInput'});

  const userEmbeddingLayer = tf.layers.embedding({
    inputDim: numUsers,
    outputDim: latentDim,
    embeddingsInitializer: 'glorotUniform',
    name: 'userEmbedding'
  });
  const movieEmbeddingLayer = tf.layers.embedding({
    inputDim: numMovies,
    outputDim: latentDim,
    embeddingsInitializer: 'glorotUniform',
    name: 'movieEmbedding'
  });

  const userEmbedding = userEmbeddingLayer.apply(userInput);
  const movieEmbedding = movieEmbeddingLayer.apply(movieInput);

  const userVec = tf.layers.flatten().apply(userEmbedding);
  const movieVec = tf.layers.flatten().apply(movieEmbedding);

  // Dot product of latent factors
  const dot = tf.layers.dot({axes: -1, name: 'dot'}).apply([userVec, movieVec]);

  // Optional linear scaling layer
  const output = tf.layers.dense({units: 1, activation: 'linear', name: 'prediction'})
                          .apply(dot);

  const mfModel = tf.model({
    inputs: [userInput, movieInput],
    outputs: output,
    name: 'mf_recommender'
  });

  return mfModel;
}

// Train the model end-to-end
async function trainModel() {
  model = createModel(numUsers, numMovies, 32);

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError'
  });

  // Prepare tensors (2D int32 for inputs, 2D float32 for labels)
  const N = ratingsData.ratings.length;

  const userTensor = tf.tensor2d(ratingsData.userIds, [N, 1], 'int32');
  const movieTensor = tf.tensor2d(ratingsData.movieIds, [N, 1], 'int32');
  const ratingTensor = tf.tensor2d(ratingsData.ratings, [N, 1], 'float32');

  const BATCH = 64;
  const EPOCHS = 8;

  setStatus(`Training model… (epochs: ${EPOCHS}, batch: ${BATCH})`);

  await model.fit([userTensor, movieTensor], ratingTensor, {
    epochs: EPOCHS,
    batchSize: BATCH,
    shuffle: true,
    verbose: 0,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        setStatus(`Training… epoch ${epoch+1}/${EPOCHS} — loss: ${logs.loss.toFixed(4)}`);
        await tf.nextFrame(); // keep UI responsive
      }
    }
  });

  userTensor.dispose();
  movieTensor.dispose();
  ratingTensor.dispose();

  setStatus('Model trained ✅ Select a user and a movie, then click “Predict Rating”.');
}

// Predict rating for selected user & movie
async function predictRating() {
  if (!model) {
    setStatus('Model not ready yet. Please wait for training to finish.');
    return;
  }

  const userSel = document.getElementById('user-select');
  const movieSel = document.getElementById('movie-select');

  const userId0 = Number(userSel.value);
  const movieId0 = Number(movieSel.value);

  // Create 2D int32 tensors of shape [1,1]
  const uT = tf.tensor2d([[userId0]], [1,1], 'int32');
  const mT = tf.tensor2d([[movieId0]], [1,1], 'int32');

  const predTensor = model.predict([uT, mT]);
  const predArr = await predTensor.data();

  uT.dispose();
  mT.dispose();
  predTensor.dispose();

  // Nicely format and clamp to [1,5] for display
  const raw = Number(predArr[0]);
  const clipped = Math.min(5, Math.max(1, raw));
  const title = movies[movieId0]?.title ?? `Movie ${movieId0+1}`;

  setStatus(`Predicted rating for <strong>User ${userId0+1}</strong> on <strong>${title}</strong>: <span style="color:#22d3ee;font-weight:800">${clipped.toFixed(2)}</span>`);
}

// Boot sequence
window.onload = (async () => {
  try {
    setStatus('Loading MovieLens data…');
    await loadData();
    populateUserSelect();
    populateMovieSelect();
    await trainModel();
  } catch (err) {
    console.error(err);
    setStatus('Error: ' + (err?.message || 'Failed to initialize.'));
  }
});
