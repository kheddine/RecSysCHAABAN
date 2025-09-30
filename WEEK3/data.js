let model;

/**
 * Initialize on window load
 */
window.onload = async () => {
  const resultEl = document.getElementById("result");
  resultEl.textContent = "Loading dataset...";

  await loadData();

  populateDropdowns();

  resultEl.textContent = "Training model...";
  await trainModel();

  resultEl.textContent = "Model trained! Select user and movie to predict rating.";
};

/**
 * Populate dropdowns with users and movies
 */
function populateDropdowns() {
  const userSelect = document.getElementById("user-select");
  const movieSelect = document.getElementById("movie-select");

  for (let u = 1; u <= numUsers; u++) {
    let opt = document.createElement("option");
    opt.value = u;
    opt.textContent = `User ${u}`;
    userSelect.appendChild(opt);
  }

  movies.forEach(m => {
    let opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.title;
    movieSelect.appendChild(opt);
  });
}

/**
 * Define MF model with embeddings
 */
function createModel(numUsers, numMovies, latentDim = 20) {
  const userInput = tf.input({ shape: [1], dtype: "int32", name: "user" });
  const movieInput = tf.input({ shape: [1], dtype: "int32", name: "movie" });

  const userEmbedding = tf.layers.embedding({
    inputDim: numUsers + 1,
    outputDim: latentDim,
    embeddingsInitializer: "heNormal"
  }).apply(userInput);

  const movieEmbedding = tf.layers.embedding({
    inputDim: numMovies + 1,
    outputDim: latentDim,
    embeddingsInitializer: "heNormal"
  }).apply(movieInput);

  const userVec = tf.layers.flatten().apply(userEmbedding);
  const movieVec = tf.layers.flatten().apply(movieEmbedding);

  const dot = tf.layers.dot({ axes: 1 }).apply([userVec, movieVec]);

  const output = tf.layers.dense({ units: 1, activation: "linear" }).apply(dot);

  const model = tf.model({ inputs: [userInput, movieInput], outputs: output });
  return model;
}

/**
 * Train MF model
 */
async function trainModel() {
  model = createModel(numUsers, numMovies, 20);
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "meanSquaredError"
  });

  const userIds = [];
  const movieIds = [];
  const labels = [];

  for (let u = 0; u < numUsers; u++) {
    for (let m = 0; m < numMovies; m++) {
      if (ratings[u][m] > 0) {
        userIds.push(u + 1);
        movieIds.push(m + 1);
        labels.push(ratings[u][m]);
      }
    }
  }

  const userTensor = tf.tensor2d(userIds, [userIds.length, 1], "int32");
  const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1], "int32");
  const ratingTensor = tf.tensor2d(labels, [labels.length, 1], "float32");

  await model.fit([userTensor, movieTensor], ratingTensor, {
    epochs: 5,
    batchSize: 64,
    verbose: 1
  });

  userTensor.dispose();
  movieTensor.dispose();
  ratingTensor.dispose();
}

/**
 * Predict rating for selected user & movie
 */
async function predictRating() {
  const userId = parseInt(document.getElementById("user-select").value);
  const movieId = parseInt(document.getElementById("movie-select").value);
  const resultEl = document.getElementById("result");

  if (!model) {
    resultEl.textContent = "Model not ready yet.";
    return;
  }

  const userTensor = tf.tensor2d([userId], [1, 1], "int32");
  const movieTensor = tf.tensor2d([movieId], [1, 1], "int32");

  const prediction = model.predict([userTensor, movieTensor]);
  const value = (await prediction.data())[0];

  resultEl.textContent = `Predicted rating for User ${userId} on "${movies[movieId - 1].title}" is ${value.toFixed(2)} ⭐`;

  userTensor.dispose();
  movieTensor.dispose();
  prediction.dispose();
}
