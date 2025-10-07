/* data.js
   - Loads MovieLens 100k files (u.data, u.item, u.user)
   - Implements:
     (A) Baseline predictor: global + user bias + item bias
     (B) Two-tower "deep" model using user demographics & movie genres with tiny on-device training
   - Computes metrics on a small hold-out
*/

const DataLoader = (() => {
  // Parse helpers
  function parseCSVLine(line, sep="|") {
    // Simple split; MovieLens 100k uses '|' in u.item and '|' in u.user
    return line.split(sep);
  }

  function parseDataContent(content) {
    // u.data: user_id\tmovie_id\trating\ttimestamp
    const rows = content.trim().split(/\r?\n/);
    const ratings = [];
    for (const r of rows) {
      if (!r) continue;
      const [u, m, y, t] = r.split(/\t/);
      ratings.push({
        userId: parseInt(u, 10),
        movieId: parseInt(m, 10),
        rating: parseInt(y, 10),
        ts: parseInt(t, 10)
      });
    }
    return ratings;
  }

  // Movie genres per ML-100k (19 flags)
  const GENRES = [
    "Unknown","Action","Adventure","Animation","Children's","Comedy","Crime","Documentary",
    "Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
  ];

  function parseItemContent(content) {
    // u.item: movie id | title | release date | video release | IMDb URL | 19 genre flags
    const rows = content.trim().split(/\r?\n/);
    const movies = new Map();
    for (const r of rows) {
      if (!r) continue;
      const cols = parseCSVLine(r, "|");
      const id = parseInt(cols[0], 10);
      const title = cols[1] || `Movie ${id}`;
      const genreFlags = cols.slice(5, 5 + GENRES.length).map(x => parseInt(x, 10));
      const genres = [];
      for (let i = 0; i < genreFlags.length; i++) {
        if (genreFlags[i] === 1) genres.push(GENRES[i]);
      }
      movies.set(id, { id, title, genres, flags: genreFlags });
    }
    return { movies, GENRES };
  }

  function parseUserContent(content) {
    // u.user: user id | age | gender | occupation | zip code
    const rows = content.trim().split(/\r?\n/);
    const users = new Map();
    const occupations = new Set();
    for (const r of rows) {
      if (!r) continue;
      const cols = parseCSVLine(r, "|");
      const id = parseInt(cols[0], 10);
      const age = parseInt(cols[1], 10);
      const gender = cols[2];
      const occ = cols[3];
      occupations.add(occ);
      users.set(id, { id, age, gender, occupation: occ });
    }
    return { users, occupations: Array.from(occupations) };
  }

  // Train/Val split per user (leave-one-out-ish)
  function splitRatingsByUser(ratings, seed=42, valFrac=0.1) {
    const byUser = new Map();
    for (const r of ratings) {
      if (!byUser.has(r.userId)) byUser.set(r.userId, []);
      byUser.get(r.userId).push(r);
    }
    const train = [];
    const val = [];
    let rng = mulberry32(seed);
    for (const [uid, arr] of byUser.entries()) {
      for (const r of arr) {
        if (rng() < valFrac) val.push(r);
        else train.push(r);
      }
    }
    return { train, val };
  }

  // Simple RNG
  function mulberry32(a) {
    return function() {
      var t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
  }

  // Baseline model: global + user bias + item bias
  function fitBaseline(train, lambda=10) {
    // Regularized least squares on biases with a few alternating passes
    const userBias = new Map();
    const itemBias = new Map();
    let mu = 0;
    for (const r of train) mu += r.rating;
    mu /= train.length;

    const byUser = new Map();
    const byItem = new Map();
    for (const r of train) {
      if (!byUser.has(r.userId)) byUser.set(r.userId, []);
      if (!byItem.has(r.movieId)) byItem.set(r.movieId, []);
      byUser.get(r.userId).push(r);
      byItem.get(r.movieId).push(r);
      if (!userBias.has(r.userId)) userBias.set(r.userId, 0);
      if (!itemBias.has(r.movieId)) itemBias.set(r.movieId, 0);
    }

    for (let it = 0; it < 8; it++) {
      // update item biases
      for (const [iid, arr] of byItem.entries()) {
        let num = 0, den = lambda + arr.length;
        for (const r of arr) {
          num += (r.rating - mu - (userBias.get(r.userId) || 0));
        }
        itemBias.set(iid, num / den);
      }
      // update user biases
      for (const [uid, arr] of byUser.entries()) {
        let num = 0, den = lambda + arr.length;
        for (const r of arr) {
          num += (r.rating - mu - (itemBias.get(r.movieId) || 0));
        }
        userBias.set(uid, num / den);
      }
    }
    function predict(uid, iid) {
      return mu + (userBias.get(uid) || 0) + (itemBias.get(iid) || 0);
    }
    return { mu, userBias, itemBias, predict };
  }

  // Utility: feature encodings
  function ageBucket(age) {
    if (age < 18) return "u<18";
    if (age <= 24) return "u18-24";
    if (age <= 34) return "u25-34";
    if (age <= 44) return "u35-44";
    if (age <= 49) return "u45-49";
    if (age <= 55) return "u50-55";
    return "u56+";
  }

  // Build user/item sparse feature spaces
  function buildFeatureSpaces(users, occupations, GENRES) {
    const uFeats = new Map(); // userId -> {keys:[], idxs:[]}
    const dict = {
      user: new Map(),  // feature -> index
      item: new Map()
    };
    let uPtr = 0, iPtr = 0;

    // Pre-add common slots for stability
    function getUIndex(token) {
      if (!dict.user.has(token)) dict.user.set(token, uPtr++);
      return dict.user.get(token);
    }
    function getIIndex(token) {
      if (!dict.item.has(token)) dict.item.set(token, iPtr++);
      return dict.item.get(token);
    }

    // user features: age bucket, gender, occupation (one-hot)
    for (const [uid, u] of users.entries()) {
      const feats = [];
      feats.push(getUIndex(`AGE:${ageBucket(u.age)}`));
      feats.push(getUIndex(`G:${u.gender || "?"}`));
      feats.push(getUIndex(`OCC:${u.occupation || "?"}`));
      uFeats.set(uid, feats);
    }

    // item features: genre one-hot
    const iFeats = []; // store per movieId later via helper
    const itemFeatCache = new Map();
    function getItemFeatIndices(genreFlags) {
      const key = genreFlags.join("");
      if (itemFeatCache.has(key)) return itemFeatCache.get(key);
      const arr = [];
      for (let g = 0; g < GENRES.length; g++) {
        if (genreFlags[g] === 1) arr.push(getIIndex(`GENRE:${GENRES[g]}`));
      }
      if (arr.length === 0) arr.push(getIIndex("GENRE:Unknown"));
      itemFeatCache.set(key, arr);
      return arr;
    }

    return { dict, uFeats, getItemFeatIndices };
  }

  // Two-tower tiny learner
  function fitTwoTower(train, users, movies, dict, uFeats, getItemFeatIndices, opts={}) {
    const dim = opts.dim || 16;
    const lr = opts.lr || 0.05;
    const epochs = opts.epochs || 2;
    const seed = opts.seed || 7;
    const posThresh = opts.posThresh || 4;
    const maxTrain = opts.maxTrain || 50000; // cap for speed

    // Matrices: W_user [dim x U], W_item [dim x I]
    const U = dict.user.size;
    const I = dict.item.size;
    const rnd = mulberry32(seed);

    function randn() {
      // Box-Muller
      let u = 0, v = 0;
      while (u === 0) u = rnd();
      while (v === 0) v = rnd();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    const Wu = new Float32Array(dim * U);
    const Wi = new Float32Array(dim * I);
    for (let k = 0; k < Wu.length; k++) Wu[k] = 0.01 * randn();
    for (let k = 0; k < Wi.length; k++) Wi[k] = 0.01 * randn();

    // Prepare samples (implicit target y in {0,1})
    const samples = [];
    for (const r of train) {
      const u = users.get(r.userId);
      const m = movies.get(r.movieId);
      if (!u || !m) continue;
      samples.push({
        uid: r.userId,
        itemFlags: m.flags,
        y: (r.rating >= posThresh) ? 1 : 0
      });
    }
    // Shuffle and cap
    for (let i = samples.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const tmp = samples[i]; samples[i] = samples[j]; samples[j] = tmp;
    }
    const trainSamples = samples.slice(0, Math.min(maxTrain, samples.length));

    // Training loop: logistic loss on dot( U(u), I(i) )
    const reg = opts.reg || 1e-4;
    const uCache = new Map();
    const iCache = new Map();

    function userVector(uid) {
      if (uCache.has(uid)) return uCache.get(uid);
      const idxs = uFeats.get(uid);
      const vec = new Float32Array(dim);
      for (const j of idxs) {
        const off = j * dim;
        for (let d = 0; d < dim; d++) vec[d] += Wu[off + d];
      }
      uCache.set(uid, vec);
      return vec;
    }
    function itemVector(flags) {
      const key = flags.join("");
      if (iCache.has(key)) return iCache.get(key);
      const idxs = getItemFeatIndices(flags);
      const vec = new Float32Array(dim);
      for (const j of idxs) {
        const off = j * dim;
        for (let d = 0; d < dim; d++) vec[d] += Wi[off + d];
      }
      iCache.set(key, vec);
      return vec;
    }

    function sigmoid(x){ return 1/(1+Math.exp(-x)); }

    for (let ep = 0; ep < epochs; ep++) {
      // mini-batch SGD with batch=1 (simple & fast enough here)
      for (const s of trainSamples) {
        const uvec = userVector(s.uid);
        const ivec = itemVector(s.itemFlags);
        // score
        let z = 0;
        for (let d = 0; d < dim; d++) z += uvec[d] * ivec[d];
        const p = sigmoid(z);
        const err = (p - s.y); // dL/dz

        // update Wu and Wi for the active feature columns only
        const uidxs = uFeats.get(s.uid);
        const iidxs = getItemFeatIndices(s.itemFlags);

        // gradients:
        // dL/dWu[:,j] = err * Ivecc + reg * Wu[:,j]
        // dL/dWi[:,j] = err * Uvecc + reg * Wi[:,j]
        for (const j of uidxs) {
          const off = j * dim;
          for (let d = 0; d < dim; d++) {
            const g = err * ivec[d] + reg * Wu[off + d];
            Wu[off + d] -= lr * g;
          }
        }
        for (const j of iidxs) {
          const off = j * dim;
          for (let d = 0; d < dim; d++) {
            const g = err * uvec[d] + reg * Wi[off + d];
            Wi[off + d] -= lr * g;
          }
        }

        // invalidate caches (cheap)
        uCache.delete(s.uid);
        iCache.clear();
      }
    }

    function embedUser(uid) {
      const idxs = uFeats.get(uid);
      const vec = new Float32Array(dim);
      for (const j of idxs) {
        const off = j * dim;
        for (let d = 0; d < dim; d++) vec[d] += Wu[off + d];
      }
      return vec;
    }
    function embedItem(flags) {
      const idxs = getItemFeatIndices(flags);
      const vec = new Float32Array(dim);
      for (const j of idxs) {
        const off = j * dim;
        for (let d = 0; d < dim; d++) vec[d] += Wi[off + d];
      }
      return vec;
    }
    function score(uid, movieFlags) {
      const u = embedUser(uid), i = embedItem(movieFlags);
      let s = 0; for (let d = 0; d < dim; d++) s += u[d]*i[d];
      return 1/(1+Math.exp(-s)); // probability for "likes"
    }

    return { score, embedUser, embedItem, dim, dictSizes: {U, I} };
  }

  // Metrics
  function computeMetrics(val, predictTopK, predictScore, k=10, posThresh=4) {
    // Evaluate Precision@k, Recall@k using predictTopK(userId,k)
    // Evaluate Accuracy on all val ratings using predictScore(userId,itemId)
    const byUser = new Map();
    for (const r of val) {
      if (!byUser.has(r.userId)) byUser.set(r.userId, []);
      byUser.get(r.userId).push(r);
    }

    let precSum=0, recSum=0, usersCount=0;

    // Accuracy
    let accCount = 0, accCorrect = 0;

    for (const [uid, arr] of byUser.entries()) {
      // positives in val for this user
      const positives = new Set(arr.filter(r => r.rating >= posThresh).map(r => r.movieId));
      if (positives.size === 0) {
        // still count accuracy below
      } else {
        const top = predictTopK(uid, k);
        const recs = new Set(top.map(x => x.movieId));
        let hit = 0;
        for (const m of positives) if (recs.has(m)) hit++;
        const prec = top.length ? (hit / top.length) : 0;
        const rec = positives.size ? (hit / positives.size) : 0;
        precSum += prec;
        recSum += rec;
        usersCount++;
      }

      // accuracy on all validation items
      for (const r of arr) {
        const score = predictScore(uid, r.movieId);
        const predPos = score >= 0.5; // classify like/dislike
        const actualPos = r.rating >= posThresh;
        if ((predPos && actualPos) || (!predPos && !actualPos)) accCorrect++;
        accCount++;
      }
    }

    return {
      precision: usersCount ? (precSum/usersCount) : 0,
      recall: usersCount ? (recSum/usersCount) : 0,
      accuracy: accCount ? (accCorrect/accCount) : 0
    };
  }

  async function fetchText(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status} ${res.statusText}`);
    return await res.text();
  }

  // Main orchestrator
  async function loadAll(basePath="") {
    const prefix = (basePath && !basePath.endsWith("/")) ? basePath + "/" : basePath;

    const [dText, iText, uText] = await Promise.all([
      fetchText(prefix + "u.data"),
      fetchText(prefix + "u.item"),
      fetchText(prefix + "u.user")
    ]);

    const ratings = parseDataContent(dText);
    const { movies, GENRES } = parseItemContent(iText);
    const { users, occupations } = parseUserContent(uText);

    // Split
    const { train, val } = splitRatingsByUser(ratings, 42, 0.1);

    // Baseline
    const baseline = fitBaseline(train, 10);
    function baselineScore(uid, movieId) {
      // Map prediction (1..5) -> [0,1] by linear scale for classification
      const p = baseline.predict(uid, movieId);
      return Math.max(0, Math.min(1, (p - 1) / 4));
    }

    // Two-tower
    const { dict, uFeats, getItemFeatIndices } = buildFeatureSpaces(users, occupations, GENRES);
    const deep = fitTwoTower(train, users, movies, dict, uFeats, getItemFeatIndices, {
      dim: 16, lr: 0.06, epochs: 3, maxTrain: 60000, reg: 1e-4, posThresh: 4
    });

    // For recommenders (exclude items already rated in train for that user)
    const trainByUser = new Map();
    for (const r of train) {
      if (!trainByUser.has(r.userId)) trainByUser.set(r.userId, new Set());
      trainByUser.get(r.userId).add(r.movieId);
    }

    function topKFor(uid, k, scorer) {
      const seen = trainByUser.get(uid) || new Set();
      const out = [];
      for (const [mid, m] of movies.entries()) {
        if (seen.has(mid)) continue;
        const s = scorer(uid, mid, m.flags);
        if (Number.isFinite(s)) {
          out.push({ movieId: mid, score: s });
        }
      }
      out.sort((a,b)=> b.score - a.score);
      return out.slice(0, k);
    }

    const baselineTopK = (uid, k=10) => topKFor(uid, k, (u, mid)=> baselineScore(u, mid));
    const deepTopK = (uid, k=10) => topKFor(uid, k, (u, mid, flags)=> deep.score(u, flags));

    // Metrics (sample users for speed)
    const metrics = {
      baseline: computeMetrics(val, baselineTopK, (u,m)=> baselineScore(u,m), 10, 4),
      deep: computeMetrics(val, deepTopK, (u,m)=> {
        const mv = movies.get(m);
        return mv ? deep.score(u, mv.flags) : 0.0;
      }, 10, 4)
    };

    return {
      users, movies, ratings, train, val,
      baseline, deep, GENRES,
      recommend: {
        baselineTopK, deepTopK
      },
      metrics
    };
  }

  return { loadAll, GENRES };
})();
