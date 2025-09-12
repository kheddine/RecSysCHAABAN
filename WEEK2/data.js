/* data.js
 * Loads and parses MovieLens 100K files.
 */

(() => {
  const DEFAULT_GENRES = [
    "unknown","Action","Adventure","Animation","Children's","Comedy","Crime","Documentary",
    "Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
  ];

  const state = {
    genreIndex: [],
    movies: [],
    ratingCountByItem: new Map(),
    ratingSumByItem: new Map(),
    loaded: false,
  };

  function splitLines(text) {
    return text.split(/\r?\n/).filter(Boolean);
  }

  function splitFields(line) {
    if (line.includes("\t")) return line.split("\t");
    return line.split("|");
  }

  async function tryFetch(path) {
    try {
      const res = await fetch(path, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.text();
    } catch {
      return null;
    }
  }

  function loadGenres(text) {
    const lines = splitLines(text);
    const pairs = lines
      .map(l => l.trim())
      .filter(l => l && !l.startsWith("#"))
      .map(l => {
        const [name, idxStr] = l.split("|");
        return { name, idx: Number(idxStr) };
      })
      .filter(p => Number.isFinite(p.idx) && p.name);

    if (!pairs.length) return DEFAULT_GENRES.slice();

    const maxIdx = Math.max(...pairs.map(p => p.idx));
    const arr = new Array(maxIdx + 1).fill(null);
    for (const { name, idx } of pairs) arr[idx] = name;
    for (let i = 0; i < arr.length; i++) {
      if (!arr[i]) arr[i] = DEFAULT_GENRES[i] ?? `Genre_${i}`;
    }
    return arr;
  }

  function parseUItem(text, genreIndex) {
    const lines = splitLines(text);
    const movies = [];

    for (const line of lines) {
      const f = splitFields(line);
      if (f.length < 5) continue;

      const id = Number(f[0]);
      const title = f[1] || `Movie ${id}`;
      const releaseDate = f[2] || "";
      const imdbUrl = f[4] || "";

      const flagsStart = 5;
      const flags = f.slice(flagsStart).map(x => Number(x) || 0);
      const vec = new Array(genreIndex.length).fill(0);
      for (let i = 0; i < Math.min(vec.length, flags.length); i++) vec[i] = flags[i] ? 1 : 0;

      const genres = new Set(
        vec.map((bit, i) => (bit ? genreIndex[i] : null)).filter(Boolean)
      );

      movies.push({ id, title, releaseDate, imdbUrl, genres, genreVec: vec });
    }
    return movies;
  }

  function parseUData(text) {
    const lines = splitLines(text);
    for (const line of lines) {
      const f = splitFields(line);
      if (f.length < 3) continue;
      const itemId = Number(f[1]);
      const rating = Number(f[2]);
      if (!Number.isFinite(itemId) || !Number.isFinite(rating)) continue;
      state.ratingCountByItem.set(itemId, (state.ratingCountByItem.get(itemId) || 0) + 1);
      state.ratingSumByItem.set(itemId, (state.ratingSumByItem.get(itemId) || 0) + rating);
    }
  }

  function getAvgRating(itemId) {
    const c = state.ratingCountByItem.get(itemId) || 0;
    if (!c) return undefined;
    const s = state.ratingSumByItem.get(itemId) || 0;
    return s / c;
  }

  function getRatingCount(itemId) {
    return state.ratingCountByItem.get(itemId) || 0;
  }

  async function loadAll() {
    if (state.loaded) return;

    const genreText = await tryFetch("u.genre");
    state.genreIndex = genreText ? loadGenres(genreText) : DEFAULT_GENRES.slice();

    const itemText = await tryFetch("u.item");
    if (!itemText) {
      throw new Error("Could not load u.item.");
    }
    state.movies = parseUItem(itemText, state.genreIndex);

    const dataText = await tryFetch("u.data");
    if (dataText) parseUData(dataText);

    state.loaded = true;
  }

  window.DataModule = {
    loadAll,
    getMovies: () => state.movies.slice(),
    getMovieById: (id) => state.movies.find(m => m.id === Number(id)),
    getAvgRating,
    getRatingCount,
    getGenreIndex: () => state.genreIndex.slice(),
  };
})();
