/* script.js
 * UI wiring + Cosine-similarity recommendations over genre vectors.
 */

(() => {
  const selectEl = document.getElementById("movie-select");
  const recBtn = document.getElementById("rec-btn");
  const resultP = document.getElementById("result");
  const recList = document.getElementById("rec-list");

  // ---------- Helpers ----------
  function cosineFromVectors(a, b) {
    let dot = 0, na = 0, nb = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      const ai = a[i] ? 1 : 0;
      const bi = b[i] ? 1 : 0;
      dot += ai * bi;
      na  += ai * ai;
      nb  += bi * bi;
    }
    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }

  function formatRating(avg) {
    return avg ? avg.toFixed(2) : "—";
  }

  function renderLoading(msg) {
    resultP.textContent = msg;
    recList.innerHTML = "";
  }

  function renderRecommendations(baseMovie, recs) {
    if (!recs.length) {
      resultP.textContent = "No similar movies found.";
      recList.innerHTML = "";
      return;
    }
    resultP.innerHTML = `Movies similar to <strong>${baseMovie.title}</strong>:`;
    recList.innerHTML = "";

    for (const r of recs) {
      const li = document.createElement("li");
      li.className = "rec-card";

      const h = document.createElement("p");
      h.className = "rec-title";
      h.textContent = r.movie.title;

      const m = document.createElement("p");
      m.className = "rec-meta";
      const overlapBadges = Array.from(r.overlap).map(g => `<span class="badge">${g}</span>`).join(" ");
      const imdb = r.movie.imdbUrl ? ` · <a href="${r.movie.imdbUrl}" target="_blank" rel="noopener">IMDb</a>` : "";
      m.innerHTML =
        `Cosine: ${r.similarity.toFixed(3)} · ` +
        `Avg ★: ${formatRating(r.avgRating)} (${r.count} ratings) · ` +
        `Release: ${r.movie.releaseDate || "n/a"}${imdb}<br/>` +
        `Overlap: ${overlapBadges || "<span class='badge'>none</span>"}`;

      li.appendChild(h);
      li.appendChild(m);
      recList.appendChild(li);
    }
  }

  function computeRecs(baseMovie, allMovies, topN = 10) {
    const baseSet = baseMovie.genres;
    const baseVec = baseMovie.genreVec;

    const out = [];
    for (const movie of allMovies) {
      if (movie.id === baseMovie.id) continue;
      const sim = cosineFromVectors(baseVec, movie.genreVec);
      if (sim <= 0) continue;

      const overlap = new Set([...baseSet].filter(g => movie.genres.has(g)));
      const avg = DataModule.getAvgRating(movie.id);
      const cnt = DataModule.getRatingCount(movie.id);
      out.push({
        movie,
        similarity: sim,
        overlap,
        avgRating: avg ?? 0,
        count: cnt ?? 0,
      });
    }

    out.sort((a, b) => {
      if (b.similarity !== a.similarity) return b.similarity - a.similarity;
      if ((b.avgRating ?? 0) !== (a.avgRating ?? 0)) return (b.avgRating ?? 0) - (a.avgRating ?? 0);
      if ((b.count ?? 0) !== (a.count ?? 0)) return (b.count ?? 0) - (a.count ?? 0);
      return a.movie.title.localeCompare(b.movie.title);
    });

    return out.slice(0, topN);
  }

  // ---------- Init ----------
  async function init() {
    try {
      renderLoading("Loading data…");
      recBtn.disabled = true;

      await DataModule.loadAll();
      const movies = DataModule.getMovies();

      movies.sort((a, b) => a.title.localeCompare(b.title));
      selectEl.innerHTML = "";
      for (const m of movies) {
        const opt = document.createElement("option");
        opt.value = String(m.id);
        opt.textContent = m.title;
        selectEl.appendChild(opt);
      }

      resultP.textContent = "Choose a movie and click “Get Recommendations”.";
      recBtn.disabled = false;

    } catch (err) {
      console.error(err);
      resultP.textContent = "Failed to load data. Place u.item / u.genre (and optionally u.data) beside this page, and open with a local web server.";
      recBtn.disabled = true;
    }
  }

  async function onRecommend() {
    const id = Number(selectEl.value);
    if (!Number.isFinite(id)) return;

    const base = DataModule.getMovieById(id);
    if (!base) return;

    renderLoading("Computing recommendations…");
    const all = DataModule.getMovies();
    const recs = computeRecs(base, all, 10);
    renderRecommendations(base, recs);
  }

  document.addEventListener("DOMContentLoaded", init);
  recBtn.addEventListener("click", onRecommend);
})();
