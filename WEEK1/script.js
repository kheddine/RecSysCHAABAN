// Random Lunch Recommender
// Tech: HTML (structure), CSS (design), JS (random logic), GitHub Pages (deploy)

/**
 * No local image files required.
 * Each item uses a Picsum seed URL so it works on GitHub Pages instantly.
 * Replace image URLs if you want fixed brand photos.
 */
const LUNCHES = [
  { name: "Pizza", emoji: "🍕", cuisine: "Italian", image: "https://picsum.photos/seed/pizza/800/500" },
  { name: "Sushi", emoji: "🍣", cuisine: "Japanese", image: "https://picsum.photos/seed/sushi/800/500" },
  { name: "Burger", emoji: "🍔", cuisine: "American", image: "https://picsum.photos/seed/burger/800/500" },
  { name: "Tacos", emoji: "🌮", cuisine: "Mexican", image: "https://picsum.photos/seed/tacos/800/500" },
  { name: "Salad", emoji: "🥗", cuisine: "Healthy", image: "https://picsum.photos/seed/salad/800/500" },
  { name: "Ramen", emoji: "🍜", cuisine: "Japanese", image: "https://picsum.photos/seed/ramen/800/500" },
  { name: "Falafel Wrap", emoji: "🧆", cuisine: "Middle Eastern", image: "https://picsum.photos/seed/falafel/800/500" },
  { name: "Poke Bowl", emoji: "🥙", cuisine: "Hawaiian", image: "https://picsum.photos/seed/poke/800/500" },
  { name: "Curry", emoji: "🍛", cuisine: "Indian", image: "https://picsum.photos/seed/curry/800/500" },
  { name: "Pad Thai", emoji: "🍜", cuisine: "Thai", image: "https://picsum.photos/seed/padthai/800/500" }
];

// --- DOM refs
const resultEl = document.getElementById("result");
const imageEl = document.getElementById("meal-image");
const tagsEl = document.getElementById("tags");
const toastEl = document.getElementById("toast");
const generateBtn = document.getElementById("generate-btn");
const copyBtn = document.getElementById("copy-btn");

// --- Core random logic
function randomPick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function renderPick(item) {
  resultEl.textContent = `Today's pick: ${item.name} ${item.emoji} 🎉`;
  imageEl.src = item.image;
  imageEl.alt = item.name + " image";

  tagsEl.innerHTML = "";
  tagsEl.appendChild(badge(item.cuisine));
  tagsEl.appendChild(badge("Random"));
}

function badge(text) {
  const span = document.createElement("span");
  span.className = "badge";
  span.textContent = text;
  return span;
}

// --- Copy to clipboard
async function copyResult() {
  const text = resultEl.textContent || "Today's pick: (none)";
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied to clipboard!");
  } catch {
    toast("Copy not supported here.");
  }
}

function toast(msg) {
  toastEl.textContent = msg;
  clearTimeout(toastEl._t);
  toastEl._t = setTimeout(() => (toastEl.textContent = ""), 1500);
}

// --- Events
generateBtn.addEventListener("click", () => {
  const pick = randomPick(LUNCHES);
  renderPick(pick);
});

copyBtn.addEventListener("click", copyResult);

// Initial state: show a random hero image but no pick
// (Optional) Auto-pick on load:
// renderPick(randomPick(LUNCHES));

/**
 * ========== EVOLVE TO AI (later) ==========
 * - Save likes/dislikes: localStorage.setItem("likes", JSON.stringify([...]))
 * - Bias randomPick by weights (simple multinomial over cuisines user likes)
 * - Replace Picsum URLs with your own /images/* or a CDN when you add assets
 */
