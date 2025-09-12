// Random Lunch Recommender with Unsplash images
const LUNCHES = [
  { name: "Pizza", emoji: "🍕", cuisine: "Italian", image: "https://source.unsplash.com/800x500/?pizza,food" },
  { name: "Sushi", emoji: "🍣", cuisine: "Japanese", image: "https://source.unsplash.com/800x500/?sushi,food" },
  { name: "Burger", emoji: "🍔", cuisine: "American", image: "https://source.unsplash.com/800x500/?burger,food" },
  { name: "Tacos", emoji: "🌮", cuisine: "Mexican", image: "https://source.unsplash.com/800x500/?tacos,food" },
  { name: "Salad", emoji: "🥗", cuisine: "Healthy", image: "https://source.unsplash.com/800x500/?salad,food" },
  { name: "Ramen", emoji: "🍜", cuisine: "Japanese", image: "https://source.unsplash.com/800x500/?ramen,food" },
  { name: "Falafel Wrap", emoji: "🧆", cuisine: "Middle Eastern", image: "https://source.unsplash.com/800x500/?falafel,food" },
  { name: "Poke Bowl", emoji: "🥙", cuisine: "Hawaiian", image: "https://source.unsplash.com/800x500/?poke,bowl,food" },
  { name: "Curry", emoji: "🍛", cuisine: "Indian", image: "https://source.unsplash.com/800x500/?curry,food" },
  { name: "Pad Thai", emoji: "🍜", cuisine: "Thai", image: "https://source.unsplash.com/800x500/?padthai,food" }
];

const resultEl = document.getElementById("result");
const imageEl = document.getElementById("meal-image");
const tagsEl = document.getElementById("tags");
const toastEl = document.getElementById("toast");
const generateBtn = document.getElementById("generate-btn");
const copyBtn = document.getElementById("copy-btn");

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

generateBtn.addEventListener("click", () => {
  const pick = randomPick(LUNCHES);
  renderPick(pick);
});
copyBtn.addEventListener("click", copyResult);
