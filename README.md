# 🎵 Spotify RecSys - Simple App

Complete music recommendation system using **your original code**.

## What's Included

- ✅ **Your Code** - recsys_core.py (unchanged)
- ✅ **Simple Backend** - server.py (all-in-one)
- ✅ **Web Interface** - simple_ui.html (clean design)
- ✅ **No Configuration** - just add your data and run

## Setup (3 Commands)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Data
Place your Spotify CSV file in this folder and rename it to:
```
data.csv
```

### 3. Run
```bash
python server.py
```

Then open your browser:
```
http://localhost:3000
```

## That's It!

- Your data loads automatically
- Your RecSys code runs unchanged
- Add your HuggingFace API key in Settings (optional)

## Features

- 💬 Chat interface to request moods
- 🎵 Get recommendations based on your requests
- ➕ Click to add tracks to playlist
- ⚙️ Settings to add API key

## Your Code

Your original code files are included unchanged:
- `recsys_core.py` - SpotifyRecSysConstrained, HuggingFaceLLM, PlaylistTransformer
- `eda_cleaner.py` - Data cleaning utilities

## How to Add API Key

1. Get free key: https://huggingface.co/settings/tokens
2. Open the app
3. Click "⚙️ Settings"
4. Paste your key
5. Click "Save"

## Questions?

Everything works out of the box. Just:
1. Add your CSV file (as data.csv)
2. Run: `python server.py`
3. Open: http://localhost:3000
