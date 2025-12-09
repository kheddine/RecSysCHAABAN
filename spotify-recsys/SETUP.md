# Spotify RecSys - Simple Setup

## What's Included
- `run.py` - Backend (your original code)
- `recsys_core.py` - Your RecSys logic (UNCHANGED)
- `eda_cleaner.py` - Data cleaning (UNCHANGED)
- `requirements.txt` - Dependencies
- `simple_ui.html` - Web interface

## Setup (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Add Your Data
```bash
# Copy your CSV file to this folder and rename it:
cp /path/to/your/spotify_data.csv ./data.csv
```

### Step 3: Run
```bash
python run.py
```

Then open browser: **http://localhost:3000**

## To Add Your API Key
1. Open the app in browser
2. Click "Settings" (⚙️)
3. Paste your HuggingFace API key
4. Save

## That's It!
- Your data loads automatically
- Your code runs unchanged
- You control the API key
