# Deploy to Render (5 Minutes)

## Step 1: Create GitHub Repository
1. Go to **github.com** (create free account if needed)
2. Click **"New repository"**
3. Name: `spotify-recsys`
4. Click **"Create repository"**

## Step 2: Upload Files
1. Click **"Add file"** → **"Upload files"**
2. Select and upload ALL files from this folder
3. Commit changes

## Step 3: Get Your Data URL
You need a CSV file link:
- **Option A:** Upload to Google Drive, get shareable link
- **Option B:** Upload to Dropbox, get download link
- **Option C:** Use a direct CSV URL

## Step 4: Deploy to Render
1. Go to **render.com**
2. Click **"New Web Service"**
3. Connect your GitHub repo
4. Name: `spotify-recsys`
5. Environment: `Python 3`
6. Build command: `pip install -r requirements.txt`
7. Start command: `python server.py`
8. Click **"Create Web Service"**

## Step 5: Add Your Data
1. After deployment, go to Render dashboard
2. Click your service
3. Go to **"Shell"** tab
4. Run:
```bash
wget "YOUR_CSV_LINK" -O data.csv
```

## Done! 🎉
Your app is live at: `https://your-service-name.onrender.com`
