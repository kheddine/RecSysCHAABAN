# 🎵 Spotify Mood Mixer

Transform your Spotify playlists based on your mood using AI-powered music recommendations.

## Features

✨ **Mood-Based Playlist Transformation** - Describe your vibe and get personalized recommendations
🎯 **Intelligent Feature Analysis** - Uses cosine similarity on Spotify audio features
🤖 **AI-Powered Mood Understanding** - LLM integration (Hugging Face) with rule-based fallback
🎨 **Beautiful UI** - Spotify-themed dark interface with smooth animations
🔄 **Constraint-Based Recommendations** - Respects feature relationships (e.g., high acousticness affects energy)
🚀 **Production-Ready** - Flask backend with CORS support, fully containerized

## Tech Stack

**Backend:**
- Flask (Python web framework)
- scikit-learn (ML for cosine similarity)
- Pandas/NumPy (data processing)
- Hugging Face Inference API (LLM)

**Frontend:**
- Vanilla HTML/CSS/JavaScript
- Spotify color palette
- Responsive design

**Deployment:**
- Docker & Docker Compose
- Gunicorn WSGI server

## Quick Start

### Local Development

#### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/spotify-mood-mixer.git
cd spotify-mood-mixer
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Prepare Data

Place your Spotify features CSV in the project root:
```
SpotifyFeatures (1)_cleaned.csv
```

The CSV should contain columns like:
- `track_name`
- `artist_name`
- `genre`
- `popularity`
- `acousticness`, `danceability`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`

#### 4. Run Backend Server

```bash
python app.py
```

Server runs on `http://localhost:5000`

#### 5. Open Frontend

Open `index.html` in your browser (or serve with a simple HTTP server):

```bash
# Python 3
python -m http.server 8000

# Then visit http://localhost:8000
```

#### 6. Configure in UI

- **API Server URL:** `http://localhost:5000`
- **CSV Path:** `SpotifyFeatures (1)_cleaned.csv`
- **HF API Key:** (optional) Get from https://huggingface.co/settings/tokens

Click "Start Mixing" and begin transforming your playlist!

---

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up --build

# Backend: http://localhost:5000
# Frontend: http://localhost:8080
```

### Manual Docker Build

```bash
# Build image
docker build -t spotify-mood-mixer .

# Run container
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -e CSV_PATH=data/SpotifyFeatures.csv \
  spotify-mood-mixer
```

---

## API Documentation

### `POST /api/init`
Initialize system with CSV data and API key.

**Request:**
```json
{
  "csv_path": "SpotifyFeatures (1)_cleaned.csv",
  "api_key": "hf_..."
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Loaded 10000 tracks",
  "playlist": [...],
  "track_count": 10000
}
```

### `POST /api/chat`
Send mood request and get recommendations.

**Request:**
```json
{
  "message": "make it more energetic"
}
```

**Response:**
```json
{
  "status": "success",
  "interpretation": "Transforming playlist: make it more energetic",
  "adjustments": {
    "energy": 0.7,
    "danceability": 0.6,
    ...
  },
  "recommendations": [
    {
      "name": "Track Name",
      "artist": "Artist Name",
      "genre": "Genre",
      "similarity": 0.89,
      "index": 42
    },
    ...
  ]
}
```

### `POST /api/add-tracks`
Add tracks to current playlist.

**Request:**
```json
{
  "indices": [10, 42, 105]
}
```

### `GET /api/playlist`
Get current playlist.

### `POST /api/reset`
Reset to initial random playlist.

### `POST /api/clear-playlist`
Clear all tracks from playlist.

---

## Mood Keywords

Try these keywords to transform your playlist:

**Energetic Vibes:** energetic, upbeat, fast, dance, electronic
**Chill Moods:** chill, relaxing, calm, slow, acoustic
**Emotional:** sad, melancholic, happy, romantic
**Dynamic:** loud, quiet, instrumental, live

Example queries:
- "make it more energetic"
- "add some sad acoustic vibes"
- "electronic and fast"
- "relaxing and happy"

---

## Configuration

### Environment Variables

Create a `.env` file:

```env
FLASK_ENV=production
CSV_PATH=SpotifyFeatures (1)_cleaned.csv
HF_API_KEY=hf_xxxxx
API_PORT=5000
```

### Feature Constraints

The system respects relationships between audio features:

- **Acousticness (positive):** Softens energy, increases instrumentalness
- **Energy (positive):** Increases valence and tempo
- **Valence (negative):** Decreases energy, increases acousticness

This prevents contradictory recommendations (e.g., high-energy acoustic music).

---

## Development

### Project Structure

```
spotify-mood-mixer/
├── app.py                 # Flask backend
├── index.html            # Frontend UI
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Multi-container setup
├── .env                 # Environment variables
└── README.md            # This file
```

### Adding New Mood Keywords

Edit `mood_rules` in `app.py` HuggingFaceLLM class:

```python
mood_rules = {
    "your_keyword": {
        "energy": 0.5,
        "danceability": 0.3,
        ...
    }
}
```

---

## Troubleshooting

### "CSV file not found"
- Ensure the CSV path is correct (relative to the Flask app)
- Use absolute paths if needed

### "HuggingFace API timeout"
- The system will automatically fall back to rule-based mood detection
- No API key needed for basic functionality

### CORS errors
- Backend is already CORS-enabled
- If issues persist, check `flask-cors` is installed

### Slow recommendations
- CSV files with 100K+ tracks may be slower
- Consider using a subset or pre-computed similarity matrices

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - see LICENSE.md

---

## Future Enhancements

- 🎧 Real Spotify API integration (read user playlists)
- 💾 Save/load playlists
- 📊 Mood analytics dashboard
- 🎯 User preference learning
- 🌐 Deploy to cloud (Heroku, AWS, GCP)
- 📱 Mobile app (React Native)

---

## Support

For issues, questions, or suggestions:
- 📝 Open an issue on GitHub
- 💬 Discuss in Discussions tab
- 📧 Contact: your-email@example.com

---

**Made with ♪ by the Spotify Mood Mixer team**
