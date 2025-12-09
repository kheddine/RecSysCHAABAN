"""
Spotify RecSys - Simple App
Uses your original code, no modifications
"""

import os
import sys
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import YOUR original code
from recsys_core import SpotifyRecSysConstrained, HuggingFaceLLM, PlaylistTransformer

app = Flask(__name__)
CORS(app)

# Global state
class AppState:
    df = None
    recsys = None
    transformer = None
    llm = None

def load_data(csv_file):
    """Load CSV file"""
    try:
        AppState.df = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(AppState.df)} tracks")
        
        # Initialize your RecSys
        AppState.recsys = SpotifyRecSysConstrained(AppState.df)
        
        # Initialize LLM (empty key - you'll set it later)
        AppState.llm = HuggingFaceLLM("")
        
        # Initialize transformer
        AppState.transformer = PlaylistTransformer(AppState.recsys, AppState.llm)
        
        # Set initial playlist (5 random tracks)
        import numpy as np
        initial = np.random.choice(len(AppState.df), 5, replace=False).tolist()
        AppState.transformer.set_initial_playlist(initial)
        
        return True
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

# API Routes
@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get recommendations based on mood"""
    if AppState.recsys is None:
        return jsonify({'error': 'Data not loaded'}), 400
    
    data = request.json
    message = data.get('message', '')
    
    try:
        result = AppState.transformer.chat(message)
        return jsonify({
            'success': True,
            'recommendations': result['new_recommendations'][:10],
            'adjustments': result['adjustments']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlist', methods=['GET'])
def get_playlist():
    """Get current playlist"""
    if AppState.recsys is None:
        return jsonify({'tracks': []}), 200
    
    tracks = []
    for idx in AppState.transformer.current_playlist:
        try:
            info = AppState.recsys.get_track_info(idx)
            tracks.append({
                'name': info['name'],
                'artist': info['artist'],
                'genre': info['genre'],
                'index': idx
            })
        except:
            pass
    
    return jsonify({'tracks': tracks})

@app.route('/api/add-tracks', methods=['POST'])
def add_tracks():
    """Add tracks to playlist"""
    if AppState.transformer is None:
        return jsonify({'error': 'Data not loaded'}), 400
    
    data = request.json
    indices = data.get('indices', [])
    
    try:
        AppState.transformer.add_tracks_to_playlist(indices)
        return jsonify({'success': True, 'playlist_size': len(AppState.transformer.current_playlist)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check if data is loaded"""
    return jsonify({
        'loaded': AppState.recsys is not None,
        'tracks': len(AppState.df) if AppState.df is not None else 0,
        'playlist_size': len(AppState.transformer.current_playlist) if AppState.transformer else 0
    })

@app.route('/api/set-api-key', methods=['POST'])
def set_api_key():
    """Set HuggingFace API key"""
    data = request.json
    api_key = data.get('api_key', '')
    
    if AppState.llm:
        AppState.llm.api_key = api_key
        return jsonify({'success': True, 'message': 'API key set'})
    return jsonify({'error': 'LLM not initialized'}), 500

if __name__ == '__main__':
    # Check if CSV file exists
    csv_file = 'data.csv'
    
    if not os.path.exists(csv_file):
        print(f"\n❌ ERROR: {csv_file} not found!")
        print("\nTo use this app:")
        print("1. Place your Spotify CSV file in this folder")
        print("2. Rename it to: data.csv")
        print("3. Run: python run.py")
        print("\nExample:")
        print("  cp /path/to/your/spotify_data.csv ./data.csv")
        print("  python run.py")
        sys.exit(1)
    
    # Load data
    print("Loading data...")
    if not load_data(csv_file):
        sys.exit(1)
    
    print("\n" + "="*50)
    print("✓ Spotify RecSys Ready!")
    print("="*50)
    print(f"📊 Tracks loaded: {len(AppState.df)}")
    print(f"🎵 Initial playlist: {len(AppState.transformer.current_playlist)} tracks")
    print(f"\n🌐 API running on: http://localhost:5000")
    print(f"📱 Open: http://localhost:3000")
    print("="*50 + "\n")
    
    app.run(debug=False, port=5000)
