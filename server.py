import os
import sys
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from recsys_core import SpotifyRecSysConstrained, HuggingFaceLLM, PlaylistTransformer

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

class AppState:
    df = None
    recsys = None
    transformer = None
    llm = None

def load_data(csv_file):
    try:
        AppState.df = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(AppState.df)} tracks")
        AppState.recsys = SpotifyRecSysConstrained(AppState.df)
        AppState.llm = HuggingFaceLLM("")
        AppState.transformer = PlaylistTransformer(AppState.recsys, AppState.llm)
        
        import numpy as np
        initial = np.random.choice(len(AppState.df), 5, replace=False).tolist()
        AppState.transformer.set_initial_playlist(initial)
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

@app.route('/')
def index():
    return send_from_directory('.', 'simple_ui.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
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
    if AppState.recsys is None:
        return jsonify({'tracks': []}), 200
    
    tracks = []
    for idx in AppState.transformer.current_playlist:
        try:
            info = AppState.recsys.get_track_info(idx)
            tracks.append({
                'name': info['name'],
                'artist': info['artist'],
                'genre': info['genre']
            })
        except:
            pass
    return jsonify({'tracks': tracks})

@app.route('/api/add-tracks', methods=['POST'])
def add_tracks():
    if AppState.transformer is None:
        return jsonify({'error': 'Data not loaded'}), 400
    
    data = request.json
    indices = data.get('indices', [])
    AppState.transformer.add_tracks_to_playlist(indices)
    return jsonify({'success': True})

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'loaded': AppState.recsys is not None,
        'tracks': len(AppState.df) if AppState.df is not None else 0
    })

@app.route('/api/set-api-key', methods=['POST'])
def set_api_key():
    data = request.json
    api_key = data.get('api_key', '')
    if AppState.llm:
        AppState.llm.api_key = api_key
        return jsonify({'success': True})
    return jsonify({'error': 'LLM not initialized'}), 500

if __name__ == '__main__':
    csv_file = 'data.csv'
    
    if os.path.exists(csv_file):
        print("Loading data...")
        load_data(csv_file)
    else:
        print("⚠️  data.csv not found - app running without data")
    
    port = int(os.environ.get('PORT', 3000))
    app.run(debug=False, port=port, host='0.0.0.0')
