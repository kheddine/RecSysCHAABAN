"""
Spotify Mood Mixer - Enhanced Backend
With Data Management, Cleaning, and Model Training
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv
import io
import traceback
from datetime import datetime

load_dotenv()

# ===================== SPOTIFY RECSYS =====================

class SpotifyRecSysConstrained:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_cols = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]

        self._normalize_features()
        self.feature_matrix = self._build_feature_matrix()
        self.scaler = StandardScaler()
        self.feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix)

        self.feature_constraints = {
            'acousticness': {
                'positive': {'energy': 'soft', 'instrumentalness': 'increase'},
                'negative': {'energy': 'flexible', 'instrumentalness': 'flexible'}
            },
            'energy': {
                'positive': {'valence': 'increase', 'tempo': 'increase'},
                'negative': {'valence': 'flexible', 'tempo': 'decrease'}
            },
            'valence': {
                'positive': {'energy': 'increase', 'instrumentalness': 'flexible'},
                'negative': {'energy': 'decrease', 'acousticness': 'increase'}
            }
        }

    def _normalize_features(self):
        if 'loudness' in self.df.columns:
            min_loudness = self.df['loudness'].min()
            max_loudness = self.df['loudness'].max()
            self.df['loudness_norm'] = (self.df['loudness'] - min_loudness) / (max_loudness - min_loudness)

        if 'tempo' in self.df.columns:
            min_tempo = self.df['tempo'].min()
            max_tempo = self.df['tempo'].max()
            self.df['tempo_norm'] = (self.df['tempo'] - min_tempo) / (max_tempo - min_tempo)

    def _build_feature_matrix(self) -> np.ndarray:
        features = []
        for col in self.feature_cols:
            if col in self.df.columns:
                features.append(self.df[col].values)
            elif col == 'loudness' and 'loudness_norm' in self.df.columns:
                features.append(self.df['loudness_norm'].values)
            elif col == 'tempo' and 'tempo_norm' in self.df.columns:
                features.append(self.df['tempo_norm'].values)

        return np.column_stack(features) if features else np.array([])

    def get_playlist_mood_vector(self, track_indices):
        if not track_indices or len(track_indices) == 0:
            return np.zeros(len(self.feature_cols))
        return self.feature_matrix_scaled[track_indices].mean(axis=0)

    def adjust_mood_vector_with_constraints(self, base_vector, adjustments):
        adjusted = base_vector.copy()

        for feature, delta in adjustments.items():
            if feature in self.feature_cols:
                idx = self.feature_cols.index(feature)
                adjusted[idx] = np.clip(adjusted[idx] + delta, -3, 3)

        for feature, delta in adjustments.items():
            if feature in self.feature_constraints and delta != 0:
                direction = 'positive' if delta > 0 else 'negative'
                constraints = self.feature_constraints[feature].get(direction, {})

                for conflicting_feature, constraint_type in constraints.items():
                    if conflicting_feature not in adjustments or adjustments[conflicting_feature] == 0:
                        if constraint_type == 'soft':
                            if conflicting_feature in self.feature_cols:
                                idx = self.feature_cols.index(conflicting_feature)
                                adjusted[idx] = adjusted[idx] * 0.5

                        elif constraint_type == 'increase':
                            if conflicting_feature in self.feature_cols:
                                idx = self.feature_cols.index(conflicting_feature)
                                delta_magnitude = abs(delta)
                                adjusted[idx] = np.clip(adjusted[idx] + delta_magnitude * 0.4, -3, 3)

                        elif constraint_type == 'decrease':
                            if conflicting_feature in self.feature_cols:
                                idx = self.feature_cols.index(conflicting_feature)
                                delta_magnitude = abs(delta)
                                adjusted[idx] = np.clip(adjusted[idx] - delta_magnitude * 0.4, -3, 3)

        return adjusted

    def recommend_by_mood_vector(self, mood_vector, n_recommendations=10, exclude_indices=None):
        if exclude_indices is None:
            exclude_indices = []

        similarities = cosine_similarity([mood_vector], self.feature_matrix_scaled)[0]

        for idx in exclude_indices:
            if idx < len(similarities):
                similarities[idx] = -1

        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        return [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0]

    def get_track_info(self, index):
        if index >= len(self.df):
            return {
                'name': 'Unknown',
                'artist': 'Unknown',
                'genre': 'Unknown',
                'popularity': 0,
                'index': index
            }
        row = self.df.iloc[index]
        return {
            'name': str(row.get('track_name', 'Unknown')),
            'artist': str(row.get('artist_name', 'Unknown')),
            'genre': str(row.get('genre', 'Unknown')),
            'popularity': int(row.get('popularity', 0)),
            'index': index
        }

    def get_random_tracks(self, n=5):
        indices = np.random.choice(len(self.df), min(n, len(self.df)), replace=False).tolist()
        return indices


# ===================== DATA CLEANING =====================

class DataCleaner:
    @staticmethod
    def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Clean and validate Spotify dataset"""
        stats = {
            'original_rows': len(df),
            'original_cols': len(df.columns),
            'removed_duplicates': 0,
            'removed_missing': 0,
            'issues': []
        }

        df_clean = df.copy()

        # Remove complete duplicates
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        stats['removed_duplicates'] = before - len(df_clean)

        # Audio feature columns
        audio_features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]

        # Remove rows with missing audio features
        available_features = [f for f in audio_features if f in df_clean.columns]
        if available_features:
            before = len(df_clean)
            df_clean = df_clean.dropna(subset=available_features)
            stats['removed_missing'] = before - len(df_clean)

        # Clip values to valid ranges
        features_0_1 = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                       'liveness', 'speechiness', 'valence']
        for feat in features_0_1:
            if feat in df_clean.columns:
                df_clean[feat] = df_clean[feat].clip(0, 1)

        if 'tempo' in df_clean.columns:
            df_clean['tempo'] = df_clean['tempo'].clip(0, 300)

        # Remove track name duplicates
        name_col = 'track_name' if 'track_name' in df_clean.columns else ('name' if 'name' in df_clean.columns else None)
        if name_col:
            before = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=[name_col], keep='first')
            stats['removed_duplicates'] += before - len(df_clean)

        stats['final_rows'] = len(df_clean)
        stats['final_cols'] = len(df_clean.columns)

        return df_clean, stats


# ===================== HUGGING FACE LLM =====================

class HuggingFaceLLM:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def extract_adjustments(self, user_message: str) -> Dict[str, float]:
        system_prompt = """You are a music mood translator. Extract audio feature adjustments.

RESPOND ONLY WITH VALID JSON (no markdown):
{
    "interpretation": "What user wants",
    "feature_adjustments": {
        "energy": 0,
        "danceability": 0,
        "valence": 0,
        "acousticness": 0,
        "instrumentalness": 0,
        "speechiness": 0,
        "liveness": 0,
        "loudness": 0,
        "tempo": 0
    },
    "explanation": "Why"
}

Guidelines: -1 to 1, negative reduces, positive increases."""

        prompt = f"{system_prompt}\n\nUser request: {user_message}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.3
            }
        }

        try:
            if not self.api_key:
                return self._fallback_extraction(user_message)

            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                output_text = result[0]['generated_text']

                try:
                    json_start = output_text.find('{')
                    json_end = output_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = output_text[json_start:json_end]
                        parsed = json.loads(json_str)
                        return parsed.get('feature_adjustments', {})
                except:
                    pass

            return self._fallback_extraction(user_message)

        except Exception as e:
            print(f"LLM API error: {e}")
            return self._fallback_extraction(user_message)

    def _fallback_extraction(self, user_message: str) -> Dict[str, float]:
        msg = user_message.lower()

        mood_rules = {
            "energetic": {"energy": 0.7, "danceability": 0.6, "valence": 0.4, "tempo": 0.5},
            "sad": {"valence": -0.8, "energy": -0.5, "acousticness": 0.3},
            "acoustic": {"acousticness": 0.7, "energy": -0.3, "instrumentalness": 0.2},
            "chill": {"energy": -0.6, "tempo": -0.5, "acousticness": 0.4},
            "dance": {"danceability": 0.8, "energy": 0.7, "tempo": 0.5},
            "electronic": {"acousticness": -0.8, "energy": 0.6, "instrumentalness": 0.5},
            "upbeat": {"energy": 0.7, "valence": 0.7, "tempo": 0.4},
            "melancholic": {"valence": -0.7, "energy": -0.4, "acousticness": 0.5},
            "relaxing": {"energy": -0.7, "valence": 0.3, "loudness": -0.4},
            "happy": {"valence": 0.8, "energy": 0.5, "danceability": 0.5},
            "slow": {"tempo": -0.6, "energy": -0.4},
            "fast": {"tempo": 0.7, "energy": 0.6},
            "quiet": {"loudness": -0.7, "valence": 0.2},
            "loud": {"loudness": 0.6, "energy": 0.5}
        }

        adjustments = {
            "energy": 0, "danceability": 0, "valence": 0, "acousticness": 0,
            "instrumentalness": 0, "speechiness": 0, "liveness": 0, "loudness": 0, "tempo": 0
        }

        for keyword, values in mood_rules.items():
            if keyword in msg:
                for key, val in values.items():
                    adjustments[key] = adjustments.get(key, 0) + val

        for key in adjustments:
            adjustments[key] = np.clip(adjustments[key], -1, 1)

        return adjustments


# ===================== FLASK APP =====================

app = Flask(__name__)
CORS(app)

# Global state
recsys = None
llm = None
current_playlist = []
excluded_tracks = set()
df_raw = None
model_trained = False


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Spotify Mood Mixer API is running'})


@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Upload CSV file"""
    global df_raw

    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400

        # Read CSV
        try:
            df = pd.read_csv(io.BytesIO(file.read()))
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Invalid CSV file: {str(e)}'}), 400

        df_raw = df
        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist()
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/clean-data', methods=['POST'])
def clean_data_endpoint():
    """Clean and validate data"""
    global df_raw

    try:
        if df_raw is None:
            return jsonify({'status': 'error', 'message': 'No data uploaded'}), 400

        df_clean, stats = DataCleaner.validate_and_clean(df_raw)

        return jsonify({
            'status': 'success',
            'message': 'Data cleaned successfully',
            'stats': stats,
            'preview': df_clean.head(5).to_dict('records')
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train recommendation model"""
    global recsys, llm, df_raw, model_trained, current_playlist, excluded_tracks

    try:
        if df_raw is None:
            return jsonify({'status': 'error', 'message': 'No data available'}), 400

        # Clean data
        df_clean, _ = DataCleaner.validate_and_clean(df_raw)

        if len(df_clean) == 0:
            return jsonify({'status': 'error', 'message': 'No valid data after cleaning'}), 400

        # Initialize RecSys
        recsys = SpotifyRecSysConstrained(df_clean)

        # Initialize LLM
        api_key = request.json.get('api_key', '') if request.json else ''
        llm = HuggingFaceLLM(api_key)

        # Reset playlist
        current_playlist = recsys.get_random_tracks(5)
        excluded_tracks = set(current_playlist)

        model_trained = True

        playlist_info = [
            {
                'name': recsys.get_track_info(idx)['name'],
                'artist': recsys.get_track_info(idx)['artist'],
                'index': idx
            }
            for idx in current_playlist
        ]

        return jsonify({
            'status': 'success',
            'message': f'Model trained on {len(df_clean)} tracks',
            'track_count': len(df_clean),
            'playlist': playlist_info
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    global recsys, llm, current_playlist, excluded_tracks

    if recsys is None or llm is None or not model_trained:
        return jsonify({
            'status': 'error',
            'message': 'Model not trained. Please train model first.'
        }), 400

    try:
        data = request.json
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'status': 'error', 'message': 'Message cannot be empty'}), 400

        adjustments = llm.extract_adjustments(user_message)
        base_mood = recsys.get_playlist_mood_vector(current_playlist)
        adjusted_mood = recsys.adjust_mood_vector_with_constraints(base_mood, adjustments)

        exclude_list = list(excluded_tracks)
        recommendations = recsys.recommend_by_mood_vector(
            adjusted_mood,
            n_recommendations=10,
            exclude_indices=exclude_list
        )

        formatted_recommendations = []
        for track_idx, similarity in recommendations:
            track_info = recsys.get_track_info(track_idx)
            formatted_recommendations.append({
                'name': track_info['name'],
                'artist': track_info['artist'],
                'genre': track_info['genre'],
                'popularity': track_info['popularity'],
                'similarity': round(float(similarity), 3),
                'index': track_idx
            })
            excluded_tracks.add(track_idx)

        return jsonify({
            'status': 'success',
            'interpretation': f'Transforming playlist: {user_message}',
            'adjustments': adjustments,
            'recommendations': formatted_recommendations
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/playlist', methods=['GET'])
def get_playlist():
    global recsys, current_playlist

    if recsys is None:
        return jsonify({'status': 'error', 'message': 'System not initialized'}), 400

    playlist_info = [
        {
            'name': recsys.get_track_info(idx)['name'],
            'artist': recsys.get_track_info(idx)['artist'],
            'index': idx
        }
        for idx in current_playlist
    ]

    return jsonify({
        'status': 'success',
        'playlist': playlist_info,
        'count': len(current_playlist)
    })


@app.route('/api/add-tracks', methods=['POST'])
def add_tracks():
    global current_playlist

    try:
        data = request.json
        indices = data.get('indices', [])
        current_playlist.extend(indices)

        return jsonify({
            'status': 'success',
            'message': f'Added {len(indices)} tracks',
            'playlist_size': len(current_playlist)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    global recsys, current_playlist, excluded_tracks

    if recsys is None:
        return jsonify({'status': 'error', 'message': 'System not initialized'}), 400

    current_playlist = recsys.get_random_tracks(5)
    excluded_tracks = set(current_playlist)

    playlist_info = [
        {
            'name': recsys.get_track_info(idx)['name'],
            'artist': recsys.get_track_info(idx)['artist'],
            'index': idx
        }
        for idx in current_playlist
    ]

    return jsonify({
        'status': 'success',
        'playlist': playlist_info
    })


@app.route('/api/clear-playlist', methods=['POST'])
def clear_playlist():
    global current_playlist, excluded_tracks

    current_playlist = []
    excluded_tracks = set()

    return jsonify({'status': 'success', 'message': 'Playlist cleared'})


@app.route('/api/model-status', methods=['GET'])
def model_status():
    global model_trained, recsys

    return jsonify({
        'status': 'success',
        'trained': model_trained,
        'track_count': len(recsys.df) if recsys else 0
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
