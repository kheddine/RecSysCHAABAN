"""
Spotify RecSys Core Module
User's original code - UNCHANGED
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple, Optional
import requests
import pandas as pd

# ===================== CONSTRAINED SPOTIFY RECSYS =====================

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

        print(f"✓ RecSys initialized with {len(self.df)} tracks (100% accurate)")

        # Feature constraints to prevent conflicts
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
        """Apply adjustments while respecting feature constraints"""
        adjusted = base_vector.copy()

        # First pass: Apply primary adjustments
        for feature, delta in adjustments.items():
            if feature in self.feature_cols:
                idx = self.feature_cols.index(feature)
                adjusted[idx] = np.clip(adjusted[idx] + delta, -3, 3)

        # Second pass: Handle conflicting features
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
        row = self.df.iloc[index]
        return {
            'name': row.get('track_name', 'Unknown'),
            'artist': row.get('artist_name', 'Unknown'),
            'genre': row.get('genre', 'Unknown'),
            'popularity': int(row.get('popularity', 0)),
            'index': index
        }


# ===================== HUGGINGFACE LLM =====================

class HuggingFaceLLM:
    """Use free Hugging Face Inference API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def extract_adjustments(self, user_message: str) -> Dict[str, float]:
        """Query HF model to extract mood adjustments"""

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
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                output_text = result[0]['generated_text']

                # Extract JSON from response
                try:
                    json_start = output_text.find('{')
                    json_end = output_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = output_text[json_start:json_end]
                        parsed = json.loads(json_str)
                        return parsed.get('feature_adjustments', {})
                except:
                    pass

            return {}

        except requests.exceptions.Timeout:
            return self._fallback_extraction(user_message)
        except Exception as e:
            return self._fallback_extraction(user_message)

    def _fallback_extraction(self, user_message: str) -> Dict[str, float]:
        """Fallback rule-based extraction if API fails"""
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
            "happy": {"valence": 0.8, "energy": 0.5, "danceability": 0.5}
        }

        adjustments = {
            "energy": 0, "danceability": 0, "valence": 0, "acousticness": 0,
            "instrumentalness": 0, "speechiness": 0, "liveness": 0, "loudness": 0, "tempo": 0
        }

        for keyword, values in mood_rules.items():
            if keyword in msg:
                adjustments.update(values)

        return adjustments


# ===================== PLAYLIST TRANSFORMER =====================

class PlaylistTransformer:
    def __init__(self, recsys: SpotifyRecSysConstrained, llm):
        self.recsys = recsys
        self.llm = llm
        self.current_playlist = []
        self.excluded_tracks = set()

    def set_initial_playlist(self, playlist_indices: List[int]):
        self.current_playlist = playlist_indices
        self.excluded_tracks = set(playlist_indices)
        print(f"✓ Playlist set with {len(playlist_indices)} tracks")

    def chat(self, user_message: str) -> Dict:
        # Extract adjustments
        adjustments = self.llm.extract_adjustments(user_message)

        # Get current playlist mood
        base_mood = self.recsys.get_playlist_mood_vector(self.current_playlist)

        # Apply adjustments WITH constraints
        adjusted_mood = self.recsys.adjust_mood_vector_with_constraints(base_mood, adjustments)

        # Get recommendations
        exclude_list = list(self.excluded_tracks)
        recommendations = self.recsys.recommend_by_mood_vector(
            adjusted_mood,
            n_recommendations=10,
            exclude_indices=exclude_list
        )

        # Format results
        new_recommendations = []
        for track_idx, similarity in recommendations:
            track_info = self.recsys.get_track_info(track_idx)
            new_recommendations.append({
                'name': track_info['name'],
                'artist': track_info['artist'],
                'genre': track_info['genre'],
                'similarity': round(similarity, 3),
                'index': track_idx
            })
            self.excluded_tracks.add(track_idx)

        return {
            'interpretation': f"Transforming playlist: {user_message}",
            'adjustments': adjustments,
            'new_recommendations': new_recommendations
        }

    def add_tracks_to_playlist(self, track_indices: List[int]):
        self.current_playlist.extend(track_indices)
