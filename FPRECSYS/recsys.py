import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

class SpotifyRecSysConstrained:
    """Content-based recommendation engine with feature constraints"""
    
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
        """Normalize features to consistent ranges"""
        if 'loudness' in self.df.columns:
            min_loudness = self.df['loudness'].min()
            max_loudness = self.df['loudness'].max()
            self.df['loudness_norm'] = (self.df['loudness'] - min_loudness) / (max_loudness - min_loudness)
        
        if 'tempo' in self.df.columns:
            min_tempo = self.df['tempo'].min()
            max_tempo = self.df['tempo'].max()
            self.df['tempo_norm'] = (self.df['tempo'] - min_tempo) / (max_tempo - min_tempo)
    
    def _build_feature_matrix(self) -> np.ndarray:
        """Build feature matrix from available features"""
        features = []
        for col in self.feature_cols:
            if col in self.df.columns:
                features.append(self.df[col].values)
            elif col == 'loudness' and 'loudness_norm' in self.df.columns:
                features.append(self.df['loudness_norm'].values)
            elif col == 'tempo' and 'tempo_norm' in self.df.columns:
                features.append(self.df['tempo_norm'].values)
        
        return np.column_stack(features) if features else np.array([])
    
    def get_playlist_mood_vector(self, track_indices: List[int]) -> np.ndarray:
        """Get average feature vector for a playlist"""
        if not track_indices or len(track_indices) == 0:
            return np.zeros(len(self.feature_cols))
        return self.feature_matrix_scaled[track_indices].mean(axis=0)
    
    def adjust_mood_vector_with_constraints(self, base_vector: np.ndarray, adjustments: dict) -> np.ndarray:
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
    
    def recommend_by_mood_vector(self, mood_vector: np.ndarray, n_recommendations: int = 10, 
                                 exclude_indices: List[int] = None) -> List[Tuple[int, float]]:
        """Recommend tracks based on specific mood vector"""
        if exclude_indices is None:
            exclude_indices = []
        
        similarities = cosine_similarity([mood_vector], self.feature_matrix_scaled)[0]
        
        for idx in exclude_indices:
            if idx < len(similarities):
                similarities[idx] = -1
        
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        return [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0]
    
    def get_track_info(self, index: int) -> dict:
        """Get track metadata"""
        row = self.df.iloc[index]
        return {
            'name': row.get('track_name', 'Unknown'),
            'artist': row.get('artist_name', 'Unknown'),
            'genre': row.get('genre', 'Unknown'),
            'popularity': int(row.get('popularity', 0)),
            'index': index
        }
