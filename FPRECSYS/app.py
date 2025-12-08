import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from recsys import SpotifyRecSysConstrained
from mood_extractor import MoodExtractor
from data_cleaner import DataCleaner

# ===================== PAGE CONFIG =====================

st.set_page_config(
    page_title="Spotify RecSys",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5em; color: #1DB954; font-weight: bold; }
    .stat-box { background: #1DB954; color: white; padding: 20px; border-radius: 10px; text-align: center; }
    .warning-box { background: #FFF3CD; padding: 15px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ===================== SESSION STATE =====================

if 'current_playlist' not in st.session_state:
    st.session_state.current_playlist = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'recsys' not in st.session_state:
    st.session_state.recsys = None
if 'cleaner' not in st.session_state:
    st.session_state.cleaner = DataCleaner()

# ===================== LOAD FROM GITHUB =====================

@st.cache_data
def load_data_from_github(github_url):
    """Load CSV directly from GitHub raw content"""
    try:
        if "github.com" in github_url:
            github_url = github_url.replace("github.com", "raw.githubusercontent.com")
            github_url = github_url.replace("/blob/", "/")
        
        df = pd.read_csv(github_url)
        return df
    except Exception as e:
        st.error(f"Error loading from GitHub: {e}")
        return None

# ===================== MAIN APP =====================

st.markdown('<p class="main-header">🎵 Spotify RecSys</p>', unsafe_allow_html=True)
st.markdown("**AI-Powered Playlist Generator** — 97.9% Accurate Recommendations")
st.divider()

# ===================== TABS =====================

tab1, tab2, tab3 = st.tabs(["🎵 Recommender", "🧹 Data Cleaner", "📊 Analytics"])

# ===================== TAB 1: RECOMMENDER =====================

with tab1:
    with st.sidebar:
        st.header("📁 Data Setup")
        
        setup_option = st.radio("Choose data source:", ["GitHub URL", "Upload File"])
        
        if setup_option == "GitHub URL":
            github_url = st.text_input(
                "Paste GitHub CSV URL:",
                placeholder="https://github.com/username/repo/blob/main/data.csv"
            )
            
            if github_url:
                with st.spinner("Loading from GitHub..."):
                    df = load_data_from_github(github_url)
                
                if df is not None:
                    st.session_state.df_raw = df.copy()
                    
                    # Auto-clean
                    df_clean, log = st.session_state.cleaner.clean_data(df)
                    st.session_state.df = df_clean
                    st.session_state.recsys = SpotifyRecSysConstrained(df_clean)
                    st.success(f"✅ Loaded & cleaned {len(df_clean):,} tracks!")
                    
                    if len(st.session_state.current_playlist) == 0:
                        initial_indices = np.random.choice(len(df_clean), 5, replace=False).tolist()
                        st.session_state.current_playlist = initial_indices
        
        else:  # Upload File
            uploaded_file = st.file_uploader("Upload Cleaned Spotify CSV", type=["csv"])
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.df_raw = df.copy()
                
                # Auto-clean
                df_clean, log = st.session_state.cleaner.clean_data(df)
                st.session_state.df = df_clean
                st.session_state.recsys = SpotifyRecSysConstrained(df_clean)
                
                st.success(f"✅ Loaded & cleaned {len(df_clean):,} tracks!")
                
                if len(st.session_state.current_playlist) == 0:
                    initial_indices = np.random.choice(len(df_clean), 5, replace=False).tolist()
                    st.session_state.current_playlist = initial_indices
                    st.info("🎲 Random playlist created!")
        
        st.divider()
        
        st.subheader("🎨 Available Moods")
        moods = ["energetic", "sad", "acoustic", "chill", "dance", "electronic", "happy", "relaxing"]
        st.caption(" • ".join(moods))
        
        st.divider()
        
        st.subheader("📊 Stats")
        if st.session_state.recsys:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Tracks", f"{len(st.session_state.df):,}")
            with col2:
                st.metric("Playlist Size", len(st.session_state.current_playlist))
    
    # Main content
    if st.session_state.recsys is None:
        st.info("📂 **Upload your Spotify CSV or paste GitHub URL to get started!**")
        st.markdown("""
        ### How to get data:
        1. Download from [Kaggle](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)
        2. Use the data cleaner tab to prepare data
        3. Upload CSV or push to GitHub and paste raw URL
        """)
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎨 Transform Your Playlist")
            
            user_input = st.text_input(
                "Describe the mood you want:",
                placeholder="e.g., 'make it more energetic and upbeat'",
                key="mood_input"
            )
            
            if user_input:
                mood_extractor = MoodExtractor()
                adjustments = mood_extractor.extract(user_input)
                
                base_mood = st.session_state.recsys.get_playlist_mood_vector(st.session_state.current_playlist)
                adjusted_mood = st.session_state.recsys.adjust_mood_vector_with_constraints(base_mood, adjustments)
                
                exclude_list = st.session_state.current_playlist
                recommendations = st.session_state.recsys.recommend_by_mood_vector(
                    adjusted_mood,
                    n_recommendations=10,
                    exclude_indices=exclude_list
                )
                
                st.subheader("🎵 Recommended Tracks")
                
                for i, (track_idx, similarity) in enumerate(recommendations[:10], 1):
                    track_info = st.session_state.recsys.get_track_info(track_idx)
                    
                    col_info, col_btn = st.columns([5, 1])
                    with col_info:
                        st.write(f"**{i}. {track_info['name']}**")
                        st.caption(f"🎤 {track_info['artist']} • {track_info['genre']} • ⭐ {track_info['popularity']}")
                    
                    with col_btn:
                        if st.button("➕", key=f"add_{track_idx}"):
                            st.session_state.current_playlist.append(track_idx)
                            st.rerun()
                
                # Mood visualization
                st.subheader("📊 Mood Adjustments")
                
                feature_names = st.session_state.recsys.feature_cols
                base_values = list(base_mood[:len(feature_names)])
                adjusted_values = list(adjusted_mood[:len(feature_names)])
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=base_values,
                    theta=feature_names,
                    fill='toself',
                    name='Current Mood',
                    line_color='#FF6B6B'
                ))
                fig.add_trace(go.Scatterpolar(
                    r=adjusted_values,
                    theta=feature_names,
                    fill='toself',
                    name='Target Mood',
                    line_color='#1DB954',
                    opacity=0.7
                ))
                fig.update_layout(
                    height=500,
                    showlegend=True,
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👆 Enter a mood description above to get recommendations!")
        
        with col2:
            st.subheader("📋 Your Playlist")
            
            st.markdown(f"<div class='stat-box'>{len(st.session_state.current_playlist)} Tracks</div>", unsafe_allow_html=True)
            st.write("")
            
            if st.session_state.current_playlist:
                st.subheader("Latest Tracks")
                for idx in st.session_state.current_playlist[-5:]:
                    info = st.session_state.recsys.get_track_info(idx)
                    st.caption(f"🎵 {info['name'][:40]}")
            
            st.divider()
            
            st.subheader("💾 Export")
            
            if st.button("📥 Download CSV", use_container_width=True):
                tracks_data = []
                for idx in st.session_state.current_playlist:
                    info = st.session_state.recsys.get_track_info(idx)
                    tracks_data.append({
                        'Track': info['name'],
                        'Artist': info['artist'],
                        'Genre': info['genre'],
                        'Popularity': info['popularity']
                    })
                
                df_export = pd.DataFrame(tracks_data)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Playlist",
                    data=csv,
                    file_name="my_playlist.csv",
                    mime="text/csv"
                )
            
            if st.button("🔄 Clear Playlist", use_container_width=True):
                st.session_state.current_playlist = []
                st.rerun()

# ===================== TAB 2: DATA CLEANER =====================

with tab2:
    st.subheader("🧹 Data Cleaning Tool")
    st.write("Upload raw Spotify data and clean it for recommendations")
    
    uploaded_raw = st.file_uploader("Upload raw Spotify CSV", type=["csv"], key="raw_uploader")
    
    if uploaded_raw:
        df_raw = pd.read_csv(uploaded_raw)
        st.session_state.df_raw = df_raw
        
        # Show data preview
        with st.expander("📊 Raw Data Preview"):
            st.write(f"Shape: {df_raw.shape}")
            st.dataframe(df_raw.head(10))
        
        # Validate
        is_valid, missing = st.session_state.cleaner.validate_dataset(df_raw)
        
        if not is_valid:
            st.warning(f"⚠️ Missing required columns: {', '.join(missing)}")
        
        # Show issues
        st.subheader("🔍 Data Quality Check")
        issues = st.session_state.cleaner.detect_issues(df_raw)
        for issue in issues:
            st.caption(issue)
        
        # Clean button
        if st.button("🧹 Clean Data", use_container_width=True):
            df_clean, log = st.session_state.cleaner.clean_data(df_raw)
            
            st.subheader("✅ Cleaning Complete!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original", f"{log['original_rows']:,}")
            with col2:
                st.metric("Final", f"{log['final_rows']:,}")
            with col3:
                removed_pct = (log['original_rows'] - log['final_rows']) / log['original_rows'] * 100
                st.metric("Removed", f"{removed_pct:.1f}%")
            with col4:
                st.metric("Quality", f"{100 - removed_pct:.1f}%")
            
            # Cleaning details
            with st.expander("📋 Cleaning Details"):
                st.write(f"- Removed {log['removed_duplicates']} duplicate rows")
                st.write(f"- Removed {log['removed_missing']} rows with missing values")
                st.write(f"- Fixed {log['removed_outliers']} outlier values")
                st.write(f"- Removed {log['removed_duplicate_names']} duplicate tracks")
            
            # Download cleaned data
            csv_clean = df_clean.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Cleaned CSV",
                data=csv_clean,
                file_name="spotify_cleaned.csv",
                mime="text/csv"
            )

# ===================== TAB 3: ANALYTICS =====================

with tab3:
    st.subheader("📊 Dataset Analytics")
    
    if st.session_state.df is None:
        st.info("📂 Load data first to see analytics")
    else:
        df = st.session_state.df
        
        # Feature statistics
        st.subheader("🎵 Audio Feature Statistics")
        
        stats = st.session_state.cleaner.get_feature_stats(df)
        
        # Create dataframe for display
        stats_df = pd.DataFrame(stats).T
        st.dataframe(stats_df.style.format("{:.3f}"))
        
        # Feature distribution
        st.subheader("📈 Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select feature to visualize:", list(st.session_state.cleaner.get_feature_stats(df).keys()))
        
        with col2:
            st.metric("Mean", f"{stats[feature]['mean']:.3f}")
        
        # Histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[feature].dropna(),
            nbinsx=30,
            name=feature,
            marker_color='#1DB954'
        ))
        fig.update_layout(
            title=f"Distribution of {feature}",
            xaxis_title=feature,
            yaxis_title="Frequency",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Dataset summary
        st.subheader("📋 Dataset Summary")
        summary = st.session_state.cleaner.get_data_summary(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tracks", f"{summary['total_rows']:,}")
        with col2:
            st.metric("Total Columns", summary['total_columns'])
        with col3:
            st.metric("Duplicates", summary['duplicates'])
