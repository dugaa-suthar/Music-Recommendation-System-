
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TuneMatch | Music Discovery",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #100c24, #17132f, #21143a);
    color: white;
}
[data-testid="stSidebar"] {
    background: #100d20;
    border-right: 1px solid #39275c;
}
h1, h2, h3, p, label {
    color: white !important;
}
.hero {
    background: linear-gradient(120deg, #7138db, #b23acb, #e34f91);
    padding: 35px;
    border-radius: 22px;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}
.hero p {
    font-size: 17px;
    opacity: 0.9;
}
.metric-card {
    background: #241b3d;
    border: 1px solid #49336b;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #d9a7ff;
}
.metric-label {
    color: #c4b5d9;
    font-size: 14px;
}
.song-card {
    background: linear-gradient(145deg, #2a2044, #1b1730);
    border: 1px solid #49336b;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 12px;
}
.song-title {
    font-size: 18px;
    font-weight: bold;
    color: white;
}
.song-artist {
    color: #c7b5df;
    font-size: 14px;
}
div.stButton > button {
    background: linear-gradient(90deg, #853ee8, #c13bd0);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: bold;
    width: 100%;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #a45cff, #e65be0);
    color: white;
    border: none;
}
.stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: #251d3b;
    color: white;
}
hr {
    border-color: #49336b;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("spotify-tracks-dataset.csv")

    # Remove unwanted unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Remove rows without song or artist information
    df = df.dropna(subset=["track_name", "artists"])

    # Remove duplicate songs
    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"])
    else:
        df = df.drop_duplicates(subset=["track_name", "artists"])

    return df.reset_index(drop=True)


# ---------------- PREPARE ML MODEL ----------------
@st.cache_data
def prepare_features(df):

    features = [
        "danceability",
        "energy",
        "acousticness",
        "valence",
        "tempo",
        "loudness",
        "speechiness",
        "instrumentalness",
        "liveness"
    ]

    # Check that required columns exist
    missing_features = [f for f in features if f not in df.columns]

    if missing_features:
        raise ValueError(
            f"Missing audio features: {missing_features}"
        )

    # Convert audio features to numeric
    data = df[features].apply(pd.to_numeric, errors="coerce")

    # Handle missing values
    data = data.fillna(data.median()).fillna(0)

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)

    # Apply K-Means clustering
    kmeans = KMeans(
        n_clusters=min(10, len(df)),
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(scaled_features)

    return scaled_features, cluster_labels


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🎧 TuneMatch")
    st.caption("Your personal music discovery platform")
    st.markdown("---")

    st.markdown("### 🏠 Navigation")
    st.markdown("🎵 Music Discovery")
    st.markdown("💜 Recommendations")
    st.markdown("📊 Music Insights")

    st.markdown("---")
    st.markdown("### ✨ About")

    st.write(
        "TuneMatch uses K-Means Clustering, "
        "StandardScaler and Cosine Similarity "
        "to recommend songs based on audio features."
    )

    st.markdown("---")
    st.caption("Powered by Python & Machine Learning")


# ---------------- MAIN DASHBOARD ----------------
st.markdown("""
<div class="hero">
    <h1>🎵 Discover Your Next Favorite Song</h1>
    <p>Explore music that matches your taste with Machine Learning.</p>
</div>
""", unsafe_allow_html=True)


try:

    # Load dataset
    df = load_data()

    # Prepare ML features and clusters
    scaled_features, cluster_labels = prepare_features(df)

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">🎵 Total Songs</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        artist_count = df["artists"].nunique()

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{artist_count:,}</div>
            <div class="metric-label">🎤 Artists</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        genres = (
            df["track_genre"].nunique()
            if "track_genre" in df.columns else 0
        )

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{genres:,}</div>
            <div class="metric-label">🎼 Music Genres</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- SONG SEARCH ----------------
    st.markdown("## 🔍 Find Music You Love")
    st.write("Search for a song and discover similar music.")

    search = st.text_input(
        "Search song or artist",
        placeholder="Example: Comedy or Gen Hoshino"
    )

    if search:
        matches = df[
            df["track_name"].str.contains(
                search, case=False, na=False
            )
            | df["artists"].str.contains(
                search, case=False, na=False
            )
        ].head(100)
    else:
        matches = df.head(100)

    if not matches.empty:

        options = matches.index.tolist()

        selected_index = st.selectbox(
            "Choose your song",
            options,
            format_func=lambda i: (
                f"{df.loc[i, 'track_name']} — "
                f"{df.loc[i, 'artists']}"
            )
        )

        number = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=10,
            value=5
        )

        # ---------------- RECOMMENDATION ----------------
        if st.button("✨ Discover Similar Songs"):

            # Identify selected song's cluster
            selected_cluster = cluster_labels[selected_index]

            # Get selected song's audio features
            selected_vector = scaled_features[
                selected_index
            ].reshape(1, -1)

            # Calculate cosine similarity
            similarities = cosine_similarity(
                selected_vector,
                scaled_features
            )[0]

            # Filter songs belonging to the same cluster
            same_cluster = cluster_labels == selected_cluster

            similarities[~same_cluster] = -1

            # Exclude selected song
            similarities[selected_index] = -1

            # Get top similar songs
            top_indices = np.argsort(
                similarities
            )[::-1][:number]

            # Remove invalid recommendations
            top_indices = [
                idx for idx in top_indices
                if similarities[idx] >= 0
            ]

            st.markdown("---")
            st.markdown("## 💜 Recommended For You")

            st.info(
                f"Selected song belongs to "
                f"Cluster {selected_cluster + 1}"
            )

            if len(top_indices) == 0:
                st.warning(
                    "No similar songs found in this cluster."
                )

            # Display each recommendation
            for rank, idx in enumerate(
                top_indices, start=1
            ):

                song = df.iloc[idx]

                genre = song.get(
                    "track_genre", "Unknown"
                )

                score = similarities[idx] * 100

                st.markdown(f"""
                <div class="song-card">
                    <div class="song-title">
                        {rank}. 🎵 {song['track_name']}
                    </div>
                    <div class="song-artist">
                        🎤 {song['artists']}
                    </div>
                    <div class="song-artist">
                        🎼 Genre: {genre}
                        &nbsp; | &nbsp;
                        💜 Similarity: {score:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Spotify music player
                if (
                    "track_id" in df.columns
                    and pd.notna(song["track_id"])
                ):

                    track_id = str(
                        song["track_id"]
                    ).strip()

                    components.iframe(
                        f"https://open.spotify.com/embed/track/{track_id}",
                        height=152,
                        scrolling=False
                    )

                else:
                    st.info(
                        "Spotify track ID is not available."
                    )

    else:
        st.warning(
            "No songs found. Try another song or artist."
        )

    # ---------------- DATASET INSIGHTS ----------------
    st.markdown("---")
    st.markdown("## 📊 Music Insights")

    if "track_genre" in df.columns:

        genre_counts = (
            df["track_genre"].value_counts().head(10)
        )

        st.markdown("### Top 10 Genres in Dataset")
        st.bar_chart(genre_counts)

    st.markdown("---")

    st.caption(
        "TuneMatch | Music Recommendation System Using Machine Learning"
    )


# ---------------- ERROR HANDLING ----------------
except FileNotFoundError:

    st.error(
        "Dataset not found! Make sure "
        "'spotify-tracks-dataset.csv' is in the same "
        "folder as song.py."
    )

except Exception as e:

    st.error(f"Something went wrong: {e}")
