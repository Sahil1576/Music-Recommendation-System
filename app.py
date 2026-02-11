import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.song-card {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 15px;
    transition: 0.2s ease-in-out;
}

.song-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

.song-title {
    font-size: 16px;
    font-weight: 600;
    color: #111827;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.song-meta {
    font-size: 13px;
    color: #6b7280;
}

.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("Final_dataset.csv")

@st.cache_resource
def load_pickle():
    indices = joblib.load("Indices.pkl")
    tfidf_matrix = joblib.load("TF-IDF_Metrix.pkl")
    return indices, tfidf_matrix

df = load_data()
indices, tfidf_matrix = load_pickle()

# ---------------- HEADER ----------------
st.title("🎵 Music Recommendation System")
st.caption("Find songs similar to your favorite track")

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Settings")
num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    5, 20, 10
)

# ---------------- SONG SELECT ----------------
song_list = df['Song Title'].values
selected_song = st.selectbox("🔎 Select a Song", song_list)

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(song):
    idx = indices[song]

    similarity_scores = list(enumerate(
        cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    ))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:num_recommendations+1]

    song_indices = [i[0] for i in similarity_scores]
    return df.iloc[song_indices]

# ---------------- BUTTON ----------------
if st.button("🎧 Recommend Songs"):
    recommendations = recommend(selected_song)

    st.markdown('<div class="section-title">✨ Recommended Songs</div>', unsafe_allow_html=True)

    cols = st.columns(3)  # 3 cards per row

    for index, row in enumerate(recommendations.iterrows()):
        i, song = row

        card_html = f"""
        <div class="song-card">
            <div class="song-title">🎵 {song['Song Title']}</div>
            <div class="song-meta">🎤 Artist: {song.get('Artist','Unknown')}</div>
            <div class="song-meta">💿 Album: {song.get('Album','Unknown')}</div>
        </div>
        """

        cols[index % 3].markdown(card_html, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.divider()
st.caption("Made with ❤️ by Sahil")