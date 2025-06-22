import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

# Load models and data
tfidf_matrix = joblib.load("cbf_tfidf_matrix.pkl")
tfidf_vectorizer = joblib.load("cbf_tfidf_vectorizer.pkl")
book_titles = joblib.load("book_titles.pkl")

# Create similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=book_titles, columns=book_titles)

# Evaluation metrics
threshold = 0.5
ground_truth = np.where(cosine_sim >= threshold, 1, 0)

sample_size = min(1000, cosine_sim.shape[0])  # avoid memory issues
cosine_sim_sample = cosine_sim[:sample_size, :sample_size]
ground_truth_sample = ground_truth[:sample_size, :sample_size]

cosine_sim_flat = cosine_sim_sample.flatten()
ground_truth_flat = ground_truth_sample.flatten()

predictions = (cosine_sim_flat >= threshold).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(
    ground_truth_flat, predictions, average='binary', zero_division=1
)

# Recommender function
def book_recommendation(book_title, k=5):
    if book_title not in cosine_sim_df.columns:
        return pd.DataFrame({'Recommended Books': ["Judul tidak ditemukan."]})
    index = cosine_sim_df.loc[:, book_title].to_numpy().argpartition(range(-1, -k-1, -1))
    closest = cosine_sim_df.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(book_title, errors='ignore')
    return pd.DataFrame({'Recommended Books': closest[:k]})

# Streamlit Interface
st.set_page_config(page_title="CBF Book Recommender", layout="centered")
st.title("📚 Content-Based Book Recommender")

book_input = st.selectbox("Pilih judul buku:", options=book_titles)

if st.button("Rekomendasikan"):
    st.subheader("📖 Rekomendasi Buku:")
    st.table(book_recommendation(book_input))

st.subheader("📊 Evaluasi Model:")
st.markdown(f"**Precision:** {precision:.4f}")
st.markdown(f"**Recall:** {recall:.4f}")
st.markdown(f"**F1-Score:** {f1:.4f}")