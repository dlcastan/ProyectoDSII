import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

image = Image.open('coderhouse.png')
st.image(image, use_container_width=True)

@st.cache_resource
def load_and_prepare_data(csv_file_path):
    try:
        df_subset = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        st.error(f"El archivo '{csv_file_path}' no se encuentra en el directorio.")
        return None

    df_subset['combined_features'] = (
        df_subset['title'] + " " + df_subset['author'] + " " + df_subset['genres']
    )

    df_subset['combined_features'] = df_subset['combined_features'].fillna("")

    df_subset['combined_features'] = df_subset['combined_features'].astype(str)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_subset['combined_features'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df_subset, cosine_sim

def recommend_books(title, df_subset, cosine_sim):
    if title not in df_subset['title'].values:
        return []

    idx = df_subset[df_subset['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]

    return df_subset['title'].iloc[book_indices].tolist()

csv_file_path = "data_model_recomendador.csv"

df_subset, cosine_sim = load_and_prepare_data(csv_file_path)

if df_subset is not None and cosine_sim is not None:

    st.title("Recomendador de Libros")
    st.write("Introduce el título de un libro y te recomendaremos otros similares.")

    selected_book = st.selectbox("Selecciona un libro", options=df_subset['title'].tolist())

    if st.button("Recomendar Libros"):
        recommendations = recommend_books(selected_book, df_subset, cosine_sim)

        if recommendations:
            st.write(f"Si te gustó '{selected_book}', también te podrían gustar:")
            for i, book in enumerate(recommendations, start=1):
                st.write(f"{i}. {book}")
        else:
            st.warning("No se encontraron recomendaciones para este título.")
