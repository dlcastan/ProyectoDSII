import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

image = Image.open('coderhouse.png')
st.image(image, use_container_width=True)

# Cargar datos y preprocesar
@st.cache_resource
def load_and_prepare_data(csv_file_path):
    try:
        df_subset = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        st.error(f"El archivo '{csv_file_path}' no se encuentra en el directorio.")
        return None

    # Combinar texto relevante para cada libro
    df_subset['combined_features'] = (
        df_subset['title'] + " " + df_subset['author'] + " " + df_subset['genres']
    )

    # Reemplazar valores NaN con una cadena vacía
    df_subset['combined_features'] = df_subset['combined_features'].fillna("")

    # Confirmar que todos los valores sean cadenas
    df_subset['combined_features'] = df_subset['combined_features'].astype(str)

    # Vectorizar texto combinado
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_subset['combined_features'])

    # Calcular similitud de coseno
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df_subset, cosine_sim

# Función para recomendar libros
def recommend_books(title, df_subset, cosine_sim):
    if title not in df_subset['title'].values:
        return []

    # Encontrar el índice del libro dado
    idx = df_subset[df_subset['title'] == title].index[0]

    # Obtener las puntuaciones de similitud para ese libro
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar libros por similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Seleccionar los 5 libros más similares (excluyendo el propio libro)
    sim_scores = sim_scores[1:6]

    # Obtener los índices de los libros recomendados
    book_indices = [i[0] for i in sim_scores]

    # Retornar los títulos de los libros recomendados
    return df_subset['title'].iloc[book_indices].tolist()

# Archivo CSV
csv_file_path = "data_model_recomendador.csv"

# Cargar datos
df_subset, cosine_sim = load_and_prepare_data(csv_file_path)

if df_subset is not None and cosine_sim is not None:
    # Interfaz de usuario
    st.title("Recomendador de Libros")
    st.write("Introduce el título de un libro y te recomendaremos otros similares.")

    # Selección de libro
    selected_book = st.selectbox("Selecciona un libro", options=df_subset['title'].tolist())

    # Botón para obtener recomendaciones
    if st.button("Recomendar Libros"):
        recommendations = recommend_books(selected_book, df_subset, cosine_sim)

        if recommendations:
            st.write(f"Si te gustó '{selected_book}', también te podrían gustar:")
            for i, book in enumerate(recommendations, start=1):
                st.write(f"{i}. {book}")
        else:
            st.warning("No se encontraron recomendaciones para este título.")
