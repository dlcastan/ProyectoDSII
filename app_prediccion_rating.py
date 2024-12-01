import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from PIL import Image

image = Image.open('coderhouse.png')
st.image(image, use_container_width=True)

# Función para cargar y entrenar el modelo
@st.cache_resource
def train_model():
    try:
        data_clean = pd.read_csv("data_model.csv")  # Asegúrate de que este archivo exista en el mismo directorio
    except FileNotFoundError:
        st.error("El archivo 'data_model.csv' no se encuentra en el directorio.")
        return None

    # Preparar los datos
    X = data_clean[['pages', 'likedPercent', 'con_serie', 'con_premio',
                    'cant_stars_5', 'cant_stars_4', 'cant_stars_3', 'cant_stars_2',
                    'cant_stars_1', 'ficcion', 'price']]
    y = data_clean['Rating_entero']

    # Dividir el dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

# Cargar el modelo
model = train_model()

# Verificar que el modelo se haya cargado correctamente
if model is None:
    st.stop()

# Función para hacer la predicción
def predict_rating(pages, likedPercent, con_serie, con_premio, cant_stars_5,
                   cant_stars_4, cant_stars_3, cant_stars_2, cant_stars_1, ficcion, price):
    input_data = np.array([[pages, likedPercent, con_serie, con_premio,
                            cant_stars_5, cant_stars_4, cant_stars_3, cant_stars_2,
                            cant_stars_1, ficcion, price]])
    prediction = model.predict(input_data)
    return prediction[0]

# Interfaz de usuario con Streamlit
st.title("Predicción de Rating")
st.write("Introduce las características del libro para predecir su rating.")

# Crear los campos de entrada
pages = st.number_input("Páginas", min_value=0, value=100, step=1)
likedPercent = st.number_input("Porcentaje de Likes", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
con_serie = st.selectbox("¿Es parte de una serie?", [0, 1])
con_premio = st.selectbox("¿Tiene premio?", [0, 1])
cant_stars_5 = st.number_input("Cantidad de estrellas 5", min_value=0, value=0, step=1)
cant_stars_4 = st.number_input("Cantidad de estrellas 4", min_value=0, value=0, step=1)
cant_stars_3 = st.number_input("Cantidad de estrellas 3", min_value=0, value=0, step=1)
cant_stars_2 = st.number_input("Cantidad de estrellas 2", min_value=0, value=0, step=1)
cant_stars_1 = st.number_input("Cantidad de estrellas 1", min_value=0, value=0, step=1)
ficcion = st.selectbox("¿Es ficción?", [0, 1])
price = st.number_input("Precio", min_value=0.0, value=10.0, step=0.1)

# Botón para predecir
if st.button("Predecir Rating"):
    prediction = predict_rating(
        pages, likedPercent, con_serie, con_premio,
        cant_stars_5, cant_stars_4, cant_stars_3,
        cant_stars_2, cant_stars_1, ficcion, price
    )
    st.success(f"El rating predicho es: {prediction:.2f}")
