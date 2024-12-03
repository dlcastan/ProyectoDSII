import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from PIL import Image

image = Image.open('coderhouse.png')
st.image(image, use_container_width=True)

@st.cache_resource
def train_model():
    try:
        data_clean = pd.read_csv("data_model.csv")
    except FileNotFoundError:
        st.error("El archivo 'data_model.csv' no se encuentra en el directorio.")
        return None

    X = data_clean[['pages', 'likedPercent', 'con_serie', 'con_premio',
                    'cant_stars_5', 'cant_stars_4', 'cant_stars_3', 'cant_stars_2',
                    'cant_stars_1', 'ficcion']]
    y = data_clean['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

model = train_model()

if model is None:
    st.stop()

def predict_price(pages, likedPercent, con_serie, con_premio, cant_stars_5,
                  cant_stars_4, cant_stars_3, cant_stars_2, cant_stars_1, ficcion):
    input_data = np.array([[pages, likedPercent, con_serie, con_premio,
                            cant_stars_5, cant_stars_4, cant_stars_3, cant_stars_2,
                            cant_stars_1, ficcion]])
    prediction = model.predict(input_data)
    return prediction[0]

st.title("Predicción de Precio")
st.write("Introduce las características del libro para predecir su precio.")

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

if st.button("Predecir Precio"):
    prediction = predict_price(
        pages, likedPercent, con_serie, con_premio,
        cant_stars_5, cant_stars_4, cant_stars_3,
        cant_stars_2, cant_stars_1, ficcion
    )
    st.success(f"El precio predicho es: ${prediction:.2f}")
