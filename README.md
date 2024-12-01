# ProyectoDSII

Data Science II: Machine Learning para la Ciencia de Datos

Comisión: 61175

Alumno: Diego Lopez Castan


# Abstract

Este conjunto de datos contiene información detallada sobre libros en el site goodreads.com. Cada registro incluye datos como el nombre el libro y el autor, el precio, las opiniones y algunos datos más.


# Objetivo

**Analizar las relaciones entre el rating de un libro y sus géneros:**
Identificar si ciertos géneros están asociados con mejores ratings en general.

**Estudio de popularidad por series literarias:** Evaluar el impacto de pertenecer a una serie en el éxito de los libros (por ejemplo, comparar el rating de los libros de una serie con otros libros del mismo autor que no pertenecen a una serie).

**Segmentación por idioma y género:** Estudiar la distribución de géneros según los diferentes idiomas de los libros y su relación con los ratings.

**Relación entre la longitud del título y el éxito de un libro:** Investigar si los libros con títulos más cortos o más largos tienen alguna ventaja en términos de popularidad (rating).

**Estudio de la relación entre el número de géneros y el rating de un libro:** Analizar si los libros que pertenecen a un genero específico recibe mejor rating.


# Campos del dataset

**title:** El título del libro. (Cadena de texto)

**series:** La serie a la que pertenece el libro, si es que pertenece a alguna. (Cadena de texto)

**author:** El autor del libro. (Cadena de texto)

**rating:** La calificación promedio del libro en GoodReads. (Flotante)

**description:** Una breve descripción del libro. (Cadena de texto)

**language:** El idioma en el que está escrito el libro. (Cadena de texto)

**isbn:** El número ISBN del libro. (Cadena de texto)

**genres:** Los géneros en los que se clasifica el libro. (Cadena de texto)

**characters:** Los personajes que aparecen en el libro. (Cadena de texto)

**bookFormat:** El formato del libro (por ejemplo, tapa blanda o libro electrónico). (Cadena de texto)

**edition:** La edición del libro. (Cadena de texto)

**pages:** El número de páginas del libro. (Entero)

**publisher:** El editor del libro. (Cadena de texto)

**publishDate:** La fecha en la que el libro fue publicado. (Fecha)

**firstPublishDate:** La fecha en la que el libro fue publicado por primera vez. (Fecha)

**awards:** Cualquier premio que el libro haya recibido. (Cadena de texto)

**numRatings:** El número de calificaciones que ha recibido el libro. (Entero)

**ratingsByStars:** Las calificaciones del libro desglosadas por número de estrellas. (Cadena de texto)

**likedPercent:** El porcentaje de lectores a los que les gustó el libro. (Flotante)

**setting:** El escenario en el que se desarrolla el libro. (Cadena de texto)

**coverImg:** La imagen de portada del libro. (Cadena de texto)

**bbeScore** La puntuación del libro en la lista de los mejores libros de GoodReads. (Flotante)

**bbeVotes:** El número de votos emitidos para el libro en la lista de los mejores libros de GoodReads. (Entero)

**price:** El precio del libro. (Flotante)

# Link del dataset
https://www.kaggle.com/datasets/thedevastator/comprehensive-overview-of-52478-goodreads-best-b



# Programas creados con Streamlit

He creado tres programas que se pueden correr en Streamlit, los he creados con el algoritmo **LinearRegression** ya que el mejor algoritmo que he encontrado para predecir el LGBMRegressor no funciona correctamente con esta aplicación. Los programas creados son tres:

**app_prediccion_precios.py:** Predictor de precio.
**app_prediccion_rating.py:** Predictor de rating.
**app_recomendador.py:** Recomendador de otros títulos de libros.


## Instalación
Dependencias que tenes que tener:
```console
pip install streamlit
pip install scikit-learn
```

## Clonar el repositorio
Generar modelos 
```console
git clone https://github.com/dlcastan/DataScience.git
```


## Correr App
Correr app para predecir precio
```console
streamlit run pp_prediccion_precios.py
```

Correr app para predecir rating
```console
streamlit run app_prediccion_rating.py
```

Correr app para recomendar otros títulos de libros
```console
streamlit run app_recomendador.py
```


