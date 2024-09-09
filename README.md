# ProyectoDSII
Data Science II: Machine Learning para la Ciencia de Datos


# ProyectoDS

Curso: Data Science I: Fundamentos para la Ciencia de Datos
Comisión 61170
Alumno: Diego Lopez Castan


# Abstract

Este conjunto de datos contiene información detallada sobre canciones en Spotify, abarcando tanto aspectos de su popularidad como características acústicas. Cada registro incluye datos como el nombre y el artista de la canción, la popularidad de la pista, y su álbum asociado. Además, proporciona información sobre las listas de reproducción en las que se incluyen las canciones, junto con su género y subgénero.


# Campos del dataset

**track_id**: Identificador único de la canción en Spotify.

**track_name:** Nombre de la canción.

**track_artist:** Artista de la canción.

**track_popularity:** Puntaje de popularidad de la canción en Spotify (de 0 a 100, donde 100 es la mayor popularidad).

**track_album_id:** Identificador único del álbum en el que se encuentra la canción.

**track_album_name:** Nombre del álbum.

**track_album_release_date:** Fecha de lanzamiento del álbum.

**playlist_name:** Nombre de la lista de reproducción en la que se encuentra la canción.

**playlist_id:** Identificador único de la lista de reproducción.

**playlist_genre:** Género musical de la lista de reproducción.

**playlist_subgenre:** Subgénero musical de la lista de reproducción.

**danceability:** La bailabilidad describe lo adecuada que es una pista para bailar basándose en una combinación de elementos musicales como el tempo, la estabilidad del ritmo, la fuerza del compás y la regularidad general. Un valor de 0,0 es el menos bailable y 1,0 el más bailable.

**energy:** La energía es una medida de 0,0 a 1,0 y representa una medida perceptiva de intensidad y actividad. Normalmente, las pistas energéticas son rápidas, ruidosas y ruidosas. Por ejemplo, el death metal tiene mucha energía, mientras que un preludio de Bach tiene una puntuación baja en la escala. Las características perceptivas que contribuyen a este atributo incluyen el rango dinámico, el volumen percibido, el timbre, la velocidad de inicio y la entropía general.

**key:** La tonalidad global estimada de la pista. Los números enteros se asignan a tonos utilizando la notación estándar Pitch Class. Por ejemplo, 0 = C, 1 = C♯/D♭, 2 = D, y así sucesivamente. Si no se detecta ninguna tonalidad, el valor es -1.

**loudness:** La sonoridad general de una pista en decibelios (dB). Los valores de sonoridad se promedian en toda la pista y son útiles para comparar la sonoridad relativa de las pistas. La sonoridad es la cualidad de un sonido que es el principal correlato psicológico de la fuerza física (amplitud). Los valores suelen oscilar entre -60 y 0 db.

**mode:** El modo indica la modalidad (mayor o menor) de una pista, el tipo de escala del que se deriva su contenido melódico. Mayor se representa con 1 y menor con 0.

**speechiness:** La locuacidad detecta la presencia de palabras habladas en una pista. Cuanto más exclusivamente hablada sea la grabación (por ejemplo, programa de entrevistas, audiolibro, poesía), más se acercará a 1,0 el valor del atributo. Los valores superiores a 0,66 describen pistas que probablemente estén compuestas en su totalidad por palabras habladas. Los valores entre 0,33 y 0,66 describen pistas que pueden contener tanto música como voz, ya sea en secciones o en capas, incluyendo casos como la música rap. Los valores por debajo de 0,33 representan probablemente música y otras pistas no habladas.

**acousticness:** Una medida de confianza de 0,0 a 1,0 para determinar si la pista es acústica. 1,0 representa una confianza alta en que la pista es acústica.

**instrumentalness:** Predice si una pista no contiene voces. Los sonidos «ooh» y «aah» se consideran instrumentales en este contexto. Las pistas de rap o spoken word son claramente «vocales». Cuanto más se acerque el valor de instrumental a 1,0, mayor será la probabilidad de que la pista no contenga voces. Los valores superiores a 0,5 representan pistas instrumentales, pero la confianza es mayor a medida que el valor se acerca a 1,0.

**liveness:** Detecta la presencia de público en la grabación. Los valores más altos representan una mayor probabilidad de que la pista se haya interpretado en directo. Un valor superior a 0,8 proporciona una gran probabilidad de que la pista sea en directo.

**valence:** Medida de 0,0 a 1,0 que describe la positividad musical que transmite una pista. Las pistas con valencia alta suenan más positivas (por ejemplo, felices, alegres, eufóricas), mientras que las pistas con valencia baja suenan más negativas (por ejemplo, tristes, deprimidas, enfadadas).

**tempo:** El tempo global estimado de una pista en pulsaciones por minuto (BPM). En terminología musical, el tempo es la velocidad o ritmo de una pieza determinada y se deriva directamente de la duración media de los tiempos.

**duration_ms:** Duración de la canción en milisegundos.

# Link del dataset
[https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset/data](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs/data)
