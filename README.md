# Proyecto 1 PLN — Recuperación de opiniones de Steam

Proyecto de **Procesamiento de Lenguaje Natural: Opinion Retrieval (etapa 1)**. Recopila reseñas de videojuegos de Steam, conserva el corpus original, lo normaliza y permite recuperar las diez opiniones más similares a una consulta mediante un Modelo de Espacio Vectorial (VSM) y similitud coseno. Incluye herramientas para construir el benchmark de la sección 9 de la práctica.

## Requisitos

- Python 3.10 o posterior (por el uso de anotaciones de tipos en `Vector.py`).
- Acceso a Internet solo para la extracción inicial desde Steam y, si faltan, recursos de NLTK.
- Ejecutar los comandos **desde la carpeta raíz del repositorio**.

En Git Bash (Windows):

```bash
git clone https://github.com/UlisesACC/proyecto_1_PLN_.git
cd proyecto_1_PLN_
python -m pip install requests langdetect nltk pandas numpy scikit-learn
```

`python -m pip` también funciona cuando Git Bash no encuentra directamente el comando `pip`.

## Flujo de ejecución

Si **ya existen** `opinions_raw_corpus.tsv` y `opinions_normalized_corpus.tsv` actualizados, **no hace falta volver a ejecutar la extracción**. Comienza directamente en el paso 3. Los archivos actuales contienen 600 reseñas (colección de reserva); el documento recomienda trabajar con 300–500. Si se decide trabajar con una muestra fija de ese tamaño, debe seleccionarse y documentarse antes de construir los juicios de relevancia, manteniendo IDs consistentes.

### Paso 1 — Obtener reseñas de Steam (opcional)

```bash
python datos.py
```

**Qué hace:** consulta la API de reseñas y los metadatos de cada juego, obtiene su género/categoría real, verifica que los comentarios tengan texto y detecta si están en inglés. Descarta reseñas insuficientes y duplicadas. Genera o **sobrescribe**:

- `opinions_raw_corpus.tsv`: opiniones aceptadas, sin lematizar, con fecha, valoración, género y procedencia.
- `steam_reviews_raw.jsonl`: respuestas originales obtenidas de la API.
- `steam_reviews_rejected.jsonl`: reseñas descartadas y motivo.

**Atención:** el script actual solicita hasta 100 reseñas de seis juegos. Ejecutarlo de nuevo puede cambiar las reseñas y el orden de los `doc_id` después de normalizar. No hacerlo si ya se prepararon `qrels.tsv` y experimentos sobre el corpus anterior, salvo que se vaya a reconstruir todo.

### Paso 2 — Normalizar el corpus (opcional si ya está actualizado)

```bash
python normalizacion.py
```

**Qué hace:** lee `opinions_raw_corpus.tsv`, limpia espacios, etiquetas de formato y caracteres de control; tokeniza, aplica etiquetas POS, elimina artículos, preposiciones, conjunciones y pronombres según esas etiquetas, y lematiza usando NLTK. Crea o sobrescribe `opinions_normalized_corpus.tsv` con `doc_id`, `review_norm`, `item_name_norm` y `item_review_norm`. El campo `review` conserva el texto original.

Opcionalmente se pueden indicar rutas:

```bash
python normalizacion.py --entrada opinions_raw_corpus.tsv --salida opinions_normalized_corpus.tsv
```

**Importante:** `doc_id` es el identificador que usan `qrels.tsv` y `rankings.tsv`; `review_id` es el ID original de Steam. Si cambian los datos o su orden, hay que revisar los juicios de relevancia anteriores.

### Paso 3 — Construir las representaciones (sección 7)

```bash
python recuperacion.py build
```

**Qué hace:** lee el corpus normalizado y genera 12 modelos `.pkl` en `modelos_recuperacion/`, combinando:

- Contenido: `review` (solo reseña) o `item_review` (nombre del juego + reseña).
- Representación: `frequency`, `binary` o `tfidf`.
- N-gramas: `unigram` o `bigram` (por separado; `bigram` significa únicamente bigramas).

**Alternativa ya presente en el repositorio:**

```bash
python Vector.py
```

`Vector.py` genera matrices con unigramas **y** bigramas juntos para las tres representaciones y los dos contenidos, como archivos `.pkl` y `.tsv` en `salidas_vectores/`. Son dos formas de generar vectores; **el comando `search` de `recuperacion.py` utiliza los modelos creados por `build`**, no los de `Vector.py`.

### Paso 4 — Probar la búsqueda (sección 8)

```bash
python recuperacion.py search "game crashes and freezes"
```

**Qué hace:** aplica a la consulta la **misma función** `normalizar_texto` del corpus, la vectoriza, calcula la similitud coseno y muestra los diez documentos mejor clasificados con `document_id`, posición, puntuación, juego y reseña.

La configuración por defecto es `--content review --representation tfidf --ngram unigram`.

Ejemplos para comparar configuraciones:

```bash
python recuperacion.py search "poor performance" --content review --representation frequency --ngram bigram
python recuperacion.py search "difficult controls" --content item_review --representation binary --ngram unigram
python recuperacion.py search "great story" --content review --representation tfidf --ngram unigram
```

Si una consulta no comparte términos con el vocabulario, el programa advierte que sus puntuaciones serán cero; ese Top-10 **no debe interpretarse como evidencia de relevancia**.

### Paso 5 — Crear las quince consultas del benchmark (sección 9)

```bash
python recuperacion.py init-queries
```

**Qué hace:** crea `queries.tsv` como plantilla con 15 consultas: cinco fáciles (`difficulty_level=1`), cinco medias (`2`) y cinco difíciles (`3`). Las columnas son `query_id`, `difficulty_level`, `query`.

Antes de evaluar, **revisar y editar las consultas** para que tengan sentido frente a las reseñas realmente recopiladas. Este comando no sobrescribe `queries.tsv` si ya existe.

### Paso 6 — Generar candidatos y evaluar relevancias manualmente

```bash
python recuperacion.py candidates
```

**Qué hace:** genera `candidatos_qrels.tsv` con hasta 30 reseñas candidatas por consulta, su `document_id`, `score`, `item_name`, texto y una columna vacía `relevance`.

Abrir este archivo en Excel (delimitador **tabulación**) o en VS Code, filtrar por `query_id` y revisar cada reseña **por su significado**, no por su puntuación. Calificar:

| Valor | Significado |
| --- | --- |
| 0 | No relevante |
| 1 | Parcialmente relevante |
| 2 | Muy relevante |

Buscar también en `opinions_normalized_corpus.tsv` si los 30 candidatos no incluyen suficientes reseñas pertinentes. No declarar relevante algo solo para cumplir una cuota.

Crear **manualmente** `qrels.tsv`, delimitado por tabulaciones, con únicamente los juicios relevantes:

```text
query_id	document_id	relevance
Q02	106	2
Q02	5	2
```

Esas filas son ejemplos reales revisados para la consulta Q02 sobre cierres del juego, **no constituyen un benchmark completo**. El archivo final debe incluir las 15 consultas, cada una con **10–15 documentos relevantes** y una mezcla de valores 1 y 2 (los de valor 0 se omiten). Total esperado: 150–225 relaciones. Verificar que no haya pares duplicados `query_id` + `document_id`.

**Atención:** el comando `candidates` se detiene con `FileExistsError` si ya existe `candidatos_qrels.tsv`, para no sobrescribir evaluaciones. Si hay que rehacerlo, respaldarlo, eliminar el anterior y volver a ejecutar; por ejemplo:

```bash
cp candidatos_qrels.tsv candidatos_qrels_backup.tsv
rm candidatos_qrels.tsv
python recuperacion.py candidates
```

Congelar `queries.tsv`, `qrels.tsv` y el corpus antes de comparar representaciones.

### Paso 7 — Guardar los rankings (sección 9)

```bash
python recuperacion.py rank
```

**Qué hace:** ejecuta las 15 consultas y guarda en `rankings.tsv` sus diez primeros resultados, con columnas `query_id`, `document_id`, `rank` y `score` (150 filas si todas se procesan correctamente). También crea `rankings_metadata.json` para detectar cambios en las consultas o el corpus.

Puede utilizarse otra configuración, por ejemplo:

```bash
python recuperacion.py rank --content review --representation frequency --ngram bigram
```

**Advertencia:** `rank` sobrescribe `rankings.tsv`. Guardar una copia de cada modelo que se desee comparar y registrar sus opciones antes de ejecutar otro `rank`.

### Paso 8 — Calcular las métricas (sección 9)

```bash
python recuperacion.py evaluate
```

**Qué hace:** requiere un `qrels.tsv` **revisado manualmente** y el `rankings.tsv` correspondiente. Comprueba las 15 consultas, los IDs, los rangos y las relevancias. Calcula:

- **Precision@10**: proporción de relevantes en los diez recuperados.
- **Recall@10**: relevantes recuperados / relevantes conocidos en el benchmark.
- **F1@10**: media armónica de precisión y recall.
- **MRR**: promedio del inverso de la posición del primer documento relevante (en la tabla por consulta aparece `rr`).
- **nDCG@10**: calidad del orden ponderada por relevancia 1 y 2.

Genera `evaluacion_por_consulta.tsv` y `evaluacion_promedios.tsv`.

Si muestra `Se requieren qrels.tsv (manual) y rankings.tsv (generado)`, **todavía falta alguno de esos archivos**: en general será `qrels.tsv`, que no se genera automáticamente.

### Paso 9 — Comparar las 12 configuraciones (sección 9)

```bash
python recuperacion.py benchmark
```

**Qué hace:** ejecuta las 15 consultas con las 12 combinaciones de contenido, representación y n-gramas, y las evalúa contra el mismo `qrels.tsv`. No modifica `rankings.tsv`; guarda en `benchmark/` los 12 rankings, `comparativa_configuraciones.tsv` (promedios ordenados por nDCG@10) y `metricas_por_consulta_y_configuracion.tsv`.

**Estado actual del benchmark:** `qrels.tsv` tiene 222 juicios (10–15 por consulta, con valores 1 y 2). `juicios_relevancia_completos.tsv` guarda todos los juicios positivos encontrados al revisar las 600 reseñas, incluidos los que no cupieron en el límite de 15. La consulta Q04 se cambió de "difficult boss fights" a "the game is too hard" porque el corpus no tenía suficientes reseñas sobre jefes. La huella del corpus ya no depende del sistema operativo; si `search`, `rank` o `benchmark` indican que el corpus cambió, basta con ejecutar una vez `python recuperacion.py build`.

## Resumen rápido (si ya hay corpus actualizado)

```bash
python recuperacion.py build
python recuperacion.py search "game crashes and freezes"
python recuperacion.py init-queries
python recuperacion.py candidates
```

Después **revisar candidatos y crear `qrels.tsv` manualmente**. Solo cuando esté listo:

```bash
python recuperacion.py rank
python recuperacion.py evaluate
```

No es necesario compilar Python: los comandos anteriores **ejecutan** directamente los scripts.

## Estructura de archivos

| Archivo o carpeta | Función |
| --- | --- |
| `datos.py` | Extracción y validación desde Steam |
| `normalizacion.py` | Normalización con NLTK |
| `Vector.py` | Vectorización alternativa con unigramas + bigramas y exportación TSV/PKL |
| `recuperacion.py` | Modelos, Top-10, benchmark y métricas |
| `opinions_raw_corpus.tsv` | Corpus original validado |
| `opinions_normalized_corpus.tsv` | Corpus normalizado |
| `modelos_recuperacion/` | Modelos que consume `recuperacion.py` |
| `salidas_vectores/` | Resultados generados por `Vector.py` |
| `queries.tsv` | 15 consultas del benchmark |
| `candidatos_qrels.tsv` | Candidatos para calificación humana |
| `qrels.tsv` | Juicios finales de relevancia, construidos manualmente |
| `rankings.tsv` | Top-10 para las 15 consultas |
| `evaluacion_por_consulta.tsv` | Métricas individuales |
| `evaluacion_promedios.tsv` | Métricas promedio |
| `juicios_relevancia_completos.tsv` | Todos los juicios positivos (incluye los que no entraron a `qrels.tsv`) |
| `benchmark/` | Rankings y métricas de las 12 configuraciones |

## Notas metodológicas

- La práctica recomienda **300–500 reseñas**. Si se mantienen 600 como reserva, indicar claramente cuántas se usan en los experimentos; **no cambiar el conjunto de documentos después de congelar el benchmark**.
- El resultado de `search` no siempre será semánticamente relevante, incluso si contiene palabras de la consulta; ese comportamiento se mide con el benchmark.
- No se debe construir el `qrels.tsv` únicamente copiando los diez primeros resultados y marcándolos relevantes.
- El análisis posterior de errores y la presentación de resultados forman parte de otras secciones de la práctica.
