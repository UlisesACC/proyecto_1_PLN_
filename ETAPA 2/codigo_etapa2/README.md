# Etapa 2 — Análisis de sentimiento supervisado (Opiniones de Steam)

Extiende el sistema de recuperación de la Etapa 1 con un clasificador de sentimiento
(positive / negative) entrenado con aprendizaje automático tradicional (scikit-learn).
No se usan LLMs ni modelos generativos.

## Requisitos

- Python 3.10 o posterior
- Desde la carpeta principal del proyecto:

```bash
python -m pip install -r "ETAPA 2/codigo_etapa2/requirements.txt"
```

Los recursos de NLTK (punkt_tab, tagger, wordnet, stopwords) se descargan solos la primera vez.

## Cómo ejecutar (desde `ETAPA 2/codigo_etapa2`)

```bash
cd "ETAPA 2/codigo_etapa2"
python experimentos.py        # corre los 66 experimentos y genera todos los entregables (~10 s)
python interfaz.py            # menú interactivo
```

Comandos directos de la interfaz:

```bash
python interfaz.py buscar "game crashes and freezes"
python interfaz.py buscar "cheaters" --sentimiento negative
python interfaz.py clasificar "not worth the money"
python interfaz.py metricas
python interfaz.py errores
```

La búsqueda usa los modelos de la Etapa 1 (`modelos_recuperacion/`). Si avisa que el corpus
cambió, ejecutar una vez `python recuperacion.py build` en la carpeta principal.

## Archivos

| Archivo | Contenido |
|---|---|
| `preprocesamiento.py` | Normalización (N0–N3), marcado de negación, rasgos del léxico y armado del Pipeline |
| `experimentos.py` | Etiquetas, partición train/dev/test, experimentos por etapas, selección y prueba final |
| `interfaz.py` | Buscador de la Etapa 1 + sentimiento predicho, filtro, métricas y errores |
| `recursos/` | Léxico de opinión de Hu & Liu (positive-words.txt, negative-words.txt) |
| `modelos/` | `modelo_final.pkl` y `config_final.json` (configuración congelada) |
| `resultados/labeled_documents.tsv` | document_id, review, text, rating, label, split + metadatos |
| `resultados/classification_predictions.tsv` | Predicciones dev (mejor config. de cada modelo) y test (`experiment_id = test`) |
| `resultados/classification_metrics.tsv` | Métricas macro, por clase y matriz de confusión (dev y test) |
| `resultados/classification_reports.txt` | `classification_report` de cada fila anterior |
| `resultados/experiments.tsv` / `.xlsx` | Bitácora completa de los 66 experimentos |
| `resultados/ablacion_mejoras.tsv` | Aporte de cada mejora lingüística frente a la línea base |
| `resultados/analisis_errores_dev.tsv` | 13 errores de dev analizados a mano |
| `resultados/revision_manual_etiquetas.tsv` | Revisión manual de 10 etiquetas dudosas |
| `resultados/recursos_linguisticos.tsv` | Cobertura del léxico y frecuencia de negadores (train) |
| `resultados/ejemplos_negacion.tsv` | Texto antes y después de marcar la negación |
| `resultados/entorno.txt` | Versiones de software y semilla |
| `resultados/figuras/` | Matrices de confusión y progreso por etapa |

## Protocolo

- Etiqueta: recomendación de Steam (`rating`: positive → positive, negative → negative). 0 valores vacíos.
  `rating` nunca entra al modelo; X es solo el texto de la reseña.
- Partición 70/15/15 estratificada, `shuffle=True`, `random_state=0`: 420 / 90 / 90 reseñas.
- Todo se ajusta con train; las decisiones se toman con macro-F1 en dev; test se usa una sola vez
  con la configuración congelada, reentrenada con train+dev.
- Etapas: E1 normalización → E2 desbalance → E3 representación → E4 mejoras lingüísticas →
  E5 hiperparámetros (GridSearchCV con PredefinedSplit: entrena en train y califica en dev).
- Los archivos `analisis_errores_dev.tsv` y `revision_manual_etiquetas.tsv` son anotaciones manuales;
  `experimentos.py` no los sobrescribe.

## Corrida ampliada de la Etapa II

Desde la raíz del repositorio, en Git Bash o una terminal de Linux:

```bash
python -m pip install -r "ETAPA 2/codigo_etapa2/requirements-ampliados.txt"
python recuperacion.py build
cd "ETAPA 2/codigo_etapa2"
python experimentos_ampliados.py
python interfaz.py --ampliado buscar "cheaters" --sentimiento negative
python interfaz.py --ampliado metricas
python interfaz.py --ampliado errores
python experimentos_ampliados.py --predecir "I didn't like this game."
```

El entorno de referencia es Python 3.12.14. NLTK requiere `punkt_tab`,
`averaged_perceptron_tagger_eng`, `wordnet` y `stopwords`. Si la red
bloquea el descargador, instala antes esos cuatro recursos y configura
`NLTK_DATA` con el directorio que los contiene. Los recursos de Hu y Liu
se leen desde `recursos/`. Conserva los avisos originales y la cita indicada
en sus cabeceras.

### Flujo del script

1. **Datos y etiquetas.** Lee las 600 filas de `opinions_raw_corpus.tsv`,
   verifica IDs únicos y que el texto y el rating estén presentes. Contrasta
   cada etiqueta con el booleano original `voted_up` en
   `steam_reviews_raw.jsonl`. `true` significa `positive` y `false`
   significa `negative`. En la corrida registrada hubo 0 ratings faltantes,
   0 votos originales inválidos y 0 discrepancias. Conserva las diez
   revisiones manuales de etiquetas ambiguas de train.
2. **Diagnóstico y partición.** El corpus contiene 509 positivas y 91
   negativas (15.17 %). Se diagnostica el desbalance sobre el corpus antes de
   dividirlo. Se estratifica por etiqueta con semilla 0 para que las tres
   particiones tengan ambas clases y proporciones cercanas a las originales.

   | Partición | Negativas | Positivas | Total |
   |---|---:|---:|---:|
   | train | 64 | 356 | 420 |
   | dev | 13 | 77 | 90 |
   | test | 14 | 76 | 90 |

3. **Pipeline.** El único predictor es `text`. El pipeline encadena
   normalización, vectorización y clasificador. Se comparan N0 (tokenización),
   N1 (limpieza y contracciones), N2 (stopwords con negadores preservados)
   y N3 (lematización POS). Se prueban binaria, frecuencia y TF-IDF con
   unigramas, bigramas y ambos juntos. El vectorizador, los escaladores,
   SVD y los clasificadores solo se ajustan con la partición de entrenamiento
   que corresponda a cada fase.
4. **Aumento lingüístico.** Cada clasificador tiene una referencia sin
   aumento. En su ablación se cambia una mejora a la vez. Se ordenan las
   mejoras individuales por macro-F1 y F1 negativo en dev. Se eligen las
   dos primeras compatibles y se prueban juntas frente a la misma referencia.
   También se prueban letras + ortografía y letras + palabras. Una mejora
   entra en la configuración candidata cuando supera a la configuración
   más simple según el criterio de selección.
5. **Desbalance, SVD e hiperparámetros.** Con el desbalance observado se
   prueba `class_weight=balanced` en regresión logística y SVM lineal. En NB,
   se comparan ponderación inversa de ejemplos mediante `sample_weight` y
   priors uniformes mediante `fit_prior=False`. El primero repondera las
   estadísticas del clasificador y el segundo modifica solo el prior. Se
   conservaron las filas originales. La búsqueda recorre una rejilla
   controlada de `alpha` o `C`, ajustando cada candidato en train y
   evaluándolo en dev. TruncatedSVD de 50 componentes se prueba para
   LogisticRegression y LinearSVC. Sus componentes pueden tener signo
   negativo, incompatible con MultinomialNB.
6. **Selección y test.** Se elige por macro-F1 de dev, luego por F1 de la
   clase negativa y, si persiste el empate, prevalece la referencia evaluada
   primero. Se guarda la configuración elegida antes de ajustar el ganador
   con train+dev. Esa instancia predice test una vez por corrida. El modelo
   guardado incluye las transformaciones que aplica a reseñas nuevas.

### Recursos y limitaciones

- **Hu y Liu.** Se preservan las listas en inglés, se retiran las entradas
  compartidas entre polaridades de ambos conjuntos y se comparan con una
  variante de Steam que neutraliza `refund` y `grind`. Los dos conteos
  de polaridad se escalan con MaxAbsScaler y se agregan al vector de texto.
  La cobertura de train fue 373/420 para Hu y Liu y 372/420 para la variante
  Steam. Cobertura significa que aparece al menos una palabra del léxico.
  Formas como `nerf` y `cheaters` ilustran brechas de jerga y morfología.
  Fuente: Minqing Hu y Bing Liu, *Mining and Summarizing Customer Reviews*,
  KDD 2004. Las cabeceras locales indican la atribución solicitada.
- **Emojis y emoticonos.** Se detectan emojis Unicode con `emoji` y
  emoticonos ASCII con una expresión regular. Se agregan tres conteos
  escalados: positivos, negativos y demás. Las asignaciones explícitas
  están en el código y se exportan a JSON. Solo 16 filas de train tenían
  algún símbolo detectado. Una cara sonriente, por ejemplo, puede cumplir
  una función sarcástica que el conteo no representa.
- **Negación.** La alternativa con marcado conserva `not`, `never`,
  `no` y otros negadores y marca hasta tres tokens léxicos o hasta una
  frontera de puntuación. La ventana fija puede alcanzar palabras fuera
  de la cláusula negada o perder contrastes entre cláusulas.
- **Ortografía y repeticiones.** El diccionario inglés de `pyspellchecker`
  propone cambios a distancia uno, con una lista fija de jerga y nombres
  protegidos. La normalización de letras reduce secuencias de tres o más
  letras iguales y consulta el diccionario para escoger entre una y dos
  letras. La de palabras colapsa duplicados consecutivos y preserva
  negadores. `transformaciones.tsv` conserva la reseña original y cada
  variante. Por ejemplo, `chalenges → challenges` corrige una falta,
  `gooooooooooooood → good` elimina el alargamiento expresivo, y
  `really really → really` pierde énfasis. También se observó
  `glados → gladys` (personaje de Portal), `larian → marian` (estudio),
  `janky → hanky` (jerga) y `ggst → gist` (sigla de producto).
  Estos ejemplos explican por qué se evalúa la corrección como ablación.

### Resultados de la corrida ampliada

| Modelo | Mejor ID dev | Macro-F1 dev | Dos mejores aumentos individuales |
|---|---|---:|---|
| MultinomialNB | A003 | 0.6496 | Hu y Liu y emojis, empatados con la referencia y otras mejoras |
| LogisticRegression | A041 | 0.7531 | Negación y Hu y Liu, este último empatado con otras mejoras |
| LinearSVC | A069 | 0.7941 | Negación y Hu y Liu, este último empatado con otras mejoras |

- En LinearSVC, la negación sola mejora la referencia de 0.7574 a 0.7941.
  Hu y Liu solo da 0.7574 y, al combinarlo con negación, se obtiene 0.7531.
  La negación mejora el F1 negativo de 0.5714 a 0.6316: mantiene 6/13
  negativas detectadas y corrige dos falsas alarmas sobre reseñas positivas.
- Letras + ortografía queda en 0.6496 para NB y 0.7574 para LinearSVC.
  Para LogisticRegression baja de 0.7352 a 0.7077. Emojis y la variante
  Steam del léxico empatan con las referencias correspondientes.
- La ponderación de muestras de NB baja macro-F1 de 0.6496 a 0.5652.
  Con prior uniforme queda en 0.6095. Los pesos de clase bajan
  LogisticRegression de 0.7531 a 0.7258 y empatan con 0.7941 en LinearSVC.
  Se conservan los ajustes sin tratamiento del desbalance en esta corrida.
- Con SVD, LogisticRegression cae a 0.6427 y LinearSVC a 0.7750.
  El ganador usa frecuencia de unigramas, N1, marcado de negación, y
  `C=1.0`. En test logra 0.6853 de macro-F1 y 84.44 % de exactitud.
  Detecta 6 de las 14 negativas. Las decisiones metodológicas y el ranking
  de mejoras proceden de dev.

### Archivos de la corrida ampliada

| Archivo bajo `resultados_ampliados/` | Evidencia |
|---|---|
| `labeled_documents.tsv`, `auditoria_corpus.json` | Reseñas, particiones, etiquetas, hash y auditoría del origen |
| `experiments.tsv`, `ablacion_mejoras.tsv`, `dos_mejores_aumentos.tsv` | 84 experimentos, parámetros, comparaciones controladas y selección |
| `classification_predictions.tsv`, `classification_metrics.tsv` | Las tres ganadoras de dev y el ganador global en test |
| `classification_reports.txt`, `figuras/` | Reportes por clase y matrices de confusión |
| `analisis_errores_dev.tsv`, `errores_dev.tsv` | 13 casos revisados y cambios de error frente a la línea base |
| `transformaciones.tsv`, `cobertura_lexicos.tsv`, `recursos_linguisticos.json` | Evidencia reproducible de recursos y comparaciones de texto |
| `distribucion_clases.tsv`, `revision_manual_etiquetas.tsv`, `entorno.json` | Distribución original, revisión de votos ambiguos y versiones |
| `config_final.json`, `modelo_final.pkl` | Configuración congelada y pipeline entrenado |
