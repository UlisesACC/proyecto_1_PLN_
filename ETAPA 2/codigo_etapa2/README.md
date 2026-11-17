# Etapa II · Sentimiento de reseñas de Steam

El proyecto compara tres clasificadores tradicionales sobre las mismas
600 reseñas de la Etapa I. El ejecutable es `experimentos_ampliados.py`.
La implementación se organiza por responsabilidad en el paquete
`sentimiento/`.

## Instalación y ejecución

Entorno de referencia: Python 3.12.14. Desde la raíz del repositorio:

```bash
python -m pip install -r "ETAPA 2/codigo_etapa2/requirements-ampliados.txt"
python recuperacion.py build
cd "ETAPA 2/codigo_etapa2"
python experimentos_ampliados.py
python interfaz.py --fuentes buscar "cheaters" --sentimiento negative
python interfaz.py --fuentes clasificar "I didn't like this game."
python interfaz.py --fuentes metricas
python interfaz.py --fuentes errores
```

También se puede predecir directamente:

```bash
python experimentos_ampliados.py --predecir "A wonderful game!"
```

El directorio predeterminado de esta corrida es `resultados_fuentes/`.
`--salida RUTA` permite indicar otro directorio al ejecutable para entrenar
o predecir. `interfaz.py --fuentes` consulta el directorio predeterminado.
`python recuperacion.py build` se ejecuta desde la raíz y prepara los
índices que necesita el buscador.

NLTK utiliza `punkt_tab`, `averaged_perceptron_tagger_eng`, `wordnet`
y `stopwords`. Se descargan si faltan. Cuando el descargador no puede
acceder a la red, se pueden instalar previamente y establecer `NLTK_DATA`
con el directorio de recursos.

## Dónde está cada parte

| Archivo | Responsabilidad |
|---|---|
| `experimentos_ampliados.py` | Ejecutar el flujo y atender los argumentos de consola |
| `sentimiento/configuracion.py` | Rutas, semilla, clases, mejoras y rejillas de hiperparámetros |
| `sentimiento/datos.py` | Validación de reseñas, auditoría contra Steam y particiones |
| `sentimiento/normalizacion.py` | Limpieza, lematización, negación, ortografía y repeticiones |
| `sentimiento/recursos.py` | Lectores de Hu y Liu, SentiWordNet y emojis, rasgos e inventario de fuentes |
| `sentimiento/modelos.py` | Construcción del Pipeline, SVD y ponderación de entrenamiento |
| `sentimiento/busqueda.py` | Ablaciones, combinaciones y selección con macro-F1 de dev |
| `sentimiento/metricas.py` | Métricas macro, por clase y matrices de confusión |
| `sentimiento/resultados.py` | Exportación de archivos, análisis de errores y figuras |
| `interfaz.py` | Búsqueda, clasificación, filtros, métricas y errores |
| `preprocesamiento.py` | Funciones compartidas de texto, vectorizadores y clasificadores |

Los alias de `Configuracion`, `NormalizadorAmpliado` y
`RasgosSentimiento` en el ejecutable permiten cargar los modelos
serializados cuando esas clases vivían en el script principal.

## Recursos de fuentes

Las rutas se resuelven desde la ubicación del código, por lo que funcionan
aunque el comando se ejecute desde otra carpeta.

| Recurso bajo `ETAPA 2/fuentes/` | Integración |
|---|---|
| `Lexicons/Lexicons/opinion-lexicon-English/` | Hu y Liu, leído directamente de las listas positivas y negativas |
| `Lexicons/Lexicons/SentiWordNet_3.0.0.txt` | Nuevo experimento con sumas de polaridad positiva y negativa por lema y POS |
| `Sentiment analysis/Sentiment analysis/polaridad_con_lexicon.py` | Adaptación de la concatenación de rasgos lingüísticos al espacio vectorial |
| `Sentiment analysis/Sentiment analysis/polaridad_con_ml.py` | Referencia para la relación entre texto, etiqueta y partición previa al ajuste |
| `Supervised text classificacion scripts/.../clasificación_texto_pipeline.py` | Referencia para encadenar representación y clasificación en Pipeline |
| `Supervised text classificacion scripts/.../clasificación_texto_pipeline_svd.py` | Referencia para colocar TruncatedSVD dentro del pipeline |
| `Lexicons/Lexicons/SEL_full.txt` | Léxico español, inventariado como referencia por diferencia de idioma |
| `Lexicons/Lexicons/LIWC2007.dic` | Diccionario español incluido, inventariado como referencia |
| `Sentiment analysis/Sentiment analysis/Rest_Mex_2022.xlsx` | Corpus de referencia en español; la colección de esta tarea es Steam en inglés |
| Ejemplos de clustering y LDA | Referencias de tareas no supervisadas |

`inventario_fuentes.tsv` registra los 13 archivos, sus hashes SHA-256,
idioma, uso y justificación. Los ejemplos aportan la estructura del
procedimiento. La implementación adapta las rutas, el idioma, las etiquetas
y el protocolo train/dev/test a Steam. Los recursos externos de sentimiento
se leen desde los archivos aportados.

### Hu y Liu

Los experimentos `hu_liu` y `hu_liu_steam` cuentan palabras positivas
y negativas. Se retiran términos que aparecen en las dos listas. La variante
Steam también neutraliza `refund` y `grind`, usos ambiguos documentados
en la revisión anterior. MaxAbsScaler ajusta la escala con train y
FeatureUnion añade las dos columnas a la representación textual.

Las listas incluyen errores ortográficos intencionales y sus avisos de
atribución. La referencia indicada en sus cabeceras es Hu y Liu,
*Mining and Summarizing Customer Reviews*, KDD 2004.

### SentiWordNet

Cada registro aporta categoría gramatical (POS), identificador de synset,
puntajes positivo y negativo y términos con número de sentido. El lector
valida el formato y las polaridades. La línea final con un marcador
`#` precedido por tabulaciones se trata como comentario.

Para cada par de lema y POS, la implementación promedia los puntajes de
sus sentidos con pesos `1 / número_de_sentido`. Etiqueta y lematiza la
reseña, consulta esos pares y suma por separado las polaridades positiva y
negativa. Ambas sumas son no negativas y compatibles con MultinomialNB.
La cabecera atribuye el recurso a Andrea Esuli e indica CC BY-SA 4.0.

La media ponderada es una decisión de implementación. Conserva información
de varios sentidos y depende del etiquetado POS. Puede mezclar acepciones
y perder expresiones de varias palabras porque se consultan unigramas.
La ablación mide su aporte en dev frente a la misma referencia sin aumento.

### Emojis, ortografía y negación

El catálogo editable de 5,244 emojis y 66 emoticonos está en
[`recursos/emojis_sentimiento.tsv`](recursos/emojis_sentimiento.tsv).
Las instrucciones y los criterios heurísticos están en
[`recursos/EMOJIS.md`](recursos/EMOJIS.md).

La corrección ortográfica usa el diccionario inglés de pyspellchecker a
distancia uno y una lista fija de palabras protegidas. La normalización
de letras reduce alargamientos; la de palabras colapsa repeticiones
consecutivas y conserva negadores. Se comparan las variantes por separado,
letras + ortografía y letras + palabras. El archivo de transformaciones
permite revisar nombres como GLaDOS, jerga y ortografía expresiva.

La negación conserva `not`, `never`, `no` y otros negadores durante la
eliminación de stopwords. La opción de marcado usa una ventana de tres
tokens léxicos o hasta puntuación. El contexto puede exceder esa ventana.

Cada transformador entrenado guarda una copia del léxico y del catálogo de
emojis que utiliza, para aplicar esos mismos recursos durante la predicción.

## Protocolo y selección

1. Se leen todas las reseñas del corpus congelado. `document_id = review_id`,
   `text = review` y el rating se conserva como metadato. Se verifica contra
   el voto original `voted_up` de Steam, con true → positive y false → negative.
2. Se diagnostica la distribución completa y se divide 70/15/15 con
   `shuffle=True`, `random_state=0` y estratificación por etiqueta.
   Se obtienen 420/90/90 reseñas. Las tres particiones contienen ambas clases.
3. Para cada modelo se comparan N0–N3 y las representaciones binaria,
   frecuencia y TF-IDF con unigramas, bigramas y ambos.
4. Se establece una referencia sin aumento y se prueba una mejora a la
   vez. Se ordenan por macro-F1 y F1 negativo en dev. Se seleccionan las dos
   primeras compatibles y se prueba su combinación. Las variantes léxicas
   compiten como alternativas del mismo componente.
5. Se prueba SVD de 50 componentes para LogisticRegression y LinearSVC.
   MultinomialNB requiere rasgos no negativos y SVD puede producir valores
   negativos. Toda reducción se ajusta dentro del pipeline en train.
6. El criterio exploratorio de desbalance es una clase minoritaria con menos
   del 25 %. Para los modelos lineales se compara class_weight=balanced.
   Para NB se comparan sample_weight inversamente proporcional a la clase
   y fit_prior=False. Los pesos se calculan con el conjunto de entrenamiento.
7. Se recorre una rejilla controlada de alpha o C. Cada candidato se ajusta
   con train y se evalúa en dev. Se registra la decisión de cada ensayo.
8. Se congela la configuración ganadora en dev, se ajusta con train+dev y
   se evalúa una vez en test durante esa corrida. La interpretación y las
   decisiones de selección se basan en dev.

## Archivos generados en resultados_fuentes

| Archivo | Contenido |
|---|---|
| `labeled_documents.tsv` | IDs originales, reseñas, rating, etiquetas y particiones |
| `auditoria_corpus.json`, `distribucion_clases.tsv` | Verificación de origen, faltantes, mapeo y distribución |
| `experiments.tsv` | Todos los ensayos, configuración serializada, parámetros, métricas y decisión |
| `ablacion_mejoras.tsv`, `dos_mejores_aumentos.tsv` | Efectos contra la referencia y las mejoras seleccionadas |
| `classification_predictions.tsv`, `classification_metrics.tsv` | Mejor configuración dev por modelo y ganador global en test |
| `classification_reports.txt`, `figuras/` | Reportes por clase y matrices de confusión |
| `errores_dev.tsv`, `analisis_errores_dev.tsv` | Errores introducidos o corregidos y casos comentados |
| `revision_manual_etiquetas.tsv` | Revisión histórica de etiquetas ambiguas en train |
| `transformaciones.tsv` | Texto original y variantes de ortografía/repeticiones |
| `inventario_fuentes.tsv`, `recursos_linguisticos.json` | Procedencia, uso, hashes y descripción de recursos |
| `cobertura_lexicos.tsv` | Filas de train con polaridad y cobertura por recurso |
| `entorno.json`, `config_final.json`, `modelo_final.pkl` | Versiones, semillas, selección y pipeline entrenado |

Las corridas previas se consultan con `interfaz.py --ampliado` para
`resultados_ampliados/`, o con `interfaz.py` para `resultados/`.
`experimentos.py` sigue siendo el ejecutable de la primera corrida.

## Resultado de la corrida con fuentes

Se ejecutaron 87 experimentos. SentiWordNet aportó polaridad a 419 de las
420 reseñas de train y empató en macro-F1 con la referencia sin aumento
de cada clasificador. Su mayor cobertura léxica representa más coincidencias
y debe interpretarse junto con las métricas de clasificación.

| Modelo | Mejor ensayo dev | Macro-F1 dev |
|---|---|---:|
| MultinomialNB | A003 | 0.649596 |
| LogisticRegression | A042 | 0.753086 |
| LinearSVC | A071 | 0.794050 |

El ganador fue LinearSVC con N1, frecuencia de unigramas, marcado de negación
y C=1.0. En test obtuvo macro-F1 de 0.685315 y accuracy de 0.844444.
Las 600 etiquetas coinciden con los votos originales de Steam, con cero
calificaciones faltantes. Se conservaron las particiones históricas y
se verificaron 13 casos de error en dev.
