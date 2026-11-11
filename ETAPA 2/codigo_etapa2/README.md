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
