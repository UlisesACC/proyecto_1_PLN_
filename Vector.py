from pathlib import Path
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def generar_representaciones_vectoriales(
    ruta_tsv: str = "opinions_normalized_corpus.tsv",
    incluir_nombre_item: bool = False,
    guardar_tsv: bool = True,
    guardar_pkl: bool = True,
    carpeta_salida: str = "salidas_vectores",
    limitar_vocabulario: int | None = None,
):
    """Genera las representaciones vectoriales para:

    - 'review text' (por defecto) o 'item name + review text' (opcional) en los
    esquemas:
    - Frequency (Unigramas + Bigramas)
    - Binary (Unigramas + Bigramas)
    - TF-IDF (Unigramas + Bigramas)

    Exporta los resultados en formatos .tsv y .pkl.
    """
    ruta = Path(ruta_tsv)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta.resolve()}")

    salida_dir = Path(carpeta_salida)
    salida_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Leyendo datos desde: {ruta.name}...")
    df = pd.read_csv(ruta, sep="\t")

    # 1. Selección de texto a vectorizar
    if incluir_nombre_item:
        columna_texto = "item_review_norm"
        etiqueta_modo = "item_name_plus_review"
        print("[i] Modo: Representación con 'item name + review text'")
    else:
        columna_texto = "review_norm"
        etiqueta_modo = "review_only"
        print("[i] Modo: Representación con 'review text'")

    # Manejo de valores vacíos y conversión a lista de cadenas
    textos = df[columna_texto].fillna("").astype(str).tolist()
    print(
        f"[✓] Se cargaron {len(textos)} textos desde la columna '{columna_texto}'."
    )

    # Conservar metadatos para las tablas de salida
    cols_meta = [
        col
        for col in ["doc_id", "review_id", "rating", "item_name"]
        if col in df.columns
    ]
    df_meta = df[cols_meta].copy().reset_index(drop=True)

    # Parámetros para unigramas y bigramas
    ngram_range = (1, 2)
    token_pattern = r"(?u)\b\w+\b"

    # Definición de los tres esquemas requeridos
    modelos = {
        "frequency": CountVectorizer(
            ngram_range=ngram_range,
            binary=False,
            token_pattern=token_pattern,
            max_features=limitar_vocabulario,
        ),
        "binary": CountVectorizer(
            ngram_range=ngram_range,
            binary=True,
            token_pattern=token_pattern,
            max_features=limitar_vocabulario,
        ),
        "tfidf": TfidfVectorizer(
            ngram_range=ngram_range,
            token_pattern=token_pattern,
            max_features=limitar_vocabulario,
        ),
    }

    # 2. Ajuste, transformación y guardado
    for nombre_esquema, vectorizador in modelos.items():
        print(f"\n[+] Procesando esquema: {nombre_esquema.upper()}...")
        matriz_vectores = vectorizador.fit_transform(textos)
        vocabulario = vectorizador.get_feature_names_out()
        print(
            f"    Vocabulario: {len(vocabulario)} características (unigramas y bigramas)."
        )

        nombre_base = f"vectores_{etiqueta_modo}_{nombre_esquema}"

        # Guardado en formato PKL (eficiente y listo para modelos de ML)
        if guardar_pkl:
            ruta_pkl = salida_dir / f"{nombre_base}.pkl"
            datos_pkl = {
                "matriz": matriz_vectores,  # Matriz dispersa (scipy.sparse)
                "vectorizador": vectorizador,  # Modelo CountVectorizer o TfidfVectorizer
                "vocabulario": vocabulario,  # Lista de términos/bigramas
                "metadatos": df_meta,  # DataFrame con doc_id, rating, etc.
                "modo": etiqueta_modo,
                "esquema": nombre_esquema,
            }
            with open(ruta_pkl, "wb") as f:
                pickle.dump(datos_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"    [✓] Archivo PKL guardado: {ruta_pkl.name}")

        # Guardado en formato TSV (legible en tablas/Excel)
        if guardar_tsv:
            ruta_tsv = salida_dir / f"{nombre_base}.tsv"
            df_matriz = pd.DataFrame(
                matriz_vectores.toarray(), columns=vocabulario
            )
            if nombre_esquema == "tfidf":
                df_matriz = df_matriz.round(4)

            df_completo = pd.concat([df_meta, df_matriz], axis=1)
            df_completo.to_csv(ruta_tsv, sep="\t", index=False)
            print(f"    [✓] Archivo TSV guardado: {ruta_tsv.name}")

    print("\n[✓] Proceso finalizado exitosamente.")


# -------------------------------------------------------------
# EJECUCIÓN
# -------------------------------------------------------------
if __name__ == "__main__":
    archivo_corpus = "opinions_normalized_corpus.tsv"

    # Caso 1: Generar solo para el texto de la reseña ('review text')
    print("OPCIÓN 1: REVIEW ONLY")
    generar_representaciones_vectoriales(
        ruta_tsv=archivo_corpus,
        incluir_nombre_item=False,  # Usa 'review_norm'
        guardar_tsv=True,
        guardar_pkl=True,
    )

    # Caso 2: Generar opcionalmente para 'item name + review text'
    print(
        "\nOPCIÓN 2: ITEM NAME + REVIEW TEXT"
    )
    generar_representaciones_vectoriales(
        ruta_tsv=archivo_corpus,
        incluir_nombre_item=True,  # Usa 'item_review_norm'
        guardar_tsv=True,
        guardar_pkl=True,
    )

 