from pathlib import Path
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def vectorizar_corpus(
    ruta_tsv: str = "opinions_normalized_corpus.tsv",
    columna_texto: str = "review_norm",
    exportar_matrices_tsv: bool = True,
    limitar_vocabulario: int | None = None,
):
    """Genera las tres representaciones solicitadas:

    - Frequency (Unigramas + Bigramas)
    - Binary (Unigramas + Bigramas)
    - TF-IDF (Unigramas + Bigramas) y exporta los archivos correspondientes en
    TSV.
    """
    ruta = Path(ruta_tsv)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta.resolve()}")

    print(f"[+] Leyendo corpus desde: {ruta.name}...")
    df = pd.read_csv(ruta, sep="\t")

    # Manejar valores nulos (había 15 filas vacías tras normalización)
    textos = df[columna_texto].fillna("").astype(str).tolist()
    print(f"[✓] Documentos cargados: {len(textos)}")

    # Columnas de identificación/metadatos que acompañarán las matrices
    cols_meta = [
        c
        for c in ["doc_id", "review_id", "rating", "item_name"]
        if c in df.columns
    ]
    df_meta = df[cols_meta].copy()

    # Patrón de tokenización para respetar unigramas y bigramas limpios
    token_pattern = r"(?u)\b\w+\b"
    ngram_range = (1, 2)  # Unigrams and bigrams

    #Frequency (Unigrams and bigrams)
    print(
        "\n[+] Generando representación: Frequency (Unigrams + Bigrams)..."
    )
    vec_freq = CountVectorizer(
        ngram_range=ngram_range,
        binary=False,
        token_pattern=token_pattern,
        max_features=limitar_vocabulario,
    )
    matriz_freq = vec_freq.fit_transform(textos)
    vocab_freq = vec_freq.get_feature_names_out()
    print(f"    Vocabulario generado: {len(vocab_freq)} características.")

    if exportar_matrices_tsv:
        archivo_freq = "corpus_vectorizado_frequency.tsv"
        df_salida_freq = pd.concat(
            [
                df_meta.reset_index(drop=True),
                pd.DataFrame(matriz_freq.toarray(), columns=vocab_freq),
            ],
            axis=1,
        )
        df_salida_freq.to_csv(archivo_freq, sep="\t", index=False)
        print(f"    [✓] Guardado en: {archivo_freq}")

    #Binary (Unigrams and bigrams)
    print("\n[+] Generando representación: Binary (Unigrams + Bigrams)...")
    vec_bin = CountVectorizer(
        ngram_range=ngram_range,
        binary=True,
        token_pattern=token_pattern,
        max_features=limitar_vocabulario,
    )
    matriz_bin = vec_bin.fit_transform(textos)
    vocab_bin = vec_bin.get_feature_names_out()

    if exportar_matrices_tsv:
        archivo_bin = "corpus_vectorizado_binary.tsv"
        df_salida_bin = pd.concat(
            [
                df_meta.reset_index(drop=True),
                pd.DataFrame(matriz_bin.toarray(), columns=vocab_bin),
            ],
            axis=1,
        )
        df_salida_bin.to_csv(archivo_bin, sep="\t", index=False)
        print(f"    [✓] Guardado en: {archivo_bin}")

    #TF-IDF (Unigrams and bigrams)
    print("\n[+] Generando representación: TF-IDF (Unigrams + Bigrams)...")
    vec_tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        token_pattern=token_pattern,
        max_features=limitar_vocabulario,
    )
    matriz_tfidf = vec_tfidf.fit_transform(textos)
    vocab_tfidf = vec_tfidf.get_feature_names_out()

    if exportar_matrices_tsv:
        archivo_tfidf = "corpus_vectorizado_tfidf.tsv"
        df_salida_tfidf = pd.concat(
            [
                df_meta.reset_index(drop=True),
                pd.DataFrame(
                    matriz_tfidf.toarray(), columns=vocab_tfidf
                ).round(4),
            ],
            axis=1,
        )
        df_salida_tfidf.to_csv(archivo_tfidf, sep="\t", index=False)
        print(f"    [✓] Guardado en: {archivo_tfidf}")

 