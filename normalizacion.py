"""
Etapa 2 del proyecto: normalizacion del corpus

Entrada : opinions_raw_corpus.tsv
Salida  : opinions_normalized_corpus.tsv

Pasos que se realizaron

  0. Normalizar espacios y caracteres de control (y limpiar marcado de Steam:
     etiquetas BBCode como [h1] o [b], URLs y entidades HTML).
  1. Tokenización.
  2. Etiquetado POS (nltk.pos_tag, etiquetas Penn Treebank).
  3. Eliminación de stopwords de las categorías gramaticales pedidas,
     identificadas con la etiqueta POS: 
        - Artículos
        - Preposiciones
        - Conjunciones
        - Pronombres
  4. Lematización (WordNetLemmatizer usando la categoría POS).

tomar en cuenta:
  - Se pasa todo a minúsculas antes de etiquetar
  - Se descartan tokens que son solo puntuación o símbolos (no aportan
    al modelo vectorial y el vectorizador los ignoraría de todos modos).
  - "dont", "cant", "im" (sin apóstrofo) se reescriben con apóstrofe.
  - Las contracciones que separa el tokenizador se llevan a su forma
    completa: n't -> not, ca -> can, wo -> will, 'm/'re/'s(verbo) -> be ...
    Así "not" se conserva, lo cual es importante en opiniones.
"""

import argparse
import html
import re
import unicodedata

import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Recursos de NLTK (solo se descargan la primera vez)
def _asegurar_recurso(recurso, rutas):
    for ruta in rutas:
        try:
            nltk.data.find(ruta)
            return
        except LookupError:
            pass
    nltk.download(recurso, quiet=True)


_asegurar_recurso("punkt_tab", ["tokenizers/punkt_tab"])
_asegurar_recurso("averaged_perceptron_tagger_eng", ["taggers/averaged_perceptron_tagger_eng"])
_asegurar_recurso("wordnet", ["corpora/wordnet", "corpora/wordnet.zip"])

lemmatizer = WordNetLemmatizer()

# ------------------------------------------------------------------
# Configuración de stopwords por categoría gramatical (Penn Treebank)
# ------------------------------------------------------------------
ARTICULOS = {"a", "an", "the"}
TAGS_PREPOSICIONES = {"IN", "TO"}
TAGS_CONJUNCIONES = {"CC"}
TAGS_PRONOMBRES = {"PRP", "PRP$", "WP", "WP$"}
TAGS_ELIMINAR = TAGS_PREPOSICIONES | TAGS_CONJUNCIONES | TAGS_PRONOMBRES

# Contracciones que word_tokenize separa ("don't" -> "do" + "n't")
CONTRACCIONES = {
    "n't": "not", "nt": "not", "ca": "can", "wo": "will", "sha": "shall",
    "'m": "be", "'re": "be", "'ve": "have", "'ll": "will", "'d": "would",
}

# Contracciones informales sin apóstrofo, muy comunes en Steam ("dont", "cant").
# Se les devuelve el apóstrofo para que la negación se conserve como "not".
RE_SIN_APOSTROFO = re.compile(
    r"\b(do|does|did|ca|wo|is|are|was|were|have|has|had|would|should|could|ai)nt\b")
RE_IM = re.compile(r"\bim\b")

RE_BBCODE = re.compile(r"\[/?[a-zA-Z0-9*]+(=[^\]]*)?\]")      # [h1], [/b], [url=...]
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_ESPACIOS = re.compile(r"\s+")
RE_TIENE_ALFANUM = re.compile(r"\w", re.UNICODE)


def limpiar_texto(texto: str) -> str:
    """Paso 0: entidades HTML, BBCode, URLs, caracteres de control y espacios."""
    if not isinstance(texto, str):
        return ""
    texto = html.unescape(texto)
    texto = RE_BBCODE.sub(" ", texto)
    texto = RE_URL.sub(" ", texto)
    # Caracteres de control / formato (categoría Unicode "C*": \n, \t, \r, zero-width, etc.)
    texto = "".join(" " if unicodedata.category(c).startswith("C") else c for c in texto)
    texto = unicodedata.normalize("NFKC", texto)
    texto = RE_ESPACIOS.sub(" ", texto).strip().lower()
    texto = RE_SIN_APOSTROFO.sub(r"\1n't", texto)
    texto = RE_IM.sub("i'm", texto)
    return texto


def get_wordnet_pos(treebank_tag: str) -> str:
    """Mapea la etiqueta Penn Treebank al formato de WordNet (igual que en clase)."""
    if treebank_tag.startswith("J"):
        return wn.ADJ
    if treebank_tag.startswith("V"):
        return wn.VERB
    if treebank_tag.startswith("R"):
        return wn.ADV
    return wn.NOUN


def es_stopword(token: str, tag: str) -> bool:
    """True si el token pertenece a artículos, preposiciones, conjunciones o pronombres."""
    if tag == "DT" and token in ARTICULOS:
        return True
    return tag in TAGS_ELIMINAR


def normalizar_texto(texto: str, devolver_detalle: bool = False):
    """
    Aplica el pipeline completo y devuelve el texto normalizado (lemas separados
    por espacio). Esta MISMA función se usa para normalizar las consultas.
    """
    limpio = limpiar_texto(texto)
    tokens = word_tokenize(limpio)
    # En minúsculas el tagger no reconoce "i" como pronombre; se etiqueta como "I"
    etiquetados = pos_tag(["I" if t == "i" else t for t in tokens])
    etiquetados = [(t.lower() if t == "I" else t, tag) for t, tag in etiquetados]

    lemas, eliminados = [], []
    for token, tag in etiquetados:
        if not RE_TIENE_ALFANUM.search(token):          # puntuación / símbolos
            continue
        if es_stopword(token, tag):
            eliminados.append((token, tag))
            continue
        if token in CONTRACCIONES:
            lema = CONTRACCIONES[token]
        elif token == "'s" and tag.startswith("V"):      # "it's" -> be
            lema = "be"
        elif token == "'s":                              # posesivo -> se descarta
            continue
        else:
            lema = lemmatizer.lemmatize(token, get_wordnet_pos(tag))
        lemas.append(lema)

    normalizado = " ".join(lemas)
    if devolver_detalle:
        return normalizado, {"tokens": tokens, "pos": etiquetados, "eliminados": eliminados}
    return normalizado


def normalizar_corpus(ruta_entrada: str, ruta_salida: str) -> pd.DataFrame:
    df = pd.read_csv(ruta_entrada, sep="\t", dtype=str, keep_default_na=False)
    print(f"Corpus crudo: {len(df)} reseñas")

    # Identificador numérico consecutivo: es el document_id que se usará en
    # qrels.tsv y rankings.tsv (más cómodo que el review_id de Steam).
    df.insert(0, "doc_id", range(1, len(df) + 1))

    print("Normalizando item_name ...")
    nombres_norm = {n: normalizar_texto(n) for n in df["item_name"].unique()}
    df["item_name_norm"] = df["item_name"].map(nombres_norm)

    print("Normalizando review (puede tardar un poco) ...")
    df["review_clean"] = df["review"].map(limpiar_texto)
    df["review_norm"] = df["review"].map(normalizar_texto)
    df["item_review_norm"] = (df["item_name_norm"] + " " + df["review_norm"]).str.strip()

    vacias = (df["review_norm"].str.strip() == "").sum()
    if vacias:
        print(f"Aviso: {vacias} reseñas quedaron vacías tras normalizar "
              f"(p. ej. solo emojis o solo pronombres/artículos).")

    columnas = ["doc_id", "review_id", "item_id", "item_name", "rating", "date",
                "category", "source", "url", "source_id", "retrieved_at",
                "review", "review_clean", "review_norm", "item_name_norm", "item_review_norm"]
    df = df[columnas]
    df.to_csv(ruta_salida, sep="\t", index=False)
    print(f"Guardado: {ruta_salida}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normaliza el corpus de opiniones.")
    parser.add_argument("--entrada", default="opinions_raw_corpus.tsv")
    parser.add_argument("--salida", default="opinions_normalized_corpus.tsv")
    args = parser.parse_args()
    normalizar_corpus(args.entrada, args.salida)
