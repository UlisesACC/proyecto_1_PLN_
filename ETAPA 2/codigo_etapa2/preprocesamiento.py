"""
preprocesamiento.py — Piezas de texto que se usan dentro de los Pipelines (Etapa 2)

Contiene tres cosas:

1. Normalizador: transforma cada reseña según una configuración de normalización
   (N0, N1, N2 o N3) y, opcionalmente, marca el alcance de la negación.
2. RasgosLexico: cuenta palabras positivas y negativas del léxico de opinión
   de Hu & Liu (2004) y las agrega como dos columnas extra.
3. construir_pipeline(): arma el Pipeline completo de scikit-learn a partir de
   un diccionario de configuración (normalización, representación, mejoras,
   clasificador e hiperparámetros).

Configuraciones de normalización (cada una agrega UN paso a la anterior):
    N0  solo tokenización (línea base: conserva mayúsculas y puntuación)
    N1  N0 + limpieza (HTML/BBCode/URLs, minúsculas, contracciones) y sin puntuación
    N2  N1 + eliminación de stopwords (CONSERVANDO negadores: not, no, never...)
    N3  N2 + lematización con WordNet (usando la etiqueta POS)
"""

import html
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import nltk
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

SEMILLA = 0
CARPETA = Path(__file__).resolve().parent
RECURSOS = CARPETA / "recursos"


# ---------------------------------------------------------------------------
# Recursos de NLTK (solo se descargan la primera vez)
# ---------------------------------------------------------------------------
def asegurar_recursos_nltk():
    necesarios = {
        "punkt_tab": "tokenizers/punkt_tab",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
        "wordnet": "corpora/wordnet",
        "stopwords": "corpora/stopwords",
    }
    for recurso, ruta in necesarios.items():
        try:
            nltk.data.find(ruta)
        except LookupError:
            try:
                nltk.data.find(ruta + ".zip")
            except LookupError:
                nltk.download(recurso, quiet=True)


asegurar_recursos_nltk()
_lematizador = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Listas de palabras
# ---------------------------------------------------------------------------
# Palabras que niegan. NUNCA se eliminan como stopwords porque cambian el sentido:
# "good" (positivo) frente a "not good" (negativo).
NEGADORES = {"not", "no", "never", "nothing", "nobody", "none", "nowhere",
             "neither", "nor", "cannot", "without"}

# Stopwords de NLTK para inglés, quitando los negadores y sus contracciones
# ("don't", "isn't", "didn"...), que también expresan negación.
RESIDUOS_NEGATIVOS = {"don", "doesn", "didn", "isn", "aren", "wasn", "weren", "hasn",
                      "haven", "hadn", "won", "wouldn", "shouldn", "couldn", "mustn",
                      "mightn", "needn", "shan", "ain"}
STOPWORDS = {
    p for p in stopwords.words("english")
    if p not in NEGADORES and "n't" not in p and p not in RESIDUOS_NEGATIVOS
} | {"'s", "'m", "'re", "'ve", "'ll", "'d"}

# Contracciones que separa word_tokenize ("don't" -> "do" + "n't")
CONTRACCIONES = {"n't": "not", "ca": "can", "wo": "will", "sha": "shall"}

# Signos que cierran el alcance de una negación
FIN_DE_ALCANCE = {".", ",", "!", "?", ";", ":", "...", "--", ")"}
VENTANA_NEGACION = 3   # cuántas palabras después del negador se marcan

# ---------------------------------------------------------------------------
# Limpieza (misma idea que normalizacion.py de la Etapa 1)
# ---------------------------------------------------------------------------
RE_BBCODE = re.compile(r"\[/?[a-zA-Z0-9*]+(=[^\]]*)?\]")      # [h1], [/b], [url=...]
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_ESPACIOS = re.compile(r"\s+")
RE_SIN_APOSTROFO = re.compile(
    r"\b(do|does|did|ca|wo|is|are|was|were|have|has|had|would|should|could|ai)nt\b")
RE_ALFANUM = re.compile(r"\w", re.UNICODE)


def limpiar(texto):
    """Quita HTML, BBCode, URLs y caracteres de control; pasa a minúsculas
    y repara contracciones sin apóstrofo ("dont" -> "don't")."""
    if not isinstance(texto, str):
        return ""
    texto = html.unescape(texto)
    texto = RE_BBCODE.sub(" ", texto)
    texto = RE_URL.sub(" ", texto)
    texto = "".join(" " if unicodedata.category(c).startswith("C") else c for c in texto)
    texto = unicodedata.normalize("NFKC", texto)
    texto = RE_ESPACIOS.sub(" ", texto).strip().lower()
    texto = RE_SIN_APOSTROFO.sub(r"\1n't", texto)
    return texto


def _pos_wordnet(etiqueta):
    """Convierte la etiqueta POS de Penn Treebank al formato de WordNet."""
    if etiqueta.startswith("J"):
        return wn.ADJ
    if etiqueta.startswith("V"):
        return wn.VERB
    if etiqueta.startswith("R"):
        return wn.ADV
    return wn.NOUN


def marcar_negacion(tokens):
    """Antepone NEG_ a las palabras que siguen a un negador, hasta un signo de
    puntuación o hasta VENTANA_NEGACION palabras.
    Ejemplo: ["not", "good", "game", "."] -> ["not", "NEG_good", "NEG_game", "."]"""
    salida, restantes = [], 0
    for tok in tokens:
        base = tok.lower()
        if base in FIN_DE_ALCANCE:
            restantes = 0
            salida.append(tok)
        elif base in NEGADORES or base == "n't":
            restantes = VENTANA_NEGACION
            salida.append(tok)
        elif restantes > 0 and RE_ALFANUM.search(tok):
            salida.append("NEG_" + tok)
            restantes -= 1
        else:
            salida.append(tok)
    return salida


@lru_cache(maxsize=None)
def normalizar(texto, config="N0", negacion=False):
    """Normaliza UNA reseña y devuelve sus tokens separados por espacios.
    El resultado se guarda en memoria (lru_cache) para no repetir el trabajo
    en cada experimento: la función no aprende nada del corpus, así que
    guardarla en caché no provoca fuga de datos."""
    if config == "N0":
        # Línea base: solo tokenizar el texto tal cual (con mayúsculas y signos)
        tokens = word_tokenize(texto if isinstance(texto, str) else "")
        if negacion:
            tokens = marcar_negacion(tokens)
        return " ".join(tokens)

    # N1, N2 y N3 parten del texto limpio
    tokens = word_tokenize(limpiar(texto))
    tokens = [CONTRACCIONES.get(t, t) for t in tokens]

    if config == "N3":   # lematización (se hace antes de quitar signos: el POS los usa)
        etiquetados = pos_tag(tokens)
        tokens = [t if not RE_ALFANUM.search(t) or t in NEGADORES
                  else _lematizador.lemmatize(t, _pos_wordnet(tag))
                  for t, tag in etiquetados]

    if negacion:          # la negación se marca mientras aún existen los signos
        tokens = marcar_negacion(tokens)

    resultado = []
    for t in tokens:
        if not RE_ALFANUM.search(t):            # puntuación y símbolos fuera
            continue
        base = t[4:] if t.startswith("NEG_") else t
        if config in ("N2", "N3") and base in STOPWORDS:
            continue
        resultado.append(t)
    return " ".join(resultado)


class Normalizador(BaseEstimator, TransformerMixin):
    """Paso de Pipeline: recibe textos crudos y devuelve textos normalizados."""

    def __init__(self, config="N0", negacion=False):
        self.config = config
        self.negacion = negacion

    def fit(self, X, y=None):
        return self          # no aprende nada: es una regla fija

    def transform(self, X):
        return [normalizar(str(t), self.config, self.negacion) for t in X]


# ---------------------------------------------------------------------------
# Léxico de opinión de Hu & Liu (inglés): 2,006 positivas y 4,783 negativas
# ---------------------------------------------------------------------------
def _leer_lexico(nombre):
    ruta = RECURSOS / nombre
    with open(ruta, encoding="latin-1") as f:
        return {linea.strip() for linea in f if linea.strip() and not linea.startswith(";")}


LEXICO_POS = _leer_lexico("positive-words.txt")
LEXICO_NEG = _leer_lexico("negative-words.txt")


@lru_cache(maxsize=None)
def contar_lexico(texto):
    """Devuelve (palabras positivas, palabras negativas) encontradas en la reseña."""
    tokens = word_tokenize(limpiar(texto))
    positivas = sum(1 for t in tokens if t in LEXICO_POS)
    negativas = sum(1 for t in tokens if t in LEXICO_NEG)
    return positivas, negativas


class RasgosLexico(BaseEstimator, TransformerMixin):
    """Paso de Pipeline: convierte cada reseña en 2 números
    [nº de palabras positivas, nº de palabras negativas]."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([contar_lexico(str(t)) for t in X], dtype=float)


# ---------------------------------------------------------------------------
# Armado del Pipeline completo
# ---------------------------------------------------------------------------
def crear_clasificador(modelo, hiper=None, balanceo=False):
    hiper = dict(hiper or {})
    if modelo == "MultinomialNB":
        # NB no tiene class_weight; su equivalente es no usar la proporción de
        # clases como probabilidad previa (fit_prior=False -> priors iguales).
        return MultinomialNB(alpha=hiper.get("alpha", 1.0), fit_prior=not balanceo)
    if modelo == "LogisticRegression":
        return LogisticRegression(C=hiper.get("C", 1.0), max_iter=5000,
                                  class_weight="balanced" if balanceo else None,
                                  random_state=SEMILLA)
    if modelo == "LinearSVC":
        return LinearSVC(C=hiper.get("C", 1.0), max_iter=20000,
                         class_weight="balanced" if balanceo else None,
                         random_state=SEMILLA)
    raise ValueError(f"Modelo desconocido: {modelo}")


def crear_vectorizador(representacion, ngramas):
    # Los textos ya vienen normalizados y separados por espacios, por eso
    # cada "palabra" es cualquier secuencia sin espacios (\S+) y no se vuelve
    # a pasar a minúsculas (N0 debe conservar las mayúsculas).
    opciones = dict(ngram_range=tuple(ngramas), token_pattern=r"\S+", lowercase=False)
    if representacion == "tfidf":
        return TfidfVectorizer(**opciones)
    if representacion == "binaria":
        return CountVectorizer(binary=True, **opciones)
    if representacion == "frecuencia":
        return CountVectorizer(**opciones)
    raise ValueError(f"Representación desconocida: {representacion}")


def construir_pipeline(cfg):
    """cfg es un diccionario como:
    {"normalizacion": "N2", "negacion": True, "representacion": "tfidf",
     "ngramas": (1, 2), "lexico": True, "balanceo": False,
     "modelo": "LogisticRegression", "hiper": {"C": 1.0}}"""
    rama_texto = Pipeline([
        ("normalizacion", Normalizador(cfg["normalizacion"], cfg.get("negacion", False))),
        ("vectorizador", crear_vectorizador(cfg["representacion"], cfg["ngramas"])),
    ])
    if cfg.get("lexico", False):
        rama_lexico = Pipeline([("conteos", RasgosLexico()), ("escala", MaxAbsScaler())])
        rasgos = FeatureUnion([("texto", rama_texto), ("lexico", rama_lexico)])
    else:
        rasgos = rama_texto
    return Pipeline([
        ("rasgos", rasgos),
        ("clasificador", crear_clasificador(cfg["modelo"], cfg.get("hiper"),
                                            cfg.get("balanceo", False))),
    ])


def describir(cfg):
    """Texto corto y legible de una configuración (para tablas y reportes)."""
    nombres_norm = {
        "N0": "N0 tokenización",
        "N1": "N1 +limpieza",
        "N2": "N2 +stopwords",
        "N3": "N3 +lematización",
    }
    ng = {(1, 1): "unigramas", (2, 2): "bigramas", (1, 2): "uni+bigramas"}[tuple(cfg["ngramas"])]
    mejoras = [m for m, activo in (("negación", cfg.get("negacion")),
                                   ("léxico", cfg.get("lexico"))) if activo]
    return {
        "normalizacion": nombres_norm[cfg["normalizacion"]],
        "representacion": f"{cfg['representacion']} {ng}",
        "mejora_linguistica": " + ".join(mejoras) if mejoras else "ninguna",
        "desbalance": ("fit_prior=False" if cfg["modelo"] == "MultinomialNB"
                       else "class_weight=balanced") if cfg.get("balanceo") else "ninguno",
        "hiperparametros": ", ".join(f"{k}={v}" for k, v in (cfg.get("hiper") or {}).items())
                           or ("alpha=1.0" if cfg["modelo"] == "MultinomialNB" else "C=1.0"),
    }
