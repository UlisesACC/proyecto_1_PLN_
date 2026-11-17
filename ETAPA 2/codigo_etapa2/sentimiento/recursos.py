"""Lectura de fuentes, polaridad léxica y catálogo editable de emojis.

Los recursos léxicos son externos al corpus. Sus puntajes nunca se ajustan
a las etiquetas de Steam. Cada transformador guarda el recurso usado al entrenar.
"""
import csv
import hashlib
import re
from collections import Counter, defaultdict
from functools import lru_cache

import emoji
import numpy as np
from nltk import pos_tag, word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

from preprocesamiento import _lematizador, _pos_wordnet, limpiar
from .configuracion import FUENTES, HU_LIU, RECURSO_EMOJIS, SENTIWORDNET

AMBIGUAS_STEAM = frozenset({"refund", "grind"})
PALABRA_LEXICO = re.compile(r"[a-z]+(?:[-'][a-z]+)*\+?")


def leer_palabras(ruta):
    with ruta.open(encoding="latin-1") as archivo:
        return frozenset(
            linea.strip().lower()
            for linea in archivo
            if linea.strip() and not linea.lstrip().startswith(";")
        )


@lru_cache(maxsize=2)
def cargar_hu_liu(recurso="hu_liu"):
    if recurso not in {"hu_liu", "hu_liu_steam"}:
        raise ValueError(f"Variante Hu y Liu desconocida: {recurso}")
    positivas = leer_palabras(HU_LIU / "positive-words.txt")
    negativas = leer_palabras(HU_LIU / "negative-words.txt")
    ambiguas = positivas & negativas
    if recurso == "hu_liu_steam":
        ambiguas = ambiguas | AMBIGUAS_STEAM
    return positivas - ambiguas, negativas - ambiguas


@lru_cache(maxsize=1)
def cargar_sentiwordnet():
    """Promedia sentidos por lema y POS, ponderados por 1/número de sentido.

    El formato POS, ID, PosScore, NegScore, SynsetTerms, Gloss está descrito
    en la cabecera del archivo SentiWordNet_3.0.0.txt proporcionado.
    """
    acumulados = defaultdict(lambda: [0.0, 0.0, 0.0])
    with SENTIWORDNET.open(encoding="utf-8") as archivo:
        for numero_linea, linea in enumerate(archivo, start=1):
            if not linea.strip() or linea.lstrip().startswith("#"):
                continue
            campos = linea.rstrip("\n").split("\t")
            if len(campos) != 6 or campos[0] not in {"a", "n", "r", "v"}:
                raise ValueError(f"Registro SentiWordNet inválido en la línea {numero_linea}.")
            pos, _, positivo, negativo, terminos, _ = campos
            positivo, negativo = float(positivo), float(negativo)
            if min(positivo, negativo) < 0 or positivo + negativo > 1:
                raise ValueError(f"Polaridad SentiWordNet inválida en la línea {numero_linea}.")
            for termino in terminos.split():
                lema, sentido = termino.rsplit("#", 1)
                peso = 1.0 / int(sentido)
                suma = acumulados[(lema, pos)]
                suma[0] += peso * positivo
                suma[1] += peso * negativo
                suma[2] += peso
    return {
        clave: (positivo / pesos, negativo / pesos)
        for clave, (positivo, negativo, pesos) in acumulados.items()
    }


@lru_cache(maxsize=5000)
def lemas_con_pos(texto):
    etiquetas = pos_tag(word_tokenize(limpiar(texto)))
    return tuple(
        (_lematizador.lemmatize(token, _pos_wordnet(tag)), _pos_wordnet(tag))
        for token, tag in etiquetas if token.isalpha()
    )


def puntajes_sentiwordnet(texto, lexico):
    pares = [lexico.get(clave, (0.0, 0.0)) for clave in lemas_con_pos(texto)]
    return sum(p[0] for p in pares), sum(p[1] for p in pares)


def conteos_lexico(texto, recurso, lexico=None):
    if recurso == "sentiwordnet":
        if lexico is None:
            lexico = cargar_sentiwordnet()
        return puntajes_sentiwordnet(texto, lexico)
    if lexico is None:
        lexico = cargar_hu_liu(recurso)
    positivas, negativas = lexico
    tokens = PALABRA_LEXICO.findall(limpiar(texto))
    return sum(t in positivas for t in tokens), sum(t in negativas for t in tokens)


def inventario_fuentes():
    """Registra rutas y hashes junto a la decisión de uso para el corpus inglés."""
    razones = {
        "positive-words.txt": ("en", "usado", "Hu y Liu: conteo de palabras positivas."),
        "negative-words.txt": ("en", "usado", "Hu y Liu: conteo de palabras negativas."),
        "SentiWordNet_3.0.0.txt": ("en", "usado", "Sumas positiva y negativa por lema y POS."),
        "SEL_full.txt": ("es", "referencia", "Léxico de emociones en español; el corpus Steam está en inglés."),
        "LIWC2007.dic": ("es", "referencia", "Diccionario español con categorías como Yo, Nosotro y Negacio."),
        "Rest_Mex_2022.xlsx": ("es", "referencia", "Corpus de reseñas en español; Steam conserva su colección e IDs."),
        "polaridad_con_lexicon.py": ("es", "adaptado", "Se adapta la concatenación de rasgos de polaridad al espacio vectorial."),
        "polaridad_con_ml.py": ("es", "adaptado", "Texto como X y etiqueta de la fuente como y, con separación previa."),
        "clasificación_texto.py": ("en", "adaptado", "Conteos o TF-IDF, Naive Bayes y regresión logística."),
        "clasificación_texto_pipeline.py": ("en", "adaptado", "Transformaciones y clasificador dentro de Pipeline."),
        "clasificación_texto_pipeline_svd.py": ("en", "adaptado", "TruncatedSVD dentro del pipeline, ajustado solo en train."),
        "clustering_texto_pipeline_svd.py": ("en", "referencia", "Agrupamiento no supervisado; objetivo distinto de la tarea de sentimiento."),
        "lda_pipeline.py": ("en", "referencia", "Modelado de temas; objetivo distinto de la clasificación supervisada."),
    }
    filas = []
    for ruta in sorted(FUENTES.rglob("*")):
        if ruta.is_file():
            idioma, uso, motivo = razones.get(
                ruta.name, ("sin determinar", "pendiente", "Recurso nuevo pendiente de revisión.")
            )
            filas.append({
                "archivo": str(ruta.relative_to(FUENTES)),
                "idioma": idioma, "uso": uso, "justificacion": motivo,
                "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest(),
            })
    return filas


@lru_cache(maxsize=1)
def cargar_emojis():
    """Valida el catálogo editable. Reiniciar el proceso después de editarlo."""
    with RECURSO_EMOJIS.open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo, delimiter="\t"))
    simbolos = set()
    for fila in filas:
        simbolo = fila["simbolo"]
        if not simbolo or simbolo in simbolos:
            raise ValueError(f"Símbolo vacío o duplicado en {RECURSO_EMOJIS}: {simbolo!r}")
        simbolos.add(simbolo)
        if fila["tipo"] not in {"emoji", "emoticono"}:
            raise ValueError(f"Tipo inválido para {simbolo!r}")
        if fila["polaridad"] not in {"positive", "negative", "sin_asignar"}:
            raise ValueError(f"Polaridad inválida para {simbolo!r}")
        esperado = {"positive": "1", "negative": "-1", "sin_asignar": ""}[fila["polaridad"]]
        if fila["puntaje"] != esperado:
            raise ValueError(f"Puntaje incompatible con la polaridad de {simbolo!r}")
    return tuple((f["simbolo"], f["tipo"], f["polaridad"]) for f in filas)


@lru_cache(maxsize=8)
def preparar_catalogo_emojis(catalogo):
    polaridades = {simbolo: polaridad for simbolo, _, polaridad in catalogo}
    emoticonos = [simbolo for simbolo, tipo, _ in catalogo if tipo == "emoticono"]
    alternativas = "|".join(re.escape(s) for s in sorted(emoticonos, key=lambda s: (-len(s), s)))
    patron = re.compile(r"(?<!\w)(?:" + alternativas + r")(?!\w)") if alternativas else None
    return polaridades, patron


def conteos_emojis(texto, catalogo=None):
    """Cuenta ocurrencias completas, incluidas variantes y secuencias ZWJ."""
    if catalogo is None:
        catalogo = cargar_emojis()
    polaridades, patron = preparar_catalogo_emojis(catalogo)
    simbolos = [entrada["emoji"] for entrada in emoji.emoji_list(texto)]
    if patron is not None:
        simbolos.extend(patron.findall(texto))
    conteos = Counter(polaridades.get(s, "sin_asignar") for s in simbolos)
    return conteos["positive"], conteos["negative"], conteos["sin_asignar"]


class RasgosSentimiento(TransformerMixin, BaseEstimator):
    """Concatena dos rasgos del léxico y/o tres conteos de emojis."""

    def __init__(self, recurso, emojis):
        self.recurso = recurso
        self.emojis = emojis

    def fit(self, X, y=None):
        self.catalogo_emojis_ = cargar_emojis() if self.emojis else ()
        self.lexico_ = None
        if self.recurso == "sentiwordnet":
            self.lexico_ = cargar_sentiwordnet()
        elif self.recurso:
            self.lexico_ = cargar_hu_liu(self.recurso)
        return self

    def transform(self, X):
        # Los modelos anteriores a este módulo no guardaban lexico_.
        lexico = getattr(self, "lexico_", None)
        filas = []
        for texto in X:
            rasgos = conteos_lexico(texto, self.recurso, lexico) if self.recurso else ()
            if self.emojis:
                rasgos += conteos_emojis(texto, self.catalogo_emojis_)
            filas.append(rasgos)
        return np.asarray(filas, dtype=float)
