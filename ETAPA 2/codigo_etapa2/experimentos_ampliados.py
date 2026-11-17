"""Etapa II: experimentos controlados de sentimiento para reseñas de Steam.

Ejecutar con python experimentos_ampliados.py. Usa un directorio de salida
independiente y conserva la partición histórica de la Etapa II.
"""

import argparse
import hashlib
import json
import pickle
import platform
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import emoji
import numpy as np
import pandas as pd
from nltk import pos_tag, word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.class_weight import compute_sample_weight
from spellchecker import SpellChecker

from experimentos import CLASES, MODELOS, REJILLA, calcular_metricas
from preprocesamiento import (
    CONTRACCIONES,
    LEXICO_NEG,
    LEXICO_POS,
    NEGADORES,
    RE_ALFANUM,
    SEMILLA,
    STOPWORDS,
    _lematizador,
    _pos_wordnet,
    crear_clasificador,
    crear_vectorizador,
    limpiar,
    marcar_negacion,
)

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parents[1]
CORPUS = RAIZ / "opinions_raw_corpus.tsv"
FUENTE = RAIZ / "steam_reviews_raw.jsonl"
SALIDA = CARPETA / "resultados_ampliados"
MAPEO = {"negative": "negative", "positive": "positive"}
NIVELES = ("N0", "N1", "N2", "N3")
PROPORCION_MINORITARIA = 0.25
SVD_COMPONENTES = 50

MEJORAS = (
    "negacion", "hu_liu", "hu_liu_steam", "emojis",
    "ortografia", "letras_repetidas", "palabras_repetidas",
)
LETRAS = re.compile(r"([a-z])\1{2,}")
PALABRA_SIMPLE = re.compile(r"[a-z]{4,24}")
PALABRA_LEXICO = re.compile(r"[a-z]+(?:[-'][a-z]+)*\+?")
EMOTICON = re.compile(r"(?<!\w)(?:[:;=][\-^]?[)D(]|</?3)(?!\w)")
EMOTICON_POS = {":)", ":-)", ";)", ";-)", ":D", ":-D", "=)", "=D", "<3"}
EMOTICON_NEG = {":(", ":-(", ";(", ";-(", "=(", "</3"}
EMOJI_POS = set(emoji.distinct_emoji_list("😀😃😄😁😊😍🥰😎🤩👍❤️❤💖🔥🎉"))
EMOJI_NEG = set(emoji.distinct_emoji_list("😞😠😡🤬😢😭😤🤮👎💔😒"))
PROTEGIDAS = frozenset("""
    steam valve dota portal counter-strike counterstrike skyrim elder scrolls bethesda
    terraria re-logic stardew minecraft roblox csgo cs2 fps pvp pve rpg mmorpg npc
    nerf nerfed buff buffed grind grinding grindy roguelike roguelite multiplayer
    gameplay matchmaking cheater cheaters smurf smurfing noob noobs gg dlc mod mods
    modding laggy iaa lmao lmfa lol rofl bruh ngl tbh ez op
""".split())
# Neutralización explícita de usos ambiguos ya identificados en el informe anterior.
AMBIGUAS_STEAM = frozenset({"refund", "grind"})
SOLAPADAS = LEXICO_POS & LEXICO_NEG
LEXICOS = {
    "hu_liu": (frozenset(LEXICO_POS - SOLAPADAS), frozenset(LEXICO_NEG - SOLAPADAS)),
    "hu_liu_steam": (
        frozenset(LEXICO_POS - SOLAPADAS - AMBIGUAS_STEAM),
        frozenset(LEXICO_NEG - SOLAPADAS - AMBIGUAS_STEAM),
    ),
}


@dataclass(frozen=True)
class Configuracion:
    modelo: str
    normalizacion: str = "N0"
    representacion: str = "frecuencia"
    ngramas: tuple = (1, 1)
    mejoras: tuple = ()
    balanceo: str = "ninguno"
    hiperparametro: float = 1.0
    svd: int = 0


# Datos y auditoría de etiquetas

def verificar_steam(datos):
    """Contrasta el TSV con los booleanos originales; protege contra null -> negative."""
    ids = set(datos["review_id"])
    observadas = {}
    with FUENTE.open(encoding="utf-8") as archivo:
        for linea in archivo:
            pagina = json.loads(linea)
            for review in pagina["data"]["reviews"]:
                id_review = str(review.get("recommendationid"))
                if id_review in ids:
                    voto = review.get("voted_up")
                    observadas.setdefault(id_review, set()).add(
                        voto if type(voto) is bool else None
                    )

    sin_fuente = len(ids - observadas.keys())
    votos_faltantes = sum(None in votos for votos in observadas.values())
    discrepancias = sum(
        observadas.get(fila.review_id, set()) != {fila.label == "positive"}
        for fila in datos.itertuples()
    )
    return {
        "ids_steam_sin_fuente": sin_fuente,
        "ids_steam_con_voto_faltante_o_invalido": votos_faltantes,
        "discrepancias_con_steam": discrepancias,
    }


def preparar_datos():
    datos = pd.read_csv(CORPUS, sep="\t", dtype=str, keep_default_na=False)
    if datos["review_id"].duplicated().any() or datos["review_id"].str.strip().eq("").any():
        raise ValueError("El corpus debe tener review_id únicos y completos.")
    if datos["review"].str.strip().eq("").any():
        raise ValueError("Hay reseñas sin texto en el corpus.")
    rating = datos["rating"].str.strip()
    datos["label"] = rating.map(MAPEO)
    auditoria = {
        "corpus": CORPUS.name,
        "sha256_corpus": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "seleccion": "Todas las reseñas del corpus congelado de la Etapa I.",
        "escala_original": "Steam voted_up booleano",
        "mapeo": {"true": "positive", "false": "negative"},
        "mapeo_rating": MAPEO,
        "rating_faltante": int(rating.eq("").sum()),
        "rating_fuera_del_mapeo": int((rating.ne("") & datos["label"].isna()).sum()),
        "distribucion_antes_de_particionar": dict(Counter(datos["label"].dropna())),
    }
    if datos["label"].isna().any():
        raise ValueError(f"Calificaciones inválidas: {auditoria}")
    auditoria.update(verificar_steam(datos))
    if auditoria["discrepancias_con_steam"]:
        raise ValueError(f"No coinciden las etiquetas con Steam: {auditoria}")

    proporcion = datos["label"].value_counts(normalize=True).min()
    auditoria["proporcion_minoritaria"] = float(proporcion)
    auditoria["desbalance_relevante"] = bool(proporcion < PROPORCION_MINORITARIA)
    auditoria["criterio_desbalance"] = (
        "Umbral exploratorio: probar tratamiento si la clase menor representa menos del 25 %."
    )
    auditoria["particion"] = {
        "train": 0.70, "dev": 0.15, "test": 0.15,
        "estratificacion": "label, para conservar ambas clases y su proporción original",
        "shuffle": True, "random_state": SEMILLA,
    }

    indices = np.arange(len(datos))
    train, resto = train_test_split(
        indices, test_size=0.30, shuffle=True, random_state=SEMILLA, stratify=datos["label"]
    )
    dev, test = train_test_split(
        resto, test_size=0.50, shuffle=True, random_state=SEMILLA,
        stratify=datos.loc[resto, "label"],
    )
    for nombre, indices_split in (("train", train), ("dev", dev), ("test", test)):
        if set(datos.loc[indices_split, "label"]) != set(CLASES):
            raise ValueError(f"La partición {nombre} debe contener ambas clases.")
        datos.loc[indices_split, "split"] = nombre

    datos = datos.rename(columns={"review_id": "document_id"})
    datos["text"] = datos["review"]
    datos["doc_id_etapa1"] = np.arange(1, len(datos) + 1)
    columnas = [
        "document_id", "review", "text", "rating", "label", "split",
        "doc_id_etapa1", "item_name", "category", "votes_up", "playtime_hours",
    ]
    return datos[columnas], auditoria


# Normalización y características lingüísticas

@lru_cache(maxsize=1)
def diccionario_ortografico():
    return SpellChecker(language="en", distance=1)


@lru_cache(maxsize=20000)
def corregir(palabra):
    if not PALABRA_SIMPLE.fullmatch(palabra) or palabra in PROTEGIDAS | NEGADORES:
        return palabra
    diccionario = diccionario_ortografico()
    if palabra in diccionario:
        return palabra
    candidatos = diccionario.candidates(palabra) or ()
    # Desempate alfabético para evitar depender del orden de un conjunto.
    return min(candidatos, key=lambda p: (-diccionario[p], p), default=palabra)


def comprimir_letras(token):
    base = token.lower()
    if not base.isalpha() or not LETRAS.search(base):
        return token
    doble = LETRAS.sub(r"\1\1", base)
    simple = LETRAS.sub(r"\1", base)
    diccionario = diccionario_ortografico()
    if doble in diccionario:
        reducido = doble
    elif simple in diccionario:
        reducido = simple
    else:
        reducido = doble
    return reducido.upper() if token.isupper() else reducido.capitalize() if token.istitle() else reducido


def comprimir_palabras(tokens):
    salida = []
    for token in tokens:
        if token.isalpha() and token.lower() not in NEGADORES and salida:
            if token.lower() == salida[-1].lower():
                continue
        salida.append(token)
    return salida


@lru_cache(maxsize=30000)
def normalizar(texto, nivel, mejoras):
    tokens = word_tokenize(texto if nivel == "N0" else limpiar(texto))
    if nivel != "N0":
        tokens = [CONTRACCIONES.get(token, token) for token in tokens]
    if "letras_repetidas" in mejoras:
        tokens = [comprimir_letras(token) for token in tokens]
    if "ortografia" in mejoras:
        tokens = [
            token if nivel == "N0" and token != token.lower() else corregir(token)
            for token in tokens
        ]
    if "palabras_repetidas" in mejoras:
        tokens = comprimir_palabras(tokens)
    if nivel == "N3":
        tokens = [
            _lematizador.lemmatize(token, _pos_wordnet(tag))
            if RE_ALFANUM.search(token) and token not in NEGADORES else token
            for token, tag in pos_tag(tokens)
        ]
    if "negacion" in mejoras:
        tokens = marcar_negacion(tokens)
    if nivel != "N0":
        tokens = [token for token in tokens if RE_ALFANUM.search(token)]
    if nivel in ("N2", "N3"):
        tokens = [token for token in tokens if token.removeprefix("NEG_") not in STOPWORDS]
    return " ".join(tokens)


class NormalizadorAmpliado(TransformerMixin, BaseEstimator):
    def __init__(self, nivel, mejoras):
        self.nivel = nivel
        self.mejoras = mejoras

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalizar(texto, self.nivel, self.mejoras) for texto in X]


@lru_cache(maxsize=5000)
def conteos_lexico(texto, recurso):
    positivas, negativas = LEXICOS[recurso]
    tokens = PALABRA_LEXICO.findall(limpiar(texto))
    return sum(t in positivas for t in tokens), sum(t in negativas for t in tokens)


@lru_cache(maxsize=5000)
def conteos_emojis(texto):
    unicode = [entrada["emoji"] for entrada in emoji.emoji_list(texto)]
    ascii = EMOTICON.findall(texto)
    positivos = sum(e in EMOJI_POS for e in unicode) + sum(e in EMOTICON_POS for e in ascii)
    negativos = sum(e in EMOJI_NEG for e in unicode) + sum(e in EMOTICON_NEG for e in ascii)
    return positivos, negativos, len(unicode) + len(ascii) - positivos - negativos


class RasgosSentimiento(TransformerMixin, BaseEstimator):
    def __init__(self, recurso, emojis):
        self.recurso = recurso
        self.emojis = emojis

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray([
            (conteos_lexico(texto, self.recurso) if self.recurso else ())
            + (conteos_emojis(texto) if self.emojis else ())
            for texto in X
        ], dtype=float)


def construir_pipeline(config):
    parametros = {REJILLA[config.modelo][0]: config.hiperparametro}
    ponderar = config.balanceo in {"prior_uniforme", "class_weight"}
    etapas_texto = [
        ("normalizacion", NormalizadorAmpliado(config.normalizacion, config.mejoras)),
        ("vectorizador", crear_vectorizador(config.representacion, config.ngramas)),
    ]
    if config.svd:
        if config.modelo == "MultinomialNB":
            raise ValueError("MultinomialNB necesita rasgos no negativos; SVD puede dar negativos.")
        etapas_texto.append(("svd", TruncatedSVD(n_components=config.svd, random_state=SEMILLA)))
    texto = Pipeline(etapas_texto)
    recurso = next((m for m in config.mejoras if m in LEXICOS), None)
    usa_emojis = "emojis" in config.mejoras
    rasgos = texto
    if recurso or usa_emojis:
        rasgos = FeatureUnion([
            ("texto", texto),
            ("sentimiento", Pipeline([
                ("conteos", RasgosSentimiento(recurso, usa_emojis)),
                ("escala", MaxAbsScaler()),
            ])),
        ])
    clasificador = crear_clasificador(config.modelo, parametros, ponderar)
    return Pipeline([("rasgos", rasgos), ("clasificador", clasificador)])


def ajustar(config, train):
    pipeline = construir_pipeline(config)
    parametros = {}
    if config.balanceo == "peso_muestras":
        parametros["clasificador__sample_weight"] = compute_sample_weight(
            class_weight="balanced", y=train["label"]
        )
    return pipeline.fit(train["text"], train["label"], **parametros)


# Experimentos y selección exclusivamente con dev

@dataclass
class Ensayo:
    identificador: str
    config: Configuracion
    etapa: str
    cambio: str
    referencia: str
    delta: float
    metricas: dict
    predicciones: np.ndarray

    def calidad(self):
        return self.metricas["macro_f1"], self.metricas["f1_negative"]


class Experimentos:
    def __init__(self, train, dev):
        self.train = train
        self.dev = dev
        self.ensayos = []
        self.ablaciones = []
        self.dos_mejoras = []

    def correr(self, config, etapa, cambio, referencia=None):
        pred = ajustar(config, self.train).predict(self.dev["text"])
        metricas = calcular_metricas(self.dev["label"], pred)
        delta = metricas["macro_f1"] - referencia.metricas["macro_f1"] if referencia else 0.0
        ensayo = Ensayo(
            f"A{len(self.ensayos) + 1:03}", config, etapa, cambio,
            referencia.identificador if referencia else "", delta, metricas, pred,
        )
        self.ensayos.append(ensayo)
        return ensayo

    def comparar_mejoras(self, base):
        individuales = [
            self.correr(replace(base.config, mejoras=(mejora,)), "aumento", mejora, base)
            for mejora in MEJORAS
        ]
        # Las dos variantes Hu–Liu son alternativas del mismo recurso.
        orden = sorted(individuales, key=Ensayo.calidad, reverse=True)
        primera = orden[0]
        segunda = next(
            e for e in orden[1:]
            if not (e.cambio in LEXICOS and primera.cambio in LEXICOS)
        )
        self.dos_mejoras.extend((primera, segunda))

        pares = dict.fromkeys((
            tuple(sorted((primera.cambio, segunda.cambio))),
            ("letras_repetidas", "ortografia"),
            ("letras_repetidas", "palabras_repetidas"),
        ))
        combinaciones = [
            self.correr(replace(base.config, mejoras=par), "combinación", " + ".join(par), base)
            for par in pares
        ]
        self.ablaciones.extend([base, *individuales, *combinaciones])
        return max([base, *individuales, *combinaciones], key=Ensayo.calidad)

    def explorar(self, modelo, hay_desbalance):
        base = self.correr(Configuracion(modelo), "normalización", "línea base sin aumento")
        niveles = [
            self.correr(replace(base.config, normalizacion=nivel), "normalización", nivel, base)
            for nivel in NIVELES[1:]
        ]
        actual = max([base, *niveles], key=Ensayo.calidad)

        referencia = actual
        representaciones = [
            self.correr(
                replace(referencia.config, representacion=rep, ngramas=ng),
                "representación", f"{rep} {ng}", referencia,
            )
            for rep in ("frecuencia", "binaria", "tfidf")
            for ng in ((1, 1), (2, 2), (1, 2))
            if (rep, ng) != (referencia.config.representacion, referencia.config.ngramas)
        ]
        actual = max([referencia, *representaciones], key=Ensayo.calidad)
        actual = self.comparar_mejoras(actual)

        if modelo != "MultinomialNB":
            reducido = self.correr(
                replace(actual.config, svd=SVD_COMPONENTES), "SVD", f"{SVD_COMPONENTES} componentes",
                actual,
            )
            actual = max([actual, reducido], key=Ensayo.calidad)

        if hay_desbalance:
            referencia = actual
            estrategias = ["peso_muestras", "prior_uniforme"] if modelo == "MultinomialNB" else ["class_weight"]
            balanceados = [
                self.correr(replace(referencia.config, balanceo=metodo), "desbalance", metodo, referencia)
                for metodo in estrategias
            ]
            actual = max([referencia, *balanceados], key=Ensayo.calidad)

        parametro, rejilla = REJILLA[modelo]
        referencia = actual
        ajustes = [
            self.correr(
                replace(referencia.config, hiperparametro=valor),
                "hiperparámetros", f"{parametro}={valor}", referencia,
            )
            for valor in rejilla
            if valor != referencia.config.hiperparametro
        ]
        return max([referencia, *ajustes], key=Ensayo.calidad)


# Resultados y diagnóstico

def guardar_tabla(salida, nombre, filas):
    tabla = filas if isinstance(filas, pd.DataFrame) else pd.DataFrame(filas)
    tabla.to_csv(salida / f"{nombre}.tsv", sep="\t", index=False, float_format="%.6f")


def guardar_json(salida, nombre, datos):
    (salida / f"{nombre}.json").write_text(
        json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def fila_ensayo(ensayo):
    config = ensayo.config
    return {
        "experiment_id": ensayo.identificador,
        "etapa": ensayo.etapa,
        "cambio": ensayo.cambio,
        "referencia_dev": ensayo.referencia,
        "delta_macro_f1_dev": ensayo.delta,
        "model": config.modelo,
        "normalizacion": config.normalizacion,
        "representacion": config.representacion,
        "ngramas": str(config.ngramas),
        "mejora_linguistica": " + ".join(config.mejoras) or "ninguna",
        "desbalance": config.balanceo,
        "hiperparametros": json.dumps({REJILLA[config.modelo][0]: config.hiperparametro}),
        "svd": config.svd,
        **ensayo.metricas,
    }


def guardar_experimentos(salida, registro, ganadoras, final):
    ids_ganadoras = {e.identificador for e in ganadoras}
    bitacora = []
    for ensayo in registro.ensayos:
        decision = "no seleccionado"
        if ensayo.identificador in ids_ganadoras:
            decision = "mejor de su modelo"
        if ensayo is final:
            decision = "ganador global"
        bitacora.append({**fila_ensayo(ensayo), "split": "dev", "decision": decision})
    guardar_tabla(salida, "experiments", bitacora)
    guardar_tabla(salida, "ablacion_mejoras", [fila_ensayo(e) for e in registro.ablaciones])
    guardar_tabla(salida, "dos_mejores_aumentos", [fila_ensayo(e) for e in registro.dos_mejoras])


def guardar_auditoria(salida, datos, train, auditoria):
    distribucion = datos.groupby(["split", "label"]).size().unstack(fill_value=0)
    distribucion = distribucion.loc[["train", "dev", "test"]]
    distribucion["total"] = distribucion[CLASES].sum(axis=1)
    distribucion["proporcion_negative"] = distribucion["negative"] / distribucion["total"]
    guardar_tabla(salida, "distribucion_clases", distribucion.reset_index())

    ruta_manual = CARPETA / "resultados" / "revision_manual_etiquetas.tsv"
    manual = pd.read_csv(ruta_manual, sep="\t", dtype=str, keep_default_na=False)
    referencias = train.set_index("document_id")
    for fila in manual.itertuples():
        if fila.document_id not in referencias.index:
            raise ValueError(f"La revisión manual {fila.document_id} debe pertenecer a train.")
        if referencias.at[fila.document_id, "label"] != fila.label:
            raise ValueError(f"Cambió la etiqueta revisada {fila.document_id}.")
    auditoria["revision_manual_etiquetas_train"] = len(manual)
    guardar_tabla(salida, "revision_manual_etiquetas", manual)
    guardar_json(salida, "auditoria_corpus", auditoria)

    filas_cobertura = []
    for recurso, (positivas, negativas) in LEXICOS.items():
        conteos = np.asarray([conteos_lexico(texto, recurso) for texto in train["text"]])
        filas_cobertura.append({
            "recurso": recurso,
            "terminos_positivos": len(positivas),
            "terminos_negativos": len(negativas),
            "filas_train_con_cobertura": int(conteos.any(axis=1).sum()),
            "proporcion_train_con_cobertura": float(conteos.any(axis=1).mean()),
        })
    guardar_tabla(salida, "cobertura_lexicos", filas_cobertura)
    emojis_train = np.asarray([conteos_emojis(texto) for texto in train["text"]])
    guardar_json(salida, "recursos_linguisticos", {
        "hu_liu": {
            "idioma": "inglés",
            "atribucion": "Conservar los avisos originales y citar a Hu y Liu.",
            "referencia": "Hu y Liu, Mining and Summarizing Customer Reviews, KDD 2004",
            "solapadas_excluidas": sorted(SOLAPADAS),
            "exclusiones_variante_steam": sorted(AMBIGUAS_STEAM),
            "archivos": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((CARPETA / "recursos").glob("*-words.txt"))
            },
        },
        "emojis": {
            "positivos": sorted(EMOJI_POS), "negativos": sorted(EMOJI_NEG),
            "emoticonos_positivos": sorted(EMOTICON_POS),
            "emoticonos_negativos": sorted(EMOTICON_NEG),
            "filas_train_con_emojis_o_emoticonos": int(emojis_train.any(axis=1).sum()),
        },
        "ortografia": {"distancia": 1, "idioma": "en", "protegidas": sorted(PROTEGIDAS)},
        "negacion": "Conserva negadores; marcado opcional por ventana de tres palabras.",
    })


def guardar_transformaciones(salida, train, dev):
    modos = {
        "ortografia": ("ortografia",),
        "letras": ("letras_repetidas",),
        "palabras": ("palabras_repetidas",),
        "letras_y_ortografia": ("letras_repetidas", "ortografia"),
    }
    filas = []
    for fila in pd.concat([train, dev]).itertuples():
        base = normalizar(fila.text, "N1", ())
        cambios = {
            nombre: normalizar(fila.text, "N1", mejoras)
            for nombre, mejoras in modos.items()
        }
        if any(texto != base for texto in cambios.values()):
            filas.append({
                "document_id": fila.document_id, "split": fila.split,
                "original": fila.text, "base_N1": base, **cambios,
            })
    guardar_tabla(salida, "transformaciones", filas)


def guardar_errores(salida, registro, ganadoras):
    filas = []
    ensayos = {ensayo.identificador: ensayo for ensayo in registro.ensayos}
    for ganadora in ganadoras:
        base = next(
            ensayo for ensayo in registro.ablaciones
            if ensayo.config.modelo == ganadora.config.modelo and not ensayo.config.mejoras
        )
        variantes = [e for e in registro.ablaciones if e.referencia == base.identificador]
        analisis = [e for e in variantes if e is not ganadora] + [ganadora]
        for ensayo in analisis:
            ref = ensayos.get(ensayo.referencia, base)
            if ensayo is ganadora:
                ref = base
            for fila, anterior, pred in zip(
                registro.dev.itertuples(), ref.predicciones, ensayo.predicciones, strict=True,
            ):
                if pred == anterior == fila.label:
                    continue
                efecto = "persiste"
                if pred != anterior:
                    efecto = "corregido" if pred == fila.label else "introducido"
                filas.append({
                    "document_id": fila.document_id, "gold_label": fila.label,
                    "predicted_label": pred, "prediccion_base": anterior,
                    "model": ensayo.config.modelo, "experiment_id": ensayo.identificador,
                    "referencia": ref.identificador, "cambio": ensayo.cambio, "efecto": efecto,
                    "modo": "seleccionado" if ensayo is ganadora else "ablacion",
                    "review": fila.text,
                    "texto_normalizado": normalizar(
                        fila.text, ensayo.config.normalizacion, ensayo.config.mejoras,
                    ),
                })
    guardar_tabla(salida, "errores_dev", filas)
    seleccionados = pd.DataFrame(filas).query(
        'modo == "seleccionado" and gold_label != predicted_label'
    )
    casos = pd.read_csv(CARPETA / "recursos" / "diagnostico_dev.tsv", sep="\t", dtype=str)
    errores = seleccionados.merge(casos, on="document_id", how="inner", validate="many_to_one")
    manual = errores.groupby(
        ["document_id", "gold_label", "predicted_label", "categoria", "explicacion", "review"],
        as_index=False,
    ).agg(modelos_que_fallan=("model", ", ".join))
    guardar_tabla(salida, "analisis_errores_dev", manual)


def guardar_evaluacion(salida, ganadoras, dev, final, test, pred_test):
    import matplotlib.pyplot as plt

    figuras = salida / "figuras"
    figuras.mkdir(exist_ok=True)
    predicciones, metricas, reportes = [], [], []
    evaluaciones = [
        (e.identificador, e.config.modelo, "dev", dev, e.predicciones)
        for e in ganadoras
    ]
    evaluaciones.append(("test", final.config.modelo, "test", test, pred_test))
    for identificador, modelo, split, datos, pred in evaluaciones:
        gold = datos["label"]
        predicciones.append(pd.DataFrame({
            "document_id": datos["document_id"], "gold_label": gold,
            "predicted_label": pred, "split": split, "model": modelo,
            "experiment_id": identificador,
        }))
        metricas.append({
            "experiment_id": identificador, "model": modelo, "split": split,
            **calcular_metricas(gold, pred),
        })
        reportes.append(f"{split} | {identificador} | {modelo}\n" + classification_report(
            gold, pred, labels=CLASES, digits=4, zero_division=0,
        ))
        figura = ConfusionMatrixDisplay.from_predictions(gold, pred, labels=CLASES, cmap="Blues")
        figura.ax_.set_title(f"{split} · {identificador} · {modelo}")
        figura.figure_.savefig(figuras / f"{split}_{modelo}.png", bbox_inches="tight", dpi=150)
        plt.close(figura.figure_)
    guardar_tabla(salida, "classification_predictions", pd.concat(predicciones))
    guardar_tabla(salida, "classification_metrics", metricas)
    (salida / "classification_reports.txt").write_text("\n".join(reportes), encoding="utf-8")


def ejecutar(salida):
    salida.mkdir(parents=True, exist_ok=True)
    datos, auditoria = preparar_datos()
    train, dev, test = (datos[datos["split"].eq(split)] for split in ("train", "dev", "test"))
    guardar_tabla(salida, "labeled_documents", datos)
    guardar_auditoria(salida, datos, train, auditoria)

    registro = Experimentos(train, dev)
    ganadoras = [registro.explorar(modelo, auditoria["desbalance_relevante"]) for modelo in MODELOS]
    final = max(ganadoras, key=Ensayo.calidad)
    guardar_experimentos(salida, registro, ganadoras, final)
    guardar_transformaciones(salida, train, dev)
    guardar_errores(salida, registro, ganadoras)
    guardar_json(salida, "entorno", {
        "python": platform.python_version(),
        "semilla": SEMILLA,
        "bibliotecas": {
            p: version(p) for p in (
                "scikit-learn", "nltk", "pandas", "numpy", "scipy",
                "matplotlib", "pyspellchecker", "emoji",
            )
        },
    })

    # Se publica la decisión de dev antes de ajustar y consultar test.
    guardar_json(salida, "config_final", {
        "experiment_id": final.identificador,
        "config": asdict(final.config),
        "macro_f1_dev": final.metricas["macro_f1"],
        "descripcion": {
            "normalizacion": final.config.normalizacion,
            "representacion": f"{final.config.representacion} {final.config.ngramas}",
            "mejoras": " + ".join(final.config.mejoras) or "ninguna",
            "desbalance": final.config.balanceo,
            "hiperparametros": fila_ensayo(final)["hiperparametros"],
            "svd": final.config.svd,
        },
        "orden_clases": CLASES,
        "entrenado_con": "train+dev",
    })
    modelo = ajustar(final.config, pd.concat([train, dev]))
    pred_test = modelo.predict(test["text"])
    guardar_evaluacion(salida, ganadoras, dev, final, test, pred_test)
    with (salida / "modelo_final.pkl").open("wb") as archivo:
        pickle.dump(modelo, archivo)
    print(f"{len(registro.ensayos)} experimentos. Ganador: {final.identificador} {final.config.modelo}")
    print(pd.read_csv(salida / "classification_metrics.tsv", sep="\t")[
        ["experiment_id", "split", "accuracy", "macro_f1", "f1_negative"]
    ].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--predecir", metavar="RESEÑA", help="Usa el modelo guardado de esa salida.")
    args = parser.parse_args()
    if args.predecir is None:
        ejecutar(args.salida)
    else:
        with (args.salida / "modelo_final.pkl").open("rb") as archivo:
            modelo = pickle.load(archivo)
        print(modelo.predict([args.predecir])[0])


if __name__ == "__main__":
    # Pickle necesita un nombre de módulo importable desde la interfaz.
    from experimentos_ampliados import main as ejecutar_cli

    ejecutar_cli()
