"""Carga, validación de etiquetas Steam y particiones estratificadas."""
import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .configuracion import CLASES, CORPUS, FUENTE, MAPEO, PROPORCION_MINORITARIA, SEMILLA


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

