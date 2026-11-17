"""Ejecuta la Etapa II utilizando los recursos locales de fuentes.

Desde codigo_etapa2: python experimentos_ampliados.py
La implementación está organizada en el paquete sentimiento.
"""
import argparse
import pickle
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from sentimiento.busqueda import Ensayo, Experimentos
from sentimiento.configuracion import Configuracion, CLASES, MODELOS, SALIDA, SEMILLA
from sentimiento.datos import preparar_datos
from sentimiento.modelos import ajustar, construir_pipeline
from sentimiento.normalizacion import NormalizadorAmpliado
from sentimiento.recursos import RasgosSentimiento
from sentimiento.resultados import (
    fila_ensayo, guardar_auditoria, guardar_errores, guardar_evaluacion,
    guardar_experimentos, guardar_json, guardar_tabla, guardar_transformaciones,
)


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
    main()
