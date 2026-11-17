"""
interfaz.py — Interfaz de consola de la Etapa 2

Extiende el buscador de la Etapa 1: ahora cada reseña recuperada muestra su
sentimiento predicho y se puede filtrar por sentimiento. También permite ver
las métricas de los experimentos y los errores representativos.

Uso con menú interactivo:
    python interfaz.py

Uso directo con comandos:
    python interfaz.py buscar "game crashes and freezes"
    python interfaz.py buscar "cheaters" --sentimiento negative
    python interfaz.py clasificar "not worth the money"
    python interfaz.py metricas
    python interfaz.py errores
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

from preprocesamiento import construir_pipeline

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parents[1]
RESULTADOS = CARPETA / "resultados"
MODELO_PKL = CARPETA / "modelos" / "modelo_final.pkl"
CONFIG_FINAL = CARPETA / "modelos" / "config_final.json"

_modelo = None


def cargar_modelo():
    """Carga el modelo final. Si el .pkl no existe o no abre (por ejemplo, otra
    versión de scikit-learn), lo vuelve a entrenar con la configuración
    congelada y los datos train+dev; el resultado es el mismo modelo."""
    global _modelo
    if _modelo is not None:
        return _modelo
    try:
        with open(MODELO_PKL, "rb") as f:
            _modelo = pickle.load(f)
    except Exception:
        if not CONFIG_FINAL.exists():
            sys.exit("No existe el modelo final. Ejecuta primero: python experimentos.py")
        cfg = json.loads(CONFIG_FINAL.read_text(encoding="utf-8"))["config"]
        datos = pd.read_csv(RESULTADOS / "labeled_documents.tsv", sep="\t", dtype=str,
                            keep_default_na=False)
        train_dev = datos[datos["split"].isin(["train", "dev"])]
        if "mejoras" in cfg:
            from experimentos_ampliados import Configuracion, ajustar

            cfg["mejoras"] = tuple(cfg["mejoras"])
            cfg["ngramas"] = tuple(cfg["ngramas"])
            _modelo = ajustar(Configuracion(**cfg), train_dev)
        else:
            _modelo = construir_pipeline(cfg).fit(train_dev["text"], train_dev["label"])
    return _modelo


def recortar(texto, n=110):
    texto = " ".join(str(texto).split())
    return texto if len(texto) <= n else texto[: n - 3] + "..."


# ---------------------------------------------------------------------------
# 1. Búsqueda (Etapa 1) + sentimiento predicho (Etapa 2)
# ---------------------------------------------------------------------------
def buscar(consulta, sentimiento="todas", top=10, ancho=110):
    # recuperacion.py usa rutas relativas a la carpeta principal del proyecto
    os.chdir(RAIZ)
    sys.path.insert(0, str(RAIZ))
    import recuperacion

    corpus = recuperacion.read_corpus()
    modelo_busqueda = recuperacion.load_model(corpus, "review", "tfidf", "unigram")
    # Se ordena TODO el corpus por similitud para poder filtrar después
    hits = recuperacion.retrieve(corpus, modelo_busqueda, consulta, top=len(corpus), quiet=True)
    hits = hits[hits["score"] > 0].copy()
    if hits.empty:
        print("La consulta no comparte palabras con el corpus; no hay resultados.")
        return
    hits["sentimiento"] = cargar_modelo().predict(hits["review"])
    if sentimiento != "todas":
        hits = hits[hits["sentimiento"] == sentimiento]
    hits = hits.head(top)
    hits["rank"] = range(1, len(hits) + 1)
    hits["score"] = hits["score"].round(3)
    hits["review"] = hits["review"].map(lambda t: recortar(t, ancho))
    print(f"\nConsulta: {consulta!r} | filtro de sentimiento: {sentimiento} | resultados: {len(hits)}")
    print(hits[["rank", "document_id", "score", "item_name", "sentimiento", "review"]]
          .to_string(index=False, justify="left",
                     formatters={"review": lambda t: f"{t:<{ancho}}",
                                 "item_name": lambda t: f"{t:<18}"}))


# ---------------------------------------------------------------------------
# 2. Clasificar un texto escrito por el usuario
# ---------------------------------------------------------------------------
def clasificar(texto):
    print(f"Sentimiento predicho: {cargar_modelo().predict([texto])[0]}")


# ---------------------------------------------------------------------------
# 3. Métricas y 4. errores representativos
# ---------------------------------------------------------------------------
def metricas():
    cfg = json.loads(CONFIG_FINAL.read_text(encoding="utf-8"))
    print(f"\nModelo final: {cfg['config']['modelo']} (configuración {cfg['experiment_id']})")
    for k, v in cfg["descripcion"].items():
        print(f"  {k}: {v}")
    m = pd.read_csv(RESULTADOS / "classification_metrics.tsv", sep="\t")
    print("\nMejor configuración de cada modelo en dev y prueba final en test:")
    print(m[["experiment_id", "model", "split", "accuracy", "macro_precision", "macro_recall",
             "macro_f1", "f1_negative", "f1_positive", "confusion_matrix"]].to_string(index=False))
    e = pd.read_csv(RESULTADOS / "experiments.tsv", sep="\t")
    print("\nLos 10 mejores experimentos en dev (de", len(e), "en total):")
    print(e.sort_values(["macro_f1", "experiment_id"], ascending=[False, True]).head(10)[
        ["experiment_id", "model", "normalizacion", "representacion", "mejora_linguistica",
         "desbalance", "hiperparametros", "macro_f1"]].to_string(index=False))


def errores():
    e = pd.read_csv(RESULTADOS / "analisis_errores_dev.tsv", sep="\t", dtype=str)
    print(f"\nErrores representativos en dev ({len(e)}):")
    for _, r in e.iterrows():
        print(f"\n[{r['document_id']}] real={r['gold_label']} -> predicho={r['predicted_label']} "
              f"| falla: {r['modelos_que_fallan']}")
        print(f"  Categoría: {r['categoria']}")
        print(f"  Reseña: {recortar(r['review'], 160)}")
        print(f"  Por qué: {r['explicacion']}")


# ---------------------------------------------------------------------------
# Menú
# ---------------------------------------------------------------------------
def menu():
    while True:
        print("\n===== Opiniones de Steam · Etapa 2 =====")
        print("1. Buscar reseñas y ver su sentimiento")
        print("2. Clasificar un texto")
        print("3. Ver métricas de los experimentos")
        print("4. Ver errores representativos")
        print("5. Salir")
        opcion = input("Elige una opción: ").strip()
        if opcion == "1":
            consulta = input("Consulta: ").strip()
            filtro = input("Filtrar por sentimiento [todas/positive/negative] (Enter = todas): ").strip()
            buscar(consulta, filtro if filtro in ("positive", "negative") else "todas")
        elif opcion == "2":
            clasificar(input("Escribe una reseña en inglés: "))
        elif opcion == "3":
            metricas()
        elif opcion == "4":
            errores()
        elif opcion == "5":
            break
        else:
            print("Opción no válida.")


def main():
    global RESULTADOS, MODELO_PKL, CONFIG_FINAL, _modelo

    parser = argparse.ArgumentParser(description="Interfaz de la Etapa 2 (análisis de sentimiento)")
    parser.add_argument("--ampliado", action="store_true", help="Usa los experimentos ampliados.")
    sub = parser.add_subparsers(dest="comando")
    b = sub.add_parser("buscar")
    b.add_argument("consulta")
    b.add_argument("--sentimiento", choices=["todas", "positive", "negative"], default="todas")
    b.add_argument("--top", type=int, default=10)
    b.add_argument("--ancho", type=int, default=110, help="caracteres de reseña a mostrar")
    c = sub.add_parser("clasificar")
    c.add_argument("texto")
    sub.add_parser("metricas")
    sub.add_parser("errores")
    args = parser.parse_args()

    if args.ampliado:
        RESULTADOS = CARPETA / "resultados_ampliados"
        MODELO_PKL = RESULTADOS / "modelo_final.pkl"
        CONFIG_FINAL = RESULTADOS / "config_final.json"
        _modelo = None
        if not CONFIG_FINAL.exists():
            parser.error("Primero ejecuta python experimentos_ampliados.py")

    if args.comando == "buscar":
        buscar(args.consulta, args.sentimiento, args.top, args.ancho)
    elif args.comando == "clasificar":
        clasificar(args.texto)
    elif args.comando == "metricas":
        metricas()
    elif args.comando == "errores":
        errores()
    else:
        menu()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    main()
