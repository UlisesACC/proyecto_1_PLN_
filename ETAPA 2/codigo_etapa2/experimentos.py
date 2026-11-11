"""
experimentos.py — Etapa 2: clasificación supervisada de sentimiento

Uso (desde esta carpeta):
    python experimentos.py

Qué hace, en orden:
  1. Lee opinions_raw_corpus.tsv (Etapa 1) y define la etiqueta a partir de la
     recomendación de Steam (columna rating: positive / negative).
  2. Separa train / dev / test (70/15/15, estratificado, shuffle=True,
     random_state=0) ANTES de ajustar cualquier vectorizador o modelo.
  3. Corre los experimentos por etapas, cambiando un factor a la vez y midiendo
     siempre en dev con macro-F1:
        E1 normalización  ->  E2 desbalance de clases  ->  E3 representación
        ->  E4 mejoras lingüísticas  ->  E5 hiperparámetros (GridSearchCV sobre dev)
  4. Elige la mejor configuración de cada modelo y la mejor global (en dev).
  5. Congela la mejor global, la reentrena con train+dev y la evalúa UNA vez
     en test.
  6. Guarda todos los entregables en resultados/ y el modelo en modelos/.
"""

import json
import warnings
import pickle
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, make_scorer, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split

from preprocesamiento import (LEXICO_NEG, LEXICO_POS, NEGADORES, SEMILLA,
                              construir_pipeline, contar_lexico, describir, limpiar,
                              normalizar)

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parents[1]                      # carpeta principal del proyecto
CORPUS = RAIZ / "opinions_raw_corpus.tsv"
SALIDA = CARPETA / "resultados"
FIGURAS = SALIDA / "figuras"
MODELOS_DIR = CARPETA / "modelos"

CLASES = ["negative", "positive"]               # orden fijo para métricas y matrices
MODELOS = ["MultinomialNB", "LogisticRegression", "LinearSVC"]
REJILLA = {                                     # valores probados en E5
    "MultinomialNB": ("alpha", [0.01, 0.1, 0.5, 1.0, 2.0]),
    "LogisticRegression": ("C", [0.01, 0.1, 1.0, 10.0, 100.0]),
    "LinearSVC": ("C", [0.01, 0.1, 1.0, 10.0, 100.0]),
}
COLORES = {"MultinomialNB": "#2a78d6", "LogisticRegression": "#eb6834", "LinearSVC": "#1baf7a"}


# ---------------------------------------------------------------------------
# 1 y 2. Datos, etiquetas y particiones
# ---------------------------------------------------------------------------
def preparar_datos():
    df = pd.read_csv(CORPUS, sep="\t", dtype=str, keep_default_na=False)
    df["doc_id_etapa1"] = range(1, len(df) + 1)     # mismo doc_id que en la Etapa 1

    # Mapeo de etiquetas: Steam da una recomendación binaria (voted_up).
    # En el corpus ya viene como texto en la columna rating.
    mapeo = {"positive": "positive", "negative": "negative"}
    faltantes = int((df["rating"].str.strip() == "").sum())
    df["label"] = df["rating"].map(mapeo)
    if df["label"].isna().any():
        raise ValueError("Hay valores de rating fuera del mapeo documentado.")

    # Partición estratificada: primero 70% train / 30% resto,
    # luego el 30% se divide a la mitad en dev y test (15% / 15%).
    indices = np.arange(len(df))
    idx_train, idx_resto = train_test_split(
        indices, test_size=0.30, shuffle=True, random_state=SEMILLA, stratify=df["label"])
    idx_dev, idx_test = train_test_split(
        idx_resto, test_size=0.50, shuffle=True, random_state=SEMILLA,
        stratify=df["label"].iloc[idx_resto])
    df["split"] = ""
    df.loc[idx_train, "split"] = "train"
    df.loc[idx_dev, "split"] = "dev"
    df.loc[idx_test, "split"] = "test"

    etiquetados = pd.DataFrame({
        "document_id": df["review_id"],          # document_id = review_id (pedido en la práctica)
        "review": df["review"],
        "text": df["review"],                    # X del modelo: SOLO el texto
        "rating": df["rating"],                  # solo metadato, nunca entra al modelo
        "label": df["label"],
        "split": df["split"],
        "doc_id_etapa1": df["doc_id_etapa1"],
        "item_name": df["item_name"],
        "category": df["category"],
        "votes_up": df["votes_up"],
        "playtime_hours": df["playtime_hours"],
    })
    etiquetados.to_csv(SALIDA / "labeled_documents.tsv", sep="\t", index=False)

    dist = (etiquetados.groupby(["split", "label"]).size().unstack(fill_value=0)
            .reindex(["train", "dev", "test"]))
    dist.loc["total"] = dist.sum()
    dist["total"] = dist.sum(axis=1)
    dist["% negative"] = (100 * dist["negative"] / dist["total"]).round(1)
    dist.to_csv(SALIDA / "distribucion_clases.tsv", sep="\t")
    print("Distribución de clases:\n", dist, "\nRatings vacíos:", faltantes)
    return etiquetados, faltantes


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def calcular_metricas(y_real, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_real, y_pred, labels=CLASES, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_real, y_pred, labels=CLASES,
                                                    average="macro", zero_division=0)
    m = {"accuracy": accuracy_score(y_real, y_pred),
         "macro_precision": mp, "macro_recall": mr, "macro_f1": mf}
    for i, c in enumerate(CLASES):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"], m[f"support_{c}"] = p[i], r[i], f[i], int(s[i])
    cm = confusion_matrix(y_real, y_pred, labels=CLASES)
    m["confusion_matrix"] = json.dumps(cm.tolist())   # filas = real, columnas = predicho
    return m


# ---------------------------------------------------------------------------
# 3. Registro de experimentos
# ---------------------------------------------------------------------------
class Registro:
    def __init__(self, X_train, y_train, X_dev, y_dev):
        self.X_train, self.y_train, self.X_dev, self.y_dev = X_train, y_train, X_dev, y_dev
        self.filas = []
        self.contador = 0

    def _nuevo_id(self):
        self.contador += 1
        return f"C{self.contador:02d}"

    def _agregar(self, cfg, etapa, cambio, metricas):
        eid = self._nuevo_id()
        fila = {"experiment_id": eid, "etapa": etapa, "model": cfg["modelo"],
                "cambio_probado": cambio, **describir(cfg)}
        fila.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metricas.items()})
        fila["config"] = json.dumps(cfg)
        self.filas.append(fila)
        print(f"{eid} {cfg['modelo']:<18} {etapa:<16} {cambio:<34} macro-F1 dev = {metricas['macro_f1']:.4f}")
        return fila

    def correr(self, cfg, etapa, cambio):
        """Entrena con train, predice dev y registra el resultado."""
        pipe = construir_pipeline(cfg)
        pipe.fit(self.X_train, self.y_train)
        pred = pipe.predict(self.X_dev)
        return self._agregar(cfg, etapa, cambio, calcular_metricas(self.y_dev, pred))

    def rejilla(self, cfg, etapa):
        """E5: GridSearchCV con PredefinedSplit (entrena en train, califica en dev)."""
        nombre, valores = REJILLA[cfg["modelo"]]
        X = pd.concat([self.X_train, self.X_dev])
        y = np.concatenate([self.y_train, self.y_dev])
        pliegue = PredefinedSplit([-1] * len(self.X_train) + [0] * len(self.X_dev))
        puntajes = {
            "macro_f1": "f1_macro", "accuracy": "accuracy",
            "macro_precision": "precision_macro", "macro_recall": "recall_macro",
            "f1_negative": make_scorer(f1_score, pos_label="negative", zero_division=0),
            "f1_positive": make_scorer(f1_score, pos_label="positive", zero_division=0),
        }
        busqueda = GridSearchCV(construir_pipeline(cfg),
                                {f"clasificador__{nombre}": valores},
                                scoring=puntajes, cv=pliegue, refit=False)
        busqueda.fit(X, y)
        filas = []
        for i, valor in enumerate(valores):
            nuevo = dict(cfg, hiper={nombre: valor})
            met = {k: float(busqueda.cv_results_[f"mean_test_{k}"][i]) for k in puntajes}
            filas.append(self._agregar(nuevo, etapa, f"GridSearch {nombre}={valor}", met))
        return filas

    def tabla(self):
        return pd.DataFrame(self.filas)


def mejor(filas):
    """La de mayor macro-F1; en empate se queda la primera (la más simple)."""
    return max(filas, key=lambda f: (round(f["macro_f1"], 4), -int(f["experiment_id"][1:])))


def decidir(filas, referencia):
    """Anota en cada fila si el cambio se conserva o se descarta frente a la referencia."""
    ganadora = mejor(filas + ([referencia] if referencia else []))
    for f in filas:
        if referencia is None:
            dif = 0.0
        else:
            dif = f["macro_f1"] - referencia["macro_f1"]
        f["dif_vs_referencia"] = round(dif, 4)
        f["decision"] = "se conserva" if f is ganadora else "se descarta"
    return ganadora


def explorar_modelo(reg, modelo):
    """Búsqueda por etapas para un modelo; devuelve la fila ganadora."""
    base = {"normalizacion": "N0", "negacion": False, "representacion": "frecuencia",
            "ngramas": (1, 1), "lexico": False, "balanceo": False,
            "modelo": modelo, "hiper": {}}

    # E1 — Normalización (representación fija: frecuencia, unigramas)
    filas = [reg.correr(dict(base, normalizacion=n), "E1 normalización", f"normalización {n}")
             for n in ["N0", "N1", "N2", "N3"]]
    for f in filas:
        f["dif_vs_referencia"] = round(f["macro_f1"] - filas[0]["macro_f1"], 4)
    actual = mejor(filas)
    for f in filas:
        f["decision"] = "se conserva" if f is actual else "se descarta"
    filas[0]["decision"] += " (línea base)"
    cfg = json.loads(actual["config"])

    # E2 — Desbalance de clases (solo afecta el entrenamiento; se prueba pronto
    #      porque el corpus tiene solo 15% de reseñas negativas)
    filas = [reg.correr(dict(cfg, balanceo=True), "E2 desbalance",
                        "fit_prior=False" if modelo == "MultinomialNB" else "class_weight=balanced")]
    actual = decidir(filas, actual)
    cfg = json.loads(actual["config"])

    # E3 — Representación vectorial y n-gramas
    filas = []
    for rep in ["binaria", "frecuencia", "tfidf"]:
        for ng in [(1, 1), (2, 2), (1, 2)]:
            if rep == cfg["representacion"] and list(ng) == list(cfg["ngramas"]):
                continue                          # ya se midió antes
            filas.append(reg.correr(dict(cfg, representacion=rep, ngramas=ng),
                                    "E3 representación", f"{rep} {ng}"))
    # Control de interacción: TF-IDF produce valores pequeños y, sin balanceo,
    # el modelo tiende a predecir solo la clase mayoritaria. Se prueba TF-IDF
    # con balanceo para no descartarlo injustamente.
    if not cfg["balanceo"]:
        filas.append(reg.correr(dict(cfg, representacion="tfidf", ngramas=(1, 2), balanceo=True),
                                "E3 representación", "tfidf (1, 2) + balanceo (control)"))
    actual = decidir(filas, actual)
    cfg = json.loads(actual["config"])

    # E4 — Mejoras lingüísticas (ablación contra la mejor sin mejoras)
    sin_mejoras = actual
    filas = [reg.correr(dict(cfg, negacion=True), "E4 mejoras", "+ manejo de negación"),
             reg.correr(dict(cfg, lexico=True), "E4 mejoras", "+ léxico de opinión"),
             reg.correr(dict(cfg, negacion=True, lexico=True), "E4 mejoras", "+ negación + léxico")]
    actual = decidir(filas, sin_mejoras)
    cfg = json.loads(actual["config"])

    # E5 — Hiperparámetros con GridSearchCV (train para ajustar, dev para calificar)
    filas = reg.rejilla(cfg, "E5 hiperparámetros")
    actual = decidir(filas, actual)
    return actual, sin_mejoras


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def figura_matriz(cm, titulo, ruta):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=200)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    ax.set_xticks([0, 1], ["negative", "positive"])
    ax.set_yticks([0, 1], ["negative", "positive"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(titulo, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_etapas(tabla, ganadoras, ruta):
    """Mejor macro-F1 en dev acumulado al terminar cada etapa, por modelo."""
    etapas = ["E1 normalización", "E2 desbalance", "E3 representación",
              "E4 mejoras", "E5 hiperparámetros"]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    for modelo in MODELOS:
        t = tabla[tabla["model"] == modelo]
        valores, mejor_hasta = [], 0
        for e in etapas:
            mejor_hasta = max(mejor_hasta, t[t["etapa"] == e]["macro_f1"].max())
            valores.append(mejor_hasta)
        ax.plot(range(len(etapas)), valores, marker="o", linewidth=2, markersize=6,
                color=COLORES[modelo], label=modelo)
        ax.annotate(f"{valores[-1]:.3f}", (len(etapas) - 1, valores[-1]),
                    textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=8, color="#52514e")
    ax.set_xticks(range(len(etapas)), [e.split(" ", 1)[1] for e in etapas], fontsize=8)
    ax.set_ylabel("Mejor macro-F1 en dev")
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.01, 0.12))
    ax.set_xlim(-0.3, len(etapas) - 0.4)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Recursos lingüísticos: cobertura del léxico y ejemplos de negación (solo train)
# ---------------------------------------------------------------------------
def analizar_recursos(etiquetados):
    train = etiquetados[etiquetados["split"] == "train"]
    conteos = np.array([contar_lexico(t) for t in train["text"]])
    train = train.assign(lex_pos=conteos[:, 0], lex_neg=conteos[:, 1])
    filas = []
    for clase in CLASES + ["todas"]:
        t = train if clase == "todas" else train[train["label"] == clase]
        filas.append({
            "clase": clase, "reseñas": len(t),
            "% con al menos 1 palabra del léxico": round(100 * ((t.lex_pos + t.lex_neg) > 0).mean(), 1),
            "promedio palabras positivas": round(t.lex_pos.mean(), 2),
            "promedio palabras negativas": round(t.lex_neg.mean(), 2),
            "% con negador": round(100 * t["text"].map(
                lambda x: any(n in normalizar(x, "N1").split() for n in NEGADORES)).mean(), 1),
        })
    pd.DataFrame(filas).to_csv(SALIDA / "recursos_linguisticos.tsv", sep="\t", index=False)

    # Palabras frecuentes del corpus que el léxico NO cubre (huecos de dominio)
    vocab = pd.Series(" ".join(normalizar(t, "N2") for t in train["text"]).split()).value_counts()
    fuera = vocab[[p not in LEXICO_POS and p not in LEXICO_NEG and p.isalpha() for p in vocab.index]]
    fuera.head(40).rename("frecuencia").to_csv(SALIDA / "palabras_frecuentes_fuera_del_lexico.tsv", sep="\t")

    # Ejemplos de cómo se ve el texto antes y después de marcar la negación
    con_neg = train[train["text"].map(lambda x: "NEG_" in normalizar(x, "N2", True))]
    ejemplos = con_neg[con_neg["text"].str.len() < 160].head(8)
    pd.DataFrame({
        "document_id": ejemplos["document_id"], "label": ejemplos["label"],
        "original": ejemplos["text"].map(lambda x: " ".join(str(x).split())),
        "N2_sin_negacion": ejemplos["text"].map(lambda x: normalizar(x, "N2")),
        "N2_con_negacion": ejemplos["text"].map(lambda x: normalizar(x, "N2", True)),
    }).to_csv(SALIDA / "ejemplos_negacion.tsv", sep="\t", index=False)

    # Candidatas para revisar etiquetas a mano: el léxico contradice la etiqueta
    contradicen = train[((train.label == "positive") & (train.lex_neg > train.lex_pos)) |
                        ((train.label == "negative") & (train.lex_pos > train.lex_neg))]
    contradicen[["document_id", "label", "lex_pos", "lex_neg", "text"]].to_csv(
        SALIDA / "revision_etiquetas_candidatas.tsv", sep="\t", index=False)
    print(f"Candidatas a revisión manual de etiqueta (train): {len(contradicen)}")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    SALIDA.mkdir(exist_ok=True)
    FIGURAS.mkdir(exist_ok=True)
    MODELOS_DIR.mkdir(exist_ok=True)

    etiquetados, faltantes = preparar_datos()
    analizar_recursos(etiquetados)

    train = etiquetados[etiquetados["split"] == "train"]
    dev = etiquetados[etiquetados["split"] == "dev"]
    test = etiquetados[etiquetados["split"] == "test"]
    reg = Registro(train["text"], train["label"].to_numpy(), dev["text"], dev["label"].to_numpy())

    ganadoras, sin_mejoras = {}, {}
    for modelo in MODELOS:
        ganadoras[modelo], sin_mejoras[modelo] = explorar_modelo(reg, modelo)

    # Mejor global en dev: macro-F1; en empate, mayor F1 de la clase minoritaria
    final = max(ganadoras.values(),
                key=lambda f: (round(f["macro_f1"], 4), round(f["f1_negative"], 4)))
    tabla = reg.tabla()
    tabla["selected"] = "no"
    for f in ganadoras.values():
        tabla.loc[tabla["experiment_id"] == f["experiment_id"], "selected"] = "yes (mejor del modelo)"
    tabla.loc[tabla["experiment_id"] == final["experiment_id"], "selected"] = "yes (final)"
    columnas = ["experiment_id", "etapa", "model", "normalizacion", "representacion",
                "mejora_linguistica", "desbalance", "hiperparametros", "cambio_probado",
                "accuracy", "macro_precision", "macro_recall", "macro_f1",
                "f1_negative", "f1_positive", "dif_vs_referencia", "decision", "selected"]
    tabla[columnas + ["config"]].to_csv(SALIDA / "experiments.tsv", sep="\t", index=False)
    try:
        tabla[columnas].to_excel(SALIDA / "experiments.xlsx", index=False)
    except ImportError:
        print("Aviso: instala openpyxl para generar experiments.xlsx (el .tsv sí se generó).")

    # Ablación de mejoras lingüísticas (contra la mejor configuración sin mejoras)
    ablacion = []
    for modelo in MODELOS:
        base = sin_mejoras[modelo]
        ablacion.append({"model": modelo, "experiment_id": base["experiment_id"],
                         "enhancement": "baseline (sin mejoras)", "dev_macro_f1": base["macro_f1"],
                         "difference_vs_baseline": 0.0, "dev_f1_negative": base.get("f1_negative"),
                         "difference_f1_negative": 0.0})
        for f in tabla[(tabla.model == modelo) & (tabla.etapa == "E4 mejoras")].to_dict("records"):
            ablacion.append({"model": modelo, "experiment_id": f["experiment_id"],
                             "enhancement": f["cambio_probado"].lstrip("+ "),
                             "dev_macro_f1": f["macro_f1"],
                             "difference_vs_baseline": round(f["macro_f1"] - base["macro_f1"], 4),
                             "dev_f1_negative": f["f1_negative"],
                             "difference_f1_negative": round(f["f1_negative"] - base["f1_negative"], 4)})
    pd.DataFrame(ablacion).to_csv(SALIDA / "ablacion_mejoras.tsv", sep="\t", index=False)

    # Predicciones y métricas en dev de la mejor configuración de cada modelo
    predicciones, metricas, errores, reportes = [], [], [], []
    for modelo in MODELOS:
        f = ganadoras[modelo]
        cfg = json.loads(f["config"])
        pipe = construir_pipeline(cfg).fit(train["text"], train["label"])
        pred = pipe.predict(dev["text"])
        m = calcular_metricas(dev["label"], pred)
        metricas.append({"experiment_id": f["experiment_id"], "model": modelo, "split": "dev", **m})
        predicciones.append(pd.DataFrame({
            "document_id": dev["document_id"], "gold_label": dev["label"], "predicted_label": pred,
            "split": "dev", "model": modelo, "experiment_id": f["experiment_id"]}))
        reportes.append(f"===== DEV | {f['experiment_id']} | {modelo}\n"
                        + classification_report(dev["label"], pred, labels=CLASES, digits=4, zero_division=0))
        figura_matriz(json.loads(m["confusion_matrix"]),
                      f"Dev · {modelo} ({f['experiment_id']})",
                      FIGURAS / f"matriz_dev_{modelo}.png")
        mal = dev[pred != dev["label"].to_numpy()]
        for _, r in mal.iterrows():
            pos, neg = contar_lexico(r["text"])
            errores.append({
                "model": modelo, "experiment_id": f["experiment_id"],
                "document_id": r["document_id"], "item_name": r["item_name"],
                "gold_label": r["label"], "predicted_label": pipe.predict([r["text"]])[0],
                "lex_pos": pos, "lex_neg": neg,
                "tiene_negador": any(n in normalizar(r["text"], "N1").split() for n in NEGADORES),
                "review": " ".join(str(r["text"]).split()),
                "texto_normalizado": normalizar(r["text"], cfg["normalizacion"], cfg["negacion"]),
            })

    # 5. Prueba final ÚNICA: se congela la mejor configuración y se reentrena con train+dev
    cfg_final = json.loads(final["config"])
    train_dev = pd.concat([train, dev])
    modelo_final = construir_pipeline(cfg_final).fit(train_dev["text"], train_dev["label"])
    pred_test = modelo_final.predict(test["text"])
    m = calcular_metricas(test["label"], pred_test)
    metricas.append({"experiment_id": "test", "model": cfg_final["modelo"], "split": "test", **m})
    predicciones.append(pd.DataFrame({
        "document_id": test["document_id"], "gold_label": test["label"], "predicted_label": pred_test,
        "split": "test", "model": cfg_final["modelo"], "experiment_id": "test"}))
    reportes.append(f"===== TEST | configuración {final['experiment_id']} congelada | "
                    f"{cfg_final['modelo']} reentrenado con train+dev\n"
                    + classification_report(test["label"], pred_test, labels=CLASES, digits=4, zero_division=0))
    figura_matriz(json.loads(m["confusion_matrix"]),
                  f"Test · {cfg_final['modelo']} (config. {final['experiment_id']})",
                  FIGURAS / "matriz_test_final.png")

    pd.concat(predicciones).to_csv(SALIDA / "classification_predictions.tsv", sep="\t", index=False)
    met = pd.DataFrame(metricas)
    for c in met.columns:
        if met[c].dtype == float:
            met[c] = met[c].round(4)
    met.to_csv(SALIDA / "classification_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(errores).to_csv(SALIDA / "errores_dev.tsv", sep="\t", index=False)
    (SALIDA / "classification_reports.txt").write_text("\n\n".join(reportes), encoding="utf-8")
    figura_etapas(tabla, ganadoras, FIGURAS / "progreso_por_etapa.png")

    # Modelo final y su configuración (los usa interfaz.py)
    with open(MODELOS_DIR / "modelo_final.pkl", "wb") as fh:
        pickle.dump(modelo_final, fh)
    (MODELOS_DIR / "config_final.json").write_text(json.dumps(
        {"experiment_id": final["experiment_id"], "config": cfg_final,
         "descripcion": describir(cfg_final), "entrenado_con": "train+dev"},
        indent=2, ensure_ascii=False), encoding="utf-8")

    # Versiones y semilla (reproducibilidad)
    (SALIDA / "entorno.txt").write_text("\n".join([
        f"python={platform.python_version()}", f"scikit-learn={sklearn.__version__}",
        f"nltk={nltk.__version__}", f"pandas={pd.__version__}", f"numpy={np.__version__}",
        f"random_state={SEMILLA}", "particion=70/15/15 estratificada, shuffle=True",
        f"ratings_vacios={faltantes}", f"experimentos={len(tabla)}",
    ]) + "\n", encoding="utf-8")

    print("\nMejor de cada modelo en dev:")
    for modelo, f in ganadoras.items():
        print(f"  {modelo:<18} {f['experiment_id']}  macro-F1={f['macro_f1']:.4f}")
    print(f"Configuración final: {final['experiment_id']} ({cfg_final['modelo']})")
    print(f"TEST  accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
