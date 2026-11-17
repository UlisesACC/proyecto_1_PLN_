"""Exportación de auditorías, experimentos, errores, predicciones y figuras."""
import hashlib
import json
from collections import Counter
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from .configuracion import CARPETA, CLASES, HU_LIU, LEXICOS, RECURSO_EMOJIS, REJILLA, SENTIWORDNET
from .metricas import calcular_metricas
from .normalizacion import PROTEGIDAS, normalizar
from .recursos import (
    AMBIGUAS_STEAM, cargar_emojis, cargar_hu_liu, cargar_sentiwordnet,
    conteos_emojis, conteos_lexico, inventario_fuentes,
)


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
        "config": json.dumps(asdict(config), ensure_ascii=False),
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

    guardar_tabla(salida, "inventario_fuentes", inventario_fuentes())
    filas_cobertura = []
    for recurso in LEXICOS:
        conteos = np.asarray([conteos_lexico(texto, recurso) for texto in train["text"]])
        if recurso == "sentiwordnet":
            lexico = cargar_sentiwordnet()
            positivos = sum(pos > 0 for pos, neg in lexico.values())
            negativos = sum(neg > 0 for pos, neg in lexico.values())
        else:
            positivas, negativas = cargar_hu_liu(recurso)
            positivos, negativos = len(positivas), len(negativas)
        filas_cobertura.append({
            "recurso": recurso,
            "entradas_con_polaridad_positiva": positivos,
            "entradas_con_polaridad_negativa": negativos,
            "unidad": "lema y POS" if recurso == "sentiwordnet" else "palabra",
            "filas_train_con_polaridad": int(conteos.any(axis=1).sum()),
            "proporcion_train_con_polaridad": float(conteos.any(axis=1).mean()),
        })
    guardar_tabla(salida, "cobertura_lexicos", filas_cobertura)
    emojis_train = np.asarray([conteos_emojis(texto) for texto in train["text"]])
    guardar_json(salida, "recursos_linguisticos", {
        "hu_liu": {
            "idioma": "inglés",
            "atribucion": "Conservar los avisos originales y citar a Hu y Liu.",
            "referencia": "Hu y Liu, Mining and Summarizing Customer Reviews, KDD 2004",
            "exclusiones_variante_steam": sorted(AMBIGUAS_STEAM),
            "archivos": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(HU_LIU.glob("*-words.txt"))
            },
        },
        "sentiwordnet": {
            "archivo": str(SENTIWORDNET.relative_to(CARPETA.parent)),
            "sha256": hashlib.sha256(SENTIWORDNET.read_bytes()).hexdigest(),
            "licencia": "CC BY-SA 4.0, según la cabecera del recurso",
            "idioma": "inglés",
            "agregacion": "Media de sentidos ponderada por 1/rango; suma por reseña con lema y POS.",
            "rasgos": ["suma_positiva", "suma_negativa"],
            "limitacion": "Se agregan sentidos sin desambiguación contextual; se consultan unigramas.",
        },
        "emojis": {
            "archivo": str(RECURSO_EMOJIS.relative_to(CARPETA)),
            "sha256": hashlib.sha256(RECURSO_EMOJIS.read_bytes()).hexdigest(),
            "entradas": len(cargar_emojis()),
            "distribucion_polaridades": dict(Counter(f[2] for f in cargar_emojis())),
            "polaridad": "Heurística manual; sin_asignar representa polaridad desconocida.",
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

