"""Métricas con un orden explícito y constante de clases."""
import json

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .configuracion import CLASES


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

