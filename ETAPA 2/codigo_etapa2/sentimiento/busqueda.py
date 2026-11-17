"""Comparaciones controladas y selección usando exclusivamente train y dev."""
from dataclasses import dataclass, replace

import numpy as np

from .configuracion import Configuracion, LEXICOS, MEJORAS, NIVELES, REJILLA, SVD_COMPONENTES
from .modelos import ajustar
from .metricas import calcular_metricas


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

