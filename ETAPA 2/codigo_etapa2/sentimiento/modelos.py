"""Pipelines de texto, rasgos lingüísticos y clasificadores supervisados."""
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.class_weight import compute_sample_weight

from preprocesamiento import crear_clasificador, crear_vectorizador
from .configuracion import LEXICOS, REJILLA, SEMILLA
from .normalizacion import NormalizadorAmpliado
from .recursos import RasgosSentimiento


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

