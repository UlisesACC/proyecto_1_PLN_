"""Rutas, semillas y configuraciones que definen los experimentos."""
from dataclasses import dataclass
from pathlib import Path

CARPETA = Path(__file__).resolve().parents[1]
RAIZ = CARPETA.parents[1]
CORPUS = RAIZ / "opinions_raw_corpus.tsv"
FUENTE = RAIZ / "steam_reviews_raw.jsonl"
FUENTES = CARPETA.parent / "fuentes"
DIRECTORIO_LEXICOS = FUENTES / "Lexicons" / "Lexicons"
HU_LIU = DIRECTORIO_LEXICOS / "opinion-lexicon-English"
SENTIWORDNET = DIRECTORIO_LEXICOS / "SentiWordNet_3.0.0.txt"
RECURSO_EMOJIS = CARPETA / "recursos" / "emojis_sentimiento.tsv"
SALIDA = CARPETA / "resultados_fuentes"

SEMILLA = 0
CLASES = ["negative", "positive"]
MODELOS = ["MultinomialNB", "LogisticRegression", "LinearSVC"]
MAPEO = {"negative": "negative", "positive": "positive"}
NIVELES = ("N0", "N1", "N2", "N3")
PROPORCION_MINORITARIA = 0.25
SVD_COMPONENTES = 50
REJILLA = {
    "MultinomialNB": ("alpha", [0.01, 0.1, 0.5, 1.0, 2.0]),
    "LogisticRegression": ("C", [0.01, 0.1, 1.0, 10.0, 100.0]),
    "LinearSVC": ("C", [0.01, 0.1, 1.0, 10.0, 100.0]),
}
LEXICOS = ("hu_liu", "hu_liu_steam", "sentiwordnet")
MEJORAS = (
    "negacion", *LEXICOS, "emojis",
    "ortografia", "letras_repetidas", "palabras_repetidas",
)


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

