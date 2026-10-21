import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from normalizacion import normalizar_texto

CORPUS = Path("opinions_normalized_corpus.tsv")
DIR = Path("modelos_recuperacion")
QUERIES = Path("queries.tsv")
QRELS = Path("qrels.tsv")
RANKINGS = Path("rankings.tsv")

SUGGESTED = [
    (1, "cheaters in competitive matches"),
    (1, "game crashes and freezes"),
    (1, "great story and characters"),
    (1, "difficult boss fights"),
    (1, "poor server connection"),
    (2, "the game runs badly on my computer"),
    (2, "playing with friends makes the experience better"),
    (2, "too many paid items and microtransactions"),
    (2, "the controls are frustrating to learn"),
    (2, "repetitive activities after many hours"),
    (3, "unfair opponents ruin online matches"),
    (3, "technical problems interrupt the experience"),
    (3, "the world rewards curiosity and exploration"),
    (3, "the game demands a lot of practice to improve"),
    (3, "the experience becomes boring over time"),
]


def read_corpus():
    df = pd.read_csv(CORPUS, sep="\t", dtype=str, keep_default_na=False)
    required = {"doc_id", "review_id", "review", "review_norm", "item_name", "item_review_norm"}
    absent = required - set(df.columns)
    if absent:
        raise ValueError(f"Faltan columnas: {sorted(absent)}")
    if df["doc_id"].duplicated().any() or df["review_id"].duplicated().any():
        raise ValueError("doc_id y review_id deben ser únicos.")
    if (df["review_norm"].str.strip() == "").any():
        raise ValueError("Hay reseñas normalizadas vacías. Corrige el corpus antes de vectorizar.")
    return df


def corpus_fingerprint(df):
    values = df[["doc_id", "review_id", "review_norm", "item_review_norm"]].to_csv(index=False, sep="\t")
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def model_path(content, representation, ngram):
    return DIR / f"{content}_{representation}_{ngram}.pkl"


def build():
    df = read_corpus()
    DIR.mkdir(exist_ok=True)
    fingerprint = corpus_fingerprint(df)
    for content, column in [("review", "review_norm"), ("item_review", "item_review_norm")]:
        for ngram, ngram_range in [("unigram", (1, 1)), ("bigram", (2, 2))]:
            for representation in ["frequency", "binary", "tfidf"]:
                kwargs = {"ngram_range": ngram_range, "token_pattern": r"(?u)\b\w+\b", "lowercase": False}
                if representation == "tfidf":
                    vectorizer = TfidfVectorizer(**kwargs)
                else:
                    vectorizer = CountVectorizer(binary=(representation == "binary"), **kwargs)
                matrix = vectorizer.fit_transform(df[column].tolist())
                output = model_path(content, representation, ngram)
                with output.open("wb") as stream:
                    pickle.dump({
                        "vectorizer": vectorizer, "matrix": matrix,
                        "doc_ids": df["doc_id"].tolist(), "fingerprint": fingerprint,
                        "content": content, "representation": representation, "ngram": ngram
                    }, stream)
                print(f"{output}: {matrix.shape[0]} documentos, {matrix.shape[1]} términos")


def load_model(df, content, representation, ngram):
    path = model_path(content, representation, ngram)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta: python recuperacion.py build")
    with path.open("rb") as stream:
        model = pickle.load(stream)
    if model["fingerprint"] != corpus_fingerprint(df):
        raise ValueError("El corpus ha cambiado. Regenera modelos con el comando build.")
    if model["doc_ids"] != df["doc_id"].tolist():
        raise ValueError("El orden de documentos no coincide con el modelo.")
    return model


def retrieve(df, model, query, top=10):
    normalized = normalizar_texto(query)
    vector = model["vectorizer"].transform([normalized])
    if vector.nnz == 0:
        print(f"Aviso: consulta sin términos en el vocabulario: {query!r}")
        print("La consulta no tiene coincidencias; los resultados tendrán puntuación 0.")
    scores = cosine_similarity(vector, model["matrix"]).ravel()
    ordered = np.argsort(-scores, kind="stable")[:top]
    return pd.DataFrame({
        "document_id": df.iloc[ordered]["doc_id"].tolist(),
        "rank": np.arange(1, len(ordered) + 1),
        "score": scores[ordered],
        "item_name": df.iloc[ordered]["item_name"].tolist(),
        "review": df.iloc[ordered]["review"].tolist(),
    })


def init_queries():
    if QUERIES.exists():
        print(f"{QUERIES} ya existe; no se sobrescribió.")
        return
    df = pd.DataFrame([
        {"query_id": f"Q{i:02d}", "difficulty_level": level, "query": q}
        for i, (level, q) in enumerate(SUGGESTED, 1)
    ])
    df.to_csv(QUERIES, sep="\t", index=False)
    print(f"Plantilla de 15 consultas creada: {QUERIES}. Revisa su pertinencia para tu corpus.")


def get_queries():
    if not QUERIES.exists():
        raise FileNotFoundError("Crea y revisa queries.tsv con: python recuperacion.py init-queries")
    df = pd.read_csv(QUERIES, sep="\t", dtype=str, keep_default_na=False)
    if {"query_id", "difficulty_level", "query"} - set(df.columns):
        raise ValueError("queries.tsv debe contener query_id, difficulty_level, query.")
    if len(df) != 15 or df["query_id"].duplicated().any():
        raise ValueError("Se requieren 15 consultas únicas.")
    levels = df["difficulty_level"].value_counts().to_dict()
    if any(levels.get(str(level), 0) != 5 for level in (1, 2, 3)):
        raise ValueError("Se requieren cinco consultas de cada dificultad: 1, 2 y 3.")
    return df


def run_queries(content, representation, ngram):
    df = read_corpus()
    model = load_model(df, content, representation, ngram)
    queries = get_queries()
    rows = []
    for _, q in queries.iterrows():
        top = retrieve(df, model, q["query"])
        for _, hit in top.iterrows():
            rows.append({
                "query_id": q["query_id"], "document_id": hit["document_id"],
                "rank": int(hit["rank"]), "score": float(hit["score"])
            })
    pd.DataFrame(rows, columns=["query_id", "document_id", "rank", "score"]).to_csv(
        RANKINGS, sep="\t", index=False
    )
    metadata = {
        "fingerprint": corpus_fingerprint(df), "content": content,
        "representation": representation, "ngram": ngram,
        "queries_hash": hashlib.sha256(QUERIES.read_bytes()).hexdigest()
    }
    Path("rankings_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Guardado: {RANKINGS} ({len(rows)} resultados)")


def candidates(content, representation, ngram):
    df = read_corpus()
    model = load_model(df, content, representation, ngram)
    queries = get_queries()
    rows = []
    for _, q in queries.iterrows():
        top = retrieve(df, model, q["query"], top=30)
        for _, hit in top.iterrows():
            rows.append({
                "query_id": q["query_id"], "document_id": hit["document_id"],
                "score": round(float(hit["score"]), 5),
                "item_name": hit["item_name"], "review": hit["review"], "relevance": ""
            })
    path = Path("candidatos_qrels.tsv")
    if path.exists():
        raise FileExistsError("candidatos_qrels.tsv ya existe; haz una copia antes de regenerarlo.")
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    print("Revisa candidatos_qrels.tsv y otros documentos del corpus. Asigna 0, 1 o 2 según relevancia.")
    print("Crea qrels.tsv con query_id, document_id, relevance, solamente valores 1 y 2.")
    print("Debe haber 10-15 documentos relevantes por consulta, con valores 1 y 2.")
    print("Congela qrels.tsv antes de comparar modelos.")


def evaluate():
    queries = get_queries()
    corpus = read_corpus()
    if not QRELS.exists() or not RANKINGS.exists():
        raise FileNotFoundError("Se requieren qrels.tsv (manual) y rankings.tsv (generado).")
    meta_path = Path("rankings_metadata.json")
    if not meta_path.exists():
        raise FileNotFoundError("Falta rankings_metadata.json. Regenera rankings.tsv.")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata["fingerprint"] != corpus_fingerprint(corpus):
        raise ValueError("El corpus difiere del usado para rankings.tsv.")
    if metadata["queries_hash"] != hashlib.sha256(QUERIES.read_bytes()).hexdigest():
        raise ValueError("Cambiaste las consultas. Regenera rankings.tsv.")
    qrels = pd.read_csv(QRELS, sep="\t", dtype=str, keep_default_na=False)
    ranks = pd.read_csv(RANKINGS, sep="\t", dtype=str, keep_default_na=False)
    if not {"query_id", "document_id", "relevance"}.issubset(qrels.columns):
        raise ValueError("qrels.tsv debe contener query_id, document_id y relevance.")
    if not {"query_id", "document_id", "rank", "score"}.issubset(ranks.columns):
        raise ValueError("rankings.tsv tiene columnas incompletas.")
    if qrels.duplicated(["query_id", "document_id"]).any():
        raise ValueError("Existen juicios duplicados en qrels.tsv.")
    if ranks.duplicated(["query_id", "document_id"]).any():
        raise ValueError("Un ranking contiene documentos repetidos.")
    if not qrels["relevance"].isin(["1", "2"]).all():
        raise ValueError("qrels.tsv solo debe guardar relevancias 1 o 2.")
    if not set(qrels["document_id"]).issubset(set(corpus["doc_id"])):
        raise ValueError("qrels.tsv contiene IDs ausentes del corpus.")
    if not set(ranks["document_id"]).issubset(set(corpus["doc_id"])):
        raise ValueError("rankings.tsv contiene IDs ausentes del corpus.")
    query_ids = set(queries["query_id"])
    if set(qrels["query_id"]) != query_ids or set(ranks["query_id"]) != query_ids:
        raise ValueError("qrels y rankings deben contener exactamente los 15 query_id.")
    results = []
    for _, q in queries.iterrows():
        relevant = qrels[qrels["query_id"] == q["query_id"]]
        if not 10 <= len(relevant) <= 15 or set(relevant["relevance"]) != {"1", "2"}:
            raise ValueError(f"{q['query_id']} requiere 10-15 documentos relevantes (con 1 y 2).")
        rel = dict(zip(relevant["document_id"], relevant["relevance"].astype(int)))
        subset = ranks[ranks["query_id"] == q["query_id"]].copy()
        subset["rank"] = subset["rank"].astype(int)
        subset = subset.sort_values("rank")
        if subset["rank"].tolist() != list(range(1, 11)):
            raise ValueError(f"{q['query_id']} necesita rangos 1 a 10 sin repetir.")
        grades = [rel.get(d, 0) for d in subset["document_id"]]
        found = sum(g > 0 for g in grades)
        p = found / 10
        r = found / len(rel)
        f1 = 2 * p * r / (p + r) if p + r else 0
        rr = next((1 / i for i, grade in enumerate(grades, 1) if grade > 0), 0)
        dcg = sum((2 ** g - 1) / np.log2(i + 1) for i, g in enumerate(grades, 1))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum((2 ** g - 1) / np.log2(i + 1) for i, g in enumerate(ideal, 1))
        ndcg = dcg / idcg if idcg else 0
        results.append({
            "query_id": q["query_id"], "difficulty_level": q["difficulty_level"],
            "precision@10": p, "recall@10": r, "f1@10": f1, "rr": rr, "ndcg@10": ndcg
        })
    report = pd.DataFrame(results)
    report.to_csv("evaluacion_por_consulta.tsv", sep="\t", index=False)
    cols = ["precision@10", "recall@10", "f1@10", "rr", "ndcg@10"]
    means = report[cols].mean().to_dict()
    pd.DataFrame([{"queries": len(report), **means}]).rename(columns={"rr": "mrr"}).to_csv(
        "evaluacion_promedios.tsv", sep="\t", index=False
    )
    print(report.to_string(index=False))
    print("\nPromedios:", {("mrr" if k == "rr" else k): round(v, 4) for k, v in means.items()})


def main():
    parser = argparse.ArgumentParser(description="Proyecto Opinion Retrieval: secciones 7-9")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("init-queries")
    q = sub.add_parser("search")
    q.add_argument("query")
    sub.add_parser("rank")
    sub.add_parser("candidates")
    sub.add_parser("evaluate")
    for name in ("search", "rank", "candidates"):
        p = sub.choices[name]
        p.add_argument("--content", choices=["review", "item_review"], default="review")
        p.add_argument("--representation", choices=["frequency", "binary", "tfidf"], default="tfidf")
        p.add_argument("--ngram", choices=["unigram", "bigram"], default="unigram")
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "init-queries":
        init_queries()
    elif args.command == "search":
        df = read_corpus()
        model = load_model(df, args.content, args.representation, args.ngram)
        hits = retrieve(df, model, args.query)
        print(hits.to_string(index=False, max_colwidth=140))
    elif args.command == "rank":
        run_queries(args.content, args.representation, args.ngram)
    elif args.command == "candidates":
        candidates(args.content, args.representation, args.ngram)
    elif args.command == "evaluate":
        evaluate()


if __name__ == "__main__":
    main()
