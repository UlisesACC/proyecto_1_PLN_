"""Extracción y validación del corpus Steam (etapa 1, Opinion Retrieval).

Instalar: python -m pip install requests langdetect
Ejecutar: python datos.py

El corpus conserva el texto ORIGINAL; normalizacion.py procesa el texto después.
"""
import csv
import json
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0  # Resultados reproducibles en langdetect.

# Solo títulos publicados con reseñas disponibles. Se pueden agregar otros App IDs.
GAMES = {
    730: "Counter-Strike 2",
    570: "Dota 2",
    620: "Portal 2",
    413150: "Stardew Valley",
    1086940: "Baldur's Gate 3",
    1384160: "Guilty Gear -Strive-",
}
REVIEWS_PER_GAME = 80          # 6 * 80 = 480 (intervalo recomendado: 300-500).
LANGUAGE = "english"
MIN_WORDS = 7                 # Evita reseñas sin contexto para búsquedas.
MIN_EN_CONFIDENCE = 0.80
MAX_PAGES_PER_GAME = 30       # Evita ciclos interminables en juegos sin resultados.
REQUEST_DELAY = 2

OUTPUT_FILE = Path("opinions_raw_corpus.tsv")
RAW_JSONL_FILE = Path("steam_reviews_raw.jsonl")
REJECTED_FILE = Path("steam_reviews_rejected.jsonl")

REVIEWS_API = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
DETAILS_API = "https://store.steampowered.com/api/appdetails"

FIELDS = [
    "review_id", "item_id", "item_name", "review", "rating",
    "date", "category", "source", "url", "source_id", "retrieved_at",
    "language", "votes_up", "playtime_hours",
]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SteamOpinionRetrievalAcademic/1.1"})
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
# Se preservan emojis y la puntuación en el corpus original: aquí solo se VALIDAN.


def get_json(url, params, attempts=4):
    """GET con timeout, manejo de 429/errores 5xx y espera limitada."""
    for i in range(attempts):
        try:
            response = SESSION.get(url, params=params, timeout=30)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = min(max(float(retry_after), 1), 120)
                except ValueError:
                    delay = min(10 * (2 ** i), 120)
                print(f"  Steam limitó solicitudes; esperar {delay:.0f}s")
                time.sleep(delay)
                continue
            if response.status_code >= 500:
                time.sleep(min(3 * (i + 1), 20))
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"  Intento {i + 1}/{attempts}: {exc}")
            if i < attempts - 1:
                time.sleep(3 * (i + 1))
    raise RuntimeError(f"No se pudo consultar {url}")


def get_game_genres(app_id):
    """Obtiene los géneros (genres) reales desde los metadatos de la tienda."""
    try:
        payload = get_json(
            DETAILS_API, {"appids": app_id, "l": "english", "cc": "us"}
        )
        entry = payload.get(str(app_id), {})
        if not entry.get("success") or not isinstance(entry.get("data"), dict):
            print(f"  Aviso: sin metadatos de géneros para {app_id}")
            return "Unknown"
        genres = entry["data"].get("genres") or []
        names = [g["description"].strip() for g in genres
                 if isinstance(g, dict) and g.get("description")]
        return " | ".join(dict.fromkeys(names)) if names else "Unknown"
    except (RuntimeError, KeyError, TypeError) as exc:
        print(f"  No fue posible obtener géneros para {app_id}: {exc}")
        return "Unknown"


def validate_english_review(text):
    """Devuelve (aceptada, motivo, confianza) sin alterar el texto de origen."""
    if not isinstance(text, str) or not text.strip():
        return False, "empty", 0.0

    words = WORD_RE.findall(text)
    if len(words) < MIN_WORDS:
        return False, "too_few_english_words", 0.0

    # Cadenas como '.....', emojis solos, números y texto ruso/chino se descartan.
    # Alfabeto latino extendido se admite, pero una proporción grande de escritura
    # no latina se rechaza antes de usar detección estadística.
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return False, "no_letters", 0.0
    latin = sum("LATIN" in unicodedata.name(c, "") for c in letters)
    if latin / len(letters) < 0.90:
        return False, "non_latin_script", 0.0

    try:
        detected = detect_langs(text)
    except LangDetectException:
        return False, "language_undetermined", 0.0

    confidence = next((d.prob for d in detected if d.lang == "en"), 0.0)
    if confidence < MIN_EN_CONFIDENCE:
        return False, "not_confident_english", round(confidence, 4)
    return True, "accepted", round(confidence, 4)


def fetch_reviews_page(app_id, cursor):
    params = {
        "appid": app_id,
        "filter": 1,
        "languages": [LANGUAGE],
        "cursor": cursor,
        "review_type": 0,
        "purchase_type": 1,
        "num_per_page": 100,
    }
    response = get_json(REVIEWS_API, {"input_json": json.dumps(params)})
    return response.get("response", response)


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_game(app_id, name, goal, global_seen, reasons):
    genres = get_game_genres(app_id)
    print(f"\n{name} ({app_id}); géneros: {genres}")
    cursor = "*"
    seen_cursors = set()
    accepted = []

    for page in range(MAX_PAGES_PER_GAME):
        if len(accepted) >= goal or cursor in seen_cursors:
            break
        seen_cursors.add(cursor)
        try:
            result = fetch_reviews_page(app_id, cursor)
        except RuntimeError as exc:
            print(f"  Error de extracción: {exc}")
            break

        reviews = result.get("reviews") or []
        if not isinstance(reviews, list) or not reviews:
            break

        append_jsonl(RAW_JSONL_FILE, {
            "appid": app_id, "game": name, "page": page + 1,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "data": result,
        })

        for review in reviews:
            review_id = str(review.get("recommendationid", "")).strip()
            if not review_id or review_id in global_seen:
                continue
            global_seen.add(review_id)

            original = review.get("review", "")
            valid, reason, probability = validate_english_review(original)
            if not valid:
                reasons[reason] += 1
                append_jsonl(REJECTED_FILE, {
                    "review_id": review_id, "appid": app_id,
                    "reason": reason, "en_confidence": probability,
                    "steam_language": review.get("language"),
                    "review": original,
                })
                continue

            timestamp = review.get("timestamp_created")
            try:
                published = datetime.fromtimestamp(
                    int(timestamp), tz=timezone.utc
                ).strftime("%Y-%m-%d") if timestamp else ""
            except (TypeError, ValueError, OverflowError, OSError):
                published = ""

            author = review.get("author") or {}
            steam_id = str(author.get("steamid", ""))
            minutes = author.get("playtime_forever") or 0
            try:
                hours = round(float(minutes) / 60, 2)
            except (TypeError, ValueError):
                hours = 0.0

            accepted.append({
                "review_id": review_id,
                "item_id": app_id,
                "item_name": name,
                "review": original,  # SIN lematizar ni quitar stopwords.
                "rating": "positive" if review.get("voted_up") else "negative",
                "date": published,
                "category": genres,
                "source": "Steam",
                "url": (f"https://steamcommunity.com/profiles/{steam_id}/recommended/{app_id}/"
                        if steam_id else f"https://store.steampowered.com/app/{app_id}/"),
                "source_id": review_id,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "language": "english",
                "votes_up": review.get("votes_up", 0),
                "playtime_hours": hours,
            })
            if len(accepted) >= goal:
                break

        print(f"  Página {page + 1}: {len(accepted)}/{goal} reseñas válidas")
        next_cursor = result.get("cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        if len(accepted) < goal:
            time.sleep(REQUEST_DELAY)

    return accepted


def main():
    # Cada ejecución produce archivos nuevos; evitar mezclar corridas.
    RAW_JSONL_FILE.write_text("", encoding="utf-8")
    REJECTED_FILE.write_text("", encoding="utf-8")

    seen = set()
    reasons = Counter()
    rows = []
    for app_id, name in GAMES.items():
        rows.extend(collect_game(
            app_id, name, REVIEWS_PER_GAME, seen, reasons
        ))

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCorpus: {len(rows)} opiniones aceptadas -> {OUTPUT_FILE}")
    print(f"Datos originales de API -> {RAW_JSONL_FILE}")
    print(f"Rechazadas con motivo -> {REJECTED_FILE}")
    print("Distribución por juego:")
    for game in GAMES.values():
        print(f"  {game}: {sum(r['item_name'] == game for r in rows)}")
    print("Rechazos:", dict(reasons))
    if len(rows) < 300:
        print("AVISO: menos de 300 opiniones. Revise rechazos o agregue juegos.")
    if len(rows) > 500:
        print("AVISO: más de 500 opiniones; ajuste REVIEWS_PER_GAME.")


if __name__ == "__main__":
    main()
