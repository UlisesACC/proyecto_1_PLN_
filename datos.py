
import requests
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

GAMES = {
    730: "Counter-Strike 2",
    570: "Dota 2",
    620: "Portal 2",
    413150: "Stardew Valley",
    1086940: "Baldur's Gate 3",
    1384160: "Guilty Gear -Strive-",
    3010850: "Gears of War: E-Day",
    4080220: "EA Sports FC 27"
}

REVIEWS_PER_GAME = 100
LANGUAGE = "english"
OUTPUT_FILE = "opinions_raw_corpus.tsv"
RAW_JSONL_FILE = "steam_reviews_raw.jsonl"

API_URL = (
    "https://api.steampowered.com/"
    "IUserReviewsService/GetAppReviews/v1/"
)

FIELDS = [
    "review_id",
    "item_id",
    "item_name",
    "review",
    "rating",
    "date",
    "category",
    "source",
    "url",
    "source_id",
    "retrieved_at",
    "language",
    "votes_up",
    "playtime_hours"
]

session = requests.Session()
session.headers.update({
    "User-Agent": "OpinionRetrievalAcademicProject/1.0"
})


def fetch_page(app_id, cursor):

    payload = {
        "appid": app_id,
        "filter": 1,
        "languages": [LANGUAGE],
        "cursor": cursor,
        "review_type": 0,
        "purchase_type": 1,
        "num_per_page": 100
    }

    for attempt in range(5):

        try:
            response = session.get(
                API_URL,
                params={
                    "input_json": json.dumps(payload)
                },
                timeout=30
            )

            if response.status_code == 429:
                retry_after = response.headers.get(
                    "Retry-After", ""
                )

                try:
                    wait = min(float(retry_after), 120)
                except ValueError:
                    wait = min(10 * (2 ** attempt), 120)

                print(f"Límite de Steam. Esperando {wait}s")
                time.sleep(wait)
                continue

            if response.status_code >= 500:
                time.sleep(3 * (attempt + 1))
                continue

            response.raise_for_status()

            data = response.json()
            return data.get("response", data)

        except (requests.RequestException, ValueError) as error:
            print(f"Error al consultar Steam: {error}")
            time.sleep(3 * (attempt + 1))

    raise RuntimeError(
        f"No fue posible consultar el juego {app_id}"
    )


def extract_reviews(app_id, game_name, limit=100):

    print(f"\nExtrayendo reseñas de {game_name}...")

    results = []
    cursor = "*"

    seen_ids = set()
    seen_cursors = set()

    while len(results) < limit:

        if cursor in seen_cursors:
            print("Cursor repetido. Finalizando.")
            break

        seen_cursors.add(cursor)

        try:
            data = fetch_page(app_id, cursor)
        except RuntimeError as error:
            print(error)
            break

        reviews = data.get("reviews", [])

        if not reviews:
            print("No hay más reseñas disponibles.")
            break

        with open(RAW_JSONL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "appid": app_id,
                "game": game_name,
                "data": data
            }, ensure_ascii=False) + "\n")

        for review in reviews:

            review_id = str(
                review.get("recommendationid", "")
            )

            text = review.get("review", "").strip()

            if not review_id or not text:
                continue

            if review_id in seen_ids:
                continue

            seen_ids.add(review_id)

            timestamp = review.get("timestamp_created")
            date = (
                datetime.fromtimestamp(
                    timestamp,
                    tz=timezone.utc
                ).strftime("%Y-%m-%d")
                if timestamp else ""
            )

            steam_id = str(
                review.get("author", {}).get("steamid", "")
            )

            rating = (
                "positive"
                if review.get("voted_up", False)
                else "negative"
            )

            url = (
                f"https://steamcommunity.com/profiles/"
                f"{steam_id}/recommended/{app_id}/"
                if steam_id else
                f"https://store.steampowered.com/app/{app_id}/"
            )

            playtime = review.get(
                "author", {}
            ).get("playtime_forever", 0)

            row = {
                "review_id": review_id,
                "item_id": app_id,
                "item_name": game_name,
                "review": text,
                "rating": rating,
                "date": date,
                "category": "Video Games",
                "source": "Steam",
                "url": url,
                "source_id": review_id,
                "retrieved_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "language": review.get("language", ""),
                "votes_up": review.get("votes_up", 0),
                "playtime_hours": round(
                    playtime / 60, 2
                )
            }

            results.append(row)

            if len(results) >= limit:
                break

        print(f"  Recolectadas: {len(results)}/{limit}")

        next_cursor = data.get("cursor")

        if not next_cursor or next_cursor == cursor:
            break

        cursor = next_cursor

        time.sleep(2)

    return results


def main():

    all_reviews = []

    Path(RAW_JSONL_FILE).write_text(
        "", encoding="utf-8"
    )

    for app_id, game_name in GAMES.items():

        reviews = extract_reviews(
            app_id,
            game_name,
            REVIEWS_PER_GAME
        )

        all_reviews.extend(reviews)

        print(
            f"{game_name}: {len(reviews)} reseñas obtenidas"
        )

    unique_reviews = {
        row["review_id"]: row
        for row in all_reviews
    }

    all_reviews = list(unique_reviews.values())

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
            delimiter="\t"
        )

        writer.writeheader()
        writer.writerows(all_reviews)

    print(f"Total: {len(all_reviews)} reseñas")
    print(f"Archivo: {OUTPUT_FILE}")
    print(f"JSON original: {RAW_JSONL_FILE}")

    print("Distribución por juego:")

    for game_name in GAMES.values():
        count = sum(
            1 for r in all_reviews
            if r["item_name"] == game_name
        )
        print(f"{game_name}: {count}")


if __name__ == "__main__":
    main()
