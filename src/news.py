"""Fetches same-day player news from ESPN's fantasy news API, which - unlike
the static injuryStatus field on a Player - can carry something that
happened this morning. Joins by ESPN's own playerId, so there's no
name-matching involved."""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import certifi
import requests

NEWS_URL = "https://site.web.api.espn.com/apis/fantasy/v2/games/ffl/news/players"
BATCH_SIZE = 25
REQUEST_TIMEOUT = 15

# Rotowire items are the terse same-day beat notes we actually want ("Mahomes
# (knee) does not have an injury designation for Monday's game"). Story/Media
# items are longer-form editorial content, not the kind of thing worth
# spending prompt tokens on.
KEPT_TYPES = {"Rotowire"}

MAX_ITEMS_PER_PLAYER = 3


@dataclass
class NewsItem:
    headline: str
    description: str
    published: datetime


def _fetch_batch(player_ids: list, days: int) -> dict:
    params = [("days", days), ("limit", 100)] + [("playerId", pid) for pid in player_ids]
    response = requests.get(
        NEWS_URL, params=params, timeout=REQUEST_TIMEOUT, verify=certifi.where(),
    )
    response.raise_for_status()
    return response.json()


def fetch_player_news(player_ids: list, days: int = 7) -> dict:
    """Returns {player_id: [NewsItem, ...]}, newest first, capped per player.

    News is an enhancement, not a dependency - any network failure is
    logged and results in an empty dict rather than breaking the run.
    """
    by_player = {}
    ids = [pid for pid in player_ids if pid]

    for start in range(0, len(ids), BATCH_SIZE):
        batch = ids[start:start + BATCH_SIZE]
        try:
            data = _fetch_batch(batch, days)
        except Exception as e:
            print(f"  Warning: failed to fetch player news: {e}", file=sys.stderr)
            continue

        for item in data.get("feed", []):
            if item.get("type") not in KEPT_TYPES:
                continue
            player_id = item.get("playerId")
            if player_id is None:
                continue
            published = _parse_timestamp(item.get("published"))
            if published is None:
                continue
            by_player.setdefault(player_id, []).append(
                NewsItem(
                    headline=item.get("headline", ""),
                    description=item.get("description", ""),
                    published=published,
                )
            )

    for player_id, items in by_player.items():
        items.sort(key=lambda i: i.published, reverse=True)
        by_player[player_id] = items[:MAX_ITEMS_PER_PLAYER]

    return by_player


def _parse_timestamp(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_age(published: datetime) -> str:
    delta = datetime.now(timezone.utc) - published
    if delta < timedelta(hours=1):
        return f"{max(int(delta.total_seconds() // 60), 1)}m ago"
    if delta < timedelta(days=1):
        return f"{int(delta.total_seconds() // 3600)}h ago"
    return f"{delta.days}d ago"
