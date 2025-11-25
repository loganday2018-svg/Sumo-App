"""Streamlit entry point for the Sumo tournament companion app."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.parse import quote_plus

import requests
import pandas as pd
import streamlit as st


def _ensure_typed_dict_closed_support() -> None:
    """Patch TypedDictMeta so Altair works on Python versions without closed."""

    def _patch_meta(module: object) -> None:
        meta = getattr(module, "_TypedDictMeta", None)
        if meta is None:
            return
        try:
            from inspect import signature

            params = signature(meta.__new__).parameters
        except Exception:
            params = {}
        if "closed" in params:
            return
        original_new = meta.__new__

        def patched_new(cls, name, bases, namespace, total=True, **kwargs):
            kwargs.pop("closed", None)
            return original_new(cls, name, bases, namespace, total=total, **kwargs)

        meta.__new__ = patched_new  # type: ignore[assignment]

    import typing

    _patch_meta(typing)
    try:
        import typing_extensions
    except ImportError:
        return
    _patch_meta(typing_extensions)


_ensure_typed_dict_closed_support()
import altair as alt


SUMO_API_BASE_URL = "https://www.sumo-api.com/api"
SUMO_API_CACHE_TTL = 60 * 60 * 24  # one day
DEFAULT_DIVISION = "Makuuchi"
CACHE_DIR = Path(".sumo_cache")
HEAD_TO_HEAD_DB = Path("head_to_head.json")
DIVISION_ORDER = {
    "Makuuchi": 0,
    "Juryo": 1,
    "Makushita": 2,
    "Sandanme": 3,
    "Jonidan": 4,
    "Jonokuchi": 5,
}

# Rank ordering for upset magnitude calculation (lower = higher rank)
RANK_ORDER = {
    "Yokozuna": 1,
    "Ozeki": 2,
    "Sekiwake": 3,
    "Komusubi": 4,
}
# Maegashira ranks get 5 + their number (M1 = 6, M2 = 7, etc.)


# ==================== Data Models ====================

@dataclass(frozen=True)
class Tournament:
    """Represents one of the six annual honbasho."""

    name: str
    city: str
    month: str
    venue: str
    basho_id: str
    start_date: date
    end_date: date
    champion: str | None = None
    highlight: str | None = None


# ==================== Business Logic Helpers ====================


def assemble_tournaments(year: int) -> List[Tournament]:
    """Return the canonical six honbasho for a given season."""

    template = [
        ("Hatsu Basho", "Tokyo", "January", "Ryogoku Kokugikan", (1, 12), (1, 26)),
        ("Haru Basho", "Osaka", "March", "Edion Arena", (3, 9), (3, 23)),
        ("Natsu Basho", "Tokyo", "May", "Ryogoku Kokugikan", (5, 11), (5, 25)),
        ("Nagoya Basho", "Nagoya", "July", "Aichi Prefectural Gym", (7, 13), (7, 27)),
        ("Aki Basho", "Tokyo", "September", "Ryogoku Kokugikan", (9, 14), (9, 28)),
        ("Kyushu Basho", "Fukuoka", "November", "Fukuoka Kokusai Center", (11, 9), (11, 23)),
    ]

    # Year-specific champion and highlight data
    champions_by_year: Dict[int, Dict[str, str]] = {
        2024: {
            "Hatsu Basho": "Terunofuji",    # 13-2
            "Haru Basho": "Onosato",        # 13-2, makuuchi debut yusho
            "Natsu Basho": "Onosato",       # 12-3
            "Nagoya Basho": "Onosato",      # 13-2
            "Aki Basho": "Onosato",         # 13-2
            "Kyushu Basho": "Kotozakura",   # 14-1, became Ozeki
        },
        2025: {
            "Hatsu Basho": "Hoshoryu",      # 12-3 playoff win, became 74th Yokozuna
            "Haru Basho": "Onosato",        # 12-3
            "Natsu Basho": "Onosato",       # 14-1, became 75th Yokozuna  
            "Nagoya Basho": "Kotoshoho",    # 13-2, first yusho
            "Aki Basho": "Onosato",         # Playoff win vs Hoshoryu
            "Kyushu Basho": "Aonishiki",    # 12-3 playoff win vs Hoshoryu
        },
        2026: {},  # Future year - no champions yet
    }
    
    highlights_by_year: Dict[int, Dict[str, str]] = {
        2024: {
            "Hatsu Basho": "Terunofuji returns from injury for his 10th yusho.",
            "Haru Basho": "Onosato's stunning makuuchi debut yusho.",
            "Natsu Basho": "Onosato continues his dominant rookie year.",
            "Nagoya Basho": "Onosato earns Ozeki promotion.",
            "Aki Basho": "Onosato's 4th straight yusho as new Ozeki.",
            "Kyushu Basho": "Kotozakura claims the cup, earns Ozeki rank.",
        },
        2025: {
            "Hatsu Basho": "Hoshoryu wins 3-way playoff to become 74th Yokozuna.",
            "Haru Basho": "Onosato's back-to-back run begins.",
            "Natsu Basho": "Onosato goes 14-1, becomes 75th Yokozuna — fastest ever.",
            "Nagoya Basho": "Kotoshoho shocks the field for his first Emperor's Cup.",
            "Aki Basho": "Epic Yokozuna vs Yokozuna playoff — Onosato prevails.",
            "Kyushu Basho": "Aonishiki defeats Hoshoryu in playoff, eyes Ozeki.",
        },
        2026: {
            "Hatsu Basho": "New Year kickoff — who will start 2026 strong?",
            "Haru Basho": "Spring tournament in Osaka.",
            "Natsu Basho": "Summer showdown in Tokyo.",
            "Nagoya Basho": "Sweltering summer meet at the new Aichi Arena.",
            "Aki Basho": "Autumn tournament at historic Kokugikan.",
            "Kyushu Basho": "Season finale and last chance at yearly awards.",
        },
    }
    
    champions = champions_by_year.get(year, {})
    highlights = highlights_by_year.get(year, {})

    tournaments: List[Tournament] = []
    for name, city, month, venue, start_parts, end_parts in template:
        month_number = start_parts[0]
        basho_id = f"{year}{month_number:02d}"
        tournaments.append(
            Tournament(
                name=name,
                city=city,
                month=month,
                venue=venue,
                basho_id=basho_id,
                start_date=date(year, start_parts[0], start_parts[1]),
                end_date=date(year, end_parts[0], end_parts[1]),
                champion=champions.get(name),
                highlight=highlights.get(name),
            )
        )
    return tournaments


def determine_default_tournament_index(tournaments: Sequence[Tournament]) -> int:
    """Pick the active basho or the next upcoming one as default."""

    today = date.today()
    for idx, tournament in enumerate(tournaments):
        if tournament.start_date <= today <= tournament.end_date:
            return idx

    for idx, tournament in enumerate(tournaments):
        if tournament.start_date > today:
            return idx

    return max(len(tournaments) - 1, 0)


def tournaments_dataframe(tournaments: Sequence[Tournament]) -> pd.DataFrame:
    """Convert tournament models to something Streamlit can chart."""

    return pd.DataFrame(
        [
            {
                "Tournament": t.name,
                "City": t.city,
                "Month": t.month,
                "Venue": t.venue,
                "Basho ID": t.basho_id,
                "Start": t.start_date,
                "End": t.end_date,
                "Champion": t.champion,
                "Highlight": t.highlight,
            }
            for t in tournaments
        ]
    )


def _is_past_tournament(basho_id: str, tournaments: Sequence[Tournament]) -> bool:
    """Check if a tournament is in the past."""
    today = date.today()
    for tournament in tournaments:
        if tournament.basho_id == basho_id:
            return tournament.end_date < today
    return False


def _get_cache_path(basho_id: str, cache_type: str, division: str | None = None, day: int | None = None) -> Path:
    """Get the cache file path for a specific data type."""
    if day is not None:
        filename = f"{basho_id}_{cache_type}_{division}_day{day}.json"
    elif division:
        filename = f"{basho_id}_{cache_type}_{division}.json"
    else:
        filename = f"{basho_id}_{cache_type}.json"
    return CACHE_DIR / filename


def _load_from_cache(cache_path: Path) -> Dict | None:
    """Load data from cache file if it exists."""
    try:
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError, OSError):
        pass
    return None


def _save_to_cache(cache_path: Path, data: Dict) -> None:
    """Save data to cache file."""
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except (IOError, OSError):
        pass  # Silently fail if cache write fails


def _sumo_api_get(path: str, params: Dict[str, str] | None = None) -> Dict:
    """Call the public Sumo API and return JSON."""

    url = f"{SUMO_API_BASE_URL}{path}"
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Sumo API request failed: {exc}") from exc
    return response.json()


def _is_past_basho(basho_id: str) -> bool:
    """Check if a basho is in the past (completed)."""
    try:
        year = int(basho_id[:4])
        month = int(basho_id[4:6])
        # Tournaments end around day 28 of the month
        tournament_end = date(year, month, 28)
        return date.today() > tournament_end
    except (ValueError, IndexError):
        return False


@st.cache_data(ttl=SUMO_API_CACHE_TTL)
def fetch_basho_overview(basho_id: str, tournaments: Sequence[Tournament] | None = None) -> Dict:
    """Fetch high-level tournament results. Uses local cache for past tournaments."""
    cache_path = CACHE_DIR / f"{basho_id}_overview.json"
    
    # Always try cache first
    cached_data = _load_from_cache(cache_path)
    if cached_data:
        return cached_data
    
    # For past tournaments, if no cache exists, still try API but cache result
    # For current tournaments, always fetch fresh
    data = _sumo_api_get(f"/basho/{basho_id}")
    
    # Save to cache (especially important for completed tournaments)
    if data and _is_past_basho(basho_id):
        _save_to_cache(cache_path, data)
    
    return data


@st.cache_data(ttl=SUMO_API_CACHE_TTL)
def fetch_torikumi_payload(basho_id: str, division: str, day: int, tournaments: Sequence[Tournament] | None = None) -> Dict:
    """Fetch torikumi data for a specific division and day. Uses local cache for past tournaments."""
    cache_path = CACHE_DIR / f"{basho_id}_torikumi_{division}_day{day}.json"
    
    # Always try cache first
    cached_data = _load_from_cache(cache_path)
    if cached_data:
        return cached_data
    
    # Fetch from API
    data = _sumo_api_get(f"/basho/{basho_id}/torikumi/{division}/{day}")
    
    # Save to cache for past tournaments
    if data and _is_past_basho(basho_id):
        _save_to_cache(cache_path, data)
    
    return data


def load_head_to_head_db() -> Dict[str, Dict]:
    """Load head-to-head database from disk."""
    if HEAD_TO_HEAD_DB.exists():
        try:
            with open(HEAD_TO_HEAD_DB, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_head_to_head_db(db: Dict[str, Dict]) -> None:
    """Save head-to-head database to disk."""
    try:
        with open(HEAD_TO_HEAD_DB, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except IOError:
        pass


def update_head_to_head_from_matches(matches: Sequence[Dict]) -> None:
    """Update head-to-head database from completed matches."""
    db = load_head_to_head_db()
    
    for match in matches:
        east_name = match.get("eastShikona")
        west_name = match.get("westShikona")
        winner_name = match.get("winnerEn")
        
        if not all([east_name, west_name, winner_name]):
            continue
        
        # Create sorted key for consistency
        pair_key = "_vs_".join(sorted([east_name, west_name]))
        
        if pair_key not in db:
            db[pair_key] = {
                "wrestler1": sorted([east_name, west_name])[0],
                "wrestler2": sorted([east_name, west_name])[1],
                "wrestler1_wins": 0,
                "wrestler2_wins": 0,
                "total_matches": 0,
                "last_winner": None
            }
        
        # Update record
        db[pair_key]["total_matches"] += 1
        db[pair_key]["last_winner"] = winner_name
        
        if winner_name == db[pair_key]["wrestler1"]:
            db[pair_key]["wrestler1_wins"] += 1
        elif winner_name == db[pair_key]["wrestler2"]:
            db[pair_key]["wrestler2_wins"] += 1
    
    save_head_to_head_db(db)


def get_head_to_head_prediction(wrestler1: str, wrestler2: str) -> Dict[str, object]:
    """Get head-to-head record and prediction for two wrestlers."""
    db = load_head_to_head_db()
    pair_key = "_vs_".join(sorted([wrestler1, wrestler2]))
    
    if pair_key not in db:
        return {
            "total_matches": 0,
            "wrestler1_wins": 0,
            "wrestler2_wins": 0,
            "prediction": "No previous matches",
            "confidence": 0.0
        }
    
    record = db[pair_key]
    total = record["total_matches"]
    
    # Determine which wrestler is which in the stored record
    w1_is_first = wrestler1 == record["wrestler1"]
    w1_wins = record["wrestler1_wins"] if w1_is_first else record["wrestler2_wins"]
    w2_wins = record["wrestler2_wins"] if w1_is_first else record["wrestler1_wins"]
    
    # Calculate win percentages
    w1_pct = w1_wins / total if total > 0 else 0.5
    w2_pct = w2_wins / total if total > 0 else 0.5
    
    # Make prediction
    if w1_wins > w2_wins:
        predicted_winner = wrestler1
        confidence = w1_pct
    elif w2_wins > w1_wins:
        predicted_winner = wrestler2
        confidence = w2_pct
    else:
        predicted_winner = "Even"
        confidence = 0.5
    
    return {
        "total_matches": total,
        "wrestler1_wins": w1_wins,
        "wrestler2_wins": w2_wins,
        "wrestler1_win_pct": w1_pct,
        "wrestler2_win_pct": w2_pct,
        "prediction": predicted_winner,
        "confidence": confidence,
        "last_winner": record.get("last_winner")
    }


@st.cache_data(ttl=SUMO_API_CACHE_TTL)
def fetch_head_to_head_pair(pair: tuple[int, int]) -> Dict:
    """Fetch head-to-head results for a pair of rikishi IDs. Uses local cache when available."""
    rikishi_id, opponent_id = pair
    cache_path = CACHE_DIR / f"h2h_{rikishi_id}_{opponent_id}.json"
    
    # Try cache first
    cached_data = _load_from_cache(cache_path)
    if cached_data:
        return cached_data
    
    # Fetch from API
    data = _sumo_api_get(f"/rikishi/{rikishi_id}/matches/{opponent_id}")
    
    # Cache the result
    if data:
        _save_to_cache(cache_path, data)
    
    return data


@st.cache_data(ttl=SUMO_API_CACHE_TTL)
def fetch_completed_torikumi(basho_id: str, division: str, tournaments: Sequence[Tournament] | None = None) -> Dict[int, List[Dict]]:
    """Return torikumi for every fully completed day."""
    completed: Dict[int, List[Dict]] = {}
    for day in range(1, 16):
        try:
            payload = fetch_torikumi_payload(basho_id, division, day, tournaments)
        except RuntimeError:
            break
        matches = payload.get("torikumi") or []
        if not matches:
            break
        day_finished = all(match.get("winnerId") is not None for match in matches)
        if not day_finished:
            break
        completed[day] = matches
    return completed


def torikumi_dataframe(matches: Sequence[Dict]) -> pd.DataFrame:
    """Return a simple dataframe for display in Streamlit."""

    return pd.DataFrame(
        [
            {
                "Bout #": match.get("matchNo"),
                "East": f"{match.get('eastShikona')} ({match.get('eastRank')})",
                "West": f"{match.get('westShikona')} ({match.get('westRank')})",
                "Winner": match.get("winnerEn") or "Pending",
                "Kimarite": match.get("kimarite") or "TBD",
            }
            for match in matches
        ]
    )


def torikumi_results_dataframe(results: Dict[int, Sequence[Dict]]) -> pd.DataFrame:
    """Flatten day-indexed results into a displayable frame."""

    rows: List[Dict[str, str | int]] = []
    for day in sorted(results.keys()):
        for match in results[day]:
            rows.append(
                {
                    "Day": day,
                    "Bout #": match.get("matchNo"),
                    "East": f"{match.get('eastShikona')} ({match.get('eastRank')})",
                    "West": f"{match.get('westShikona')} ({match.get('westRank')})",
                    "Winner": match.get("winnerEn") or "Pending",
                    "Kimarite": match.get("kimarite") or "TBD",
                }
            )
    return pd.DataFrame(rows)


def compute_rikishi_records(results: Dict[int, Sequence[Dict]]) -> Dict[str, Dict[str, object]]:
    """Aggregate wins/losses for each rikishi from completed days."""

    stats: Dict[str, Dict[str, object]] = {}
    for matches in results.values():
        for match in matches:
            east_name = match.get("eastShikona") or f"east-{match.get('eastId')}"
            west_name = match.get("westShikona") or f"west-{match.get('westId')}"
            east_id = match.get("eastId")
            west_id = match.get("westId")
            east_rank = match.get("eastRank")
            west_rank = match.get("westRank")
            winner_name = match.get("winnerEn")
            if not winner_name:
                continue

            for name, rank, rid in ((east_name, east_rank, east_id), (west_name, west_rank, west_id)):
                if name not in stats:
                    stats[name] = {
                        "rank": rank or "-",
                        "wins": 0,
                        "losses": 0,
                        "id": rid,
                    }

            if winner_name == east_name:
                stats[east_name]["wins"] = int(stats[east_name]["wins"]) + 1
                stats[west_name]["losses"] = int(stats[west_name]["losses"]) + 1
            elif winner_name == west_name:
                stats[west_name]["wins"] = int(stats[west_name]["wins"]) + 1
                stats[east_name]["losses"] = int(stats[east_name]["losses"]) + 1

    return stats


def build_match_history(results: Dict[int, Sequence[Dict]]) -> Dict[str, List[Dict[str, object]]]:
    """Build chronological match history with opponent details for each rikishi."""
    
    history: Dict[str, List[Dict[str, object]]] = {}
    
    # Process matches in chronological order (by day, then by match order)
    for day in sorted(results.keys()):
        for match in results[day]:
            east_name = match.get("eastShikona") or f"east-{match.get('eastId')}"
            west_name = match.get("westShikona") or f"west-{match.get('westId')}"
            winner_name = match.get("winnerEn")
            
            if not winner_name:
                continue
            
            # Initialize history if needed
            if east_name not in history:
                history[east_name] = []
            if west_name not in history:
                history[west_name] = []
            
            # Record match details
            east_won = winner_name == east_name
            history[east_name].append({
                "is_win": east_won,
                "opponent": west_name,
                "day": day,
                "result": "Win" if east_won else "Loss"
            })
            history[west_name].append({
                "is_win": not east_won,
                "opponent": east_name,
                "day": day,
                "result": "Win" if not east_won else "Loss"
            })
    
    return history


def generate_match_bubbles(match_history: List[Dict[str, object]]) -> str:
    """Generate HTML bubbles with tooltips for match results."""
    
    if not match_history:
        return ""
    
    bubbles = []
    for match_info in match_history:
        is_win = match_info["is_win"]
        opponent = escape(str(match_info["opponent"]))
        day = match_info["day"]
        result = match_info["result"]
        
        tooltip = escape(f"Day {day}: {result} vs {opponent}")
        
        if is_win:
            # Filled green circle for win with tooltip
            bubbles.append(
                f'<span title="{tooltip}" style="cursor:pointer;color:#4CAF50;font-size:28px;">●</span>'
            )
        else:
            # Empty circle for loss with tooltip
            bubbles.append(
                f'<span title="{tooltip}" style="cursor:pointer;color:#666;font-size:28px;">○</span>'
            )
    
    return " ".join(bubbles)


def calculate_momentum_score(
    match_history: List[Dict[str, object]], 
    records: Dict[str, Dict[str, object]]
) -> Dict[str, object]:
    """
    Calculate a momentum score that weighs recent matches more heavily.
    Returns a dict with score, label, description, and streak info.
    """
    if not match_history:
        return {
            "score": 0,
            "label": "➖",
            "description": "No matches yet",
            "streak": 0,
            "streak_type": None
        }
    
    # Calculate weighted score (recent matches count more)
    weighted_score = 0
    total_weight = 0
    
    for i, match in enumerate(match_history):
        # Weight increases for more recent matches (last match = highest weight)
        weight = i + 1
        total_weight += weight
        if match["is_win"]:
            weighted_score += weight
        else:
            weighted_score -= weight * 0.5  # Losses hurt less than wins help
    
    # Normalize to -1 to 1 scale
    if total_weight > 0:
        normalized = weighted_score / total_weight
    else:
        normalized = 0
    
    # Calculate current streak
    streak = 0
    streak_type = None
    if match_history:
        last_result = match_history[-1]["is_win"]
        streak_type = "W" if last_result else "L"
        for match in reversed(match_history):
            if match["is_win"] == last_result:
                streak += 1
            else:
                break
    
    # Determine momentum label and description
    if normalized >= 0.6:
        label = "🔥🔥"
        description = f"On fire! {streak}{'W' if streak_type == 'W' else 'L'} streak" if streak >= 2 else "Hot form"
    elif normalized >= 0.3:
        label = "🔥"
        description = f"Strong momentum" + (f" ({streak}W streak)" if streak >= 2 and streak_type == 'W' else "")
    elif normalized >= -0.1:
        label = "➖"
        description = "Steady form"
    elif normalized >= -0.4:
        label = "❄️"
        description = f"Cooling off" + (f" ({streak}L streak)" if streak >= 2 and streak_type == 'L' else "")
    else:
        label = "❄️❄️"
        description = f"Ice cold! {streak}L streak" if streak >= 2 and streak_type == 'L' else "Struggling"
    
    return {
        "score": round(normalized, 2),
        "label": label,
        "description": description,
        "streak": streak,
        "streak_type": streak_type
    }


# ==================== NEW FEATURES: Momentum, Upsets, Tale of the Tape ====================


def parse_rank_to_number(rank: str | None) -> int:
    """Convert a rank string to a numeric value for comparison (lower = higher rank)."""
    if not rank:
        return 99
    
    rank = rank.strip()
    
    # Check named ranks first
    for named_rank, value in RANK_ORDER.items():
        if rank.startswith(named_rank):
            # Handle East/West distinction (East is slightly higher)
            if "East" in rank or "e" in rank.split()[-1] if len(rank.split()) > 1 else False:
                return value
            return value + 0.5
    
    # Parse Maegashira ranks (M1, M2, etc.)
    if rank.startswith("M") or rank.startswith("Maegashira"):
        try:
            # Extract number from rank like "M1" or "Maegashira 1"
            parts = rank.replace("Maegashira", "M").split()
            num_part = parts[0].replace("M", "") if parts else "17"
            if not num_part:
                num_part = parts[1] if len(parts) > 1 else "17"
            num = int(num_part)
            base = 5 + num
            # East/West distinction
            if len(parts) > 1 and parts[1].lower().startswith("e"):
                return base
            return base + 0.5
        except (ValueError, IndexError):
            return 20
    
    # Juryo and below
    if rank.startswith("J") or rank.startswith("Juryo"):
        return 25
    
    return 30  # Unknown ranks


def calculate_upset_magnitude(winner_rank: str | None, loser_rank: str | None) -> Dict[str, object]:
    """Calculate if a match result was an upset and its magnitude."""
    winner_num = parse_rank_to_number(winner_rank)
    loser_num = parse_rank_to_number(loser_rank)
    
    # Upset = lower-ranked wrestler (higher number) beats higher-ranked (lower number)
    rank_diff = winner_num - loser_num
    
    if rank_diff <= 0:
        # Favorite won, no upset
        return {
            "is_upset": False,
            "magnitude": 0,
            "label": None,
            "rank_diff": rank_diff
        }
    
    # Determine upset severity
    if rank_diff >= 8:
        label = "🚨 MASSIVE UPSET"
        magnitude = 3
    elif rank_diff >= 4:
        label = "⚠️ Major Upset"
        magnitude = 2
    elif rank_diff >= 2:
        label = "📢 Upset"
        magnitude = 1
    else:
        label = "Minor upset"
        magnitude = 0.5
    
    return {
        "is_upset": True,
        "magnitude": magnitude,
        "label": label,
        "rank_diff": rank_diff
    }


def analyze_upsets(results: Dict[int, Sequence[Dict]]) -> Dict[str, object]:
    """Analyze all upsets in the tournament results."""
    upsets: List[Dict] = []
    giant_killers: Dict[str, int] = {}  # wrestler -> upset wins count
    vulnerable: Dict[str, int] = {}  # high-ranked wrestler -> losses to lower-ranked
    
    for day, matches in results.items():
        for match in matches:
            winner_name = match.get("winnerEn")
            if not winner_name:
                continue
            
            east_name = match.get("eastShikona")
            west_name = match.get("westShikona")
            east_rank = match.get("eastRank")
            west_rank = match.get("westRank")
            
            # Determine winner/loser ranks
            if winner_name == east_name:
                winner_rank, loser_rank = east_rank, west_rank
                loser_name = west_name
            else:
                winner_rank, loser_rank = west_rank, east_rank
                loser_name = east_name
            
            upset_info = calculate_upset_magnitude(winner_rank, loser_rank)
            
            if upset_info["is_upset"] and upset_info["magnitude"] >= 0.5:
                upsets.append({
                    "day": day,
                    "winner": winner_name,
                    "winner_rank": winner_rank,
                    "loser": loser_name,
                    "loser_rank": loser_rank,
                    "magnitude": upset_info["magnitude"],
                    "label": upset_info["label"],
                    "kimarite": match.get("kimarite")
                })
                
                # Track giant killers
                giant_killers[winner_name] = giant_killers.get(winner_name, 0) + 1
                
                # Track vulnerable wrestlers (only for significant upsets)
                if upset_info["magnitude"] >= 1:
                    vulnerable[loser_name] = vulnerable.get(loser_name, 0) + 1
    
    # Sort upsets by magnitude and day
    upsets.sort(key=lambda x: (-x["magnitude"], x["day"]))
    
    # Get top giant killers and vulnerable
    top_giant_killers = sorted(giant_killers.items(), key=lambda x: -x[1])[:5]
    top_vulnerable = sorted(vulnerable.items(), key=lambda x: -x[1])[:5]
    
    return {
        "upsets": upsets,
        "giant_killers": top_giant_killers,
        "vulnerable": top_vulnerable,
        "total_upsets": len(upsets),
        "biggest_upset": upsets[0] if upsets else None
    }


def calculate_momentum_score(
    match_history: List[Dict[str, object]],
    records: Dict[str, Dict[str, object]]
) -> Dict[str, object]:
    """
    Calculate a momentum score for a wrestler based on recent performance.
    
    Factors:
    - Recent matches weighted more heavily (exponential decay)
    - Opponent strength (beating higher-ranked = more momentum)
    - Win streaks bonus
    """
    if not match_history:
        return {
            "score": 0,
            "label": "➡️",
            "description": "No matches yet",
            "trend": "neutral",
            "streak": 0,
            "streak_type": None
        }
    
    # Calculate weighted score (recent matches count more)
    weights = []
    base_weight = 1.0
    decay = 0.8  # Each older match is worth 80% of the next
    
    for i, match in enumerate(reversed(match_history)):  # Most recent first
        weight = base_weight * (decay ** i)
        weights.append(weight)
    
    weights.reverse()  # Back to chronological order
    
    # Calculate momentum
    momentum = 0.0
    total_weight = sum(weights)
    
    for i, match in enumerate(match_history):
        is_win = match["is_win"]
        opponent = match["opponent"]
        
        # Base value: win = +1, loss = -1
        base_value = 1.0 if is_win else -1.0
        
        # Opponent strength multiplier
        opponent_record = records.get(opponent, {})
        opponent_wins = int(opponent_record.get("wins", 0))
        opponent_losses = int(opponent_record.get("losses", 0))
        opponent_total = opponent_wins + opponent_losses
        
        if opponent_total > 0:
            opponent_win_rate = opponent_wins / opponent_total
            # Beating strong opponents (>60% win rate) gives bonus
            # Losing to weak opponents (<40% win rate) gives penalty
            if is_win and opponent_win_rate > 0.6:
                base_value *= 1.3  # 30% bonus for quality win
            elif not is_win and opponent_win_rate < 0.4:
                base_value *= 1.3  # 30% extra penalty for bad loss
        
        momentum += base_value * weights[i]
    
    # Normalize by total weight
    normalized_score = momentum / total_weight if total_weight > 0 else 0
    
    # Calculate current streak
    streak = 0
    streak_type = None
    for match in reversed(match_history):
        if streak == 0:
            streak_type = "win" if match["is_win"] else "loss"
            streak = 1
        elif (match["is_win"] and streak_type == "win") or (not match["is_win"] and streak_type == "loss"):
            streak += 1
        else:
            break
    
    # Streak bonus/penalty
    if streak >= 3:
        streak_modifier = 0.2 * (streak - 2)  # +0.2 per match beyond 2
        if streak_type == "win":
            normalized_score += streak_modifier
        else:
            normalized_score -= streak_modifier
    
    # Determine label and trend
    if normalized_score >= 0.6:
        label = "🔥🔥"
        trend = "hot"
        description = "On fire!"
    elif normalized_score >= 0.3:
        label = "🔥"
        trend = "warming"
        description = "Building momentum"
    elif normalized_score >= -0.3:
        label = "➡️"
        trend = "neutral"
        description = "Steady"
    elif normalized_score >= -0.6:
        label = "❄️"
        trend = "cooling"
        description = "Struggling"
    else:
        label = "❄️❄️"
        trend = "cold"
        description = "Ice cold"
    
    return {
        "score": round(normalized_score, 2),
        "label": label,
        "description": description,
        "trend": trend,
        "streak": streak,
        "streak_type": streak_type
    }


def generate_tale_of_tape_html(
    east_name: str,
    west_name: str,
    east_data: Dict[str, object],
    west_data: Dict[str, object],
    east_momentum: Dict[str, object],
    west_momentum: Dict[str, object],
    h2h_summary: Dict[str, object] | None = None
) -> str:
    """Generate a visually rich 'Tale of the Tape' comparison card."""
    
    east_photo = east_data.get("photo_url", build_rikishi_avatar_url(east_name))
    west_photo = west_data.get("photo_url", build_rikishi_avatar_url(west_name))
    east_avatar = build_rikishi_avatar_url(east_name)
    west_avatar = build_rikishi_avatar_url(west_name)
    
    # Format stats with fallbacks
    def fmt(val, suffix=""):
        return f"{val}{suffix}" if val else "—"
    
    east_height = fmt(east_data.get("height"), " cm")
    west_height = fmt(west_data.get("height"), " cm")
    east_weight = fmt(east_data.get("weight"), " kg")
    west_weight = fmt(west_data.get("weight"), " kg")
    east_age = fmt(east_data.get("age"))
    west_age = fmt(west_data.get("age"))
    east_record = f"{east_data.get('wins', 0)}-{east_data.get('losses', 0)}"
    west_record = f"{west_data.get('wins', 0)}-{west_data.get('losses', 0)}"
    
    # H2H record
    h2h_text = "First meeting"
    if h2h_summary and h2h_summary.get("total", 0) > 0:
        h2h_text = h2h_summary.get("record", "")
    
    html = f'''
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #1a1a2e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 3px solid #c41e3a;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        font-family: 'Segoe UI', sans-serif;
        color: #f5f5f5;
        margin: 10px 0;
    ">
        <!-- Header -->
        <div style="text-align:center;margin-bottom:15px;">
            <div style="font-size:11px;color:#c41e3a;text-transform:uppercase;letter-spacing:3px;">Tale of the Tape</div>
            <div style="font-size:10px;color:#888;margin-top:4px;">{escape(str(h2h_text))}</div>
        </div>
        
        <!-- Main comparison -->
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <!-- East wrestler -->
            <div style="text-align:center;flex:1;">
                <img src="{escape(str(east_photo))}" 
                     onerror="this.src='{escape(str(east_avatar))}'"
                     style="width:80px;height:80px;border-radius:50%;border:3px solid #c41e3a;object-fit:cover;margin-bottom:8px;">
                <div style="font-size:18px;font-weight:bold;color:#f5ede0;">{escape(east_name)}</div>
                <div style="font-size:12px;color:#d4a574;">{escape(str(east_data.get('rank', '')))}</div>
                <div style="font-size:20px;margin-top:4px;">{east_momentum.get('label', '➡️')}</div>
                <div style="font-size:10px;color:#888;">{east_momentum.get('description', '')}</div>
            </div>
            
            <!-- VS divider -->
            <div style="text-align:center;padding:0 20px;">
                <div style="font-size:28px;font-weight:bold;color:#c41e3a;">VS</div>
            </div>
            
            <!-- West wrestler -->
            <div style="text-align:center;flex:1;">
                <img src="{escape(str(west_photo))}" 
                     onerror="this.src='{escape(str(west_avatar))}'"
                     style="width:80px;height:80px;border-radius:50%;border:3px solid #c41e3a;object-fit:cover;margin-bottom:8px;">
                <div style="font-size:18px;font-weight:bold;color:#f5ede0;">{escape(west_name)}</div>
                <div style="font-size:12px;color:#d4a574;">{escape(str(west_data.get('rank', '')))}</div>
                <div style="font-size:20px;margin-top:4px;">{west_momentum.get('label', '➡️')}</div>
                <div style="font-size:10px;color:#888;">{west_momentum.get('description', '')}</div>
            </div>
        </div>
        
        <!-- Stats comparison table -->
        <div style="margin-top:20px;border-top:1px solid #333;padding-top:15px;">
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
                <tr style="border-bottom:1px solid #333;">
                    <td style="padding:8px;text-align:right;color:#f5ede0;font-weight:bold;">{east_record}</td>
                    <td style="padding:8px;text-align:center;color:#888;width:100px;">Record</td>
                    <td style="padding:8px;text-align:left;color:#f5ede0;font-weight:bold;">{west_record}</td>
                </tr>
                <tr style="border-bottom:1px solid #333;">
                    <td style="padding:8px;text-align:right;color:#f5ede0;">{east_height}</td>
                    <td style="padding:8px;text-align:center;color:#888;">Height</td>
                    <td style="padding:8px;text-align:left;color:#f5ede0;">{west_height}</td>
                </tr>
                <tr style="border-bottom:1px solid #333;">
                    <td style="padding:8px;text-align:right;color:#f5ede0;">{east_weight}</td>
                    <td style="padding:8px;text-align:center;color:#888;">Weight</td>
                    <td style="padding:8px;text-align:left;color:#f5ede0;">{west_weight}</td>
                </tr>
                <tr style="border-bottom:1px solid #333;">
                    <td style="padding:8px;text-align:right;color:#f5ede0;">{east_age}</td>
                    <td style="padding:8px;text-align:center;color:#888;">Age</td>
                    <td style="padding:8px;text-align:left;color:#f5ede0;">{west_age}</td>
                </tr>
                <tr>
                    <td style="padding:8px;text-align:right;color:#f5ede0;">{east_momentum.get('streak', 0)} {east_momentum.get('streak_type', '') or ''}</td>
                    <td style="padding:8px;text-align:center;color:#888;">Streak</td>
                    <td style="padding:8px;text-align:left;color:#f5ede0;">{west_momentum.get('streak', 0)} {west_momentum.get('streak_type', '') or ''}</td>
                </tr>
            </table>
        </div>
    </div>
    '''
    return html


def render_upset_summary(upset_analysis: Dict[str, object]) -> None:
    """Render the upset analysis summary in Streamlit."""
    import streamlit.components.v1 as components
    
    upsets = upset_analysis.get("upsets", [])
    giant_killers = upset_analysis.get("giant_killers", [])
    vulnerable = upset_analysis.get("vulnerable", [])
    
    if not upsets:
        st.info("No significant upsets recorded yet this tournament.")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Upsets", upset_analysis.get("total_upsets", 0))
    with col2:
        biggest = upset_analysis.get("biggest_upset")
        if biggest:
            st.metric("Biggest Upset", f"{biggest['winner']} def. {biggest['loser']}")
    with col3:
        if giant_killers:
            top_killer = giant_killers[0]
            st.metric("Top Giant Killer", f"{top_killer[0]} ({top_killer[1]})")
    
    # Giant Killers & Vulnerable sections
    gk_col, vul_col = st.columns(2)
    
    with gk_col:
        st.markdown("**🗡️ Giant Killers**")
        if giant_killers:
            for name, count in giant_killers:
                st.caption(f"• {name}: {count} upset win{'s' if count > 1 else ''}")
        else:
            st.caption("None yet")
    
    with vul_col:
        st.markdown("**🎯 Vulnerable Favorites**")
        if vulnerable:
            for name, count in vulnerable:
                st.caption(f"• {name}: {count} upset loss{'es' if count > 1 else ''}")
        else:
            st.caption("None yet")
    
    # Recent upsets list
    st.markdown("**📋 Upset Log**")
    for upset in upsets[:10]:  # Show top 10
        st.markdown(
            f"{upset['label']} Day {upset['day']}: **{upset['winner']}** ({upset['winner_rank']}) "
            f"def. {upset['loser']} ({upset['loser_rank']}) via {upset['kimarite'] or 'TBD'}"
        )



    """Generate HTML bubbles with tooltips for match results."""
    
    if not match_history:
        return ""
    
    bubbles = []
    for match_info in match_history:
        is_win = match_info["is_win"]
        opponent = escape(str(match_info["opponent"]))
        day = match_info["day"]
        result = match_info["result"]
        
        tooltip = escape(f"Day {day}: {result} vs {opponent}")
        
        if is_win:
            # Filled green circle for win with tooltip
            bubbles.append(
                f'<span title="{tooltip}" style="cursor:pointer;color:#4CAF50;font-size:28px;">●</span>'
            )
        else:
            # Empty circle for loss with tooltip
            bubbles.append(
                f'<span title="{tooltip}" style="cursor:pointer;color:#666;font-size:28px;">○</span>'
            )
    
    return " ".join(bubbles)


def scoreboard_dataframe(
    records: Dict[str, Dict[str, object]], 
    match_history: Dict[str, List[Dict[str, object]]] | None = None,
    limit: int | None = 15
) -> pd.DataFrame:
    """Convert aggregated records into a dataframe with momentum indicators."""

    if not records:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for name, payload in records.items():
        wins = int(payload["wins"])
        losses = int(payload["losses"])
        rikishi_id = payload.get("id")
        
        # Get match history bubbles for this wrestler
        history = match_history.get(name, []) if match_history else []
        match_bubbles = generate_match_bubbles(history)
        
        # Calculate momentum score
        momentum = calculate_momentum_score(history, records)
        
        # Use real photo if we have an ID, fallback to avatar
        if rikishi_id:
            photo_url = build_rikishi_photo_url(int(rikishi_id))
        else:
            photo_url = build_rikishi_avatar_url(name)
        
        rows.append(
            {
                "Photo": photo_url,
                "Avatar": build_rikishi_avatar_url(name),  # Fallback
                "RikishiId": rikishi_id,
                "Shikona": name,
                "Rank": payload.get("rank") or "-",
                "Wins": wins,
                "Losses": losses,
                "Matches": match_bubbles,
                "Momentum": momentum["label"],
                "MomentumScore": momentum["score"],
                "MomentumDesc": momentum["description"],
                "Streak": momentum["streak"],
                "StreakType": momentum["streak_type"],
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["Wins", "Losses"], ascending=[False, True])
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def estimate_win_probability(
    east_name: str,
    west_name: str,
    summary: Dict[str, object],
    records: Dict[str, Dict[str, object]],
) -> Tuple[float, float, str]:
    """Return estimated win probabilities using simple heuristics."""

    def _win_pct(record: Dict[str, object] | None) -> float | None:
        if not record:
            return None
        wins = int(record.get("wins", 0))
        losses = int(record.get("losses", 0))
        total = wins + losses
        return wins / total if total else None

    east_form = _win_pct(records.get(east_name))
    west_form = _win_pct(records.get(west_name))

    base = 0.5
    if east_form is not None and west_form is not None:
        base += 0.2 * (east_form - west_form)

    total_series = summary.get("total", 0) or 0
    east_series = summary.get("east_wins", 0) or 0
    west_series = summary.get("west_wins", 0) or 0
    if total_series > 0:
        series_ratio = (east_series + 1) / (total_series + 2)
        base += 0.3 * (series_ratio - 0.5)

    base = max(0.1, min(0.9, base))
    east_prob = round(base, 3)
    west_prob = round(1 - base, 3)
    reason_parts = []
    if east_form is not None and west_form is not None:
        reason_parts.append(f"form diff {east_form:.2f}-{west_form:.2f}")
    if total_series:
        reason_parts.append(f"H2H {east_series}-{west_series}")
    rationale = "; ".join(reason_parts) or "insufficient data"
    return east_prob, west_prob, rationale


def build_rikishi_avatar_url(name: str) -> str:
    """Return a deterministic avatar URL for a given shikona (fallback)."""

    encoded = quote_plus(name)
    return (
        "https://ui-avatars.com/api/"
        f"?name={encoded}&background=EB3223&color=FFFFFF&size=64&rounded=true&bold=true"
    )


def build_rikishi_photo_url(rikishi_id: int) -> str:
    """Return the official photo URL from sumo-api for a rikishi."""
    return f"https://www.sumo-api.com/images/rikishi/{rikishi_id}.jpg"


def _division_from_rank(rank: str | None) -> str:
    """Map specific rank labels to their broader division."""

    if not rank:
        return "Unknown"
    token = rank.split()[0]
    if token in {"Yokozuna", "Ozeki", "Sekiwake", "Komusubi", "Maegashira"}:
        return "Makuuchi"
    if token in DIVISION_ORDER:
        return token
    return token


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned).date()
    except ValueError:
        return None


def _calculate_age(birth_date: date | None) -> int | None:
    if not birth_date:
        return None
    today = date.today()
    return (
        today.year
        - birth_date.year
        - ((today.month, today.day) < (birth_date.month, birth_date.day))
    )


def build_rikishi_dataframe(records: Sequence[Dict]) -> pd.DataFrame:
    """Transform API records into a tabular dataframe."""

    rows: List[Dict[str, object]] = []
    for record in records:
        rank = record.get("currentRank")
        division = _division_from_rank(rank)
        birth = _parse_iso_date(record.get("birthDate"))
        rows.append(
            {
                "ID": record.get("id"),
                "Shikona": record.get("shikonaEn") or "Unknown",
                "Rank": rank or "Unknown",
                "Division": division,
                "Heya": record.get("heya") or "Unknown",
                "Shusshin": record.get("shusshin") or "Unknown",
                "Height (cm)": record.get("height"),
                "Weight (kg)": record.get("weight"),
                "Age": _calculate_age(birth),
                "Birthdate": birth,
                "Division Order": DIVISION_ORDER.get(division, 99),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["Division Order", "Rank"], ascending=[True, True])
    return df


@st.cache_data(ttl=SUMO_API_CACHE_TTL)
def fetch_rikishi_directory(limit: int = 600, include_retired: bool = False) -> Dict:
    """Load rikishi records. Uses local cache when available."""
    
    # Determine cache file name
    if include_retired:
        cache_path = CACHE_DIR / "rikishi_directory_with_retired.json"
    else:
        cache_path = CACHE_DIR / "rikishi_directory.json"
    
    # Try cache first
    cached_data = _load_from_cache(cache_path)
    if cached_data:
        return cached_data
    
    # Fetch from API
    params = {"limit": limit, "skip": 0, "ranks": "true"}
    if include_retired:
        params["intai"] = "true"
    data = _sumo_api_get("/rikishis", params=params)
    
    # Cache the result
    if data:
        _save_to_cache(cache_path, data)
    
    return data


@st.cache_data(ttl=SUMO_API_CACHE_TTL)
def fetch_rikishi_stats(rikishi_id: int) -> Dict:
    """Fetch aggregated stats for a specific rikishi. Uses local cache when available."""
    cache_path = CACHE_DIR / f"rikishi_{rikishi_id}_stats.json"
    
    # Try cache first
    cached_data = _load_from_cache(cache_path)
    if cached_data:
        return cached_data
    
    # Fetch from API
    data = _sumo_api_get(f"/rikishi/{rikishi_id}/stats")
    
    # Cache the result
    if data:
        _save_to_cache(cache_path, data)
    
    return data


def summarize_head_to_head(
    east_id: int, west_id: int, east_name: str, west_name: str
) -> Dict[str, object | None]:
    """Return a lightweight record summary for a match-up."""

    ordered_pair = tuple(sorted((east_id, west_id)))
    payload = fetch_head_to_head_pair(ordered_pair)
    east_is_first = east_id == ordered_pair[0]
    east_wins = payload.get("rikishiWins", 0) if east_is_first else payload.get("opponentWins", 0)
    west_wins = payload.get("opponentWins", 0) if east_is_first else payload.get("rikishiWins", 0)
    total = payload.get("total", east_wins + west_wins)

    if total == 0:
        record = "First-time meeting"
    elif east_wins > west_wins:
        record = f"{east_name} leads {east_wins}-{west_wins}"
    elif west_wins > east_wins:
        record = f"{west_name} leads {west_wins}-{east_wins}"
    else:
        record = f"Series tied {east_wins}-{west_wins}"

    matches = payload.get("matches") or []
    last_result = None
    if matches:
        latest = matches[0]
        last_result = (
            f"Last meeting {latest.get('bashoId')} Day {latest.get('day')}: "
            f"{latest.get('winnerEn')} via {latest.get('kimarite')}"
        )

    recent = []
    for match in matches[:5]:
        recent.append(
            f"{match.get('bashoId')} D{match.get('day')}: "
            f"{match.get('winnerEn')} ({match.get('kimarite')})"
        )

    if east_is_first:
        east_kimarite = payload.get("kimariteWins", {}) or {}
        west_kimarite = payload.get("kimariteLosses", {}) or {}
    else:
        east_kimarite = payload.get("kimariteLosses", {}) or {}
        west_kimarite = payload.get("kimariteWins", {}) or {}

    technique_rows: List[Dict[str, object]] = []
    for technique, count in east_kimarite.items():
        technique_rows.append(
            {"Rikishi": east_name, "Technique": technique, "Wins": count}
        )
    for technique, count in west_kimarite.items():
        technique_rows.append(
            {"Rikishi": west_name, "Technique": technique, "Wins": count}
        )
    technique_rows = sorted(technique_rows, key=lambda row: row["Wins"], reverse=True)[:10]

    return {
        "total": total,
        "record": record,
        "last_result": last_result,
        "east_wins": east_wins,
        "west_wins": west_wins,
        "recent_results": recent,
        "technique_breakdown": technique_rows,
    }


def init_session_state() -> None:
    """Seed Streamlit session state used across tabs."""

    defaults = {
        "favorite_rikishi": [],
        "daily_notes": {},
        "pick_log": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, dict) else list(value)


def toggle_favorite(shikona: str, add: bool) -> None:
    """Add or remove a rikishi from the favorites list."""

    favorites: List[str] = st.session_state["favorite_rikishi"]
    if add and shikona not in favorites:
        favorites.append(shikona)
    elif not add and shikona in favorites:
        favorites.remove(shikona)


def record_pick(tournament_id: str, day: int, east: str, west: str, pick: str | None) -> None:
    """Persist the user's picks for a bout."""

    bout_id = f"{tournament_id}-day{day}:{east}vs{west}"
    if pick:
        st.session_state["pick_log"][bout_id] = pick
    elif bout_id in st.session_state["pick_log"]:
        st.session_state["pick_log"].pop(bout_id)


def save_daily_note(tournament_id: str, day: int, note: str) -> None:
    """Store a short note keyed by tournament/day."""

    key = f"{tournament_id}-day{day}"
    if note.strip():
        st.session_state["daily_notes"][key] = note.strip()
    elif key in st.session_state["daily_notes"]:
        st.session_state["daily_notes"].pop(key)


# ==================== Streamlit Layout ====================


def get_next_tournament(tournaments: Sequence[Tournament]) -> Tournament | None:
    """Find the next upcoming tournament."""
    today = date.today()
    for t in tournaments:
        if t.start_date > today:
            return t
        if t.start_date <= today <= t.end_date:
            return t  # Currently active
    return None


def get_tournament_status(tournament: Tournament) -> Dict[str, str]:
    """Determine tournament status and styling."""
    today = date.today()
    if today < tournament.start_date:
        days_until = (tournament.start_date - today).days
        return {
            "status": "upcoming",
            "label": f"In {days_until} days",
            "color": "#6c757d",
            "bg": "rgba(108, 117, 125, 0.1)",
            "border": "#6c757d",
            "icon": "🔜"
        }
    elif today > tournament.end_date:
        return {
            "status": "completed",
            "label": "Completed",
            "color": "#28a745",
            "bg": "rgba(40, 167, 69, 0.1)",
            "border": "#28a745",
            "icon": "✅"
        }
    else:
        day_num = (today - tournament.start_date).days + 1
        return {
            "status": "live",
            "label": f"Day {day_num} LIVE",
            "color": "#c41e3a",
            "bg": "rgba(196, 30, 58, 0.15)",
            "border": "#c41e3a",
            "icon": "🔴"
        }


def generate_ical_event(tournament: Tournament) -> str:
    """Generate an iCal event string for a tournament."""
    start_str = tournament.start_date.strftime("%Y%m%d")
    end_str = tournament.end_date.strftime("%Y%m%d")
    uid = f"{tournament.basho_id}@sumo-companion"
    
    return f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Sumo Companion//EN
BEGIN:VEVENT
UID:{uid}
DTSTART;VALUE=DATE:{start_str}
DTEND;VALUE=DATE:{end_str}
SUMMARY:{tournament.name}
DESCRIPTION:{tournament.highlight or 'Grand Sumo Tournament'}
LOCATION:{tournament.venue}, {tournament.city}, Japan
END:VEVENT
END:VCALENDAR"""


def render_countdown_banner(tournament: Tournament) -> None:
    """Render a countdown banner for the next tournament."""
    import streamlit.components.v1 as components
    
    today = date.today()
    status = get_tournament_status(tournament)
    
    if status["status"] == "live":
        day_num = (today - tournament.start_date).days + 1
        days_remaining = 15 - day_num
        countdown_html = f'''
        <div style="text-align:center;">
            <div style="font-size:14px;color:#d4a574;text-transform:uppercase;letter-spacing:2px;">Now Live</div>
            <div style="font-size:48px;font-weight:bold;color:#f5ede0;margin:8px 0;">Day {day_num} of 15</div>
            <div style="font-size:16px;color:#d4a574;">{days_remaining} days remaining</div>
        </div>
        '''
    else:
        days_until = (tournament.start_date - today).days
        countdown_html = f'''
        <div style="text-align:center;">
            <div style="font-size:14px;color:#d4a574;text-transform:uppercase;letter-spacing:2px;">Next Tournament</div>
            <div style="font-size:64px;font-weight:bold;color:#f5ede0;margin:8px 0;">{days_until}</div>
            <div style="font-size:18px;color:#d4a574;">days until {escape(tournament.name)}</div>
        </div>
        '''
    
    banner_html = f'''
    <div style="
        background: linear-gradient(135deg, #2d1b1b 0%, #4a2c2c 50%, #1a1a2e 100%);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        border: 2px solid #c41e3a;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    ">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:20px;">
            <div style="flex:1;min-width:200px;">
                <div style="font-size:28px;font-weight:bold;color:#f5ede0;">🏯 {escape(tournament.name)}</div>
                <div style="font-size:14px;color:#d4a574;margin-top:6px;">
                    📍 {escape(tournament.city)} • {escape(tournament.venue)}
                </div>
                <div style="font-size:13px;color:#a0a0a0;margin-top:4px;">
                    📅 {tournament.start_date.strftime('%B %d')} - {tournament.end_date.strftime('%B %d, %Y')}
                </div>
            </div>
            <div style="flex:1;min-width:200px;">
                {countdown_html}
            </div>
        </div>
    </div>
    '''
    components.html(banner_html, height=160)


def render_tournament_card(tournament: Tournament, champion_id: int | None = None) -> str:
    """Generate HTML for a single tournament card."""
    status = get_tournament_status(tournament)
    
    # Champion photo or placeholder
    if champion_id:
        champion_photo = build_rikishi_photo_url(champion_id)
        champion_avatar = build_rikishi_avatar_url(tournament.champion or "TBD")
    else:
        champion_photo = build_rikishi_avatar_url(tournament.champion or "TBD")
        champion_avatar = champion_photo
    
    champion_name = escape(tournament.champion or "TBD")
    champion_section = f'''
        <div style="display:flex;align-items:center;gap:10px;margin-top:12px;padding-top:12px;border-top:1px solid rgba(212,165,116,0.3);">
            <img src="{escape(champion_photo)}" 
                 onerror="this.src='{escape(champion_avatar)}'"
                 style="width:40px;height:40px;border-radius:50%;border:2px solid #c41e3a;object-fit:cover;">
            <div>
                <div style="font-size:10px;color:#888;text-transform:uppercase;">Champion</div>
                <div style="font-size:14px;font-weight:bold;color:#2d1b1b;">{champion_name}</div>
            </div>
        </div>
    ''' if status["status"] == "completed" else '''
        <div style="margin-top:12px;padding-top:12px;border-top:1px solid rgba(212,165,116,0.3);">
            <div style="font-size:11px;color:#888;font-style:italic;">Champion to be determined</div>
        </div>
    '''
    
    # City emoji mapping
    city_emoji = {
        "Tokyo": "🗼",
        "Osaka": "🏯",
        "Nagoya": "🌸",
        "Fukuoka": "🌊"
    }.get(tournament.city, "📍")
    
    return f'''
    <div style="
        background: {status['bg']};
        border: 2px solid {status['border']};
        border-radius: 12px;
        padding: 16px;
        height: 100%;
        box-sizing: border-box;
        transition: transform 0.2s, box-shadow 0.2s;
    ">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div style="font-size:18px;font-weight:bold;color:#2d1b1b;">{escape(tournament.name)}</div>
            <div style="
                background:{status['color']};
                color:white;
                padding:3px 8px;
                border-radius:10px;
                font-size:10px;
                font-weight:bold;
            ">{status['icon']} {status['label']}</div>
        </div>
        
        <div style="margin-top:10px;font-size:13px;color:#555;">
            <div>{city_emoji} {escape(tournament.city)}</div>
            <div style="margin-top:4px;color:#777;font-size:12px;">{escape(tournament.venue)}</div>
        </div>
        
        <div style="margin-top:10px;font-size:12px;color:#666;">
            📅 {tournament.start_date.strftime('%b %d')} - {tournament.end_date.strftime('%b %d')}
        </div>
        
        {champion_section}
    </div>
    '''


@st.cache_data(ttl=SUMO_API_CACHE_TTL)
def fetch_champion_ids(champion_names: List[str]) -> Dict[str, int]:
    """Look up rikishi IDs for champion names."""
    try:
        directory = fetch_rikishi_directory(limit=800, include_retired=True)
        records = directory.get("records") or []
        name_to_id = {}
        for record in records:
            shikona = record.get("shikonaEn")
            if shikona and shikona in champion_names:
                name_to_id[shikona] = record.get("id")
        return name_to_id
    except Exception:
        return {}


def render_calendar_tab(tournaments: Sequence[Tournament]) -> None:
    """Render the redesigned honbasho calendar with cards, countdown, and champion showcase."""
    import streamlit.components.v1 as components
    
    st.subheader("🗓️ Tournament Calendar")
    
    # === COUNTDOWN BANNER ===
    next_tournament = get_next_tournament(tournaments)
    if next_tournament:
        render_countdown_banner(next_tournament)
    
    # === TOURNAMENT CARDS GRID ===
    st.markdown("### All Tournaments")
    
    # Fetch champion IDs for photos
    champion_names = [t.champion for t in tournaments if t.champion]
    champion_ids = fetch_champion_ids(champion_names)
    
    # Create 3-column grid
    cards_html = ""
    for i, tournament in enumerate(tournaments):
        champion_id = champion_ids.get(tournament.champion) if tournament.champion else None
        card_html = render_tournament_card(tournament, champion_id)
        cards_html += f'<div style="flex:1;min-width:280px;max-width:400px;">{card_html}</div>'
    
    grid_html = f'''
    <div style="
        display:flex;
        flex-wrap:wrap;
        gap:16px;
        justify-content:center;
    ">
        {cards_html}
    </div>
    '''
    components.html(grid_html, height=520, scrolling=True)
    
    # === CHAMPION SHOWCASE ===
    st.markdown("### 🏆 Champion Showcase")
    
    completed_tournaments = [t for t in tournaments if get_tournament_status(t)["status"] == "completed"]
    
    if completed_tournaments:
        showcase_html = ""
        for tournament in completed_tournaments:
            if not tournament.champion:
                continue
            champion_id = champion_ids.get(tournament.champion)
            if champion_id:
                photo_url = build_rikishi_photo_url(champion_id)
            else:
                photo_url = build_rikishi_avatar_url(tournament.champion)
            avatar_url = build_rikishi_avatar_url(tournament.champion)
            
            showcase_html += f'''
            <div style="
                text-align:center;
                padding:16px;
                background:rgba(255,255,255,0.9);
                border-radius:12px;
                border:1px solid #d4a574;
                min-width:140px;
            ">
                <img src="{escape(photo_url)}" 
                     onerror="this.src='{escape(avatar_url)}'"
                     style="width:80px;height:80px;border-radius:50%;border:3px solid #c41e3a;object-fit:cover;margin-bottom:8px;">
                <div style="font-size:16px;font-weight:bold;color:#2d1b1b;">{escape(tournament.champion)}</div>
                <div style="font-size:12px;color:#666;margin-top:4px;">{escape(tournament.name)}</div>
                <div style="font-size:11px;color:#888;">{tournament.month}</div>
            </div>
            '''
        
        if showcase_html:
            components.html(f'''
            <div style="
                display:flex;
                flex-wrap:wrap;
                gap:16px;
                justify-content:center;
                padding:10px 0;
            ">
                {showcase_html}
            </div>
            ''', height=200)
    else:
        st.info("No completed tournaments yet this season.")
    
    # === CALENDAR EXPORT ===
    st.markdown("### 📥 Export to Calendar")
    
    export_cols = st.columns([2, 1, 1])
    with export_cols[0]:
        export_choice = st.selectbox(
            "Choose tournament to export",
            ["All tournaments"] + [t.name for t in tournaments],
            key="calendar_export_choice"
        )
    
    with export_cols[1]:
        if st.button("📅 Download .ics", use_container_width=True):
            if export_choice == "All tournaments":
                # Combine all tournaments into one ical
                events = ""
                for t in tournaments:
                    start_str = t.start_date.strftime("%Y%m%d")
                    end_str = t.end_date.strftime("%Y%m%d")
                    uid = f"{t.basho_id}@sumo-companion"
                    events += f"""BEGIN:VEVENT
UID:{uid}
DTSTART;VALUE=DATE:{start_str}
DTEND;VALUE=DATE:{end_str}
SUMMARY:{t.name}
DESCRIPTION:{t.highlight or 'Grand Sumo Tournament'}
LOCATION:{t.venue}, {t.city}, Japan
END:VEVENT
"""
                ical_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Sumo Companion//EN
{events}END:VCALENDAR"""
                filename = f"sumo_calendar_{tournaments[0].start_date.year}.ics"
            else:
                selected = next(t for t in tournaments if t.name == export_choice)
                ical_content = generate_ical_event(selected)
                filename = f"{selected.basho_id}_{selected.name.replace(' ', '_')}.ics"
            
            st.download_button(
                label="⬇️ Save File",
                data=ical_content,
                file_name=filename,
                mime="text/calendar",
                key="ics_download"
            )
    
    with export_cols[2]:
        st.caption("Import into Google Calendar, Outlook, or Apple Calendar")
    
    # === SEASON OVERVIEW (collapsible) ===
    with st.expander("📊 Season Overview & Storylines"):
        for tournament in tournaments:
            status = get_tournament_status(tournament)
            status_badge = f'<span style="background:{status["color"]};color:white;padding:2px 6px;border-radius:8px;font-size:10px;">{status["label"]}</span>'
            
            highlight = tournament.highlight or "A key stop on the grand sumo circuit."
            champion_text = f"**Champion:** {tournament.champion}" if tournament.champion else "*Champion TBD*"
            
            st.markdown(
                f"**{tournament.name}** ({tournament.city}) {status_badge}",
                unsafe_allow_html=True
            )
            st.caption(f"{highlight}")
            st.caption(champion_text)
            st.markdown("---")


def render_rikishi_tab() -> None:
    """Allow the user to explore the banzuke and set favorites."""

    st.subheader("Banzuke Spotlight")
    include_retired = st.checkbox("Include retired rikishi", value=False)
    try:
        directory = fetch_rikishi_directory(include_retired=include_retired)
        records = directory.get("records") or []
    except RuntimeError as exc:
        st.error(f"Unable to load rikishi list: {exc}")
        return

    rikishi_df = build_rikishi_dataframe(records)
    if rikishi_df.empty:
        st.info("No rikishi records available from the API right now.")
        return

    filter_cols = st.columns(3)
    with filter_cols[0]:
        division_options = ["All"] + sorted(
            rikishi_df["Division"].unique(), key=lambda d: DIVISION_ORDER.get(d, 99)
        )
        default_division = division_options.index("Makuuchi") if "Makuuchi" in division_options else 0
        selected_division = st.selectbox("Division", division_options, index=default_division)
    with filter_cols[1]:
        heya_options = sorted(rikishi_df["Heya"].unique())
        selected_heya = st.multiselect("Heya", heya_options)
    with filter_cols[2]:
        search_query = st.text_input("Search", placeholder="Shikona or hometown").strip()

    filtered_df = rikishi_df.copy()
    if selected_division != "All":
        filtered_df = filtered_df[filtered_df["Division"] == selected_division]
    if selected_heya:
        filtered_df = filtered_df[filtered_df["Heya"].isin(selected_heya)]
    if search_query:
        filtered_df = filtered_df[
            filtered_df["Shikona"].str.contains(search_query, case=False, na=False)
            | filtered_df["Shusshin"].str.contains(search_query, case=False, na=False)
        ]

    st.caption(f"Showing {len(filtered_df)} of {len(rikishi_df)} rikishi")
    display_cols = [
        "Shikona",
        "Rank",
        "Division",
        "Heya",
        "Shusshin",
        "Age",
        "Height (cm)",
        "Weight (kg)",
    ]
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    if filtered_df.empty:
        st.warning("Adjust the filters to see rikishi.")
        return

    st.markdown("### Rikishi Details")
    detail_names = filtered_df["Shikona"].tolist()
    selected_name = st.selectbox("Choose a rikishi", detail_names)
    detail_row = filtered_df[filtered_df["Shikona"] == selected_name].iloc[0]
    st.markdown(f"**{selected_name}** - {detail_row['Rank']} ({detail_row['Heya']})")
    stats = {}
    try:
        stats = fetch_rikishi_stats(int(detail_row["ID"]))
    except RuntimeError as exc:
        st.warning(f"Could not load stats: {exc}")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Total matches", stats.get("totalMatches", "-"))
    metric_cols[1].metric("Career wins", stats.get("totalWins", "-"))
    metric_cols[2].metric("Yusho", stats.get("yusho", 0))

    wins_by_division = stats.get("winsByDivision") or {}
    if wins_by_division:
        chart_df = pd.DataFrame(
            [{"Division": division, "Wins": wins} for division, wins in wins_by_division.items()]
        )
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(x="Division", y="Wins")
            .properties(height=180)
        )
        st.altair_chart(chart, use_container_width=True)

    sansho = stats.get("sansho") or {}
    if sansho:
        st.caption(
            "Sansho: "
            + ", ".join(f"{name} ({count})" for name, count in sansho.items())
        )

    st.markdown("### Matchup Lab")
    if len(detail_names) < 2:
        st.info("Adjust filters to load at least two rikishi for comparisons.")
    else:
        matchup_cols = st.columns(2)
        with matchup_cols[0]:
            contender_a = st.selectbox("Rikishi A", detail_names, key="matchup_a")
        with matchup_cols[1]:
            contender_b = st.selectbox(
                "Rikishi B", detail_names, index=1 if len(detail_names) > 1 else 0, key="matchup_b"
            )
        if contender_a == contender_b:
            st.warning("Select two different rikishi to run the matchup analysis.")
        else:
            row_a = filtered_df[filtered_df["Shikona"] == contender_a].iloc[0]
            row_b = filtered_df[filtered_df["Shikona"] == contender_b].iloc[0]
            try:
                summary = summarize_head_to_head(
                    int(row_a["ID"]),
                    int(row_b["ID"]),
                    contender_a,
                    contender_b,
                )
                st.caption(summary["record"])
                if summary["last_result"]:
                    st.caption(summary["last_result"])
                recent_results = summary.get("recent_results") or []
                if recent_results:
                    st.markdown("Recent meetings:")
                    for result in recent_results:
                        st.caption(f"- {result}")
                technique_rows = summary.get("technique_breakdown") or []
                if technique_rows:
                    technique_df = pd.DataFrame(technique_rows)
                    chart = (
                        alt.Chart(technique_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Wins:Q", title="Wins"),
                            y=alt.Y("Technique:N", sort="-x"),
                            color=alt.Color("Rikishi:N", legend=alt.Legend(title="Winner")),
                        )
                        .properties(height=150)
                    )
                    st.altair_chart(chart, use_container_width=True)
            except RuntimeError as exc:
                st.error(f"Head-to-head data unavailable: {exc}")

            stats_a = fetch_rikishi_stats(int(row_a["ID"]))
            stats_b = fetch_rikishi_stats(int(row_b["ID"]))
            stat_cols = st.columns(2)
            for col, name, stats_payload in zip(stat_cols, (contender_a, contender_b), (stats_a, stats_b)):
                win_pct = (
                    stats_payload.get("totalWins", 0) / stats_payload.get("totalMatches", 1)
                    if stats_payload.get("totalMatches")
                    else 0
                )
                col.metric(
                    f"{name} win rate",
                    f"{win_pct * 100:.1f}% ({stats_payload.get('totalWins', 0)}-{stats_payload.get('totalLosses', 0)})",
                )
                sansho = stats_payload.get("sansho") or {}
                if sansho:
                    col.caption(
                        "Sansho: " + ", ".join(f"{title} ({count})" for title, count in sansho.items())
                    )

    st.markdown("### Favorites")
    favorite_container = st.columns(2)
    with favorite_container[0]:
        choice = st.selectbox("Add a favorite", sorted(rikishi_df["Shikona"].tolist()))
        if st.button("Save favorite", use_container_width=True):
            toggle_favorite(choice, add=True)
    with favorite_container[1]:
        remove_choice = st.selectbox(
            "Remove favorite", ["(none)"] + st.session_state["favorite_rikishi"]
        )
        if st.button("Remove selected", use_container_width=True) and remove_choice != "(none)":
            toggle_favorite(remove_choice, add=False)

    if st.session_state["favorite_rikishi"]:
        st.success("Current favorites: " + ", ".join(st.session_state["favorite_rikishi"]))
    else:
        st.info("You have not saved any favorites yet.")


def render_tournament_banner(
    tournament: Tournament,
    latest_day: int,
    records: Dict[str, Dict[str, object]]
) -> None:
    """Render a live tournament progress banner with yusho race leaders."""
    
    days_remaining = 15 - latest_day
    
    # Get top 5 leaders by wins
    leaders = sorted(
        [(name, data) for name, data in records.items()],
        key=lambda x: (int(x[1].get("wins", 0)), -int(x[1].get("losses", 0))),
        reverse=True
    )[:5]
    
    # Determine tournament status
    today = date.today()
    if today < tournament.start_date:
        status = "🔜 Upcoming"
        status_color = "#6c757d"
    elif today > tournament.end_date:
        status = "✅ Completed"
        status_color = "#28a745"
    else:
        status = "🔴 LIVE"
        status_color = "#c41e3a"
    
    # Build leader photos HTML
    leader_html_parts = []
    for name, data in leaders:
        wins = int(data.get("wins", 0))
        losses = int(data.get("losses", 0))
        # Try to get rikishi ID from records if available
        rikishi_id = data.get("id")
        if rikishi_id:
            photo_url = build_rikishi_photo_url(int(rikishi_id))
        else:
            photo_url = build_rikishi_avatar_url(name)
        
        leader_html_parts.append(f'''
            <div style="text-align:center;margin:0 8px;">
                <img src="{escape(photo_url)}" 
                     onerror="this.src='{escape(build_rikishi_avatar_url(name))}'"
                     style="width:50px;height:50px;border-radius:50%;border:2px solid #c41e3a;object-fit:cover;">
                <div style="font-size:11px;font-weight:bold;color:#2d1b1b;margin-top:4px;">{escape(name)}</div>
                <div style="font-size:10px;color:#666;">{wins}-{losses}</div>
            </div>
        ''')
    
    leaders_html = "".join(leader_html_parts) if leader_html_parts else "<span style='color:#666;'>No results yet</span>"
    
    banner_html = f'''
    <div style="
        background: linear-gradient(135deg, #2d1b1b 0%, #4a2c2c 50%, #1a1a2e 100%);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        border: 2px solid #c41e3a;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    ">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <!-- Tournament Info -->
            <div style="flex:1;min-width:200px;">
                <div style="font-size:22px;font-weight:bold;color:#f5ede0;">🏯 {escape(tournament.name)}</div>
                <div style="font-size:13px;color:#d4a574;margin-top:4px;">
                    {escape(tournament.city)} • {escape(tournament.venue)}
                </div>
            </div>
            
            <!-- Status & Day -->
            <div style="text-align:center;">
                <div style="
                    background:{status_color};
                    color:white;
                    padding:4px 12px;
                    border-radius:12px;
                    font-size:12px;
                    font-weight:bold;
                    display:inline-block;
                ">{status}</div>
                <div style="font-size:28px;font-weight:bold;color:#f5ede0;margin-top:4px;">Day {latest_day}/15</div>
                <div style="font-size:11px;color:#d4a574;">{days_remaining} days remaining</div>
            </div>
            
            <!-- Yusho Race Leaders -->
            <div style="flex:1;min-width:280px;">
                <div style="font-size:11px;color:#d4a574;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                    🏆 Yusho Race Leaders
                </div>
                <div style="display:flex;justify-content:flex-end;align-items:flex-start;">
                    {leaders_html}
                </div>
            </div>
        </div>
    </div>
    '''
    
    import streamlit.components.v1 as components
    components.html(banner_html, height=140)


def render_tracker_tab(tournaments: Sequence[Tournament]) -> None:
    """Daily picks and journaling for the active basho."""

    st.subheader("Daily Tracker")
    default_index = determine_default_tournament_index(tournaments)
    current = st.selectbox(
        "Select a tournament",
        [t.name for t in tournaments],
        index=default_index,
    )
    chosen = next(t for t in tournaments if t.name == current)
    completed_days = fetch_completed_torikumi(chosen.basho_id, DEFAULT_DIVISION, tournaments)
    latest_completed_day = max(completed_days.keys()) if completed_days else 0
    
    # Update head-to-head database with completed matches
    if completed_days:
        all_matches = []
        for day_matches in completed_days.values():
            all_matches.extend(day_matches)
        update_head_to_head_from_matches(all_matches)
    scoreboard_records = compute_rikishi_records(completed_days)
    match_history = build_match_history(completed_days)
    
    # === LIVE TOURNAMENT PROGRESS BANNER ===
    render_tournament_banner(chosen, latest_completed_day, scoreboard_records)
    
    scoreboard = scoreboard_dataframe(scoreboard_records, match_history=match_history, limit=15)
    if not scoreboard.empty:
        st.markdown("### Scoreboard (Top Performers)")
        
        # Create custom HTML table with tooltips for match bubbles
        html_rows = []
        for _, row in scoreboard.iterrows():
            matches_html = row["Matches"]
            photo_url = escape(str(row['Photo']))
            avatar_url = escape(str(row['Avatar']))
            shikona = escape(str(row['Shikona']))
            rank = escape(str(row['Rank']))
            momentum = row.get("Momentum", "➡️")
            momentum_desc = escape(str(row.get("MomentumDesc", "")))
            html_rows.append(
                f'<tr>'
                f'<td style="padding:8px;"><img src="{photo_url}" onerror="this.src=\'{avatar_url}\'" width="50" height="50" style="border-radius:50%;object-fit:cover;border:2px solid #c41e3a;"></td>'
                f'<td style="padding:8px;"><strong>{shikona}</strong></td>'
                f'<td style="padding:8px;">{rank}</td>'
                f'<td style="padding:8px;text-align:center;">{row["Wins"]}</td>'
                f'<td style="padding:8px;text-align:center;">{row["Losses"]}</td>'
                f'<td style="padding:8px;text-align:center;" title="{momentum_desc}">{momentum}</td>'
                f'<td style="padding:8px;">{matches_html}</td>'
                f'</tr>'
            )
        
        html_table = (
            '<div style="overflow-x:auto;">'
            '<table style="width:100%;border-collapse:collapse;border:1px solid #d4a574;background:rgba(255,255,255,0.9);">'
            '<thead>'
            '<tr style="background:linear-gradient(135deg,#2d1b1b,#4a2c2c);color:#f5ede0;">'
            '<th style="padding:10px;text-align:left;">Photo</th>'
            '<th style="padding:10px;text-align:left;">Shikona</th>'
            '<th style="padding:10px;text-align:left;">Rank</th>'
            '<th style="padding:10px;text-align:center;">Wins</th>'
            '<th style="padding:10px;text-align:center;">Losses</th>'
            '<th style="padding:10px;text-align:center;">Form</th>'
            '<th style="padding:10px;text-align:left;">Match Results</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            + ''.join(html_rows) +
            '</tbody>'
            '</table>'
            '</div>'
        )
        try:
            import streamlit.components.v1 as components
            components.html(html_table, height=600, scrolling=True)
        except ImportError:
            st.markdown(html_table, unsafe_allow_html=True)
        top_rows = scoreboard.head(min(3, len(scoreboard)))
        columns = st.columns(len(top_rows))
        for col, (_, row) in zip(columns, top_rows.iterrows()):
            record = f"{row['Wins']}-{row['Losses']}"
            col.metric(label=row["Shikona"], value=record)
    else:
        st.info("Scoreboard will appear once winners are recorded.")

    default_slider_value = latest_completed_day or 1
    day = st.slider("Completed day", min_value=1, max_value=15, value=default_slider_value)
    st.caption(
        f"{chosen.name} in {chosen.city} - {chosen.start_date.strftime('%b %d')} to "
        f"{chosen.end_date.strftime('%b %d')} at {chosen.venue} (Basho ID {chosen.basho_id})"
    )
    st.caption("Sumo-API data is cached for 24 hours to respect the free tier.")

    try:
        overview = fetch_basho_overview(chosen.basho_id, tournaments)
    except RuntimeError as exc:
        st.warning(f"Could not load tournament overview: {exc}")
        overview = None

    if overview:
        st.markdown("### Tournament Results Snapshot")
        yusho_entries = overview.get("yusho") or []
        makuuchi_champion = next(
            (entry for entry in yusho_entries if entry.get("type") == DEFAULT_DIVISION), None
        )
        if makuuchi_champion:
            st.success(f"{DEFAULT_DIVISION} yusho: {makuuchi_champion.get('shikonaEn')}")
        else:
            st.info("Champion not announced yet.")

        special_prizes = overview.get("specialPrizes") or []
        if special_prizes:
            st.caption(
                "Special prizes: "
                + ", ".join(
                    f"{prize.get('type')}: {prize.get('shikonaEn')}" for prize in special_prizes
                )
            )

    if completed_days:
        st.markdown(f"### Results through Day {latest_completed_day}")
        st.dataframe(
            torikumi_results_dataframe(completed_days),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No completed match results are available for this basho yet.")

    st.markdown(f"### Day {day} Results")
    if day in completed_days:
        day_results = completed_days[day]
    else:
        try:
            day_payload = fetch_torikumi_payload(chosen.basho_id, DEFAULT_DIVISION, day, tournaments)
            day_results = day_payload.get("torikumi") or []
        except RuntimeError as exc:
            st.error(f"Unable to load results for day {day}: {exc}")
            day_results = []

    if day_results:
        st.dataframe(torikumi_dataframe(day_results), use_container_width=True, hide_index=True)
    else:
        st.info("No results published for this day yet.")

    next_day = day + 1
    if next_day <= 15:
        st.markdown(f"### Day {next_day} Matches & Head-to-heads")
        try:
            next_payload = fetch_torikumi_payload(chosen.basho_id, DEFAULT_DIVISION, next_day, tournaments)
            upcoming = next_payload.get("torikumi") or []
        except RuntimeError as exc:
            st.warning(f"Unable to load day {next_day} torikumi: {exc}")
            upcoming = []

        if not upcoming:
            st.info("No torikumi posted for the next day yet.")
        else:
            for match in upcoming:
                east_name = match.get("eastShikona")
                west_name = match.get("westShikona")
                east_rank = match.get("eastRank")
                west_rank = match.get("westRank")
                
                # Get prediction from local database
                h2h_pred = get_head_to_head_prediction(east_name, west_name)
                
                # Get current tournament records
                east_record = scoreboard_records.get(east_name, {})
                west_record = scoreboard_records.get(west_name, {})
                east_wins = east_record.get("wins", 0)
                east_losses = east_record.get("losses", 0)
                west_wins = west_record.get("wins", 0)
                west_losses = west_record.get("losses", 0)
                
                # Get recent form (last 5 matches)
                east_history = match_history.get(east_name, [])[-5:] if match_history.get(east_name) else []
                west_history = match_history.get(west_name, [])[-5:] if match_history.get(west_name) else []
                
                # Calculate combined confidence
                h2h_confidence = h2h_pred["confidence"]
                
                # Determine predicted winner with icon
                if h2h_pred["prediction"] == east_name:
                    predicted_winner = east_name
                    confidence = h2h_confidence
                    winner_side = "east"
                elif h2h_pred["prediction"] == west_name:
                    predicted_winner = west_name
                    confidence = h2h_confidence
                    winner_side = "west"
                else:
                    predicted_winner = "Even matchup"
                    confidence = 0.5
                    winner_side = "none"
                
                # Create expander for each match
                with st.expander(
                    f"**Bout {match.get('matchNo')}: {east_name} vs {west_name}**" +
                    (f" → Predicted: {predicted_winner} ({confidence*100:.0f}%)" if winner_side != "none" else " → Even matchup"),
                    expanded=False
                ):
                    # Visual prediction gauge
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        st.markdown(f"**{east_name}**")
                        st.caption(f"{east_rank}")
                        if east_wins + east_losses > 0:
                            st.caption(f"Record: {east_wins}-{east_losses}")
                    with col2:
                        # Confidence bar
                        if winner_side == "east":
                            bar_value = confidence
                            st.progress(bar_value, text=f"← {confidence*100:.0f}% favored")
                        elif winner_side == "west":
                            bar_value = 1 - confidence
                            st.progress(bar_value, text=f"{confidence*100:.0f}% favored →")
                        else:
                            st.progress(0.5, text="Even odds (50%)")
                    with col3:
                        st.markdown(f"**{west_name}**")
                        st.caption(f"{west_rank}")
                        if west_wins + west_losses > 0:
                            st.caption(f"Record: {west_wins}-{west_losses}")
                    
                    st.divider()
                    
                    # Head-to-head stats
                    if h2h_pred["total_matches"] > 0:
                        st.markdown("**📊 Head-to-Head Record**")
                        h2h_col1, h2h_col2 = st.columns(2)
                        with h2h_col1:
                            st.metric(
                                east_name,
                                f"{h2h_pred['wrestler1_wins']} wins",
                                f"{h2h_pred['wrestler1_win_pct']*100:.0f}%"
                            )
                        with h2h_col2:
                            st.metric(
                                west_name,
                                f"{h2h_pred['wrestler2_wins']} wins",
                                f"{h2h_pred['wrestler2_win_pct']*100:.0f}%"
                            )
                        st.caption(f"Total meetings: {h2h_pred['total_matches']}")
                        if h2h_pred.get("last_winner"):
                            st.caption(f"Last winner: {h2h_pred['last_winner']}")
                    else:
                        st.info("**First-time meeting** - No historical head-to-head data")
                    
                    # Recent form
                    if east_history or west_history:
                        st.markdown("**📈 Recent Form (Last 5 Matches)**")
                        form_col1, form_col2 = st.columns(2)
                        with form_col1:
                            if east_history:
                                form_bubbles = " ".join(["🟢" if m["is_win"] else "⚪" for m in east_history])
                                st.caption(f"{east_name}: {form_bubbles}")
                            else:
                                st.caption(f"{east_name}: No data")
                        with form_col2:
                            if west_history:
                                form_bubbles = " ".join(["🟢" if m["is_win"] else "⚪" for m in west_history])
                                st.caption(f"{west_name}: {form_bubbles}")
                            else:
                                st.caption(f"{west_name}: No data")
                
                    
                    # Make your pick
                    st.markdown("**🎯 Make Your Pick**")
                    pick_options = [east_name, west_name, "(skip)"]
                    bout_key = f"{chosen.basho_id}-day{next_day}:{east_name}vs{west_name}"
                    saved_pick = st.session_state["pick_log"].get(bout_key)
                    if saved_pick == east_name:
                        default_index = 0
                    elif saved_pick == west_name:
                        default_index = 1
                    else:
                        default_index = 2

                    pick = st.radio(
                        label="Who will win?",
                        options=pick_options,
                        horizontal=True,
                        key=f"pick-{bout_key}",
                        index=default_index,
                    )
                    record_pick(
                        tournament_id=chosen.basho_id,
                        day=next_day,
                        east=east_name,
                        west=west_name,
                        pick=None if pick == "(skip)" else pick,
                    )
                    
                    if pick != "(skip)":
                        agrees = (pick == predicted_winner) if winner_side != "none" else False
                        if agrees:
                            st.success(f"✓ You agree with the prediction ({predicted_winner})")
                        elif winner_side != "none":
                            st.warning(f"⚠ You're going against the prediction (favors {predicted_winner})")
    else:
        st.info("Day 15 is the final day of the tournament.")

    note = st.text_area("Daily notes", placeholder="Upsets, yusho race, injuries...")
    if st.button("Save note"):
        save_daily_note(chosen.basho_id, day, note)
        st.toast(f"Note saved for day {day}")

    stored_key = f"{chosen.basho_id}-day{day}"
    if stored_key in st.session_state["daily_notes"]:
        st.write("Saved note:")
        st.info(st.session_state["daily_notes"][stored_key])


def render_learning_tab() -> None:
    """Reference material for fans who are learning the sport."""

    st.subheader("Learn the Lingo")
    glossary = {
        "Banzuke": "Monthly ranking sheet that sets matchups.",
        "Kimarite": "The winning technique used to end a bout.",
        "Kachi-koshi": "A winning record (8+ wins) that moves a rikishi up the ranks.",
        "Make-koshi": "Losing record that drops a wrestler down the banzuke.",
        "Sansho": "Special prizes awarded on day 15 for technique, fighting spirit, and achievement.",
    }
    for term, description in glossary.items():
        st.markdown(f"- **{term}** - {description}")

    st.markdown("### How to Use the Companion")
    st.write(
        "Use the calendar to plan viewing parties, save your favorite rikishi, then log daily "
        "picks while you stream the live broadcast. The learning corner keeps terminology handy."
    )


def main() -> None:
    """Bootstraps Streamlit tabs for the Sumo companion app."""

    st.set_page_config(page_title="Sumo Tournament Companion", page_icon="🏯", layout="wide")
    init_session_state()

    # Japanese-themed CSS: cream/paper background with subtle seigaiha wave pattern
    st.markdown("""
    <style>
    /* Main app background - cream paper with subtle wave pattern */
    .stApp {
        background: 
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='50' viewBox='0 0 100 50'%3E%3Cpath d='M0 25 Q25 0 50 25 T100 25' fill='none' stroke='%23d4a574' stroke-width='0.5' opacity='0.3'/%3E%3Cpath d='M0 35 Q25 10 50 35 T100 35' fill='none' stroke='%23d4a574' stroke-width='0.5' opacity='0.2'/%3E%3Cpath d='M0 45 Q25 20 50 45 T100 45' fill='none' stroke='%23d4a574' stroke-width='0.5' opacity='0.15'/%3E%3C/svg%3E"),
            linear-gradient(135deg, #faf6f0 0%, #f5ede0 50%, #faf6f0 100%);
        background-size: 100px 50px, cover;
    }
    
    /* Sidebar styling - darker traditional feel */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1b1b 0%, #1a1a2e 100%);
    }
    
    /* Header area accent */
    header[data-testid="stHeader"] {
        background: rgba(250, 246, 240, 0.95);
        border-bottom: 2px solid #c41e3a;
    }
    
    /* Cards and containers - subtle paper texture effect */
    [data-testid="stExpander"], .stDataFrame, [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 1px solid #d4a574 !important;
        border-radius: 4px;
    }
    
    /* Tab styling with traditional red accent */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #c41e3a;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #2d1b1b;
    }
    
    .stTabs [aria-selected="true"] {
        background: #c41e3a !important;
        color: white !important;
        border-radius: 4px 4px 0 0;
    }
    
    /* Button accents */
    .stButton > button {
        background: linear-gradient(180deg, #c41e3a 0%, #8b0000 100%);
        color: white;
        border: none;
    }
    
    .stButton > button:hover {
        background: linear-gradient(180deg, #d4342e 0%, #a00000 100%);
    }
    
    /* Slider and select styling */
    .stSlider > div > div > div {
        background: #c41e3a !important;
    }
    
    /* Title styling */
    h1 {
        color: #2d1b1b !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Sumo Tournament Companion")
    st.caption("Plan each honbasho, spotlight your favorite rikishi, and log bout picks.")

    year = st.radio("Season", options=[2024, 2025, 2026], index=1, horizontal=True)
    tournaments = assemble_tournaments(year)

    tabs = st.tabs(["Daily Tracker", "Calendar", "Banzuke", "Learn"], default="Daily Tracker")
    with tabs[0]:
        render_tracker_tab(tournaments)
    with tabs[1]:
        render_calendar_tab(tournaments)
    with tabs[2]:
        render_rikishi_tab()
    with tabs[3]:
        render_learning_tab()


if __name__ == "__main__":
    main()
