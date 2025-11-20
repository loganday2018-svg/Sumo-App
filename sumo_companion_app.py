"""Streamlit entry point for the Sumo tournament companion app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Sequence, Tuple

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
DIVISION_ORDER = {
    "Makuuchi": 0,
    "Juryo": 1,
    "Makushita": 2,
    "Sandanme": 3,
    "Jonidan": 4,
    "Jonokuchi": 5,
}


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

    highlights: Dict[str, str] = {
        "Hatsu Basho": "New Year kickoff with debuts from the latest banzuke.",
        "Haru Basho": "Crowd-favorite crowd noise inside Edion Arena.",
        "Nagoya Basho": "Sweltering summer meet known for marathon bouts.",
        "Kyushu Basho": "Season finale and last chance at yearly awards.",
    }

    champions = {
        "Hatsu Basho": "Terunofuji",
        "Haru Basho": "Kirishima",
        "Natsu Basho": "Takakeisho",
        "Nagoya Basho": "Hoshoryu",
        "Aki Basho": "Kotonowaka",
        "Kyushu Basho": "Daieisho",
    }

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


def _sumo_api_get(path: str, params: Dict[str, str] | None = None) -> Dict:
    """Call the public Sumo API and return JSON."""

    url = f"{SUMO_API_BASE_URL}{path}"
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Sumo API request failed: {exc}") from exc
    return response.json()


@st.cache_data(ttl=SUMO_API_CACHE_TTL)
def fetch_basho_overview(basho_id: str) -> Dict:
    """Fetch high-level tournament results once per day."""

    return _sumo_api_get(f"/basho/{basho_id}")


@st.cache_data(ttl=SUMO_API_CACHE_TTL)
def fetch_torikumi_payload(basho_id: str, division: str, day: int) -> Dict:
    """Fetch torikumi data for a specific division and day."""

    return _sumo_api_get(f"/basho/{basho_id}/torikumi/{division}/{day}")


@st.cache_data(ttl=SUMO_API_CACHE_TTL)
def fetch_head_to_head_pair(pair: tuple[int, int]) -> Dict:
    """Fetch head-to-head results for a pair of rikishi IDs."""

    rikishi_id, opponent_id = pair
    return _sumo_api_get(f"/rikishi/{rikishi_id}/matches/{opponent_id}")


@st.cache_data(ttl=SUMO_API_CACHE_TTL)
def fetch_completed_torikumi(basho_id: str, division: str) -> Dict[int, List[Dict]]:
    """Return torikumi for every fully completed day."""

    completed: Dict[int, List[Dict]] = {}
    for day in range(1, 16):
        try:
            payload = fetch_torikumi_payload(basho_id, division, day)
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


def scoreboard_dataframe(
    results: Dict[int, Sequence[Dict]], limit: int = 15
) -> pd.DataFrame:
    """Aggregate wins/losses for each rikishi from completed days."""

    stats: Dict[int, Dict[str, object]] = {}
    for matches in results.values():
        for match in matches:
            east_id = match.get("eastId")
            west_id = match.get("westId")
            winner_id = match.get("winnerId")
            if east_id is None or west_id is None or winner_id is None:
                continue

            for side, rikishi_id in (("east", east_id), ("west", west_id)):
                if rikishi_id not in stats:
                    stats[rikishi_id] = {
                        "shikona": match.get(f"{side}Shikona"),
                        "rank": match.get(f"{side}Rank"),
                        "wins": 0,
                        "losses": 0,
                    }

            if winner_id == east_id:
                stats[east_id]["wins"] = int(stats[east_id]["wins"]) + 1
                stats[west_id]["losses"] = int(stats[west_id]["losses"]) + 1
            elif winner_id == west_id:
                stats[west_id]["wins"] = int(stats[west_id]["wins"]) + 1
                stats[east_id]["losses"] = int(stats[east_id]["losses"]) + 1

    rows: List[Dict[str, object]] = []
    for payload in stats.values():
        wins = int(payload["wins"])
        losses = int(payload["losses"])
        rows.append(
            {
                "Shikona": payload.get("shikona") or "Unknown",
                "Rank": payload.get("rank") or "-",
                "Wins": wins,
                "Losses": losses,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["Wins", "Losses"], ascending=[False, True])
    return df.head(limit).reset_index(drop=True)


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
    """Load active rikishi records from the API."""

    params = {"limit": limit, "skip": 0, "ranks": "true"}
    if include_retired:
        params["intai"] = "true"
    return _sumo_api_get("/rikishis", params=params)


@st.cache_data(ttl=SUMO_API_CACHE_TTL)
def fetch_rikishi_stats(rikishi_id: int) -> Dict:
    """Fetch aggregated stats for a specific rikishi."""

    return _sumo_api_get(f"/rikishi/{rikishi_id}/stats")


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


def render_calendar_tab(tournaments: Sequence[Tournament]) -> None:
    """Render the full honbasho calendar and champion history."""

    st.subheader("Tournament Calendar")
    df = tournaments_dataframe(tournaments)
    st.dataframe(df, use_container_width=True, hide_index=True)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("Tournament", sort=None),
            x=alt.X("Start", title="Date"),
            x2="End",
            color=alt.Color("City", legend=alt.Legend(title="Host City")),
            tooltip=["Tournament", "City", "Champion", "Start", "End", "Highlight"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Champion storyline"):
        for _, row in df.iterrows():
            st.markdown(
                f"**{row['Tournament']}** - {row['Champion']} ( {row['Highlight'] or 'Season staple'} )"
            )


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
    completed_days = fetch_completed_torikumi(chosen.basho_id, DEFAULT_DIVISION)
    latest_completed_day = max(completed_days.keys()) if completed_days else 0
    scoreboard = scoreboard_dataframe(completed_days)
    if not scoreboard.empty:
        st.markdown("### Scoreboard (Top Performers)")
        st.dataframe(scoreboard, use_container_width=True, hide_index=True)
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
        overview = fetch_basho_overview(chosen.basho_id)
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
            day_payload = fetch_torikumi_payload(chosen.basho_id, DEFAULT_DIVISION, day)
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
            next_payload = fetch_torikumi_payload(chosen.basho_id, DEFAULT_DIVISION, next_day)
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
                st.markdown(
                    f"**Bout {match.get('matchNo')}: {east_name} ({match.get('eastRank')}) "
                    f"vs {west_name} ({match.get('westRank')})**"
                )
                east_id = match.get("eastId")
                west_id = match.get("westId")
                if east_id is None or west_id is None:
                    st.caption("Head-to-head data unavailable for this bout.")
                else:
                    try:
                        summary = summarize_head_to_head(east_id, west_id, east_name, west_name)
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
                                .properties(height=120)
                            )
                            st.altair_chart(chart, use_container_width=True)
                    except RuntimeError as exc:
                        st.caption(f"Head-to-head unavailable: {exc}")

                pick_options = ["(skip)", east_name, west_name]
                bout_key = f"{chosen.basho_id}-day{next_day}:{east_name}vs{west_name}"
                saved_pick = st.session_state["pick_log"].get(bout_key)
                if saved_pick == east_name:
                    default_index = 1
                elif saved_pick == west_name:
                    default_index = 2
                else:
                    default_index = 0

                pick = st.radio(
                    label="Select your pick",
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
                st.divider()
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

    st.set_page_config(page_title="Sumo Tournament Companion", page_icon="S", layout="wide")
    init_session_state()

    st.title("Sumo Tournament Companion")
    st.caption("Plan each honbasho, spotlight your favorite rikishi, and log bout picks.")

    year = st.select_slider("Season", options=[2024, 2025, 2026], value=2025)
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
