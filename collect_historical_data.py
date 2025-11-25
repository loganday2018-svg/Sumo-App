"""
Sumo Historical Data Collection Script

Run this script to download and cache all historical tournament data
from the Sumo API. This should be run once initially, then after each
tournament completes to grab that new data.

Usage:
    python collect_historical_data.py

The script will:
1. Fetch basho overview for each completed tournament
2. Fetch all 15 days of torikumi (match results) for Makuuchi division
3. Fetch the rikishi directory (all wrestlers)
4. Save everything to .sumo_cache/ folder
"""

import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List

import requests

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
SUMO_API_BASE_URL = "https://www.sumo-api.com/api"
CACHE_DIR = Path(".sumo_cache")
REQUEST_DELAY = 1.0  # Seconds between API calls to be respectful

# Tournaments to collect (basho_id format: YYYYMM)
TOURNAMENTS_TO_COLLECT = [
    # 2024 tournaments
    "202401",  # Hatsu
    "202403",  # Haru
    "202405",  # Natsu
    "202407",  # Nagoya
    "202409",  # Aki
    "202411",  # Kyushu
    # 2025 tournaments
    "202501",  # Hatsu
    "202503",  # Haru
    "202505",  # Natsu
    "202507",  # Nagoya
    "202509",  # Aki
    "202511",  # Kyushu
]

DIVISIONS = ["Makuuchi"]  # Add "Juryo" etc. if you want more divisions


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)
    print(f"✓ Cache directory: {CACHE_DIR.absolute()}")


def api_get(path: str, params: Dict = None) -> Dict | None:
    """Make an API request with error handling."""
    url = f"{SUMO_API_BASE_URL}{path}"
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"  ✗ API error: {e}")
        return None


def save_to_cache(filename: str, data: Dict) -> bool:
    """Save data to cache file."""
    filepath = CACHE_DIR / filename
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except IOError as e:
        print(f"  ✗ Failed to save {filename}: {e}")
        return False


def load_from_cache(filename: str) -> Dict | None:
    """Load data from cache if it exists."""
    filepath = CACHE_DIR / filename
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def collect_basho_overview(basho_id: str, force: bool = False) -> bool:
    """Fetch and cache basho overview."""
    filename = f"{basho_id}_overview.json"
    
    if not force and load_from_cache(filename):
        print(f"  → Overview already cached")
        return True
    
    print(f"  → Fetching overview...")
    data = api_get(f"/basho/{basho_id}")
    if data:
        save_to_cache(filename, data)
        print(f"  ✓ Overview saved")
        return True
    return False


def collect_torikumi(basho_id: str, division: str, force: bool = False) -> int:
    """Fetch and cache all 15 days of torikumi for a division."""
    days_collected = 0
    
    for day in range(1, 16):
        filename = f"{basho_id}_torikumi_{division}_day{day}.json"
        
        if not force and load_from_cache(filename):
            days_collected += 1
            continue
        
        print(f"  → Fetching {division} Day {day}...")
        data = api_get(f"/basho/{basho_id}/torikumi/{division}/{day}")
        
        if data and data.get("torikumi"):
            save_to_cache(filename, data)
            days_collected += 1
            time.sleep(REQUEST_DELAY)
        else:
            # No data for this day - tournament might not have this day yet
            print(f"  → No data for Day {day}")
            time.sleep(REQUEST_DELAY / 2)
    
    return days_collected


def collect_rikishi_directory(force: bool = False) -> bool:
    """Fetch and cache the full rikishi directory."""
    filename = "rikishi_directory.json"
    
    if not force and load_from_cache(filename):
        print("  → Rikishi directory already cached")
        return True
    
    print("  → Fetching rikishi directory...")
    data = api_get("/rikishis", params={"limit": 1000, "skip": 0, "ranks": "true"})
    
    if data:
        save_to_cache(filename, data)
        record_count = len(data.get("records", []))
        print(f"  ✓ Saved {record_count} rikishi records")
        return True
    return False


def collect_rikishi_directory_with_retired(force: bool = False) -> bool:
    """Fetch and cache rikishi directory including retired wrestlers."""
    filename = "rikishi_directory_with_retired.json"
    
    if not force and load_from_cache(filename):
        print("  → Rikishi directory (with retired) already cached")
        return True
    
    print("  → Fetching rikishi directory (including retired)...")
    data = api_get("/rikishis", params={"limit": 1000, "skip": 0, "ranks": "true", "intai": "true"})
    
    if data:
        save_to_cache(filename, data)
        record_count = len(data.get("records", []))
        print(f"  ✓ Saved {record_count} rikishi records (including retired)")
        return True
    return False



def is_tournament_complete(basho_id: str) -> bool:
    """Check if a tournament is in the past (completed)."""
    year = int(basho_id[:4])
    month = int(basho_id[4:6])
    
    # Tournament typically ends around the 4th Sunday of the month
    # Approximate end date as the 28th of the tournament month
    tournament_end = date(year, month, 28)
    return date.today() > tournament_end


def collect_tournament(basho_id: str, force: bool = False) -> Dict[str, int]:
    """Collect all data for a single tournament."""
    results = {
        "overview": 0,
        "torikumi_days": 0,
    }
    
    # Overview
    if collect_basho_overview(basho_id, force):
        results["overview"] = 1
    time.sleep(REQUEST_DELAY)
    
    # Torikumi for each division
    for division in DIVISIONS:
        days = collect_torikumi(basho_id, division, force)
        results["torikumi_days"] += days
    
    return results


def get_tournament_name(basho_id: str) -> str:
    """Convert basho_id to human-readable name."""
    month_names = {
        "01": "Hatsu (January)",
        "03": "Haru (March)",
        "05": "Natsu (May)",
        "07": "Nagoya (July)",
        "09": "Aki (September)",
        "11": "Kyushu (November)",
    }
    year = basho_id[:4]
    month = basho_id[4:6]
    name = month_names.get(month, f"Month {month}")
    return f"{year} {name}"


def main():
    """Main collection routine."""
    print("=" * 60)
    print("SUMO HISTORICAL DATA COLLECTOR")
    print("=" * 60)
    print()
    
    ensure_cache_dir()
    print()
    
    # Collect rikishi directory first
    print("COLLECTING RIKISHI DIRECTORY")
    print("-" * 40)
    collect_rikishi_directory()
    time.sleep(REQUEST_DELAY)
    collect_rikishi_directory_with_retired()
    time.sleep(REQUEST_DELAY)
    print()
    
    # Collect tournament data
    total_stats = {
        "tournaments": 0,
        "overviews": 0,
        "torikumi_days": 0,
    }
    
    for basho_id in TOURNAMENTS_TO_COLLECT:
        tournament_name = get_tournament_name(basho_id)
        is_complete = is_tournament_complete(basho_id)
        status = "✓ Complete" if is_complete else "⏳ May be ongoing"
        
        print(f"COLLECTING: {tournament_name} ({basho_id}) [{status}]")
        print("-" * 40)
        
        if not is_complete:
            print("  → Skipping (tournament may not be complete)")
            print()
            continue
        
        results = collect_tournament(basho_id)
        
        total_stats["tournaments"] += 1
        total_stats["overviews"] += results["overview"]
        total_stats["torikumi_days"] += results["torikumi_days"]
        
        print(f"  ✓ Complete: {results['overview']} overview, {results['torikumi_days']} days of matches")
        print()
        
        time.sleep(REQUEST_DELAY)
    
    # Summary
    print("=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"  Tournaments processed: {total_stats['tournaments']}")
    print(f"  Overviews collected:   {total_stats['overviews']}")
    print(f"  Torikumi days:         {total_stats['torikumi_days']}")
    print()
    
    # List cache contents
    cache_files = list(CACHE_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in cache_files)
    print(f"  Cache files: {len(cache_files)}")
    print(f"  Total size:  {total_size / 1024:.1f} KB")
    print()
    print("Data saved to:", CACHE_DIR.absolute())


if __name__ == "__main__":
    main()
