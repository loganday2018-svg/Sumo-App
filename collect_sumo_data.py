"""
Sumo Data Collection Script
============================
Run this script to download and cache historical tournament data from sumo-api.com.
This reduces API calls when using the Sumo Companion app.

Usage:
    python collect_sumo_data.py

The script will:
1. Fetch basho overview data for all completed tournaments
2. Fetch daily torikumi (match) data for each tournament
3. Fetch rikishi directory (wrestler list)
4. Save everything to .sumo_cache/ folder
"""

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Configuration
SUMO_API_BASE_URL = "https://www.sumo-api.com/api"
CACHE_DIR = Path(".sumo_cache")
RATE_LIMIT_DELAY = 1.0  # Seconds between API calls to be respectful

# Tournament definitions
TOURNAMENTS_2024 = [
    ("202401", "Hatsu Basho", "January"),
    ("202403", "Haru Basho", "March"),
    ("202405", "Natsu Basho", "May"),
    ("202407", "Nagoya Basho", "July"),
    ("202409", "Aki Basho", "September"),
    ("202411", "Kyushu Basho", "November"),
]

TOURNAMENTS_2025 = [
    ("202501", "Hatsu Basho", "January"),
    ("202503", "Haru Basho", "March"),
    ("202505", "Natsu Basho", "May"),
    ("202507", "Nagoya Basho", "July"),
    ("202509", "Aki Basho", "September"),
    ("202511", "Kyushu Basho", "November"),
]

DIVISIONS = ["Makuuchi"]  # Add "Juryo" if you want lower division data


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)
    print(f"[OK] Cache directory: {CACHE_DIR.absolute()}")


def api_get(path: str, params: Dict = None) -> Optional[Dict]:
    """Make an API request with rate limiting."""
    url = f"{SUMO_API_BASE_URL}{path}"
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        time.sleep(RATE_LIMIT_DELAY)  # Be nice to the API
        return response.json()
    except requests.RequestException as e:
        print(f"  [X] API error: {e}")
        return None


def save_to_cache(filename: str, data: Dict) -> bool:
    """Save data to cache file."""
    filepath = CACHE_DIR / filename
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except IOError as e:
        print(f"  [X] Failed to save {filename}: {e}")
        return False


def cache_exists(filename: str) -> bool:
    """Check if a cache file already exists."""
    return (CACHE_DIR / filename).exists()


def collect_basho_overview(basho_id: str, name: str) -> bool:
    """Fetch and cache basho overview data."""
    filename = f"{basho_id}_overview.json"
    
    if cache_exists(filename):
        print(f"  - {name} overview: already cached")
        return True
    
    print(f"  - Fetching {name} overview...", end=" ", flush=True)
    data = api_get(f"/basho/{basho_id}")
    
    if data:
        save_to_cache(filename, data)
        print("[OK]")
        return True
    else:
        print("[FAILED]")
        return False


def collect_torikumi(basho_id: str, name: str, division: str) -> int:
    """Fetch and cache all 15 days of torikumi data."""
    days_collected = 0
    
    for day in range(1, 16):
        filename = f"{basho_id}_torikumi_{division}_day{day}.json"
        
        if cache_exists(filename):
            days_collected += 1
            continue
        
        print(f"  - Fetching {name} {division} Day {day}...", end=" ", flush=True)
        data = api_get(f"/basho/{basho_id}/torikumi/{division}/{day}")
        
        if data and data.get("torikumi"):
            save_to_cache(filename, data)
            print("[OK]")
            days_collected += 1
        else:
            print("[NO DATA]")
            break  # Stop if no data for this day
    
    return days_collected


def collect_rikishi_directory() -> bool:
    """Fetch and cache the full rikishi directory."""
    filename = "rikishi_directory.json"
    
    print("  - Fetching rikishi directory...", end=" ", flush=True)
    data = api_get("/rikishis", params={"limit": 800, "skip": 0, "ranks": "true"})
    
    if data:
        save_to_cache(filename, data)
        count = len(data.get("records", []))
        print(f"[OK] ({count} wrestlers)")
        return True
    else:
        print("[FAILED]")
        return False


def collect_rikishi_directory_with_retired() -> bool:
    """Fetch and cache rikishi directory including retired wrestlers."""
    filename = "rikishi_directory_with_retired.json"
    
    print("  - Fetching rikishi directory (with retired)...", end=" ", flush=True)
    data = api_get("/rikishis", params={"limit": 800, "skip": 0, "ranks": "true", "intai": "true"})
    
    if data:
        save_to_cache(filename, data)
        count = len(data.get("records", []))
        print(f"[OK] ({count} wrestlers)")
        return True
    else:
        print("[FAILED]")
        return False


def is_tournament_completed(basho_id: str) -> bool:
    """Check if a tournament has completed based on current date."""
    year = int(basho_id[:4])
    month = int(basho_id[4:6])
    
    # Tournament end dates (approximate - last day is around 23-28 of the month)
    end_day = 28
    tournament_end = date(year, month, end_day)
    
    return date.today() > tournament_end


def collect_tournament_data(basho_id: str, name: str, month: str):
    """Collect all data for a single tournament."""
    print(f"\n{'='*50}")
    print(f"  {name} ({basho_id})")
    print(f"{'='*50}")
    
    if not is_tournament_completed(basho_id):
        print("  [SKIP] Tournament not yet completed")
        return
    
    # Basho overview
    collect_basho_overview(basho_id, name)
    
    # Torikumi for each division
    for division in DIVISIONS:
        days = collect_torikumi(basho_id, name, division)
        print(f"  --> {division}: {days}/15 days cached")


def print_cache_summary():
    """Print a summary of cached files."""
    if not CACHE_DIR.exists():
        print("\nNo cache directory found.")
        return
    
    files = list(CACHE_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in files)
    
    print(f"\n{'='*50}")
    print("  CACHE SUMMARY")
    print(f"{'='*50}")
    print(f"  Total files: {len(files)}")
    print(f"  Total size: {total_size / 1024:.1f} KB")
    
    # Count by type
    overviews = len(list(CACHE_DIR.glob("*_overview.json")))
    torikumi = len(list(CACHE_DIR.glob("*_torikumi_*.json")))
    rikishi = len(list(CACHE_DIR.glob("rikishi_*.json")))
    
    print(f"\n  By type:")
    print(f"    - Basho overviews: {overviews}")
    print(f"    - Torikumi files: {torikumi}")
    print(f"    - Rikishi files: {rikishi}")


def main():
    """Main entry point for data collection."""
    print("\n" + "="*50)
    print("  SUMO DATA COLLECTION SCRIPT")
    print("="*50)
    print(f"\nThis script will download historical tournament data")
    print(f"from sumo-api.com and cache it locally.")
    print(f"\nRate limit: {RATE_LIMIT_DELAY}s delay between requests")
    
    ensure_cache_dir()
    
    # Collect 2024 tournaments
    print("\n\n" + " 2024 TOURNAMENTS ".center(50, "="))
    for basho_id, name, month in TOURNAMENTS_2024:
        collect_tournament_data(basho_id, name, month)
    
    # Collect 2025 tournaments
    print("\n\n" + " 2025 TOURNAMENTS ".center(50, "="))
    for basho_id, name, month in TOURNAMENTS_2025:
        collect_tournament_data(basho_id, name, month)
    
    # Collect rikishi directory
    print("\n\n" + " RIKISHI DIRECTORY ".center(50, "="))
    collect_rikishi_directory()
    collect_rikishi_directory_with_retired()
    
    # Print summary
    print_cache_summary()
    
    print("\n[DONE] Data collection complete!")
    print("       You can now run the Sumo Companion app with cached data.\n")


if __name__ == "__main__":
    main()
