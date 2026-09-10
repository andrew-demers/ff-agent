"""CLI entrypoint: loops over every league in config.yaml, pulls the current
roster/free-agent data from ESPN, asks Claude for start/sit + waiver
recommendations, and writes a report per league.

Usage:
    python main.py
"""

import os
import sys

import yaml
from dotenv import load_dotenv

from src.espn_client import fetch_league_snapshot
from src.claude_analyst import get_recommendations
from src.report import write_report


def main():
    load_dotenv()

    espn_s2 = os.environ.get("ESPN_S2")
    swid = os.environ.get("SWID")
    if not espn_s2 or not swid:
        print("Missing ESPN_S2 or SWID in your .env file. See .env.example.", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Missing ANTHROPIC_API_KEY in your .env file. See .env.example.", file=sys.stderr)
        sys.exit(1)

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    for league_cfg in config["leagues"]:
        name = league_cfg["name"]
        print(f"\n=== {name} ===")
        try:
            snapshot = fetch_league_snapshot(
                name=name,
                league_id=league_cfg["league_id"],
                year=league_cfg["year"],
                team_id=league_cfg["team_id"],
                espn_s2=espn_s2,
                swid=swid,
            )
        except Exception as e:
            print(f"  Failed to fetch data: {e}", file=sys.stderr)
            continue

        print(f"  Week {snapshot.week} - {snapshot.team_name}. Asking Claude for recommendations...")
        recommendations = get_recommendations(snapshot)

        path = write_report(snapshot, recommendations)
        print(f"  Report written to {path}")
        print("\n" + recommendations + "\n")


if __name__ == "__main__":
    main()
