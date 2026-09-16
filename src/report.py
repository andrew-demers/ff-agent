"""Writes the per-league recommendation to a markdown file under reports/."""

import os
from datetime import datetime

from .espn_client import LeagueSnapshot
from .history import extract_calls_block, league_slug

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


def write_report(snapshot: LeagueSnapshot, lineup_text: str, waiver_text: str) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(
        REPORTS_DIR, f"{timestamp}_week{snapshot.week}_{league_slug(snapshot.league_name)}.md"
    )

    # Each agent's trailing <!--calls--> block is for src/history.py, not
    # the owner - keep it out of the file they actually read.
    lineup_prose, _ = extract_calls_block(lineup_text)
    waiver_prose, _ = extract_calls_block(waiver_text)

    with open(path, "w") as f:
        f.write(f"# {snapshot.league_name} - {snapshot.team_name} - Week {snapshot.week}\n\n")
        f.write(
            f"Record: {snapshot.team_record} | This week vs {snapshot.opponent_name} "
            f"({snapshot.opponent_record})\n\n"
        )
        f.write(lineup_prose.strip() + "\n\n")
        f.write(waiver_prose.strip() + "\n")

    return path
