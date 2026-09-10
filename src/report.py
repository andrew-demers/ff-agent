"""Writes the per-league recommendation to a markdown file under reports/."""

import os
from datetime import datetime

from .espn_client import LeagueSnapshot

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


def write_report(snapshot: LeagueSnapshot, recommendations: str) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    safe_name = snapshot.league_name.lower().replace(" ", "_")
    path = os.path.join(REPORTS_DIR, f"{timestamp}_week{snapshot.week}_{safe_name}.md")

    with open(path, "w") as f:
        f.write(f"# {snapshot.league_name} - {snapshot.team_name} - Week {snapshot.week}\n\n")
        f.write(recommendations.strip() + "\n")

    return path
