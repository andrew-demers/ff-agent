"""Sends a league snapshot to the Claude API and gets back start/sit and
waiver-wire recommendations."""

import os
from anthropic import Anthropic

from .espn_client import LeagueSnapshot

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are a fantasy football analyst. You will be given one \
team's current roster and a list of available free agents for a single ESPN \
league, week, including each player's position, pro team, injury status, \
season average points, and projected points for the coming week.

Produce a concise report with three sections:
1. Start/Sit: for each roster spot that has a real decision to make (bench \
   players who could reasonably start, or starters who should be benched), \
   say who to start and who to sit, and why in one sentence.
2. Injury/Bye Watch: flag any rostered player who is injured, questionable, \
   or on a bye, and what to do about it (drop, stash, or plan around).
3. Waiver Wire Targets: rank the top 3-5 free agents worth adding this week, \
   with a one-sentence reason each, and who on the current roster (if anyone) \
   they'd replace.

Be direct and specific. Use the data given; do not invent stats or news you \
were not given. If the data doesn't support a confident call, say so briefly \
instead of guessing."""


def _format_player(p) -> str:
    return (
        f"- {p.name} ({p.position}, {p.pro_team}) | slot: {p.lineup_slot} | "
        f"injury: {p.injury_status} | avg pts: {p.avg_points} | "
        f"proj pts: {p.projected_points} | owned: {p.percent_owned}% | "
        f"started: {p.percent_started}%"
    )


def build_user_prompt(snapshot: LeagueSnapshot) -> str:
    roster_lines = "\n".join(_format_player(p) for p in snapshot.roster)
    fa_lines = "\n".join(_format_player(p) for p in snapshot.free_agents[:20])

    return (
        f"League: {snapshot.league_name}\n"
        f"Team: {snapshot.team_name}\n"
        f"Week: {snapshot.week}\n\n"
        f"CURRENT ROSTER:\n{roster_lines}\n\n"
        f"TOP AVAILABLE FREE AGENTS:\n{fa_lines}\n"
    )


def get_recommendations(snapshot: LeagueSnapshot, client: Anthropic = None) -> str:
    client = client or Anthropic()
    message = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(snapshot)}],
    )
    return message.content[0].text
