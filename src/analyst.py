"""Sends a league snapshot to an LLM (Anthropic's API or a local Ollama
server) and gets back start/sit and waiver-wire recommendations."""

import os

from .espn_client import LeagueSnapshot

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

SYSTEM_PROMPT = """You are a fantasy football analyst. You will be given: \
this team's season record and week-by-week scoring so far, this week's \
opponent (record and their projected starters), the team's own roster, and \
a list of available free agents for a single ESPN league. Each player is \
listed with position, pro team, injury status, season average points, \
projected points for the coming week, that player's real NFL opponent this \
week ("this wk vs"), and (for rostered players) actual points from recent \
weeks. The opponent given is factual data from the league; you were not \
given that defense's own stats, so only use it to name the matchup, and \
lean on the projected points (which already account for matchup difficulty) \
rather than asserting how strong or weak that defense is.

Produce a concise report with three sections:
1. Starting Lineup Changes: for each roster spot that has a real decision \
   to make (bench players who could reasonably start, or starters who \
   should be benched), say who to start and who to sit, and why in one \
   sentence. Weigh recent-week trends alongside projections and, where it \
   matters, how a player's matchup stacks up against the opponent's \
   starters at the same position. Explicitly cover the kicker and D/ST \
   slots every week, even if the call is just to confirm the current \
   starter is still the right one.
2. Waiver Wire Pickups: rank the top 3-5 free agents worth adding this \
   week, with a one-sentence reason each, and who on the current roster \
   (if anyone) they'd replace. Consider the kicker and D/ST free-agent \
   pools every week alongside skill positions, not just when there's an \
   injury forcing the issue.
3. Things to Watch: flag anything else worth keeping an eye on - injuries, \
   byes, hot or cold streaks in the recent-week data, and any part of this \
   week's matchup against the opponent's roster that could swing the \
   outcome.

Be direct and specific. Use the data given; do not invent stats or news you \
were not given. If the data doesn't support a confident call, say so briefly \
instead of guessing."""


def _format_player(p) -> str:
    recent = f" | recent pts: {p.recent_points}" if p.recent_points else ""
    opponent = f" | this wk vs: {p.pro_opponent}" if p.pro_opponent else ""
    return (
        f"- {p.name} ({p.position}, {p.pro_team}) | slot: {p.lineup_slot} | "
        f"injury: {p.injury_status} | avg pts: {p.avg_points} | "
        f"proj pts: {p.projected_points} | owned: {p.percent_owned}% | "
        f"started: {p.percent_started}%{recent}{opponent}"
    )


def build_user_prompt(snapshot: LeagueSnapshot) -> str:
    roster_lines = "\n".join(_format_player(p) for p in snapshot.roster)
    fa_lines = "\n".join(_format_player(p) for p in snapshot.free_agents[:20])
    k_lines = "\n".join(_format_player(p) for p in snapshot.free_agent_kickers[:5])
    dst_lines = "\n".join(_format_player(p) for p in snapshot.free_agent_defenses[:5])
    opponent_lines = "\n".join(_format_player(p) for p in snapshot.opponent_starters)

    return (
        f"League: {snapshot.league_name}\n"
        f"Team: {snapshot.team_name} (record {snapshot.team_record})\n"
        f"Week: {snapshot.week}\n"
        f"Scoring so far this season (oldest to most recent): {snapshot.team_weekly_scores}\n\n"
        f"THIS WEEK'S OPPONENT: {snapshot.opponent_name} (record {snapshot.opponent_record})\n"
        f"OPPONENT'S PROJECTED STARTERS:\n{opponent_lines}\n\n"
        f"CURRENT ROSTER:\n{roster_lines}\n\n"
        f"TOP AVAILABLE FREE AGENTS (skill positions):\n{fa_lines}\n\n"
        f"AVAILABLE FREE AGENT KICKERS:\n{k_lines}\n\n"
        f"AVAILABLE FREE AGENT DEFENSES/D-ST:\n{dst_lines}\n"
    )


def _get_anthropic_recommendations(user_prompt: str, client=None) -> str:
    from anthropic import Anthropic

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    client = client or Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def _get_ollama_recommendations(user_prompt: str, client=None) -> str:
    from ollama import Client

    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    client = client or Client(host=host)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


def get_recommendations(snapshot: LeagueSnapshot, client=None) -> str:
    user_prompt = build_user_prompt(snapshot)
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()

    if provider == "ollama":
        return _get_ollama_recommendations(user_prompt, client)
    if provider == "anthropic":
        return _get_anthropic_recommendations(user_prompt, client)

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic' or 'ollama'.")
