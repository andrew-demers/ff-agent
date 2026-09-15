"""Wraps espn_api to pull the data we need for a single league: current
roster/lineup, injury status, top available free agents, this week's
matchup, and each rostered player's week-by-week scoring history so far
this season."""

from dataclasses import dataclass, field
from espn_api.football import League

BENCH_SLOTS = {"BE", "IR"}

# How many past weeks of actual points to include per roster player.
RECENT_WEEKS_LOOKBACK = 4


@dataclass
class PlayerSnapshot:
    name: str
    position: str
    pro_team: str
    lineup_slot: str
    injury_status: str
    projected_points: float
    avg_points: float
    percent_owned: float
    percent_started: float
    recent_points: list = field(default_factory=list)
    pro_opponent: str = ""


@dataclass
class LeagueSnapshot:
    league_name: str
    week: int
    team_name: str
    team_record: str
    team_weekly_scores: list
    opponent_name: str
    opponent_record: str
    roster: list = field(default_factory=list)
    free_agents: list = field(default_factory=list)
    opponent_starters: list = field(default_factory=list)


def _recent_points(player, current_week: int) -> list:
    """Actual points for the most recent finished weeks, oldest first.

    Only populated when the player object carries per-week stats (true for
    rostered players fetched off a team; free agents only carry the season
    total and the current week, so this stays empty for them).
    """
    first_week = max(1, current_week - RECENT_WEEKS_LOOKBACK)
    points = []
    for week in range(first_week, current_week):
        week_stats = player.stats.get(week)
        if week_stats and "points" in week_stats:
            points.append(round(week_stats["points"], 1))
    return points


def _pro_opponent(player, current_week: int = None) -> str:
    """This week's real NFL opponent for the player's pro team, so the LLM
    reasons about actual matchups instead of guessing at them.

    Free agents come back as BoxPlayer objects with `pro_opponent` already
    resolved; rostered players are plain Player objects that carry a
    `schedule` dict of {week: {"team": abbrev}} instead.
    """
    direct = getattr(player, "pro_opponent", None)
    if direct and direct != "None":
        return direct
    if current_week is not None:
        entry = player.schedule.get(str(current_week))
        if entry:
            return entry["team"]
        return "BYE"
    return ""


def _snapshot_player(player, current_week: int = None) -> PlayerSnapshot:
    return PlayerSnapshot(
        name=player.name,
        position=player.position,
        pro_team=player.proTeam,
        lineup_slot=getattr(player, "lineupSlot", ""),
        injury_status=getattr(player, "injuryStatus", "ACTIVE") or "ACTIVE",
        projected_points=round(getattr(player, "projected_avg_points", 0) or 0, 1),
        avg_points=round(getattr(player, "avg_points", 0) or 0, 1),
        percent_owned=round(getattr(player, "percent_owned", 0) or 0, 1),
        percent_started=round(getattr(player, "percent_started", 0) or 0, 1),
        recent_points=_recent_points(player, current_week) if current_week else [],
        pro_opponent=_pro_opponent(player, current_week),
    )


def fetch_league_snapshot(name: str, league_id: int, year: int, team_id: int,
                           espn_s2: str, swid: str, free_agent_size: int = 30) -> LeagueSnapshot:
    league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
    current_week = league.current_week

    team = next((t for t in league.teams if t.team_id == team_id), None)
    if team is None:
        raise ValueError(
            f"No team with team_id={team_id} found in league '{name}'. "
            f"Available team_ids: {[t.team_id for t in league.teams]}"
        )

    roster = [_snapshot_player(p, current_week) for p in team.roster]

    free_agents = [
        _snapshot_player(p)
        for p in league.free_agents(size=free_agent_size)
    ]
    # Surface the most relevant waiver targets first: highest recent trend, then projection.
    free_agents.sort(key=lambda p: (p.percent_started, p.projected_points), reverse=True)

    opponent = team.schedule[current_week - 1]
    opponent_starters = sorted(
        (
            _snapshot_player(p, current_week)
            for p in opponent.roster
            if getattr(p, "lineupSlot", "") not in BENCH_SLOTS
        ),
        key=lambda p: p.lineup_slot,
    )

    # Only fully-played weeks have a real score; drop the current (in-progress) week.
    team_weekly_scores = [round(s, 1) for s in team.scores[: current_week - 1]]

    return LeagueSnapshot(
        league_name=name,
        week=current_week,
        team_name=team.team_name,
        team_record=f"{team.wins}-{team.losses}-{team.ties}",
        team_weekly_scores=team_weekly_scores,
        opponent_name=opponent.team_name,
        opponent_record=f"{opponent.wins}-{opponent.losses}-{opponent.ties}",
        roster=roster,
        free_agents=free_agents,
        opponent_starters=opponent_starters,
    )
