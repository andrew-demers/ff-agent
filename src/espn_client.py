"""Wraps espn_api to pull the data we need for a single league: current
roster/lineup, injury status, and top available free agents."""

from dataclasses import dataclass, field
from espn_api.football import League


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


@dataclass
class LeagueSnapshot:
    league_name: str
    week: int
    team_name: str
    roster: list = field(default_factory=list)
    free_agents: list = field(default_factory=list)


def _snapshot_player(player) -> PlayerSnapshot:
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
    )


def fetch_league_snapshot(name: str, league_id: int, year: int, team_id: int,
                           espn_s2: str, swid: str, free_agent_size: int = 30) -> LeagueSnapshot:
    league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)

    team = next((t for t in league.teams if t.team_id == team_id), None)
    if team is None:
        raise ValueError(
            f"No team with team_id={team_id} found in league '{name}'. "
            f"Available team_ids: {[t.team_id for t in league.teams]}"
        )

    roster = [_snapshot_player(p) for p in team.roster]

    free_agents = [
        _snapshot_player(p)
        for p in league.free_agents(size=free_agent_size)
    ]
    # Surface the most relevant waiver targets first: highest recent trend, then projection.
    free_agents.sort(key=lambda p: (p.percent_started, p.projected_points), reverse=True)

    return LeagueSnapshot(
        league_name=name,
        week=league.current_week,
        team_name=team.team_name,
        roster=roster,
        free_agents=free_agents,
    )
