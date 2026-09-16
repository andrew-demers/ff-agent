"""Wraps espn_api to pull the data we need for a single league: current
roster/lineup, injury status, top available free agents (as dedicated
per-position pools - QB/RB/WR/TE/K/D-ST - so no position's best option gets
crowded out by a single list sorted across all of them), this week's
matchup, each rostered player's week-by-week scoring history so far this
season, and - when games for the current week are underway or finished -
live actual scoring pulled from ESPN's box scores."""

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from espn_api.football import League

from .news import fetch_player_news

BENCH_SLOTS = {"BE", "IR"}

# How many past weeks of actual points to include per roster player.
RECENT_WEEKS_LOOKBACK = 4

# How many weeks ahead to check for a roster player's next bye, so waiver
# adds can get ahead of a bye before that position gets crowded on the wire
# closer to it.
FUTURE_WEEKS_LOOKAHEAD = 4

# Fetched as their own guaranteed per-position pools (see below) so a top
# option at a thin position can't get crowded out of a single list sorted
# by overall ownership.
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")


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
    player_id: int = 0
    recent_points: list = field(default_factory=list)
    pro_opponent: str = ""
    next_bye_week: int = None
    live_points: float = None
    game_status: str = ""
    news: list = field(default_factory=list)


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
    free_agents_by_position: dict = field(default_factory=dict)
    opponent_starters: list = field(default_factory=list)
    free_agent_kickers: list = field(default_factory=list)
    free_agent_defenses: list = field(default_factory=list)
    live_score: float = None
    live_opponent_score: float = None
    any_games_started: bool = False
    first_kickoff: datetime = None


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


def _next_bye_week(player, current_week: int) -> int:
    """The next week within FUTURE_WEEKS_LOOKAHEAD that this rostered player
    has no scheduled game, i.e. their upcoming bye - so waiver adds can plan
    for a thin position's bye before it's this week's problem.

    Only rostered Player objects carry the full-season `schedule` dict this
    depends on; free agents only carry pro_opponent for the current week, so
    this always returns None for them.
    """
    schedule = getattr(player, "schedule", None)
    if not schedule:
        return None
    for week in range(current_week + 1, current_week + 1 + FUTURE_WEEKS_LOOKAHEAD):
        if str(week) not in schedule:
            return week
    return None


def _snapshot_player(player, current_week: int = None, live_player=None) -> PlayerSnapshot:
    live_points = None
    game_status = ""
    if live_player is not None:
        live_points = round(getattr(live_player, "points", 0) or 0, 1)
        game_status = _game_status(live_player)

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
        player_id=getattr(player, "playerId", 0),
        recent_points=_recent_points(player, current_week) if current_week else [],
        pro_opponent=_pro_opponent(player, current_week),
        next_bye_week=_next_bye_week(player, current_week) if current_week else None,
        live_points=live_points,
        game_status=game_status,
    )


def _game_status(box_player) -> str:
    """Where a player's real-world game stands, for a box-score player.

    `box_player.game_played` is not ESPN data - it's the library's own
    kickoff-plus-3-hours heuristic - so we reproduce that same heuristic
    here rather than lean on a flag that looks more authoritative than it is.
    """
    if getattr(box_player, "on_bye_week", False):
        return "BYE"
    game_date = getattr(box_player, "game_date", None)
    if game_date is None:
        return "unknown"
    now = datetime.now()
    if now < game_date:
        return "not started"
    if now < game_date + timedelta(hours=3):
        return "in progress"
    return "final"


def _live_lineups(league, team_id: int):
    """Best-effort live box-score data for this week's matchup: per-player
    live points/status for both this team's and the opponent's lineup, plus
    each team's live score. Live data is an enhancement, never something
    that should break a run, so any failure here just falls back to
    projections-only (empty dict, None scores).

    Returns (live_by_player_id, own_live_score, opponent_live_score).
    """
    try:
        box_scores = league.box_scores()
    except Exception as e:
        print(f"  Warning: failed to fetch live box scores: {e}", file=sys.stderr)
        return {}, None, None

    for box in box_scores:
        home_id = getattr(box.home_team, "team_id", None)
        away_id = getattr(box.away_team, "team_id", None)
        if team_id not in (home_id, away_id):
            continue
        if home_id == team_id:
            own_lineup, own_score, opp_lineup, opp_score = (
                box.home_lineup, box.home_score, box.away_lineup, box.away_score,
            )
        else:
            own_lineup, own_score, opp_lineup, opp_score = (
                box.away_lineup, box.away_score, box.home_lineup, box.home_score,
            )
        live_by_id = {p.playerId: p for p in own_lineup + opp_lineup}
        return live_by_id, own_score, opp_score

    # No matchup found for this team this week (e.g. a bye week matchup).
    return {}, None, None


def build_league(league_id: int, year: int, espn_s2: str, swid: str) -> League:
    return League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)


def fetch_league_snapshot(name: str, league: League, team_id: int) -> LeagueSnapshot:
    current_week = league.current_week

    team = next((t for t in league.teams if t.team_id == team_id), None)
    if team is None:
        raise ValueError(
            f"No team with team_id={team_id} found in league '{name}'. "
            f"Available team_ids: {[t.team_id for t in league.teams]}"
        )

    live_by_id, live_score, live_opponent_score = _live_lineups(league, team_id)

    roster = [_snapshot_player(p, current_week, live_by_id.get(p.playerId)) for p in team.roster]

    kickoffs = [
        live_by_id[p.playerId].game_date
        for p in team.roster
        if p.playerId in live_by_id and getattr(live_by_id[p.playerId], "game_date", None) is not None
    ]
    first_kickoff = min(kickoffs) if kickoffs else None
    any_games_started = any(p.game_status in ("in progress", "final") for p in roster)

    # Pulled as separate per-position pools, not one list sorted by overall
    # ownership - otherwise a thin position's best available option can get
    # buried under a flood of higher-owned players at a deeper position.
    free_agents_by_position = {}
    for position in SKILL_POSITIONS:
        pool = [_snapshot_player(p) for p in league.free_agents(position=position, size=8)]
        pool.sort(key=lambda p: (p.percent_started, p.projected_points), reverse=True)
        free_agents_by_position[position] = pool

    # Kickers and D/ST rarely crack the top of the general free-agent pool
    # (it's sorted by percent owned across all positions), so pull them as
    # their own guaranteed groups rather than risk them getting crowded out.
    free_agent_kickers = [_snapshot_player(p) for p in league.free_agents(position="K", size=8)]
    free_agent_kickers.sort(key=lambda p: (p.projected_points, p.percent_owned), reverse=True)

    free_agent_defenses = [_snapshot_player(p) for p in league.free_agents(position="D/ST", size=8)]
    free_agent_defenses.sort(key=lambda p: (p.projected_points, p.percent_owned), reverse=True)

    opponent = team.schedule[current_week - 1]
    opponent_starters = sorted(
        (
            _snapshot_player(p, current_week, live_by_id.get(p.playerId))
            for p in opponent.roster
            if getattr(p, "lineupSlot", "") not in BENCH_SLOTS
        ),
        key=lambda p: p.lineup_slot,
    )

    # Only fully-played weeks have a real score; drop the current (in-progress) week -
    # its live number comes from the box score above instead.
    team_weekly_scores = [round(s, 1) for s in team.scores[: current_week - 1]]

    # Same-day news, joined by ESPN's own player id - only for the players
    # that actually reach the prompt, so we're not paginating news for 30
    # free agents when only the top slice of each pool gets shown.
    news_candidates = (
        roster
        + [p for pool in free_agents_by_position.values() for p in pool[:5]]
        + free_agent_kickers[:5] + free_agent_defenses[:5]
    )
    news_by_id = fetch_player_news([p.player_id for p in news_candidates])
    for p in news_candidates:
        p.news = news_by_id.get(p.player_id, [])

    return LeagueSnapshot(
        league_name=name,
        week=current_week,
        team_name=team.team_name,
        team_record=f"{team.wins}-{team.losses}-{team.ties}",
        team_weekly_scores=team_weekly_scores,
        opponent_name=opponent.team_name,
        opponent_record=f"{opponent.wins}-{opponent.losses}-{opponent.ties}",
        roster=roster,
        free_agents_by_position=free_agents_by_position,
        opponent_starters=opponent_starters,
        free_agent_kickers=free_agent_kickers,
        free_agent_defenses=free_agent_defenses,
        live_score=live_score,
        live_opponent_score=live_opponent_score,
        any_games_started=any_games_started,
        first_kickoff=first_kickoff,
    )
