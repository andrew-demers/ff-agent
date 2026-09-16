"""Sends a league snapshot to an LLM (Anthropic's API or a local Ollama
server) and gets back recommendations, as two independent calls: one for
this week's starting lineup and things to watch, one for waiver-wire
pickups. Splitting them keeps each prompt focused on only the data it
actually needs - the lineup call never sees free-agent pools, the waiver
call never sees the live in-game score - and keeps a long waiver take from
crowding out the lineup call's token budget, or vice versa."""

import os
from collections import defaultdict
from datetime import datetime

from .espn_client import LeagueSnapshot
from .news import format_age

HEALTHY_STATUSES = {"ACTIVE", ""}

# A free agent whose ownership is running at least this many points ahead
# of their start rate is being quietly stashed leaguewide. Surfaced as its
# own computed block rather than left for the model to spot by eyeballing
# ~30 rows per position - in practice it missed the single largest gap on
# the board (a 40-point owned/started split) when asked to do that.
STASH_GAP_THRESHOLD = 15.0

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

LINEUP_SYSTEM_PROMPT = """You are a fantasy football analyst focused on this \
week's starting lineup. You will be given: this team's season record and \
week-by-week scoring so far, this week's opponent (record and their \
projected starters), and the team's own roster for a single ESPN league. \
Each player is listed with position, pro team, injury status, season \
average points, projected points for the coming week, that player's real \
NFL opponent this week ("this wk vs"), and actual points from recent \
weeks. The opponent given is factual data from the league; you were not \
given that defense's own stats, so only use it to name the matchup, and \
lean on the projected points (which already account for matchup difficulty) \
rather than asserting how strong or weak that defense is.

If a LIVE section is present, the week is already underway: points shown \
there are realized, not predicted. Treat any player whose game is "in \
progress" or "final" as locked into their actual result - never suggest \
benching or starting a player whose game has already started or finished; \
lineup advice only applies to players whose games have not started yet. \
When the live score margin matters given who's left to play, say so.

If a RECENT PLAYER NEWS section is present, treat each item as a reported \
fact from today or the last few days, not speculation. When a news item \
conflicts with a player's static injury designation, trust the news item - \
the designation can go stale within a day.

If a PODCAST ANALYSIS section is present, it is expert opinion from The \
Fantasy Footballers' start/sit episode this week, transcribed by an \
automated speech-to-text system from their YouTube captions - not verified \
data, and player names may be misspelled or garbled in transcription. \
Attribute it explicitly ("the Footballers flagged X as a start") rather \
than presenting it as your own finding, and when it conflicts with the \
ESPN projections or your own read, say so and pick a side rather than \
blending them into mush.

If a PAST RESULTS section is present, it's this team's own history: bench \
regret and ESPN's own projection error, computed from actual box scores, \
not a report card on individual calls. Use it to calibrate - for example, \
lean less on projections at a position where they've been running hot or \
cold this season - but don't over-correct on a handful of weeks of data.

Produce a concise report with two sections:
1. Starting Lineup Changes: for each roster spot that has a real decision \
   to make (bench players who could reasonably start, or starters who \
   should be benched), say who to start and who to sit, and why in one \
   sentence. Weigh recent-week trends alongside projections and, where it \
   matters, how a player's matchup stacks up against the opponent's \
   starters at the same position. Explicitly cover the kicker and D/ST \
   slots every week, even if the call is just to confirm the current \
   starter is still the right one.
2. Things to Watch: flag anything else worth keeping an eye on - injuries, \
   byes, hot or cold streaks in the recent-week data, and any part of this \
   week's matchup against the opponent's roster that could swing the \
   outcome.

Be direct and specific. Use the data given; do not invent stats or news you \
were not given. If the data doesn't support a confident call, say so briefly \
instead of guessing.

After the two sections, append exactly one machine-readable block \
recording every start/sit call you actually made above (omit it entirely \
if you made none), in this exact form and nothing else after it:

<!--calls
{"start_sit": [{"start": "<player name>", "sit": "<player name>", "slot": "<slot>"}]}
-->

Use each player's exact name as given in the data above. This block is \
parsed by code, not read by the owner - it must be the last thing in your \
response, valid JSON inside the HTML comment, with no other text after it."""


WAIVER_SYSTEM_PROMPT = """You are a fantasy football analyst focused on \
this week's waiver wire. You will be given: this team's current roster \
(with bench depth by position, any starter at risk with no healthy \
backup, and any roster player's upcoming bye), and free agents for a \
single ESPN league pulled as separate pools per position - QB, RB, WR, \
TE, plus dedicated kicker and D/ST pools - so a strong option at a thin \
position can't get lost behind whichever position is deepest. Each player \
is listed with position, pro team, injury status, season average points, \
projected points for the coming week, percent owned/started, and that \
player's real NFL opponent this week ("this wk vs").

If a RECENT PLAYER NEWS section is present, treat each item as a reported \
fact from today or the last few days, not speculation. When a news item \
conflicts with a player's static injury designation, trust the news item - \
the designation can go stale within a day.

If a PODCAST ANALYSIS section is present, it is expert opinion from The \
Fantasy Footballers' waiver-wire episode this week, transcribed by an \
automated speech-to-text system from their YouTube captions - not verified \
data, and player names may be misspelled or garbled in transcription. \
Attribute it explicitly ("the Footballers flagged X as a top add") rather \
than presenting it as your own finding, and when it conflicts with the \
ESPN projections or your own read, say so and pick a side rather than \
blending them into mush.

Produce a concise report with one section:
Waiver Wire Pickups: start from the ROSTER DEPTH data - name the \
position(s) that are thinnest (fewest bench players) or carry the biggest \
single-point-of-failure risk (a starter who's hurt or on bye with no \
healthy same-position backup), and prioritize free agents that add \
insurance there. Check the top available free agent at each of QB, RB, \
WR, and TE below - they're broken into separate pools per position so a \
strong option at a thin position can't get lost behind whichever position \
is deepest. Then list the top 3-5 free agents worth adding this week as a \
numbered list in strict priority order - #1 is the add you'd make first if \
you could only make one - with a one-sentence reason each and who on the \
current roster (if anyone) they'd replace. Weigh roster need (from ROSTER \
DEPTH) alongside player quality and opportunity when ordering, not name \
recognition.
Treat the kicker and D/ST slots as weekly streaming spots, not just \
injury or bye backfills: every week, name the free-agent kicker and the \
free-agent D/ST with the best matchup - use projected points, which \
already bake in matchup strength, to find whoever is facing the weakest \
opponent - and include them in the list even when the current starter is \
healthy and has a game. Because a good matchup this week says nothing \
about next week, these two streaming adds always rank last, below every \
skill-position pickup - a QB/RB/WR/TE add that fills a real roster need \
outranks a one-week K or D/ST streamer every time.
Also look past this week: check UPCOMING BYES for a thin position (from \
ROSTER DEPTH) about to lose its starter to a bye, and check OWNERSHIP \
MOMENTUM if it's present - it's a computed list of free agents whose \
owned% is already running well ahead of their started%, meaning the \
league is quietly stashing them before their role is starting-caliber \
yet; trust that list over eyeballing owned/started yourself, since it's \
computed across every pool, not just whichever one you happened to scan. \
Where either signal points to a free agent worth grabbing now before \
they're gone, name them as a separate, explicitly-labeled "stash for \
later" pick with the week or reason it should pay off - and rank it below \
this week's immediate-need adds (though still ahead of the K/D-ST \
streamers), since it isn't solving anything yet. Skip this note entirely \
if neither signal turns up anything rather than forcing a speculative \
pick that isn't there.

Be direct and specific. Use the data given; do not invent stats or news you \
were not given. If the data doesn't support a confident call, say so briefly \
instead of guessing.

After the section, append exactly one machine-readable block recording \
every waiver call you actually made above (omit it entirely if you made \
none), in this exact form and nothing else after it:

<!--calls
{"waivers": [{"add": "<player name>", "drop": "<player name or null>"}]}
-->

Use each player's exact name as given in the data above. This block is \
parsed by code, not read by the owner - it must be the last thing in your \
response, valid JSON inside the HTML comment, with no other text after it."""


def _format_player(p) -> str:
    recent = f" | recent pts: {p.recent_points}" if p.recent_points else ""
    opponent = f" | this wk vs: {p.pro_opponent}" if p.pro_opponent else ""
    live = f" | live: {p.live_points} ({p.game_status})" if p.live_points is not None else ""
    return (
        f"- {p.name} ({p.position}, {p.pro_team}) | slot: {p.lineup_slot} | "
        f"injury: {p.injury_status} | avg pts: {p.avg_points} | "
        f"proj pts: {p.projected_points} | owned: {p.percent_owned}% | "
        f"started: {p.percent_started}%{recent}{opponent}{live}"
    )


def _format_roster_depth(snapshot: LeagueSnapshot) -> str:
    """Bench depth by position, any starter who's hurt or on bye with no
    healthy same-position bench player behind them, and any roster player
    with a bye coming up in the next few weeks. Computed directly from the
    roster rather than left for the model to eyeball from a flat 15-player
    list, so 'what am I lacking, now or soon' is grounded in an actual
    count instead of guessed at."""
    by_position = defaultdict(list)
    for p in snapshot.roster:
        if p.lineup_slot != "IR":
            by_position[p.position].append(p)

    depth_lines, risk_lines, bye_lines = [], [], []
    for position, players in sorted(by_position.items()):
        starters = [p for p in players if p.lineup_slot != "BE"]
        bench = [p for p in players if p.lineup_slot == "BE"]
        depth_lines.append(f"  {position}: {len(starters)} starting, {len(bench)} bench")

        healthy_bench = any(
            p.injury_status in HEALTHY_STATUSES and p.pro_opponent != "BYE" for p in bench
        )
        for starter in starters:
            at_risk = starter.injury_status not in HEALTHY_STATUSES or starter.pro_opponent == "BYE"
            if at_risk and not healthy_bench:
                reason = "BYE" if starter.pro_opponent == "BYE" else starter.injury_status
                risk_lines.append(f"  {position}: {starter.name} ({reason}), no healthy bench behind them")

        for p in sorted(players, key=lambda p: p.next_bye_week or 0):
            if p.next_bye_week:
                bye_lines.append(f"  {position}: {p.name} on bye in week {p.next_bye_week}")

    section = "ROSTER DEPTH (bench count by position):\n" + "\n".join(depth_lines)
    if risk_lines:
        section += "\n\nNO BACKUP AT RISK:\n" + "\n".join(risk_lines)
    if bye_lines:
        section += "\n\nUPCOMING BYES (next 4 weeks):\n" + "\n".join(bye_lines)
    return section


def _format_live_block(snapshot: LeagueSnapshot) -> str:
    as_of = datetime.now().strftime("%I:%M %p").lstrip("0")
    return (
        f"LIVE - WEEK {snapshot.week} IN PROGRESS (as of {as_of})\n"
        f"Live score: {snapshot.team_name} {snapshot.live_score} - "
        f"{snapshot.opponent_name} {snapshot.live_opponent_score}\n\n"
    )


def _format_news_section(players: list) -> str:
    lines = []
    for p in players:
        if not p.news:
            continue
        lines.append(f"{p.name} ({p.position}):")
        for item in p.news:
            age = format_age(item.published)
            headline = f"{item.headline} {item.description}".strip()
            lines.append(f"  - [{age}] {headline}")
    return "\n".join(lines)


def _format_podcast_section(podcast_excerpts: list) -> str:
    if not podcast_excerpts:
        return ""
    lines = ["PODCAST ANALYSIS - The Fantasy Footballers:"]
    for excerpt in podcast_excerpts:
        lines.append(
            f"- {excerpt['player']} [{excerpt['episode']} @ {excerpt['timestamp']}]: {excerpt['text']}"
        )
    return "\n".join(lines)


def _format_stash_signals(snapshot: LeagueSnapshot) -> str:
    """Free agents whose ownership is running well ahead of their start
    rate, across every pool (skill positions, K, D/ST) - a sign the league
    is already quietly stashing them before their role is starting-caliber
    yet. Computed directly rather than left for the model to eyeball across
    every position's pool, where the single biggest gap is easy to miss."""
    pools = list(snapshot.free_agents_by_position.values()) + [
        snapshot.free_agent_kickers, snapshot.free_agent_defenses,
    ]
    signals = [
        (p, p.percent_owned - p.percent_started)
        for pool in pools for p in pool
    ]
    signals = [(p, gap) for p, gap in signals if gap >= STASH_GAP_THRESHOLD]
    signals.sort(key=lambda pair: pair[1], reverse=True)
    if not signals:
        return ""

    lines = [
        f"  {p.name} ({p.position}): owned {p.percent_owned}%, started {p.percent_started}% "
        f"(+{gap:.1f} gap)"
        for p, gap in signals[:5]
    ]
    return (
        "OWNERSHIP MOMENTUM (owned% well ahead of started% - league is "
        "already stashing this player):\n" + "\n".join(lines)
    )


def _format_free_agents_by_position(snapshot: LeagueSnapshot) -> str:
    """One labeled section per skill position rather than a single list
    sorted across all of them, so the top available QB/TE can't get buried
    under a flood of higher-owned RBs/WRs."""
    sections = []
    for position, players in snapshot.free_agents_by_position.items():
        lines = "\n".join(_format_player(p) for p in players[:5])
        sections.append(f"TOP AVAILABLE FREE AGENT {position}s:\n{lines}")
    return "\n\n".join(sections)


def build_lineup_prompt(snapshot: LeagueSnapshot, podcast_excerpts: list = None,
                         feedback: str = "") -> str:
    roster_lines = "\n".join(_format_player(p) for p in snapshot.roster)
    opponent_lines = "\n".join(_format_player(p) for p in snapshot.opponent_starters)

    news_section = _format_news_section(snapshot.roster)
    podcast_section = _format_podcast_section(
        [e for e in (podcast_excerpts or []) if e.get("kind") == "start_sit"]
    )

    parts = []
    if feedback:
        parts.append(f"{feedback}\n\n")
    if snapshot.any_games_started:
        parts.append(_format_live_block(snapshot))

    parts.append(
        f"League: {snapshot.league_name}\n"
        f"Team: {snapshot.team_name} (record {snapshot.team_record})\n"
        f"Week: {snapshot.week}\n"
        f"Scoring so far this season (oldest to most recent): {snapshot.team_weekly_scores}\n\n"
        f"THIS WEEK'S OPPONENT: {snapshot.opponent_name} (record {snapshot.opponent_record})\n"
        f"OPPONENT'S PROJECTED STARTERS:\n{opponent_lines}\n\n"
        f"CURRENT ROSTER:\n{roster_lines}\n"
    )

    if news_section:
        parts.append(f"\nRECENT PLAYER NEWS (last 7 days):\n{news_section}\n")
    if podcast_section:
        parts.append(f"\n{podcast_section}\n")

    return "".join(parts)


def build_waiver_prompt(snapshot: LeagueSnapshot, podcast_excerpts: list = None) -> str:
    roster_lines = "\n".join(_format_player(p) for p in snapshot.roster)
    fa_sections = _format_free_agents_by_position(snapshot)
    k_lines = "\n".join(_format_player(p) for p in snapshot.free_agent_kickers[:5])
    dst_lines = "\n".join(_format_player(p) for p in snapshot.free_agent_defenses[:5])

    news_players = (
        snapshot.roster
        + [p for pool in snapshot.free_agents_by_position.values() for p in pool[:5]]
        + snapshot.free_agent_kickers[:5] + snapshot.free_agent_defenses[:5]
    )
    news_section = _format_news_section(news_players)
    stash_section = _format_stash_signals(snapshot)
    podcast_section = _format_podcast_section(
        [e for e in (podcast_excerpts or []) if e.get("kind") == "waivers"]
    )

    parts = [
        f"League: {snapshot.league_name}\n"
        f"Team: {snapshot.team_name} (record {snapshot.team_record})\n"
        f"Week: {snapshot.week}\n\n"
        f"CURRENT ROSTER:\n{roster_lines}\n\n"
        f"{_format_roster_depth(snapshot)}\n\n"
        f"{fa_sections}\n\n"
        f"AVAILABLE FREE AGENT KICKERS:\n{k_lines}\n\n"
        f"AVAILABLE FREE AGENT DEFENSES/D-ST:\n{dst_lines}\n"
    ]

    if stash_section:
        parts.append(f"\n{stash_section}\n")
    if news_section:
        parts.append(f"\nRECENT PLAYER NEWS (last 7 days):\n{news_section}\n")
    if podcast_section:
        parts.append(f"\n{podcast_section}\n")

    return "".join(parts)


def _get_anthropic_recommendations(system_prompt: str, user_prompt: str, client=None) -> str:
    from anthropic import Anthropic

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    client = client or Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=2048,
        # This task is a single structured-output pass, not multi-step
        # reasoning - extended thinking on some models otherwise defaults to
        # claiming the whole token budget for itself, leaving nothing for
        # the actual report.
        thinking={"type": "disabled"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return next(block.text for block in message.content if block.type == "text")


def _get_ollama_recommendations(system_prompt: str, user_prompt: str, client=None) -> str:
    from ollama import Client

    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    client = client or Client(host=host)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


def _dispatch(system_prompt: str, user_prompt: str, client=None) -> str:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()

    if provider == "ollama":
        return _get_ollama_recommendations(system_prompt, user_prompt, client)
    if provider == "anthropic":
        return _get_anthropic_recommendations(system_prompt, user_prompt, client)

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic' or 'ollama'.")


def get_lineup_recommendations(snapshot: LeagueSnapshot, podcast_excerpts: list = None,
                                feedback: str = "", client=None) -> str:
    user_prompt = build_lineup_prompt(snapshot, podcast_excerpts, feedback)
    return _dispatch(LINEUP_SYSTEM_PROMPT, user_prompt, client)


def get_waiver_recommendations(snapshot: LeagueSnapshot, podcast_excerpts: list = None,
                                client=None) -> str:
    user_prompt = build_waiver_prompt(snapshot, podcast_excerpts)
    return _dispatch(WAIVER_SYSTEM_PROMPT, user_prompt, client)
