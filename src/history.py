"""Tracks the model's own start/sit and waiver calls across weeks, grades
them once the actual results are in, and computes ESPN-derived signals
(bench regret, projection accuracy) that feed back into future prompts.

The model's report ends with a machine-readable `<!--calls ...-->` block.
Capture resolves every named player against the snapshot in hand (or the
league's own name/id map) at record time and stores a canonical player_id,
so grading later joins on id rather than fuzzy name matching. Nothing here
raises out of `grade_pending` - a grading failure should never take down a
run that's just trying to fetch this week's recommendations.

One file per league per season: history/<league_slug>_<season>.jsonl.
"""

import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime

from .espn_client import BENCH_SLOTS

HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history")

CALLS_BLOCK_RE = re.compile(r"<!--calls\s*(.*?)-->", re.DOTALL)

# A win by less than this many points is closer to a coin flip than a real
# read on the call, so it's scored as a push rather than a win or a loss.
PUSH_THRESHOLD = 2.0


def league_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# --- storage -----------------------------------------------------------

def history_path(slug: str, season: int) -> str:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    return os.path.join(HISTORY_DIR, f"{slug}_{season}.jsonl")


def load_records(slug: str, season: int) -> list:
    path = history_path(slug, season)
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_record(slug: str, season: int, record: dict) -> None:
    with open(history_path(slug, season), "a") as f:
        f.write(json.dumps(record) + "\n")


def rewrite_records(slug: str, season: int, records: list) -> None:
    path = history_path(slug, season)
    fd, tmp_path = tempfile.mkstemp(dir=HISTORY_DIR, prefix=".tmp_history_")
    try:
        with os.fdopen(fd, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


# --- capture -------------------------------------------------------------

def extract_calls_block(report_text: str):
    """Splits a report into (prose, raw_json_or_None)."""
    match = CALLS_BLOCK_RE.search(report_text)
    if not match:
        return report_text.strip(), None
    prose = report_text[: match.start()].rstrip()
    return prose, match.group(1).strip()


def parse_calls(raw: str):
    """Returns (calls, capture_error). calls is [] when raw is missing or
    malformed - a bad block should never cost the report itself."""
    if raw is None:
        return [], "no calls block found"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return [], f"invalid JSON: {e}"

    calls = []
    for entry in data.get("start_sit", []):
        calls.append({
            "kind": "start_sit",
            "slot": entry.get("slot", ""),
            "start_name": entry.get("start"),
            "sit_name": entry.get("sit"),
        })
    for entry in data.get("waivers", []):
        calls.append({
            "kind": "waiver",
            "add_name": entry.get("add"),
            "drop_name": entry.get("drop"),
            "faab_bid": entry.get("faab_bid"),
        })
    return calls, None


def _normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower().strip()
    name = name.replace(".", "").replace("'", "")
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


def build_player_index(snapshot) -> dict:
    """norm_name -> (player_id, source), covering everyone the model could
    plausibly have named: the full roster and every free-agent pool, not
    just the slice that made it into the prompt."""
    index = {}
    for p in snapshot.roster:
        index[_normalize_name(p.name)] = (p.player_id, "roster")
    free_agent_groups = (
        list(snapshot.free_agents_by_position.values())
        + [snapshot.free_agent_kickers, snapshot.free_agent_defenses]
    )
    for group in free_agent_groups:
        for p in group:
            index.setdefault(_normalize_name(p.name), (p.player_id, "free_agents"))
    return index


def resolve_player(name: str, index: dict, player_map_by_name: dict = None) -> dict:
    if not name:
        return None
    norm = _normalize_name(name)
    if norm in index:
        player_id, source = index[norm]
        return {"name": name, "player_id": player_id, "resolved_from": source}
    if player_map_by_name and norm in player_map_by_name:
        return {"name": name, "player_id": player_map_by_name[norm], "resolved_from": "player_map"}
    return {"name": name, "player_id": None, "resolved_from": "unresolved"}


def record_calls(slug: str, season: int, week: int, snapshot, report_texts: list, *,
                  player_map: dict = None, model: str = None, report_path: str = None) -> dict:
    """report_texts is one raw model response per agent (lineup, waivers,
    ...) - each carries its own trailing <!--calls--> block, so every one
    gets extracted and merged into a single record."""
    parsed_calls, capture_errors = [], []
    for report_text in report_texts:
        _, raw = extract_calls_block(report_text)
        calls, error = parse_calls(raw)
        parsed_calls.extend(calls)
        if error:
            capture_errors.append(error)
    capture_error = "; ".join(capture_errors) if capture_errors else None

    index = build_player_index(snapshot)
    player_map_by_name = {}
    if player_map:
        for key, value in player_map.items():
            if isinstance(key, str) and isinstance(value, int):
                player_map_by_name[_normalize_name(key)] = value

    resolved_calls = []
    for call in parsed_calls:
        if call["kind"] == "start_sit":
            resolved_calls.append({
                "kind": "start_sit",
                "slot": call["slot"],
                "start": resolve_player(call["start_name"], index, player_map_by_name),
                "sit": resolve_player(call["sit_name"], index, player_map_by_name),
            })
        else:
            resolved_calls.append({
                "kind": "waiver",
                "add": resolve_player(call["add_name"], index, player_map_by_name),
                "drop": (resolve_player(call["drop_name"], index, player_map_by_name)
                         if call["drop_name"] else None),
                "faab_bid": call.get("faab_bid"),
            })

    record = {
        "schema_version": 1,
        "league_slug": slug,
        "season": season,
        "week": week,
        "recorded_at": datetime.now().isoformat(),
        "first_kickoff": snapshot.first_kickoff.isoformat() if snapshot.first_kickoff else None,
        "report_path": report_path,
        "model": model,
        "capture_error": capture_error,
        "calls": resolved_calls,
        "graded": False,
        "graded_at": None,
        "grading_error": None,
        "results": None,
        "team_week": None,
    }
    append_record(slug, season, record)
    return record


# --- grading ---------------------------------------------------------------

def _wanted_ids(record: dict) -> set:
    ids = set()
    for call in record["calls"]:
        for key in ("start", "sit", "add", "drop"):
            entry = call.get(key)
            if entry and entry.get("player_id"):
                ids.add(entry["player_id"])
    return ids


def build_week_points(league, week: int, wanted_ids: set, team_id: int, player_team_cache: dict):
    """{player_id: {"points", "slot", "source"}} for wanted_ids that week,
    plus this team's own lineup that week (for team_week metrics). Actual
    points come from box_scores; a free agent nobody rostered falls back to
    player_info. Presence of a 'points' key - not its value - is what marks
    a player as having actually played that week, since a bye or inactive
    week defaults to 0, indistinguishable from a real zero."""
    box_scores = league.box_scores(week, player_team_cache=player_team_cache)
    own_lineup = []
    points_index = {}
    for box in box_scores:
        home_id = getattr(box.home_team, "team_id", None)
        away_id = getattr(box.away_team, "team_id", None)
        if team_id in (home_id, away_id):
            own_lineup = box.home_lineup if home_id == team_id else box.away_lineup
        for p in box.home_lineup + box.away_lineup:
            week_stats = p.stats.get(week, {})
            if "points" in week_stats:
                points_index[p.playerId] = {
                    "points": round(week_stats["points"], 2),
                    "slot": p.slot_position,
                    "source": "box_scores",
                }

    missing = [pid for pid in wanted_ids if pid not in points_index]
    if missing:
        fallback = league.player_info(playerId=missing) or []
        if not isinstance(fallback, list):
            fallback = [fallback]
        for p in fallback:
            week_stats = p.stats.get(week, {})
            if "points" in week_stats:
                points_index[p.playerId] = {
                    "points": round(week_stats["points"], 2),
                    "slot": None,
                    "source": "player_info",
                }

    return points_index, own_lineup


def grade_record(record: dict, points_index: dict) -> list:
    results = []
    for call in record["calls"]:
        if call["kind"] == "start_sit":
            results.append(_grade_start_sit(call, points_index))
        else:
            results.append(_grade_waiver(call, points_index))
    return results


def _verdict(delta: float) -> str:
    if abs(delta) < PUSH_THRESHOLD:
        return "push"
    return "correct" if delta > 0 else "wrong"


def _grade_start_sit(call: dict, points_index: dict) -> dict:
    start_id = (call.get("start") or {}).get("player_id")
    sit_id = (call.get("sit") or {}).get("player_id")
    start_info = points_index.get(start_id)
    sit_info = points_index.get(sit_id)
    if start_info is None or sit_info is None:
        return {**call, "verdict": "ungraded", "ungraded_reason": "missing actual points"}

    delta = round(start_info["points"] - sit_info["points"], 2)
    return {
        **call,
        "start_points": start_info["points"],
        "sit_points": sit_info["points"],
        "delta": delta,
        "start_slot_actual": start_info.get("slot"),
        "sit_slot_actual": sit_info.get("slot"),
        "followed": start_info.get("slot") not in (None, "BE", "IR"),
        "verdict": _verdict(delta),
        "source": {"start": start_info["source"], "sit": sit_info["source"]},
    }


def _grade_waiver(call: dict, points_index: dict) -> dict:
    add_id = (call.get("add") or {}).get("player_id")
    add_info = points_index.get(add_id)
    if add_info is None:
        return {**call, "verdict": "ungraded", "ungraded_reason": "missing actual points"}

    drop = call.get("drop")
    if not drop or not drop.get("player_id"):
        return {**call, "add_points": add_info["points"], "verdict": "ungraded",
                "ungraded_reason": "no drop named"}

    drop_info = points_index.get(drop["player_id"])
    if drop_info is None:
        return {**call, "add_points": add_info["points"], "verdict": "ungraded",
                "ungraded_reason": "missing actual points"}

    delta = round(add_info["points"] - drop_info["points"], 2)
    return {
        **call,
        "add_points": add_info["points"],
        "drop_points": drop_info["points"],
        "delta": delta,
        "verdict": _verdict(delta),
    }


def _optimal_lineup(starting_slots: list, candidates: list):
    """Exact max-points assignment of candidates to starting_slots via DP
    over (slot_index, used-player bitmask). Roster/slot counts here are
    small (~15-20 players, ~9-10 slots), so this is fast and exact - a
    greedy fill gets FLEX-vs-RB2/WR2 wrong, which is precisely the case
    that matters.

    Returns (total_points, assignment) where assignment[slot_index] is the
    chosen candidate's index into `candidates`, or None.
    """
    points = [round(getattr(c, "points", 0) or 0, 2) for c in candidates]
    memo = {}

    def solve(slot_idx, mask):
        if slot_idx == len(starting_slots):
            return 0.0
        key = (slot_idx, mask)
        if key in memo:
            return memo[key]
        label = starting_slots[slot_idx]
        best = solve(slot_idx + 1, mask)
        for i, c in enumerate(candidates):
            if mask & (1 << i) or label not in c.eligibleSlots:
                continue
            val = points[i] + solve(slot_idx + 1, mask | (1 << i))
            if val > best:
                best = val
        memo[key] = best
        return best

    total = solve(0, 0)

    assignment = []
    mask = 0
    for slot_idx, label in enumerate(starting_slots):
        target = solve(slot_idx, mask)
        chosen = None
        if target != solve(slot_idx + 1, mask):
            for i, c in enumerate(candidates):
                if mask & (1 << i) or label not in c.eligibleSlots:
                    continue
                if abs(points[i] + solve(slot_idx + 1, mask | (1 << i)) - target) < 1e-6:
                    chosen = i
                    break
        assignment.append(chosen)
        if chosen is not None:
            mask |= (1 << chosen)

    return total, assignment


def _team_week_metrics(own_lineup: list, week: int) -> dict:
    starting = [p for p in own_lineup if getattr(p, "slot_position", "") not in BENCH_SLOTS]
    starting_slots = [p.slot_position for p in starting]
    candidates = [p for p in own_lineup if getattr(p, "slot_position", "") != "IR"]

    started_points = round(sum(round(getattr(p, "points", 0) or 0, 2) for p in starting), 2)
    optimal_points, assignment = _optimal_lineup(starting_slots, candidates)

    optimal_by_slot, actual_by_slot = defaultdict(float), defaultdict(float)
    for slot_idx, label in enumerate(starting_slots):
        idx = assignment[slot_idx]
        if idx is not None:
            optimal_by_slot[label] += round(getattr(candidates[idx], "points", 0) or 0, 2)
    for p in starting:
        actual_by_slot[p.slot_position] += round(getattr(p, "points", 0) or 0, 2)

    regret_by_slot = {
        label: round(optimal_by_slot.get(label, 0) - actual_by_slot.get(label, 0), 1)
        for label in set(starting_slots)
    }
    regret_slots = sorted(
        (label for label, regret in regret_by_slot.items() if regret > 0.5),
        key=lambda label: regret_by_slot[label], reverse=True,
    )[:2]

    errors = defaultdict(list)
    for p in own_lineup:
        week_stats = p.stats.get(week, {})
        if "points" in week_stats and "projected_points" in week_stats:
            errors[p.position].append(week_stats["projected_points"] - week_stats["points"])
    projection_error_by_position = {
        pos: {"n": len(deltas), "mean_signed": round(sum(deltas) / len(deltas), 2)}
        for pos, deltas in errors.items()
    }

    return {
        "started_points": started_points,
        "optimal_points": round(optimal_points, 2),
        "bench_regret": round(optimal_points - started_points, 2),
        "regret_slots": regret_slots,
        "projection_error_by_position": projection_error_by_position,
    }


def grade_pending(league, slug: str, season: int, team_id: int, force_week: int = None) -> list:
    """Grades every ungraded record for a completed week. Never raises - a
    single week's grading failure is recorded on that record and skipped,
    since a report run should never be blocked by history bookkeeping.
    Pass force_week to (re)grade one specific week regardless of its
    current graded state (covers --regrade and the post-season case where
    league.current_week stops advancing past the final week)."""
    records = load_records(slug, season)
    if not records:
        return []

    if force_week is not None:
        targets = [r for r in records if r["week"] == force_week]
    else:
        targets = [r for r in records if not r.get("graded") and r["week"] < league.current_week]
    if not targets:
        return []

    player_team_cache = {}
    graded = []
    for record in sorted(targets, key=lambda r: r["week"]):
        week = record["week"]
        if week >= league.current_week and force_week is None:
            continue
        try:
            points_index, own_lineup = build_week_points(
                league, week, _wanted_ids(record), team_id, player_team_cache,
            )
            record["results"] = grade_record(record, points_index)
            record["team_week"] = _team_week_metrics(own_lineup, week) if own_lineup else None
            record["graded"] = True
            record["graded_at"] = datetime.now().isoformat()
            record["grading_error"] = None
        except Exception as e:
            print(f"  Warning: failed to grade week {week}: {e}", file=sys.stderr)
            record["grading_error"] = str(e)
        graded.append(record)

    rewrite_records(slug, season, records)
    return graded


# --- feedback ---------------------------------------------------------------

def format_scorecard(slug: str, season: int, max_weeks: int = 4) -> str:
    records = load_records(slug, season)
    graded = sorted(
        (r for r in records if r.get("graded") and r.get("team_week")),
        key=lambda r: r["week"],
    )
    if not graded:
        return ""
    recent = graded[-max_weeks:]

    lines = [
        "PAST RESULTS (computed from this team's own history against ESPN's "
        "actual scoring - use for calibration, don't over-correct on a "
        "handful of weeks):",
        "",
        "Bench regret (optimal legal lineup vs. what was actually started):",
    ]
    regret_total = 0.0
    for r in recent:
        tw = r["team_week"]
        slots = f" ({', '.join(tw['regret_slots'])})" if tw.get("regret_slots") else ""
        lines.append(f"  Week {r['week']}: {tw['bench_regret']:+.1f}{slots}")
        regret_total += tw["bench_regret"]
    lines.append(f"  Avg over {len(recent)} wk(s): {regret_total / len(recent):+.1f}/wk")
    lines.append("")

    lines.append("ESPN projection error, signed (projected minus actual; positive = ESPN ran high):")
    combined = defaultdict(lambda: [0, 0.0])
    for r in recent:
        for pos, stat in r["team_week"].get("projection_error_by_position", {}).items():
            combined[pos][0] += stat["n"]
            combined[pos][1] += stat["mean_signed"] * stat["n"]
    for pos, (n, total) in sorted(combined.items()):
        if n:
            lines.append(f"  {pos}: {total / n:+.1f} (n={n})")

    deltas, followed, total_calls = [], 0, 0
    for r in recent:
        for result in r.get("results") or []:
            if result["kind"] != "start_sit":
                continue
            total_calls += 1
            if result.get("followed"):
                followed += 1
            if "delta" in result:
                deltas.append(result["delta"])
    if total_calls:
        lines.append("")
        record_str = f"{sum(1 for d in deltas if d > 0)}-{sum(1 for d in deltas if d < 0)}"
        lines.append(
            f"Your own start/sit calls, {len(recent)} wk(s): {record_str} "
            f"(full detail in history/), adherence {followed}/{total_calls}"
        )

    return "\n".join(lines)
