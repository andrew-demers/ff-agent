"""CLI entrypoint: loops over every league in config.yaml, pulls the current
roster/free-agent/live/news data from ESPN plus this week's relevant
Fantasy Footballers podcast analysis, grades any of its own past calls that
have now played out, asks the LLM for start/sit + waiver recommendations,
and writes a report per league.

Usage:
    python main.py
    python main.py --regrade-week 3   # force-(re)grade one past week for
                                       # every league and exit without
                                       # fetching new recommendations; also
                                       # the escape hatch for grading the
                                       # final week of a season, since
                                       # league.current_week stops advancing
                                       # once the season ends.
"""

import argparse
import os
import sys

import yaml
from dotenv import load_dotenv

from src import analyst, history, podcast
from src.espn_client import build_league, fetch_league_snapshot
from src.report import write_report

DEFAULT_PODCAST_CHANNEL = "https://www.youtube.com/@TheFantasyFootballers/videos"
PODCAST_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "podcast")


def _news_and_podcast_player_names(snapshot) -> list:
    players = (
        snapshot.roster + snapshot.free_agents[:20]
        + snapshot.free_agent_kickers[:5] + snapshot.free_agent_defenses[:5]
    )
    return [p.name for p in players]


def main():
    parser = argparse.ArgumentParser(description="Weekly fantasy football recommendations")
    parser.add_argument(
        "--regrade-week", type=int, default=None,
        help="Force-(re)grade this week for every league and exit, skipping new recommendations.",
    )
    args = parser.parse_args()

    load_dotenv()

    espn_s2 = os.environ.get("ESPN_S2")
    swid = os.environ.get("SWID")
    if not espn_s2 or not swid:
        print("Missing ESPN_S2 or SWID in your .env file. See .env.example.", file=sys.stderr)
        sys.exit(1)

    llm_provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Missing ANTHROPIC_API_KEY in your .env file. See .env.example.", file=sys.stderr)
        sys.exit(1)
    model_name = (
        os.environ.get("OLLAMA_MODEL", analyst.DEFAULT_OLLAMA_MODEL) if llm_provider == "ollama"
        else os.environ.get("ANTHROPIC_MODEL", analyst.DEFAULT_ANTHROPIC_MODEL)
    )

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    podcast_cfg = config.get("podcast") or {}
    podcast_enabled = podcast_cfg.get("enabled", True)
    podcast_channel = podcast_cfg.get("channel", DEFAULT_PODCAST_CHANNEL)
    podcast_max_episodes = podcast_cfg.get("max_episodes", 8)

    for league_cfg in config["leagues"]:
        name = league_cfg["name"]
        team_id = league_cfg["team_id"]
        slug = history.league_slug(name)
        print(f"\n=== {name} ===")

        try:
            league = build_league(
                league_id=league_cfg["league_id"], year=league_cfg["year"],
                espn_s2=espn_s2, swid=swid,
            )
        except Exception as e:
            print(f"  Failed to connect to league: {e}", file=sys.stderr)
            continue

        if args.regrade_week is not None:
            graded = history.grade_pending(league, slug, league.year, team_id,
                                            force_week=args.regrade_week)
            print(f"  Regraded week {args.regrade_week}." if graded
                  else f"  No history record found for week {args.regrade_week}.")
            continue

        try:
            history.grade_pending(league, slug, league.year, team_id)
        except Exception as e:
            print(f"  Warning: grading past weeks failed: {e}", file=sys.stderr)

        try:
            snapshot = fetch_league_snapshot(name=name, league=league, team_id=team_id)
        except Exception as e:
            print(f"  Failed to fetch data: {e}", file=sys.stderr)
            continue

        scorecard = history.format_scorecard(slug, league.year)

        podcast_excerpts = []
        if podcast_enabled:
            podcast_excerpts = podcast.fetch_podcast_excerpts(
                _news_and_podcast_player_names(snapshot), snapshot.week,
                podcast_channel, PODCAST_CACHE_DIR, podcast_max_episodes,
            )

        print(f"  Week {snapshot.week} - {snapshot.team_name}. Asking {llm_provider} for recommendations...")
        recommendations = analyst.get_recommendations(snapshot, podcast_excerpts, scorecard)

        path = write_report(snapshot, recommendations)
        history.record_calls(
            slug, league.year, snapshot.week, snapshot, recommendations,
            player_map=league.player_map, model=model_name, report_path=path,
        )

        prose, _ = history.extract_calls_block(recommendations)
        print(f"  Report written to {path}")
        print("\n" + prose + "\n")


if __name__ == "__main__":
    main()
