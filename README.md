# ff-agent

Pulls your roster, injury status, live in-week scoring, same-day player
news, and available free agents from ESPN Fantasy Football (across any
number of leagues) - plus this week's relevant Fantasy Footballers podcast
analysis - and asks Claude for recommendations as two focused calls: one
for this week's starting lineup and things to watch, one for waiver-wire
pickups. Splitting them keeps each call's prompt scoped to only the data
it needs and keeps a long waiver take from crowding out the lineup call,
or vice versa.
It also tracks its own past calls and grades them once the results are in,
so later runs can calibrate against how its projections have actually
played out. Run it manually whenever you want an update - the morning
waivers process, mid-week, or Sunday morning before lineups lock; it just
reflects whatever has happened by the time you run it.

## Setup

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Get your ESPN session cookies (required for private leagues, which is
   almost all leagues):

   - Log into https://fantasy.espn.com in your browser.
   - Open dev tools -> Application/Storage -> Cookies -> `fantasy.espn.com`.
   - Copy the values of the `espn_s2` and `SWID` cookies (SWID includes the
     curly braces, e.g. `{ABC-123...}`).

3. Copy `.env.example` to `.env` and fill in:

   - `ESPN_S2`, `SWID` - from step 2. These are the same across all your
     ESPN leagues since they're tied to your ESPN login, not a specific league.
   - `LLM_PROVIDER` - `anthropic` (default) or `ollama`.
   - `ANTHROPIC_API_KEY` - from https://console.anthropic.com. Required if
     `LLM_PROVIDER=anthropic`.
   - `OLLAMA_MODEL`, `OLLAMA_HOST` - only used if `LLM_PROVIDER=ollama`. Requires
     a local Ollama server running (`ollama serve`) with the model pulled
     (`ollama pull llama3.1`).

4. Edit `config.yaml` with your three leagues. For each one you need:

   - `league_id` - in the URL when viewing the league, e.g.
     `...leagueId=123456...`.
   - `year` - the season year.
   - `team_id` - in the URL when viewing your team, e.g.
     `...teamId=4...`.

   The optional `podcast:` block controls the Fantasy Footballers podcast
   integration (see below) - set `enabled: false` to turn it off.

5. Podcast analysis needs [`yt-dlp`](https://github.com/yt-dlp/yt-dlp)
   installed and on your `PATH` (`brew install yt-dlp`). If it's missing,
   or a network/captions fetch fails, that run just skips the podcast
   section - it never blocks a report.

## Run

```
python main.py
```

This prints a recommendation report for each league to the console and
saves a copy under `reports/`. Run it as often as you like during the
week - Thursday night, Sunday morning, whenever - it reflects whatever has
actually happened by the time you run it (see "Live data" below).

To run just one of the two agents instead of both:

```
python main.py --only lineup    # start/sit + things to watch only
python main.py --only waivers   # waiver-wire pickups only
```

To force a specific past week to be (re)graded - e.g. after ESPN applies a
stat correction, or to grade the season's final week once
`league.current_week` stops advancing - run:

```
python main.py --regrade-week 3
```

This regrades week 3 for every league and exits without fetching new
recommendations.

## What's in a report

A report has two parts, each written by its own focused call to the model:

- **Starting Lineup Changes / Things to Watch.** Start/sit calls for any
  roster spot with a real decision to make, weighing recent-week trends,
  projections, and the opponent's starters at the same position - plus
  anything else worth watching (injuries, byes, hot/cold streaks).
- **Waiver Wire Pickups.** Free agents are pulled as separate pools per
  position (QB/RB/WR/TE, plus dedicated K and D/ST pools) so a strong
  option at a thin position can't get crowded out by a deeper one. Picks
  are prioritized by roster need (bench depth, single-point-of-failure
  risk) over raw player quality. Kicker and D/ST are treated as weekly
  streaming spots - the model names whoever has the best matchup every
  week, not just when there's an injury - but those two always rank last,
  since a QB/RB/WR/TE pickup that fills a real need outranks a one-week
  streamer. The model is also asked to flag a speculative "stash for
  later" pick when a thin position has a bye coming up in the next few
  weeks, or when a free agent's ownership is running well ahead of their
  start rate (a sign the league is already quietly stashing them). In a
  league that uses FAAB (free-agent budget bidding, detected from your
  league's own settings - no config needed), each pick also gets a
  suggested dollar bid sized against your remaining budget; leagues on
  rotating waiver priority just get the priority order.

Supporting both of the above:

- **Live data.** If any of this week's games have started, a `LIVE` block
  reflects real, already-scored points (via ESPN's live box scores) rather
  than pre-week projections, and the model is told to treat those players
  as locked - no more "bench him" for someone whose game already happened.
- **Same-day news.** Recent player news (injury updates, beat-reporter
  notes) is pulled from ESPN's own player-news feed, joined by player ID -
  more current than the static injury designation, which can go stale.
- **Podcast analysis.** This week's relevant Fantasy Footballers episodes
  are found on their YouTube channel and classified as start/sit or
  waiver-wire episodes; their auto-generated captions are searched for
  mentions of your roster and top free agents, and each excerpt only feeds
  the matching half of the report. It's expert opinion, not data - the
  report attributes it explicitly, and captions are auto-transcribed so
  player names can be garbled. Captions are cached under `cache/podcast/`
  so re-runs in the same week don't re-fetch.
- **Past results.** Each report's start/sit and waiver calls are logged to
  `history/<league>_<season>.jsonl`. Once a week's games are over, those
  calls get graded against what actually happened, and a summary (bench
  regret vs. the best legal lineup, and how far off ESPN's projections ran
  by position) feeds into the next lineup call as calibration. `history/`
  is this tool's only durable state, so unlike `reports/` it isn't
  gitignored.

## Notes

- `espn_s2` and `SWID` are session cookies and will eventually expire -
  if fetches start failing with an auth error, grab fresh values from your
  browser and update `.env`.
- All of the above - live scoring, news, podcast analysis, past-results
  grading - degrade gracefully: a failure in any one of them prints a
  warning and the run continues with whatever it does have.
