# ff-agent

Pulls your roster, injury status, and available free agents from ESPN
Fantasy Football (across any number of leagues) and asks Claude for
start/sit and waiver-wire recommendations. Run it manually once a week,
e.g. the morning waivers process.

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
   - `ANTHROPIC_API_KEY` - from https://console.anthropic.com.

4. Edit `config.yaml` with your three leagues. For each one you need:

   - `league_id` - in the URL when viewing the league, e.g.
     `...leagueId=123456...`.
   - `year` - the season year.
   - `team_id` - in the URL when viewing your team, e.g.
     `...teamId=4...`.

## Run

```
python main.py
```

This prints a recommendation report for each league to the console and
saves a copy under `reports/`.

## Notes

- `espn_s2` and `SWID` are session cookies and will eventually expire -
  if fetches start failing with an auth error, grab fresh values from your
  browser and update `.env`.
- The recommendations are based only on ESPN's own stats, projections, and
  injury designations - there's no external news feed wired in yet. If you
  want headline-level news (e.g. beat-reporter reports) factored in, that
  would need a separate news source added to `src/`.
