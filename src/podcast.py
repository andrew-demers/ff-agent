"""Pulls same-week start/sit and waiver-wire analysis from The Fantasy
Footballers podcast, as expert opinion to weigh alongside ESPN's own data.

Episode descriptions are marketing copy with no player-level verdicts, and
the show's own site blocks non-browser traffic, so the only same-day path
is YouTube's auto-generated captions via yt-dlp (already installed on this
machine). Captions are auto-generated, not human, so player names are
sometimes garbled - matching falls back to fuzzy string matching, and the
system prompt is told to treat this as approximate.

Every step here is best-effort: a missing yt-dlp, a network failure, or an
episode whose captions haven't posted yet all just mean no podcast section
in the prompt, never a broken run.
"""

import difflib
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict

YT_DLP = "yt-dlp"
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

MIN_EPISODE_SECONDS = 600  # drop Shorts/clips, keep full episodes
DISCOVER_TIMEOUT = 30
CAPTIONS_TIMEOUT = 120

WAIVER_RE = re.compile(r"week\s+\d+\s+waiver", re.I)
START_SIT_RE = re.compile(r"start\s*/?\s*sit|starts of the week", re.I)
WEEK_NUM_RE = re.compile(r"week\s+(\d+)", re.I)

WINDOW_BEFORE_SECONDS = 20
WINDOW_AFTER_SECONDS = 40
MAX_EXCERPTS_PER_PLAYER = 2
MAX_EXCERPT_CHARS = 600

SRT_TIME_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)


def discover_episodes(channel: str, max_episodes: int = 8) -> list:
    cmd = [
        YT_DLP, "--flat-playlist", "--print", "%(id)s|%(duration)s|%(title)s",
        "--playlist-end", str(max_episodes), channel,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=DISCOVER_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:300] or "yt-dlp failed to list episodes")

    episodes = []
    for line in result.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        video_id, duration_str, title = parts
        if not VIDEO_ID_RE.match(video_id):
            continue
        try:
            duration = float(duration_str)
        except ValueError:
            continue
        if duration < MIN_EPISODE_SECONDS:
            continue
        episodes.append({"video_id": video_id, "title": title, "duration": duration})
    return episodes


def classify_episode(title: str) -> str:
    if WAIVER_RE.search(title):
        return "waivers"
    if START_SIT_RE.search(title):
        return "start_sit"
    return "other"


def select_relevant_episodes(episodes: list, current_week: int) -> list:
    relevant = []
    for ep in episodes:
        kind = classify_episode(ep["title"])
        if kind == "other":
            continue
        match = WEEK_NUM_RE.search(ep["title"])
        if not match or int(match.group(1)) != current_week:
            continue
        relevant.append({**ep, "kind": kind})
    return relevant


def fetch_captions(video_id: str, cache_dir: str) -> list:
    """Returns [(start_seconds, end_seconds, text), ...], cached to disk so
    repeat runs in the same week don't re-fetch."""
    if not VIDEO_ID_RE.match(video_id):
        raise ValueError(f"invalid video id: {video_id!r}")

    os.makedirs(cache_dir, exist_ok=True)
    existing = glob.glob(os.path.join(cache_dir, f"{video_id}*.srt"))
    if not existing:
        url = f"https://www.youtube.com/watch?v={video_id}"
        cmd = [
            YT_DLP, "--skip-download", "--write-auto-subs", "--sub-langs", "en-en",
            "--sub-format", "srt", "-o", os.path.join(cache_dir, f"{video_id}.%(ext)s"), url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=CAPTIONS_TIMEOUT)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip()[:300] or "yt-dlp failed to fetch captions")
        existing = glob.glob(os.path.join(cache_dir, f"{video_id}*.srt"))

    if not existing:
        return []
    return _parse_srt(existing[0])


def _parse_srt(path: str) -> list:
    with open(path, encoding="utf-8", errors="ignore") as f:
        blocks = f.read().split("\n\n")

    cues = []
    for block in blocks:
        lines = [line for line in block.strip().splitlines() if line.strip()]
        time_match, text_lines = None, []
        for line in lines:
            match = SRT_TIME_RE.search(line)
            if match:
                time_match = match
            elif not line.strip().isdigit():
                text_lines.append(line.strip())
        if not time_match or not text_lines:
            continue
        start = _srt_timestamp_to_seconds(*time_match.groups()[0:4])
        end = _srt_timestamp_to_seconds(*time_match.groups()[4:8])
        text = re.sub(r"<[^>]+>", "", " ".join(text_lines))
        cues.append((start, end, text))
    return cues


def _srt_timestamp_to_seconds(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _clean_word(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower())


def _merge_windows(windows: list) -> list:
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def find_player_mentions(cues: list, player_names: list) -> dict:
    """{player_name: [(window_start, window_end), ...]}, merged. Matches on
    last name first; anyone still unmatched gets one fuzzy pass against the
    transcript's actual vocabulary, since auto-captions garble names."""
    last_name_map = defaultdict(list)
    for name in player_names:
        parts = name.split()
        if len(parts) >= 2:
            last_name_map[_clean_word(parts[-1])].append(name)

    hits = defaultdict(list)
    for start, end, text in cues:
        words = {_clean_word(w) for w in text.split()}
        for word in words & last_name_map.keys():
            for name in last_name_map[word]:
                hits[name].append((max(0, start - WINDOW_BEFORE_SECONDS), end + WINDOW_AFTER_SECONDS))

    unmatched = [n for n in player_names if n not in hits]
    if unmatched:
        vocab = set()
        for _, _, text in cues:
            vocab.update(_clean_word(w) for w in text.split())
        for name in unmatched:
            parts = name.split()
            if not parts:
                continue
            close = difflib.get_close_matches(_clean_word(parts[-1]), vocab, n=1, cutoff=0.82)
            if not close:
                continue
            for start, end, text in cues:
                if close[0] in {_clean_word(w) for w in text.split()}:
                    hits[name].append((max(0, start - WINDOW_BEFORE_SECONDS), end + WINDOW_AFTER_SECONDS))

    return {name: _merge_windows(windows) for name, windows in hits.items()}


def _excerpt_text(cues: list, start: float, end: float) -> str:
    text = " ".join(c[2] for c in cues if start <= c[0] <= end)
    return re.sub(r"\s+", " ", text).strip()[:MAX_EXCERPT_CHARS]


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fetch_podcast_excerpts(player_names: list, current_week: int, channel: str,
                            cache_dir: str, max_episodes: int = 8) -> list:
    """Best-effort list of {player, episode, timestamp, text} excerpts from
    this week's waiver/start-sit episodes. Returns [] on any failure."""
    try:
        episodes = discover_episodes(channel, max_episodes)
    except Exception as e:
        print(f"  Warning: failed to list podcast episodes: {e}", file=sys.stderr)
        return []

    excerpts = []
    for ep in select_relevant_episodes(episodes, current_week):
        try:
            cues = fetch_captions(ep["video_id"], cache_dir)
        except Exception as e:
            print(f"  Warning: failed to fetch captions for '{ep['title']}': {e}", file=sys.stderr)
            continue
        if not cues:
            continue

        for name, windows in find_player_mentions(cues, player_names).items():
            for start, end in windows[:MAX_EXCERPTS_PER_PLAYER]:
                text = _excerpt_text(cues, start, end)
                if text:
                    excerpts.append({
                        "player": name,
                        "episode": ep["title"],
                        "timestamp": _format_timestamp(start),
                        "text": text,
                        # "start_sit" or "waivers" - lets each downstream
                        # analyst prompt show only the excerpts from its
                        # own kind of episode.
                        "kind": ep["kind"],
                    })

    return excerpts
