"""Everything that talks to yt-dlp: metadata, captions, audio download."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yt_dlp


@dataclass
class Segment:
    start: float  # seconds from the beginning of the video
    text: str


@dataclass
class Chapter:
    start: float
    end: float
    title: str


@dataclass
class Video:
    id: str
    title: str
    url: str
    channel: str | None
    duration: float
    upload_date: str | None  # YYYY-MM-DD
    description: str
    language: str | None
    is_youtube: bool
    chapters: list[Chapter] = field(default_factory=list)

    def link_at(self, seconds: float) -> str | None:
        """URL that opens the video at the given time (YouTube only)."""
        if not self.is_youtube:
            return None
        return f"https://www.youtube.com/watch?v={self.id}&t={int(seconds)}s"


@dataclass
class Transcript:
    segments: list[Segment]
    source: str  # human-readable, e.g. "YouTube captions (en, auto-generated)"
    language: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "source": self.source,
                "language": self.language,
                "segments": [[s.start, s.text] for s in self.segments],
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "Transcript":
        data = json.loads(raw)
        return cls(
            segments=[Segment(start, text) for start, text in data["segments"]],
            source=data["source"],
            language=data.get("language"),
        )


class _SilentLogger:
    """yt-dlp prints errors even when quiet; they are raised and reported by us instead."""

    def debug(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


def _ydl(**extra) -> yt_dlp.YoutubeDL:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "logger": _SilentLogger(),
        "noplaylist": True,
        # YouTube extraction needs a JavaScript runtime; yt-dlp only enables deno
        # by default, so also allow node when that is what's installed.
        "js_runtimes": {"deno": {}, "node": {}},
    }
    opts.update(extra)
    return yt_dlp.YoutubeDL(opts)


def _to_video(info: dict) -> Video:
    date = info.get("upload_date")
    if date and len(date) == 8:
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    duration = float(info.get("duration") or 0)
    chapters = [
        Chapter(float(c.get("start_time") or 0), float(c.get("end_time") or duration), c.get("title") or "")
        for c in info.get("chapters") or []
    ]
    return Video(
        id=str(info["id"]),
        title=info.get("title") or str(info["id"]),
        url=info.get("webpage_url") or info.get("original_url") or "",
        channel=info.get("channel") or info.get("uploader"),
        duration=duration,
        upload_date=date,
        description=info.get("description") or "",
        language=info.get("language"),
        is_youtube=(info.get("extractor_key") or "").lower() == "youtube",
        chapters=chapters,
    )


def _pick_caption_track(info: dict, lang: str | None) -> tuple[str, str, bool] | None:
    """Choose the caption track closest to the spoken language.

    Returns (language key, json3 url, is_auto_generated) or None.
    """
    manual = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"}
    auto = info.get("automatic_captions") or {}

    # YouTube marks the untranslated auto-caption track as "<lang>-orig".
    orig_auto = next((k for k in auto if k.endswith("-orig")), None)
    wanted = lang or info.get("language") or (orig_auto[: -len("-orig")] if orig_auto else "en")
    base = wanted.split("-")[0]

    def json3_url(tracks: list[dict]) -> str | None:
        return next((t["url"] for t in tracks if t.get("ext") == "json3" and t.get("url")), None)

    candidates: list[tuple[str, dict, bool]] = []
    candidates += [(k, manual, False) for k in manual if k == wanted]
    candidates += [(k, manual, False) for k in manual if k.split("-")[0] == base]
    candidates += [(k, auto, True) for k in (f"{wanted}-orig", f"{base}-orig", wanted, base) if k in auto]
    if not lang:  # nothing in the expected language: take whatever the uploader provided
        candidates += [(k, manual, False) for k in manual]
        if orig_auto:
            candidates.append((orig_auto, auto, True))

    for key, tracks, is_auto in candidates:
        url = json3_url(tracks[key])
        if url:
            return key, url, is_auto
    return None


def parse_json3(raw: bytes | str) -> list[Segment]:
    """Parse YouTube's json3 caption format into segments."""
    segments: list[Segment] = []
    for event in json.loads(raw).get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        text = re.sub(r"\s+", " ", "".join(s.get("utf8", "") for s in segs)).strip()
        if text:
            segments.append(Segment(event.get("tStartMs", 0) / 1000, text))
    return segments


def is_playlist(info: dict) -> bool:
    """True for playlists, channels and channel tabs: anything that lists videos."""
    return info.get("_type") in ("playlist", "multi_video")


def probe(url: str) -> dict:
    """Look a URL up: a full record for a video, or a playlist whose entries are only listed.

    A video link that also names a playlist (watch?v=...&list=...) counts as the video.
    """
    with _ydl(extract_flat="in_playlist") as ydl:
        return ydl.extract_info(url, download=False)


def load(info: dict, lang: str | None = None, want_captions: bool = True, log=print) -> tuple[Video, Transcript | None]:
    """Turn a probed video record into a Video and, when available, its captions."""
    video = _to_video(info)
    if not want_captions:
        return video, None

    track = _pick_caption_track(info, lang)
    if not track:
        return video, None
    key, caption_url, is_auto = track
    try:
        with _ydl() as ydl:
            segments = parse_json3(ydl.urlopen(caption_url).read())
    except Exception as exc:  # captions are best-effort; Whisper is the fallback
        log(f"  Could not download captions ({exc}).")
        return video, None

    # A handful of cues on a long video means the track is a stub, not a transcript.
    if sum(len(s.text) for s in segments) < max(200, video.duration * 2):
        return video, None
    kind = "auto-generated" if is_auto else "uploaded"
    lang_code = key.removesuffix("-orig")
    return video, Transcript(segments, f"YouTube captions ({lang_code}, {kind})", lang_code)


def download_audio(url: str, dest_dir: Path, log=print) -> Path:
    """Download the best audio-only stream and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    last_decile = -1

    def hook(d: dict) -> None:
        nonlocal last_decile
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        if d.get("status") == "downloading" and total:
            decile = int(d.get("downloaded_bytes", 0) * 10 / total)
            if decile > last_decile:
                last_decile = decile
                log(f"  downloading audio... {decile * 10}%")

    for attempt in range(3):
        try:
            with _ydl(
                format="bestaudio/best",
                outtmpl=str(dest_dir / "audio.%(ext)s"),
                progress_hooks=[hook],
                noprogress=True,
            ) as ydl:
                info = ydl.extract_info(url, download=True)
                downloads = info.get("requested_downloads") or []
                path = Path(downloads[0]["filepath"]) if downloads else Path(ydl.prepare_filename(info))
            break
        except yt_dlp.utils.DownloadError as exc:
            # YouTube sometimes refuses a stream URL (HTTP 403); a fresh extraction
            # gets new URLs and usually works.
            if "403" not in str(exc) or attempt == 2:
                raise
            log("  YouTube refused the audio stream (HTTP 403); retrying...")
            time.sleep(3 * (attempt + 1))
    if not path.exists():
        raise RuntimeError("yt-dlp reported success but the audio file is missing.")
    return path
