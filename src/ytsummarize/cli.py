"""Command line entry point: ytsummarize URL_OR_LIST [URL_OR_LIST ...]

Each argument can be a video, a playlist or channel, or a Markdown file listing
videos under headings. Videos are handled one at a time, transcript then notes,
so a long batch can be stopped at any point and every finished video is saved.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from yt_dlp.extractor.youtube import YoutubeIE
from yt_dlp.utils import DownloadError, sanitize_filename

from . import __version__, hardware, notes, sources, video as yt
from .llm import DEFAULT_CLAUDE_MODEL, DEFAULT_CONTEXT_LENGTH, DEFAULT_LOCAL_URL, LLM, LLMError
from .video import Transcript, Video

_ERRORS = (DownloadError, LLMError, ValueError, RuntimeError, OSError)
_UNAVAILABLE = {"[Private video]", "[Deleted video]", "[Unavailable video]"}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ytsummarize",
        description="Write organised Markdown notes on YouTube videos, playlists, or a Markdown list of them.",
    )
    p.add_argument("sources", nargs="*", metavar="URL_OR_LIST", help="video or playlist URL, or a .md file listing videos under headings")
    p.add_argument("--version", action="version", version=f"ytsummarize {__version__}")
    p.add_argument("--check", action="store_true", help="report what this machine will use (GPU, Whisper, notes model) and exit")
    p.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)  # offline check of a build
    p.add_argument("-o", "--output-dir", type=Path, default=Path("notes"), help="where to write the .md files (default: ./notes)")
    p.add_argument("--overwrite", action="store_true", help="redo videos whose notes already exist, and ticked items of list files")
    p.add_argument("--no-mark-done", dest="mark_done", action="store_false", help="do not tick finished items in list files (they are ticked by default)")
    p.add_argument("--lang", help="spoken language of the video, e.g. en or fr (default: detect)")
    p.add_argument("--notes-lang", help="language to write the notes in, e.g. English (default: same as the video)")
    p.add_argument("--transcribe", action="store_true", help="ignore YouTube captions and transcribe the audio locally with Whisper")
    p.add_argument("--whisper-model", default="auto", help="faster-whisper model (default: large-v3-turbo on an NVIDIA GPU, small on the CPU)")
    p.add_argument("--cpu", action="store_true", help="run Whisper on the CPU even when an NVIDIA GPU is present")
    p.add_argument("--rocm", action="store_true", help="EXPERIMENTAL: run Whisper on an AMD GPU (downloads CTranslate2's ROCm build on first use)")
    p.add_argument("--backend", choices=["local", "claude-code", "api"], default="local", help="who writes the notes: a local model (default), the Claude Code CLI, or the Anthropic API")
    p.add_argument("--model", help=f"model to use (default: for LM Studio, the loaded model or the largest that fits your GPU; {DEFAULT_CLAUDE_MODEL} for the Claude backends)")
    p.add_argument("--local-url", default=DEFAULT_LOCAL_URL, help=f"OpenAI-compatible endpoint of the local model server (default: {DEFAULT_LOCAL_URL}, LM Studio)")
    p.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH, help=f"context window, in tokens, to load the local model with (default: {DEFAULT_CONTEXT_LENGTH})")
    p.add_argument("--think", action="store_true", help="local backend only: let a reasoning model think before writing (more detailed notes, several times slower)")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium", help="Claude backends only: how hard the model works on the notes (default: medium)")
    p.add_argument("--section-minutes", type=float, help="for long videos, approximate length of each section noted separately (default: 15 local, 30 Claude)")
    p.add_argument("--jobs", type=int, help="sections written in parallel (default: 1 local, 3 Claude)")
    p.add_argument("--save-transcript", action="store_true", help="also save the timestamped transcript next to the notes")
    p.add_argument("--no-cache", action="store_true", help="ignore cached transcripts and notes from earlier runs")
    p.add_argument("--cache-dir", type=Path, default=Path(".cache"), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not args.sources and not (args.check or args.self_test):
        p.error("give at least one video, playlist or list file (or --check)")
    return args


@dataclass(eq=False)
class Entry:
    """A bullet of a list file: ticked once every video it stands for is done."""

    path: Path
    item: sources.ListItem
    items: list[Item] = field(default_factory=list)
    failed: bool = False
    ticked: bool = False


@dataclass(eq=False)
class Item:
    """One video to process; nothing is fetched until its turn comes."""

    url: str
    out_dir: Path
    prefix: str = ""  # e.g. "03 - " to keep a playlist's order
    video_id: str | None = None
    entry: Entry | None = None  # the list bullet it came from, if any
    entries: list[Entry] = field(default_factory=list)
    finished: bool = False

    def __post_init__(self) -> None:
        if self.entry:
            self.entries.append(self.entry)


@dataclass
class Batch:
    args: argparse.Namespace
    llm: LLM
    items: list[Item] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    skipped: int = 0
    written: int = 0
    ticked: int = 0
    _queued: dict[tuple[Path, str], Item] = field(default_factory=dict)

    # -- listing what to do (fast: playlists are listed, videos are not looked up) --

    def collect(self, url: str, out_dir: Path, prefix: str = "", depth: int = 0, entry: Entry | None = None) -> None:
        url = url.strip()
        video_id = YoutubeIE.get_temp_id(url)
        if video_id:  # a YouTube video link, recognised without any network request
            if not YoutubeIE.suitable(url):  # opened from inside a playlist: take just the video
                playlist = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("list", [""])[0]
                url = f"https://www.youtube.com/watch?v={video_id}"
                if playlist and not playlist.startswith("RD"):  # RD... are YouTube's endless auto-mixes
                    print(f"  note: {video_id} was linked from inside a playlist; only that video is processed. "
                          f"For the whole playlist, pass https://www.youtube.com/playlist?list={playlist}")  # fmt: skip
            self._queue(Item(url, out_dir, prefix, video_id, entry))
            return
        try:
            info = yt.probe(url)
        except _ERRORS as exc:
            if entry:
                entry.failed = True
            self._fail(url, exc)
            return
        if yt.is_playlist(info):
            self._collect_playlist(info, out_dir, depth, source=entry)
        else:  # a video on another site
            self._queue(Item(url, out_dir, prefix, str(info.get("id") or "") or None, entry))

    def _collect_playlist(self, info: dict, out_dir: Path, depth: int, source: Entry | None = None) -> None:
        entries = [e for e in info.get("entries") or [] if e]
        title = info.get("title") or info.get("id") or "Playlist"
        folder = out_dir / sources.folder_name(title)
        kind = "sections" if any(yt.is_playlist(e) for e in entries) else "videos"
        print(f"Playlist: {title} - {len(entries)} {kind} -> {folder}")
        width = len(str(len(entries)))
        for index, entry in enumerate(entries, 1):
            prefix = f"{index:0{width}d} - "
            if yt.is_playlist(entry):  # a channel's tabs (Videos, Shorts, Live) are playlists too
                if depth < 2:
                    if entry.get("entries") is not None:
                        self._collect_playlist(entry, folder, depth + 1, source)
                    elif entry.get("url"):
                        self.collect(entry["url"], folder, depth=depth + 1, entry=source)
                continue
            if entry.get("title") in _UNAVAILABLE:
                print(f"  {prefix}{entry['title']} - skipped")
                continue
            url = entry.get("url") or entry.get("webpage_url") or entry.get("id")
            if url:
                self._queue(Item(url, folder, prefix, entry.get("id"), source))

    def _queue(self, item: Item) -> None:
        if item.video_id:
            key = (item.out_dir, item.video_id)
            if key in self._queued:  # the same video twice for the same folder: done once,
                existing = self._queued[key]  # and both list bullets wait for it
                for entry in item.entries:
                    existing.entries.append(entry)
                    entry.items.append(existing)
                return
            self._queued[key] = item
        for entry in item.entries:
            entry.items.append(item)
        self.items.append(item)

    # -- doing it, one video at a time --------------------------------------

    def run(self) -> None:
        """Transcript then notes for each video in turn, so every finished video is saved."""
        total = len(self.items)
        for number, item in enumerate(self.items, 1):
            tag = f"[{number}/{total}]"
            if item.video_id and self._existing(item, tag):
                self._finished(item)
                continue
            label, announced = item.url, False
            try:
                info = yt.probe(item.url)
                if yt.is_playlist(info):
                    raise ValueError("expected a single video here but found a playlist")
                if not item.video_id:
                    item.video_id = str(info.get("id"))
                    if self._existing(item, tag):
                        self._finished(item)
                        continue
                video, captions = yt.load(info, lang=self.args.lang, want_captions=not self.args.transcribe)
                label = f"{item.prefix}{video.title}"
                print(f"\n{tag} {label}  ({notes.fmt_ts(video.duration)})")
                announced = True
                transcript = self._transcript(video, captions)
                self._write(video, transcript, item)
                self._finished(item)
            except _ERRORS as exc:
                if not announced:
                    print(f"\n{tag} {label}")
                for entry in item.entries:
                    entry.failed = True
                self._fail(label, exc)

    def _finished(self, item: Item) -> None:
        """Tick the list bullets whose videos are now all done."""
        item.finished = True
        for entry in item.entries:
            if entry.ticked or entry.failed or not all(i.finished for i in entry.items):
                continue
            entry.ticked = True
            if not self.args.mark_done:
                continue
            if sources.tick(entry.path, entry.item):
                self.ticked += 1
                print(f"  ticked in {entry.path.name} (line {entry.item.line})")
            else:
                print(f"  note: could not tick this item in {entry.path.name} (file changed or in use)")

    def _existing(self, item: Item, tag: str) -> bool:
        """Report and skip a video whose notes are already in its folder."""
        if self.args.overwrite or not item.out_dir.is_dir():
            return False
        pattern = "*" + glob.escape(f"[{sanitize_filename(item.video_id, restricted=True)}].md")
        done = next(item.out_dir.glob(pattern), None)
        if done:
            print(f"{tag} already done: {done.name}")
            self.skipped += 1
        return done is not None

    def _transcript(self, video: Video, captions: Transcript | None) -> Transcript:
        args = self.args
        work_dir = args.cache_dir / sanitize_filename(video.id, restricted=True)
        cache_file = work_dir / "transcript.json"
        if not args.no_cache and cache_file.exists():
            cached = Transcript.from_json(cache_file.read_text(encoding="utf-8"))
            if not args.transcribe or cached.source.startswith("Whisper"):
                print(f"  transcript: {cached.source} (cached)")
                return cached

        transcript = captions
        if transcript is None:
            from . import whisper  # heavy import, only when needed

            print("  transcribing the audio locally with Whisper..." if args.transcribe
                  else "  no usable captions; transcribing the audio locally with Whisper...")  # fmt: skip
            audio = yt.download_audio(video.url, work_dir)
            # Whisper and a local LLM share the GPU. Swap them only when they do not
            # fit side by side; a small enough LLM simply stays loaded.
            swap = whisper.device(args.cpu, args.rocm) == "cuda" and not self.llm.leaves_room_for(whisper.GPU_BYTES)
            if swap:
                self.llm.free_gpu()
            try:
                transcript = whisper.transcribe(
                    audio, args.whisper_model, args.lang or video.language, video.duration,
                    force_cpu=args.cpu, use_rocm=args.rocm,
                )  # fmt: skip
            finally:
                audio.unlink(missing_ok=True)
                if swap:
                    whisper.release()
            if not transcript.segments:
                raise RuntimeError("Whisper found no speech in this video.")

        print(f"  transcript: {transcript.source}, {sum(len(s.text) for s in transcript.segments):,} characters")
        work_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(transcript.to_json(), encoding="utf-8")
        return transcript

    def _write(self, video: Video, transcript: Transcript, item: Item) -> None:
        args, started = self.args, time.monotonic()
        work_dir = args.cache_dir / sanitize_filename(video.id, restricted=True)
        document = notes.generate(
            video,
            transcript,
            self.llm,
            notes_language=args.notes_lang,
            section_minutes=args.section_minutes,
            jobs=args.jobs,
            cache_dir=None if args.no_cache else work_dir / "notes",
        )
        item.out_dir.mkdir(parents=True, exist_ok=True)
        title = sanitize_filename(video.title)[:120].strip()
        stem = f"{item.prefix}{title} [{sanitize_filename(video.id, restricted=True)}]"
        out = item.out_dir / f"{stem}.md"
        out.write_text(document, encoding="utf-8")
        if args.save_transcript:
            blocks = notes.to_blocks(transcript.segments, video.chapters)
            (item.out_dir / f"{stem}.transcript.txt").write_text(
                "\n".join(f"[{notes.fmt_ts(b.start)}] {b.text}" for b in blocks) + "\n", encoding="utf-8"
            )
        self.written += 1
        print(f"  done in {time.monotonic() - started:.0f}s -> {out}")

    def _fail(self, label: str, exc: Exception) -> None:
        message = re.sub(r"^ERROR:\s*", "", str(exc)).strip() or type(exc).__name__
        print(f"  error: {message}", file=sys.stderr)
        self.failures.append((label, message))


def _collect_list_file(batch: Batch, path: Path) -> None:
    items, warnings = sources.parse_list(path.read_text(encoding="utf-8-sig"))
    ticked = [] if batch.args.overwrite else [item for item in items if item.done]
    todo = [item for item in items if item not in ticked]
    folders = {item.folder for item in todo}
    print(f"{path.name}: {len(todo)} link(s) to do in {len(folders)} folder(s)"
          + (f", {len(ticked)} already ticked" if ticked else ""))  # fmt: skip
    for warning in warnings:
        print(f"  {path.name}, {warning}")
    for item in todo:
        entry = Entry(path, item, ticked=item.done)  # with --overwrite, a ticked item stays ticked
        batch.collect(item.url, batch.args.output_dir.joinpath(*item.folder), entry=entry)


def _summary(batch: Batch) -> None:
    if batch.written + batch.skipped + len(batch.failures) > 1:
        print(f"\n{batch.written} written, {batch.skipped} already done, {len(batch.failures)} failed.")
        for label, message in batch.failures:
            print(f"  failed: {label}: {message}", file=sys.stderr)


def _self_test() -> int:
    """Offline smoke test for the standalone builds: every bundled piece loads and works."""
    import importlib.resources
    import math
    import struct
    import tempfile
    import wave

    import certifi
    import ctranslate2
    from faster_whisper.vad import get_vad_model

    from . import whisper

    get_vad_model()  # faster-whisper's bundled model, run through onnxruntime
    print(f"ok  faster-whisper voice detection model; CTranslate2 {ctranslate2.__version__}")
    with tempfile.TemporaryDirectory() as tmp:  # PyAV decoding a generated tone
        path = Path(tmp) / "tone.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1), out.setsampwidth(2), out.setframerate(44100)
            out.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 20))) for i in range(44100)))
        samples = whisper._decode(path)
    assert abs(len(samples) - 16000) < 400, len(samples)
    print(f"ok  audio decoding ({len(samples)} samples at 16 kHz)")
    scripts = list(Path(str(importlib.resources.files("yt_dlp_ejs"))).rglob("*.js"))
    assert scripts, "yt-dlp's JavaScript challenge solver files are missing"
    assert Path(certifi.where()).is_file()
    print("ok  yt-dlp challenge solver scripts and certificates")
    print(f"ok  GPU detection: {hardware.describe()}")
    return 0


def _check(args: argparse.Namespace) -> int:
    """Report what a run on this machine would use, without downloading anything."""
    import shutil

    from . import rocm, whisper

    print(f"ytsummarize {__version__}, Python {sys.version.split()[0]}, {sys.platform}")
    print(f"GPU: {hardware.describe()}")
    for gpu in hardware.gpus():
        print(f"  - {gpu.vendor}: {gpu.name}, {gpu.vram_mb / 1024:.1f} GB")
    js = shutil.which("deno") or shutil.which("node")
    print(f"JavaScript runtime for YouTube: {js or 'none found (some videos may fail; install Node.js or Deno)'}")

    if args.cpu:
        where = "CPU (--cpu)"
    elif args.rocm:
        if rocm.is_installed():
            where = "AMD GPU (experimental)" if rocm.activate(print, allow_download=False) else "CPU (AMD build unusable here)"
        else:
            where = f"AMD GPU, experimental (its files are not downloaded yet: {rocm.install_dir()})"
    elif hardware.has_nvidia() and whisper._has_cuda():
        state = "ready" if hardware.has_cuda_libs() else "downloaded on first use, about 2 GB"
        where = f"NVIDIA GPU, model {whisper.GPU_MODEL} (CUDA libraries: {state})"
    else:
        where = f"CPU, model {whisper.CPU_MODEL}"
    print(f"Whisper (only for videos without captions): {where}")

    try:
        llm = LLM(args.backend, args.model, args.effort, args.local_url, args.context_length, args.think)
        print(f"Notes written by: {llm.description}")
    except LLMError as exc:
        print(f"Notes: not ready - {exc}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # video titles can contain anything
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace", line_buffering=True)
    args = _parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.check:
        return _check(args)

    for source in args.sources:  # catch a mistyped list file before anything slow happens
        if "://" not in source and source.lower().endswith((".md", ".markdown", ".txt")) and not Path(source).is_file():
            print(f"error: list file not found: {source}", file=sys.stderr)
            return 1

    print(f"GPU: {hardware.describe()}")
    try:
        llm = LLM(args.backend, args.model, args.effort, args.local_url, args.context_length, args.think)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Notes written by: {llm.description}\n")

    batch = Batch(args, llm)
    try:
        for source in args.sources:
            path = Path(source)
            if "://" not in source and path.is_file():
                _collect_list_file(batch, path)
            else:
                batch.collect(source, args.output_dir)
        if len(batch.items) > 1:
            print(f"\n{len(batch.items)} videos to go through.")
        batch.run()
    except KeyboardInterrupt:
        print(
            f"\nStopped. The {batch.written} note file(s) finished in this run are saved. Re-run the "
            "same command to continue: finished videos are skipped, and the interrupted one picks "
            "up from its cached transcript and sections.",
            file=sys.stderr,
        )
        return 130
    _summary(batch)
    if llm.input_tokens or llm.output_tokens:
        print(f"Model usage: {llm.input_tokens:,} input / {llm.output_tokens:,} output tokens")
    return 1 if batch.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
