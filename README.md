# ytsummarize

Give it a YouTube video, a playlist, or a Markdown list of them, and get clean Markdown study notes: an overview, the key
takeaways, then detailed notes organised by topic, with every heading linked to the
moment in the video where that topic starts. Built to handle multi-hour videos, and runs
entirely on your own machine by default: local Whisper for speech, a local LLM for the notes.

## Install

**Ready-made program (no Python needed):** download the archive for your system from
the [Releases page](../../releases), unpack it, and run `ytsummarize` from a terminal
in the unpacked `ytsummarize` folder (`.\ytsummarize.exe` in PowerShell,
`./ytsummarize` on Linux). It is a command-line program: double-clicking it does nothing
useful, and is also what triggers Windows' "Windows protected your PC" warning, which a
terminal launch does not. If Windows 11's Smart App Control is on, it blocks unsigned
programs like this one outright; use the from-source install in that case.

**From source**, with [uv](https://docs.astral.sh/uv/):

```powershell
uv sync
uv run ytsummarize --check      # what this machine will use
```

The examples below use `ytsummarize`; from source, put `uv run` in front.

## Usage

```powershell
ytsummarize "https://www.youtube.com/watch?v=..."
ytsummarize "https://www.youtube.com/playlist?list=..."
ytsummarize courses.md
ytsummarize --check             # GPU, Whisper and notes model this machine will use
```

Each argument can be:

- **A video** -> `notes/<title> [<id>].md`
- **A playlist or channel** -> a folder named after it, files numbered in playlist order:
  `notes/<playlist>/01 - <title> [<id>].md`. A video link that also carries `&list=`
  counts as just that video; the tool prints the playlist link if you wanted all of it.
- **A Markdown list file**, where headings become folders and bullet points are the
  videos (or playlists) to process:

  ```markdown
  # Machine learning
  ## Foundations
  - https://www.youtube.com/watch?v=aircAruvnKk
  - [Backpropagation](https://youtu.be/Ilg3gGewQ5U)
  ## Transformers
  - https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
  ```

  gives `notes/Machine learning/Foundations/…` and
  `notes/Machine learning/Transformers/Neural networks/01 - …`. A `##` heading nests
  inside the `#` above it; bullets before any heading go straight into `notes/`. Any of
  `-`, `*`, `+` or `1.` work, the link can be bare, `<…>` or `[text](…)`, and the first
  link of each bullet is used. Lines that are not bullets, code blocks and HTML comments
  are ignored; the tool warns about links it skips.

Videos whose notes already exist in their folder are skipped, so you can add to a list
or a playlist and simply re-run it. `--overwrite` redoes them.

In a list file, each bullet is ticked (`- [x] link`) as soon as all its videos are done
(for a playlist, after its last video), and ticked bullets are skipped without even
contacting YouTube. Only that bullet's line is changed; the rest of the file is left as
it was. Untick a playlist to check it for new videos. A bullet with a failed video stays
unticked so the next run retries it. `--no-mark-done` leaves the file alone.

Useful options (`ytsummarize --help` for all of them):

| Option | What it does |
|---|---|
| `--notes-lang English` | Write the notes in a given language (default: the video's language) |
| `--transcribe` | Ignore YouTube's captions and transcribe the audio locally with Whisper |
| `--cpu` | Run Whisper on the CPU even when an NVIDIA GPU is present |
| `--rocm` | **Experimental:** run Whisper on an AMD GPU (see *Matching your hardware*) |
| `--overwrite` | Redo videos whose notes already exist, including ticked list items |
| `--no-mark-done` | Do not tick finished items in list files |
| `--lang fr` | Spoken language, if detection gets it wrong |
| `--section-minutes 20` | Finer (or coarser) sections on long videos; smaller means more detail |
| `--model <name>` | Use a specific model instead of the default one |
| `--context-length 32768` | Load the local model with a bigger context window (needs more VRAM) |
| `--think` | Let a local reasoning model think before writing: more detail, several times slower |
| `--backend claude-code` | Have Claude write the notes instead of the local model (see below) |
| `--save-transcript` | Also save the timestamped transcript next to the notes |
| `-o some/dir` | Output directory |

## How it works

1. **Transcript.** yt-dlp fetches the video's metadata, chapters and captions (uploaded
   captions first, YouTube's auto-generated ones otherwise). If there are none, or with
   `--transcribe`, it downloads the audio track and transcribes it locally with
   faster-whisper (see *Matching your hardware*).
2. **Notes.** A short video is noted in one model call. A longer one is cut into sections
   along the uploader's chapters, each section gets its own detailed notes, and a final call
   writes the overview and key takeaways from all of them. Detail therefore stays constant
   however long the video is. Section size follows the model: about 15 minutes of video for
   a local model, 30 for Claude, and never more than fits the model's context window.
3. **Assembly.** The parts are stitched into one document and `[HH:MM:SS]` markers become
   links into the video.

With several videos, each one is finished (transcript, then notes, then saved) before the
next starts, so a long batch can be stopped with Ctrl+C at any time. Re-running the same
command skips the finished videos and resumes the interrupted one from its cache.

Transcripts and finished sections are cached in `.cache/`, so an interrupted run resumes
where it stopped. `--no-cache` starts fresh.

## Matching your hardware

The tool detects your GPUs at start-up (it prints what it found) and adapts:

| | NVIDIA GPU | AMD or Intel GPU | No dedicated GPU |
|---|---|---|---|
| Local LLM (LM Studio) runs on | the GPU | the GPU (LM Studio supports these too) | the CPU |
| LM Studio model, if none is loaded | the largest one you have that fits in GPU memory | same | the first one listed |
| Whisper runs on | the GPU | the CPU (AMD: the GPU with `--rocm`, experimental) | the CPU |
| Whisper model | `large-v3-turbo` (most accurate) | `small` (keeps hours of audio practical) | `small` |
| CUDA libraries (~2 GB) | downloaded once, the first time Whisper is needed | never downloaded | never downloaded |

Whisper is only needed for videos without captions (or with `--transcribe`); most
YouTube videos have captions, so on most machines it rarely runs.

**AMD GPUs (`--rocm`, experimental, untested on real AMD hardware).** Whisper's engine,
CTranslate2, supports NVIDIA in its normal builds, but also publishes AMD ROCm builds.
With `--rocm`, the tool downloads the one for your Python (and, on Windows, AMD's ROCm
runtime: about 1.3 GB in all, once) and uses it instead. On Linux it needs AMD's ROCm 7
installed (`rocm-hip-libraries`). Supported cards are those CTranslate2's ROCm builds
target, mainly the Radeon RX 6000, 7000 and 9000 series. Anything that goes wrong falls
back to the CPU with a message saying why; `ytsummarize --check --rocm` shows the
result without transcribing anything. If you try it, reports of what happened are very
welcome.

When Whisper and the local LLM both need the GPU, the LLM is unloaded while Whisper runs
only if the two do not fit in GPU memory together.

The CUDA libraries go to `%LOCALAPPDATA%\ytsummarize` on Windows and
`~/.local/share/ytsummarize` on Linux, shared by every install of the tool. If the
download fails, Whisper simply runs on the CPU. `--whisper-model` and `--model` override
the automatic choices, and `--cpu` keeps Whisper off the GPU.

If the local model you use is too big for your GPU, the tool says so and suggests a size
that would fit. A model that spills into system RAM runs several times slower.

## Who writes the notes

**A local model, by default** (`--backend local`). The tool talks to
[LM Studio](https://lmstudio.ai): it starts LM Studio's server if needed, uses the model
named with `--model`, or the one already loaded, or the best fit for your GPU among those
you have downloaded, loads it with a 16k-token context window, and unloads it again after
15 idle minutes.
Nothing leaves your machine except the request to YouTube.

Any other OpenAI-compatible server works too, for example Ollama:

```powershell
ytsummarize URL --local-url http://localhost:11434/v1 --model <model name>
```

With those servers you must make sure yourself that the model runs with a context window
of at least `--context-length` tokens (16384 by default). Local servers silently drop the
start of a prompt that does not fit, and Ollama's default window is far smaller.

**Claude, if you prefer** — faster and more thorough on dense material:

- `--backend claude-code` uses the `claude` CLI and the account it is logged into.
- `--backend api` uses the Anthropic API (needs `ANTHROPIC_API_KEY`, pay per token).
  `--effort` and `--model claude-sonnet-5-5` style options apply to these two.

## Limits

The notes come from what is *said*. Anything shown but never spoken (slides, code on
screen, diagrams) is not captured.

## Requirements

- The ready-made program, or [uv](https://docs.astral.sh/uv/), which fetches a suitable
  Python (3.11+) and every dependency by itself. Plain `pip install .` in a Python 3.11+
  environment works too.
- [LM Studio](https://lmstudio.ai) with at least one model downloaded, for the default
  local backend (or another local server, or one of the Claude backends).

No system ffmpeg is needed. Optional but helpful:

- **An NVIDIA GPU** makes Whisper transcription fast (see *Matching your hardware*). The
  program is about 270 MB unpacked (330 MB from source); GPU machines download about
  2 GB more on first use.
- **Node.js or Deno** lets yt-dlp solve YouTube's JavaScript challenges. Many videos work
  without it; if a download fails with a message about a JavaScript runtime, install one.

Tested on Windows 11 (Python 3.11 to 3.14, and the ready-made program) and Linux
(Python 3.13 and the ready-made program, via WSL). Not tested on macOS. Building and
publishing releases is described in [RELEASING.md](RELEASING.md).
