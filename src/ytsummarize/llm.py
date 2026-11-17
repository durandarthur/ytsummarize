"""The model call behind the notes, with three interchangeable backends.

* ``local``        - a model running on this machine behind an OpenAI-compatible
                     server. LM Studio is managed automatically (server start, model
                     load, context size); Ollama, llama.cpp etc. work via --local-url.
* ``claude-code``  - the Claude Code CLI in print mode, using the account it is logged into.
* ``api``          - the Anthropic API through the official SDK (needs ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import hardware

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_LOCAL_URL = "http://localhost:1234/v1"  # LM Studio's default
DEFAULT_CONTEXT_LENGTH = 16_384
# GPU memory a model needs beyond its file size: the context window's cache and runtime
# buffers at the default context length, plus what the desktop already uses.
VRAM_HEADROOM = 3 * 1024**3
MAX_OUTPUT_TOKENS = 64_000

# Models where the API can transparently re-run a request another model declined.
_SERVER_FALLBACK_MODELS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5")

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    pass


class LLM:
    """One configured model. ``complete()`` is thread-safe.

    The planning attributes tell the note-writer how much text this model can
    take per call, so a small local context window gets smaller pieces.
    """

    def __init__(
        self,
        backend: str = "local",
        model: str | None = None,
        effort: str = "medium",
        local_url: str = DEFAULT_LOCAL_URL,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        think: bool = False,
        log=print,
    ):
        self.backend = backend
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()
        self._client = None
        self._local: _LocalServer | None = None

        if backend == "local":
            self._local = _LocalServer(local_url, context_length, think, log)
            self.model = self._local.resolve_model(model)
            self.effort = "think" if think else ""
            # Leave room for the notes themselves (and the model's reasoning, which can be
            # longer than the notes) and for the instructions; 3 characters per token is
            # conservative for any language.
            reserve = 0.6 if think else 0.4
            budget_tokens = context_length - max(4_000, int(context_length * reserve)) - 1_500
            if budget_tokens < 2_000:
                raise LLMError("--context-length is too small to take notes with; use at least 8192.")
            self.input_budget_chars = budget_tokens * 3
            self.single_pass_chars = min(50_000, self.input_budget_chars)
            self.section_minutes = 15.0
            self.jobs = 1
            self.description = (
                f"local model {self.model} via {self._local.kind}, context {context_length:,} tokens"
                + (", reasoning on" if think else "")
            )
            return

        self.model = model or DEFAULT_CLAUDE_MODEL
        self.effort = effort
        self.input_budget_chars = 1_500_000
        self.single_pass_chars = 50_000  # roughly 50 minutes of speech
        self.section_minutes = 30.0
        self.jobs = 3
        if backend == "api":
            import anthropic

            # Section calls on long videos run for minutes; be generous with retries.
            self._client = anthropic.Anthropic(max_retries=4)
            self.description = f"Anthropic API, model {self.model}, effort {effort}"
        elif backend == "claude-code":
            if not shutil.which("claude"):
                raise LLMError("The claude-code backend needs the `claude` CLI on PATH.")
            self.description = f"Claude Code CLI, model {self.model}, effort {effort}"
        else:
            raise LLMError(f"Unknown backend {backend!r}.")

    def leaves_room_for(self, nbytes: int) -> bool:
        """Whether the local model and `nbytes` of other GPU work fit in GPU memory together."""
        return self._local is None or self._local.fits_with(self.model, nbytes)

    def free_gpu(self) -> None:
        """Unload a local model so Whisper can have the GPU; it is reloaded on the next call."""
        if self._local:
            self._local.unload(self.model)

    def complete(self, system: str, user: str) -> str:
        """Run one system+user prompt and return the model's text."""
        call = {"local": self._complete_local, "api": self._complete_api, "claude-code": self._complete_cli}
        text, used_in, used_out = call[self.backend](system, user)
        with self._lock:
            self.input_tokens += used_in
            self.output_tokens += used_out
        if not text.strip():
            raise LLMError("The model returned an empty response.")
        return text.strip()

    def _complete_local(self, system: str, user: str) -> tuple[str, int, int]:
        return self._local.chat(self.model, system, user)

    def _complete_api(self, system: str, user: str) -> tuple[str, int, int]:
        import anthropic

        kwargs = dict(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
        )
        stream = self._client.messages.stream
        if self.model.startswith(_SERVER_FALLBACK_MODELS):
            # If a safety classifier declines the request (e.g. a security or biology
            # lecture), let the API retry it on a fallback model instead of failing.
            stream = self._client.beta.messages.stream
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")

        try:
            # Streaming keeps long generations clear of HTTP timeouts.
            with stream(**kwargs) as s:
                message = s.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic API rejected the credentials - check ANTHROPIC_API_KEY.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic API rate limit hit; re-run later (finished sections are cached).") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc

        if message.stop_reason == "refusal":
            raise LLMError("The model declined to process this content.")
        if message.stop_reason == "max_tokens":
            raise LLMError("The notes for one section exceeded the output limit; try a smaller --section-minutes.")
        text = "".join(block.text for block in message.content if block.type == "text")
        return text, message.usage.input_tokens, message.usage.output_tokens

    def _complete_cli(self, system: str, user: str) -> tuple[str, int, int]:
        cmd = [
            shutil.which("claude"),
            "--print",
            "--model", self.model,
            "--effort", self.effort,
            "--system-prompt", system,
            "--tools", "",  # pure text generation: the transcript is untrusted input
            "--strict-mcp-config",
            "--no-session-persistence",
            "--output-format", "json",
        ]  # fmt: skip
        # Run from an empty directory so no project CLAUDE.md or settings leak in.
        with tempfile.TemporaryDirectory(prefix="ytsummarize-") as cwd:
            proc = subprocess.run(
                cmd, input=user, capture_output=True, text=True, encoding="utf-8", cwd=cwd, timeout=3600
            )
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            detail = (proc.stderr or proc.stdout or "no output").strip()[-800:]
            raise LLMError(f"claude CLI failed (exit {proc.returncode}): {detail}") from None
        if result.get("is_error") or proc.returncode != 0:
            raise LLMError(f"claude CLI error: {str(result.get('result') or proc.stderr).strip()[-800:]}")
        usage = result.get("usage") or {}
        used_in = sum(
            usage.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        return result.get("result") or "", used_in, usage.get("output_tokens") or 0


class _LocalServer:
    """An OpenAI-compatible server on this machine, with extra care for LM Studio.

    A local model silently drops whatever does not fit its context window, and
    servers load models with a small default window. So for LM Studio this class
    loads the model itself with the context length the note-writer planned for.
    """

    def __init__(self, base_url: str, context_length: int, think: bool, log):
        self.base = base_url.rstrip("/")
        self.root = self.base[: -len("/v1")] if self.base.endswith("/v1") else self.base
        self.context_length = context_length
        self.log = log
        self.kind = "OpenAI-compatible server"
        self._is_lmstudio = False
        self._ready = False
        self._reasoning_off = not think
        self._sizes: dict[str, int] | None = None
        self._lock = threading.Lock()
        bundled = Path.home() / ".lmstudio" / "bin" / ("lms.exe" if os.name == "nt" else "lms")
        self._lms = shutil.which("lms") or (str(bundled) if bundled.exists() else None)

    # -- plumbing ---------------------------------------------------------

    def _get(self, url: str) -> dict:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.load(response)

    def _reachable(self) -> bool:
        try:
            self._get(f"{self.base}/models")
            return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _run_lms(self, *args: str) -> None:
        proc = subprocess.run([self._lms, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            detail = _ANSI.sub("", proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            raise LLMError(f"`lms {' '.join(args)}` failed: {' '.join(detail) or 'no output'}")

    def _lmstudio_models(self) -> list[dict] | None:
        """LM Studio's own model list (with load state), or None if this is another server."""
        try:
            data = self._get(f"{self.root}/api/v0/models").get("data")
        except (urllib.error.URLError, OSError, ValueError):
            return None
        return [m for m in data if m.get("type") in ("llm", "vlm")] if isinstance(data, list) else None

    def _start_server(self) -> None:
        # Only LM Studio's own default address is ours to start; any other URL belongs
        # to a server the user runs themselves.
        if not self._lms or self.base != DEFAULT_LOCAL_URL:
            raise LLMError(
                f"No local model server is answering at {self.base}.\n"
                "  - LM Studio (https://lmstudio.ai): install it, download a model, and re-run; "
                "the server is then started for you.\n"
                "  - Another server (Ollama, llama.cpp...): start it and pass --local-url and --model.\n"
                "  - Or let Claude write the notes: --backend claude-code or --backend api."
            )
        self.log("Starting the LM Studio server...")
        self._run_lms("server", "start", "--port", str(urllib.parse.urlsplit(self.base).port))
        for _ in range(30):
            if self._reachable():
                return
            time.sleep(1)
        raise LLMError(f"LM Studio's server was started but is not answering at {self.base}.")

    # -- public -----------------------------------------------------------

    def resolve_model(self, requested: str | None) -> str:
        """Make sure a server is up and decide which model to use (without loading it yet)."""
        if not self._reachable():
            self._start_server()

        models = self._lmstudio_models()
        if models is not None:
            self._is_lmstudio = True
            self.kind = "LM Studio"
            ids = [m["id"] for m in models]
            if requested and requested not in ids:
                raise LLMError(f"LM Studio has no model {requested!r}. Available: {', '.join(ids) or 'none'}.")
            if not ids:
                raise LLMError(
                    "LM Studio has no language model downloaded. Get one in the app or with `lms get`"
                    + self._size_advice()
                )
            loaded = [m["id"] for m in models if m.get("state") == "loaded"]
            # An explicit or already-loaded choice wins; otherwise pick for the GPU.
            choice = requested or (loaded[0] if loaded else self._best_fit(ids))
            self._warn_if_too_big(choice)
            return choice

        if requested:
            return requested
        ids = [m["id"] for m in self._get(f"{self.base}/models").get("data", [])]
        if not ids:
            raise LLMError(f"The server at {self.base} lists no models; name one with --model.")
        return ids[0]

    # -- matching the model to the GPU ------------------------------------

    def _model_sizes(self) -> dict[str, int]:
        """File size of each downloaded LM Studio model, in bytes ({} if unknown)."""
        if self._sizes is None:
            self._sizes = {}
            if self._lms:
                proc = subprocess.run(
                    [self._lms, "ls", "--json"], capture_output=True, text=True, encoding="utf-8", errors="replace"
                )
                try:
                    listing = json.loads(proc.stdout[proc.stdout.index("[") :])
                    self._sizes = {m["modelKey"]: int(m["sizeBytes"]) for m in listing if m.get("sizeBytes")}
                except (ValueError, KeyError, TypeError):
                    pass
        return self._sizes

    def fits_with(self, model: str, nbytes: int) -> bool:
        """True only when the model is known to leave `nbytes` of GPU memory free."""
        budget, size = self._gpu_budget(), self._model_sizes().get(model)
        return budget is not None and size is not None and size + nbytes <= budget

    def _gpu_budget(self) -> int | None:
        vram = hardware.vram_bytes()
        return vram - VRAM_HEADROOM if vram else None

    def _best_fit(self, ids: list[str]) -> str:
        """The largest downloaded model that fits in GPU memory, else the smallest one."""
        budget, sizes = self._gpu_budget(), self._model_sizes()
        if budget is None or not sizes:
            return ids[0]
        known = [i for i in ids if i in sizes]
        fitting = [i for i in known if sizes[i] <= budget]
        if fitting:
            return max(fitting, key=sizes.get)
        return min(known, key=sizes.get) if known else ids[0]

    def _warn_if_too_big(self, model: str) -> None:
        budget, size = self._gpu_budget(), self._model_sizes().get(model)
        if budget is None or size is None or size <= budget:
            return
        self.log(
            f"warning: {model} ({size / 1024**3:.1f} GB) does not fit in your GPU's memory with room "
            f"for its context, so part of it runs on the CPU: expect it to be several times slower"
            f"{self._size_advice()}"
        )

    def _size_advice(self) -> str:
        budget = self._gpu_budget()
        if budget is None:
            return "."
        return f". For your GPU, a model file under about {max(budget, 0) / 1024**3:.0f} GB runs fully on the GPU."

    def _ensure_loaded(self, model: str) -> None:
        if not self._is_lmstudio:
            return  # other servers manage their own models; see the README about context size
        info = next((m for m in self._lmstudio_models() or [] if m["id"] == model), None)
        if info is None:
            raise LLMError(f"LM Studio no longer lists the model {model!r}.")
        loaded = info.get("state") == "loaded"
        if loaded and (info.get("loaded_context_length") or 0) >= self.context_length:
            return
        if (info.get("max_context_length") or self.context_length) < self.context_length:
            raise LLMError(
                f"{model} supports at most {info['max_context_length']:,} tokens of context; "
                "lower --context-length."
            )
        if not self._lms:
            raise LLMError(
                f"Load {model} in LM Studio with a context length of at least {self.context_length:,} "
                "(the `lms` command was not found, so it cannot be done automatically)."
            )
        if loaded:  # loaded, but with a window too small for the planned prompts
            self._run_lms("unload", model)
        self.log(f"  loading {model} into LM Studio (context {self.context_length:,} tokens)...")
        # The TTL frees the GPU again after 15 idle minutes.
        self._run_lms("load", model, "--context-length", str(self.context_length), "--ttl", "900", "-y")

    def unload(self, model: str) -> None:
        if not (self._is_lmstudio and self._lms):
            return
        with self._lock:
            info = next((m for m in self._lmstudio_models() or [] if m["id"] == model), None)
            if info and info.get("state") == "loaded":
                self._run_lms("unload", model)
            self._ready = False

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                return json.load(response)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError) as exc:
            raise LLMError(f"Lost the connection to the local model server at {self.base}: {exc}") from exc

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> LLMError:
        detail = exc.read().decode(errors="replace")[:600]
        return LLMError(f"Local model server returned HTTP {exc.code}: {detail}")

    def chat(self, model: str, system: str, user: str) -> tuple[str, int, int]:
        with self._lock:
            if not self._ready:
                self._ensure_loaded(model)
                self._ready = True

        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
        }
        # Reasoning models think at length before writing, which on local hardware
        # multiplies the time per section and eats the context window. Unless --think
        # was given, ask for a direct answer.
        if self._reasoning_off:
            try:
                result = self._post({**body, "reasoning_effort": "none"})
            except urllib.error.HTTPError as exc:
                if exc.code != 400:
                    raise self._http_error(exc) from exc
                self._reasoning_off = False  # this server or model does not take the setting
        if not self._reasoning_off:
            try:
                result = self._post(body)
            except urllib.error.HTTPError as exc:
                raise self._http_error(exc) from exc

        choice = (result.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise LLMError(
                "The local model ran out of context before finishing a section. "
                "Use a smaller --section-minutes or a larger --context-length."
            )
        # Reasoning models may inline their thinking; only the answer belongs in the notes.
        text = _THINK.sub("", (choice.get("message") or {}).get("content") or "")
        usage = result.get("usage") or {}
        return text, usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
