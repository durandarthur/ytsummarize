"""Local speech-to-text with faster-whisper, used when a video has no captions."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np

from . import hardware
from .video import Segment, Transcript

# The model follows the hardware unless --whisper-model names one: large-v3-turbo is
# accurate and fast on a GPU but slow on a CPU, where "small" keeps hours of audio
# practical at some cost in accuracy.
GPU_MODEL = "large-v3-turbo"
CPU_MODEL = "small"
# GPU memory Whisper needs while transcribing with the GPU model (weights + workspace).
GPU_BYTES = 3 * 2**30

_models: dict[tuple[str, str, str], object] = {}
_device: str | None = None  # decided once per run


def _load_cuda_libs() -> None:
    """Make the CUDA libraries findable by CTranslate2.

    They live in folders neither Windows nor Linux searches on their own.
    """
    roots = hardware.cuda_lib_roots()
    if os.name == "nt":
        for bin_dir in (d for root in roots for d in root.glob("*/bin")):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        return

    # Linux: the dynamic loader's search path is fixed at process start, so load the
    # libraries now; CTranslate2's later dlopen("libcublas.so.12") then reuses them.
    import ctypes

    libs = sorted({p for root in roots for p in root.glob("*/lib/lib*.so.*")})
    # A library whose own dependency is not loaded yet fails (cuBLAS needs cuBLASLt),
    # so retry until a pass makes no progress.
    while libs:
        failed = []
        for lib in libs:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(lib)
        if len(failed) == len(libs):
            break  # whatever is left is unusable here; the CPU fallback handles it
        libs = failed


def _has_cuda() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # no NVIDIA driver, or a CPU-only build
        return False


def device(force_cpu: bool = False, use_rocm: bool = False, log=print) -> str:
    """'cuda' (an NVIDIA GPU, or an AMD one with --rocm) or 'cpu'. Decided once per run."""
    global _device
    if _device is None:
        _device = _pick_device(force_cpu, use_rocm, log)
    return _device


def _pick_device(force_cpu: bool, use_rocm: bool, log) -> str:
    if force_cpu:
        return "cpu"
    if use_rocm:  # experimental AMD path; must run before ctranslate2 is first imported
        from . import rocm

        return "cuda" if rocm.activate(log) else "cpu"
    # CTranslate2's GPU support on PyPI is NVIDIA (CUDA) only.
    if not hardware.has_nvidia() or not _has_cuda():
        if hardware.has_amd():
            log("  (AMD GPU detected: --rocm tries experimental GPU transcription on it)")
        return "cpu"
    if not hardware.has_cuda_libs() and not hardware.install_cuda_libs(log):
        return "cpu"
    _load_cuda_libs()
    return "cuda"


def _decode(audio: Path) -> np.ndarray:
    """Decode any audio file to the 16 kHz mono float32 samples Whisper expects.

    Uses PyAV, whose wheels bundle the FFmpeg libraries, so no ffmpeg install is
    needed. (faster-whisper has its own PyAV-based decoder, but release 1.2.1 passes
    an argument that current PyAV no longer accepts.)
    """
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks: list[np.ndarray] = []
    try:
        with av.open(str(audio)) as container:
            for frame in container.decode(audio=0):
                chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(frame)]
            chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(None)]  # flush
    except (av.FFmpegError, IndexError) as exc:
        raise RuntimeError(f"Could not decode the downloaded audio: {exc}") from exc
    if not chunks:
        raise RuntimeError("The downloaded file contains no audio.")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _model(name: str, device: str, compute_type: str):
    """Load a model once and keep it for the following videos of the same run."""
    key = (name, device, compute_type)
    if key not in _models:
        from faster_whisper import WhisperModel

        _models[key] = WhisperModel(name, device=device, compute_type=compute_type)
    return _models[key]


def release() -> None:
    """Free the memory held by loaded Whisper models (a local LLM needs the GPU next)."""
    _models.clear()
    gc.collect()


def _run(samples: np.ndarray, model_name: str, language: str | None, device: str, compute_type: str, duration: float, log):
    pieces, info = _model(model_name, device, compute_type).transcribe(samples, language=language, vad_filter=True)
    segments: list[Segment] = []
    next_report = 0.0
    for piece in pieces:  # generator: the actual decoding happens here
        text = piece.text.strip()
        if text:
            segments.append(Segment(piece.start, text))
        if duration and piece.end >= next_report:
            log(f"  transcribing... {min(100, int(piece.end * 100 / duration))}%")
            next_report = piece.end + duration / 10
    return segments, info.language


def transcribe(
    audio: Path,
    model_name: str = "auto",
    language: str | None = None,
    duration: float = 0,
    force_cpu: bool = False,
    use_rocm: bool = False,
    log=print,
) -> Transcript:
    global _device
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    language = language.split("-")[0] if language else None
    samples = _decode(audio)
    device(force_cpu, use_rocm, log)

    if _device == "cuda":
        name = GPU_MODEL if model_name == "auto" else model_name
        try:
            segments, detected = _run(samples, name, language, "cuda", "float16", duration, log)
            return Transcript(segments, f"Whisper {name} ({detected}, local GPU)", detected)
        except (RuntimeError, ValueError) as exc:
            # The GPU is there but unusable: libraries failing to load, or too little free VRAM.
            log(f"  GPU transcription failed ({str(exc).splitlines()[0]}).")
            _device = "cpu"
            release()

    name = CPU_MODEL if model_name == "auto" else model_name
    log(f"  transcribing on the CPU with the Whisper {name} model (much slower than on an NVIDIA GPU).")
    segments, detected = _run(samples, name, language, "cpu", "int8", duration, log)
    return Transcript(segments, f"Whisper {name} ({detected}, local CPU)", detected)
