"""What this machine can run, and the downloads that go with it.

GPUs of any vendor are detected, because LM Studio runs the local LLM on NVIDIA,
AMD and Intel GPUs alike, and the model choice depends on their memory. Whisper's
GPU acceleration (CTranslate2) needs NVIDIA; elsewhere it runs on the CPU.

The CUDA libraries GPU transcription needs are about 2 GB, so they are not part of
the install: they are fetched once, the first time Whisper runs on a machine that
actually has an NVIDIA GPU. Machines without one never download them.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

# CTranslate2 4.x (the engine under faster-whisper) is built against CUDA 12 and cuDNN 9.
# (PyPI package, major version) - the newest release of each with a build for this platform.
CUDA_PACKAGES = [("nvidia-cublas-cu12", 12), ("nvidia-cuda-nvrtc-cu12", 12), ("nvidia-cudnn-cu12", 9)]

_VENDORS = {"10de": "NVIDIA", "1002": "AMD", "8086": "Intel"}
# Less dedicated memory than this means an integrated GPU sharing system RAM.
_MIN_DEDICATED_MB = 2048


@dataclass(frozen=True)
class GPU:
    name: str
    vram_mb: int
    vendor: str = "NVIDIA"


def _nvidia_smi() -> list[GPU]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    for line in proc.stdout.splitlines():
        name, _, memory = line.rpartition(",")
        try:
            found.append(GPU(name.strip(), int(float(memory)), "NVIDIA"))
        except ValueError:
            continue
    return found


def _windows_adapters() -> list[GPU]:
    """Display adapters as recorded by Windows, with their exact dedicated memory."""
    import winreg

    key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    found = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key)
    except OSError:
        return []
    with root:
        for index in range(256):
            try:
                sub = winreg.EnumKey(root, index)
            except OSError:
                break
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as adapter:
                    values = {}
                    for name in ("DriverDesc", "MatchingDeviceId", "HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                        try:
                            values[name] = winreg.QueryValueEx(adapter, name)[0]
                        except OSError:
                            pass
            except OSError:
                continue
            device = str(values.get("MatchingDeviceId", "")).lower()
            vendor = next((v for code, v in _VENDORS.items() if f"ven_{code}" in device), None)
            if not vendor:
                continue  # virtual and remote-desktop adapters
            memory = values.get("HardwareInformation.qwMemorySize") or values.get("HardwareInformation.MemorySize") or 0
            if isinstance(memory, bytes):
                memory = int.from_bytes(memory[:8], "little")
            found.append(GPU(str(values.get("DriverDesc") or f"{vendor} GPU"), int(memory) // 2**20, vendor))
    return found


def _linux_adapters() -> list[GPU]:
    found = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if "-" in card.name:  # connectors such as card0-HDMI-A-1
            continue
        device = card / "device"
        try:
            vendor = _VENDORS.get(device.joinpath("vendor").read_text().strip().lower().removeprefix("0x"))
        except OSError:
            continue
        if not vendor:
            continue
        try:  # amdgpu reports its memory here; other drivers do not
            memory = int(device.joinpath("mem_info_vram_total").read_text()) // 2**20
        except (OSError, ValueError):
            memory = 0
        try:
            name = device.joinpath("product_name").read_text().strip()
        except OSError:
            name = ""
        found.append(GPU(name or f"{vendor} GPU ({card.name})", memory, vendor))
    return found


@cache
def gpus() -> tuple[GPU, ...]:
    """Every GPU found, NVIDIA ones as reported by their driver."""
    nvidia = _nvidia_smi()
    if os.name == "nt":
        others = _windows_adapters()
    elif sys.platform == "linux":
        others = _linux_adapters()
    else:
        others = []
    if nvidia:  # nvidia-smi's figures are the authoritative ones
        others = [g for g in others if g.vendor != "NVIDIA"]
    return tuple(nvidia + others)


def has_nvidia() -> bool:
    return any(g.vendor == "NVIDIA" for g in gpus())


def has_amd() -> bool:
    return any(g.vendor == "AMD" and g.vram_mb >= _MIN_DEDICATED_MB for g in gpus())


def vram_bytes() -> int | None:
    """Dedicated memory of the largest GPU (any vendor), or None when there is none."""
    sizes = [g.vram_mb for g in gpus() if g.vram_mb >= _MIN_DEDICATED_MB]
    return max(sizes) * 2**20 if sizes else None


def describe() -> str:
    found = [g for g in gpus() if g.vram_mb >= _MIN_DEDICATED_MB or g.vendor == "NVIDIA"]
    if not found:
        return "no dedicated GPU detected (Whisper runs on the CPU)"
    text = ", ".join(f"{g.name} ({g.vram_mb / 1024:.0f} GB)" for g in found)
    if has_amd() and not has_nvidia():
        text += " - the local LLM uses it; Whisper runs on the CPU unless you try --rocm (experimental)"
    elif not has_nvidia():
        text += " - the local LLM uses it; Whisper runs on the CPU (its GPU support needs NVIDIA, or AMD with --rocm)"
    return text


def data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "ytsummarize"


def _managed_cuda_dir() -> Path:
    return data_dir() / "cuda12"


def cuda_lib_roots() -> list[Path]:
    """Folders laid out as nvidia/<package>/{bin,lib}: ours, plus any installed into this environment."""
    roots = []
    managed = _managed_cuda_dir()
    if (managed / ".complete").exists():
        roots.append(managed / "nvidia")
    spec = importlib.util.find_spec("nvidia")
    if spec and spec.submodule_search_locations:
        roots += [Path(p) for p in spec.submodule_search_locations]
    return roots


def has_cuda_libs() -> bool:
    return any(
        any(root.glob("cublas/*/*cublas*")) and any(root.glob("cudnn/*/*cudnn*")) for root in cuda_lib_roots()
    )


def install_cuda_libs(log=print) -> bool:
    """Download the CUDA libraries into the user data folder. Returns True on success."""
    if sys.platform not in ("win32", "linux"):
        return False
    from . import downloads

    log("  NVIDIA GPU found: downloading the CUDA libraries for GPU transcription (one time, about 2 GB)...")
    target = _managed_cuda_dir()
    try:
        wheels = [downloads.pypi_wheel(name, major) for name, major in CUDA_PACKAGES]
        downloads.install([(url, None) for url, _ in wheels], target, log)
    except (OSError, RuntimeError, ValueError) as exc:
        log(f"  Could not download the CUDA libraries ({exc}); using the CPU instead.")
        return False
    (target / ".complete").write_text("\n".join(name for _, name in wheels) + "\n", encoding="utf-8")
    log(f"  CUDA libraries saved in {target}")
    return True
