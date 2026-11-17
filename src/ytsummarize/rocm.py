"""EXPERIMENTAL: Whisper on AMD GPUs (--rocm). Untested on real AMD hardware.

faster-whisper runs on CTranslate2, whose PyPI builds support NVIDIA (CUDA) only.
CTranslate2 also publishes AMD ROCm builds on its GitHub release page. This module
downloads the one matching the running Python and loads it *instead of* the regular
CTranslate2, which must therefore happen before anything imports ctranslate2.

It needs AMD's HIP runtime:
  - Windows: AMD's ROCm 7.2 runtime packages, downloaded here too (the same ones
    CTranslate2's own CI tests its ROCm builds with); only their DLL folders are kept.
  - Linux: a system ROCm 7 install providing libamdhip64.so.7, libhipblas.so.3 and
    libhiprand.so.1 (AMD's `rocm-hip-libraries` package), usually under /opt/rocm.

Supported GPUs are whatever CTranslate2's ROCm builds target (mostly Radeon RX 6000,
7000 and 9000 series). Any failure falls back to the CPU.
"""

from __future__ import annotations

import ctypes
import importlib.machinery
import importlib.util
import os
import platform
import sys
from pathlib import Path

from . import downloads, hardware

CT2_VERSION = "4.9.0"
_CT2_ZIP = "https://github.com/OpenNMT/CTranslate2/releases/download/v{version}/rocm-python-wheels-{os}.zip"
_AMD_RUNTIME_WINDOWS = [
    # (wheel URL, the only part of it that is kept)
    ("https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl", "_rocm_sdk_core/bin/"),
    ("https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl", "_rocm_sdk_libraries_custom/bin/"),
]
_LINUX_LIBS = ["libamdhip64.so.7", "libhipblas.so.3", "libhiprand.so.1"]


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def install_dir() -> Path:
    return hardware.data_dir() / "rocm" / f"ctranslate2-{CT2_VERSION}-{_python_tag()}"


def is_installed() -> bool:
    return (install_dir() / ".complete").exists()


def _linux_rocm_lib_dir() -> Path | None:
    for base in (os.environ.get("ROCM_PATH"), "/opt/rocm"):
        if base and all((Path(base) / "lib" / lib).exists() for lib in _LINUX_LIBS):
            return Path(base) / "lib"
    return None


def _supported(log) -> bool:
    if platform.machine().lower() not in ("amd64", "x86_64"):
        log("  --rocm: AMD GPU builds exist for x86-64 Windows and Linux only.")
        return False
    if sys.platform == "linux" and _linux_rocm_lib_dir() is None:
        log(f"  --rocm: needs AMD's ROCm 7 libraries ({', '.join(_LINUX_LIBS)}), not found under "
            "/opt/rocm or $ROCM_PATH. Install ROCm (package rocm-hip-libraries) from https://rocm.docs.amd.com")  # fmt: skip
        return False
    return sys.platform in ("win32", "linux")


def download(log=print) -> bool:
    """Fetch CTranslate2's ROCm build (and, on Windows, AMD's runtime). Returns True on success."""
    target = install_dir()
    zip_os = "Windows" if sys.platform == "win32" else "Linux"
    log(f"  --rocm (experimental): downloading CTranslate2's AMD build{' and AMD runtime (about 1.3 GB)' if zip_os == 'Windows' else ''}, one time...")
    try:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="ytsummarize-") as tmp:
            bundle = downloads.fetch(_CT2_ZIP.format(version=CT2_VERSION, os=zip_os), Path(tmp) / "ct2.zip",
                                     "downloading CTranslate2 ROCm builds", log)  # fmt: skip
            import zipfile

            with zipfile.ZipFile(bundle) as zf:
                arch = "win_amd64" if zip_os == "Windows" else "x86_64"
                tag = _python_tag()
                wheel = next((n for n in zf.namelist() if f"-{tag}-{tag}-" in n and n.endswith(".whl") and arch in n), None)
                if wheel is None:
                    raise RuntimeError(f"no ROCm build of CTranslate2 {CT2_VERSION} for Python {tag}")
                wheel_path = Path(zf.extract(wheel, tmp))
            # Everything goes into one folder: ctranslate2/ from the wheel, plus AMD's DLL folders.
            sources: list = [(wheel_path, None)]
            if zip_os == "Windows":
                sources += _AMD_RUNTIME_WINDOWS
            downloads.install(sources, target, log)
    except (OSError, RuntimeError, ValueError) as exc:
        log(f"  --rocm: download failed ({exc}); using the CPU.")
        return False
    (target / ".complete").write_text(f"ctranslate2 {CT2_VERSION} ROCm\n", encoding="utf-8")
    return True


def _forget_ctranslate2() -> None:
    for name in [n for n in sys.modules if n == "ctranslate2" or n.startswith("ctranslate2.")]:
        del sys.modules[name]


def activate(log=print, allow_download: bool = True) -> bool:
    """Load the ROCm build of CTranslate2 in place of the regular one.

    Returns True when it loaded and sees at least one AMD GPU. Must run before
    anything imports ctranslate2 (faster-whisper does).
    """
    if "ctranslate2" in sys.modules:
        log("  --rocm: CTranslate2 was already loaded, so the AMD build cannot replace it.")
        return False
    if not _supported(log):
        return False
    if not hardware.has_amd():
        log("  --rocm: no AMD GPU detected; trying anyway.")
    if not is_installed() and not (allow_download and download(log)):
        return False

    root = install_dir()
    if sys.platform == "win32":
        for bin_dir in (root / "_rocm_sdk_core" / "bin", root / "_rocm_sdk_libraries_custom" / "bin"):
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    else:
        lib_dir = _linux_rocm_lib_dir()
        for lib in _LINUX_LIBS:  # loaded globally so CTranslate2's own lookup finds them
            ctypes.CDLL(str(lib_dir / lib), mode=ctypes.RTLD_GLOBAL)

    # Load the package from our folder explicitly: a plain `import` could still pick the
    # regular build (in the standalone builds, PyInstaller's importer comes first).
    try:
        spec = importlib.machinery.PathFinder.find_spec("ctranslate2", [str(root)])
        if spec is None or spec.loader is None:
            raise ImportError(f"no ctranslate2 package in {root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["ctranslate2"] = module
        spec.loader.exec_module(module)
        devices = module.get_cuda_device_count()  # HIP devices; CTranslate2 calls them "cuda"
    except Exception as exc:  # DLL/so load failures surface as ImportError or OSError
        _forget_ctranslate2()
        log(f"  --rocm: the AMD build of CTranslate2 failed to load ({str(exc).splitlines()[0]}); using the CPU.")
        return False
    if devices == 0:
        log("  --rocm: the AMD build loaded but sees no supported AMD GPU; using the CPU.")
        return False
    log(f"  --rocm: AMD GPU transcription enabled (experimental), {devices} device(s).")
    return True
