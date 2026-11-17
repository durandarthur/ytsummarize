"""Build the standalone executable for the current platform.

    uv sync --group build
    uv run python scripts/build.py

Produces dist/ytsummarize-<version>-<os>-<arch>.zip (Windows) or .tar.gz (Linux,
macOS): a folder with the `ytsummarize` program and everything it needs, Python
included. The GPU libraries are not bundled; the program downloads them on first
use, only on machines that can use them.

PyInstaller cannot cross-compile, so each platform's archive must be built on that
platform; .github/workflows/release.yml does this on GitHub's machines.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DIST = ROOT / "dist"


def _version() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from ytsummarize import __version__

    return __version__


def _platform_name() -> str:
    os_name = {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform, sys.platform)
    machine = platform.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)
    return f"{os_name}-{arch}"


def main() -> int:
    version, target = _version(), _platform_name()
    shutil.rmtree(BUILD, ignore_errors=True)
    app_dir = BUILD / "dist" / "ytsummarize"

    subprocess.run(
        [
            sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "entry.py"),
            "--name", "ytsummarize",
            "--onedir",  # a folder: starts instantly, unlike --onefile which unpacks ~300 MB each run
            "--console",
            "--noconfirm",
            "--distpath", str(BUILD / "dist"),
            "--workpath", str(BUILD / "work"),
            "--specpath", str(BUILD),
            "--paths", str(ROOT / "src"),
            # Data files the libraries load at run time:
            "--collect-data", "faster_whisper",  # voice-activity model (silero_vad_v6.onnx)
            "--collect-all", "ctranslate2",
            "--collect-all", "av",
            "--copy-metadata", "anthropic",
            # Never bundled: the GPU runtimes are downloaded on demand, per machine.
            "--exclude-module", "nvidia",
            "--exclude-module", "torch",
        ],
        check=True,
        cwd=ROOT,
    )  # fmt: skip

    # Smoke test the result before packing it: every bundled piece must load.
    exe = app_dir / ("ytsummarize.exe" if sys.platform == "win32" else "ytsummarize")
    subprocess.run([str(exe), "--version"], check=True)
    subprocess.run([str(exe), "--self-test"], check=True)

    DIST.mkdir(exist_ok=True)
    name = f"ytsummarize-{version}-{target}"
    if sys.platform == "win32":
        archive = Path(shutil.make_archive(str(DIST / name), "zip", app_dir.parent, "ytsummarize"))
    else:  # tar keeps the executable bit
        archive = DIST / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(app_dir, arcname="ytsummarize")
    print(f"\nBuilt {archive} ({archive.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
