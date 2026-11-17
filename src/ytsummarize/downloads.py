"""Fetching optional GPU runtimes without pip.

The standalone builds have no pip (and users may not have uv), so the GPU libraries
are downloaded as plain files: wheels from PyPI or a release page, unpacked with
zipfile. A wheel is just a zip archive laid out the way pip --target would install it.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import ssl
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_USER_AGENT = "ytsummarize (https://pypi.org/project/yt-dlp/ based tool)"


def _ssl_context() -> ssl.SSLContext:
    try:  # standalone builds cannot rely on the system's certificate locations
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _open(url: str, timeout: float = 60):
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    return urllib.request.urlopen(request, context=_ssl_context(), timeout=timeout)


def fetch(url: str, dest: Path, label: str, log=print) -> Path:
    """Download a file, reporting progress every 25%."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _open(url) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, reported = 0, 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total and done * 4 // total > reported:
                reported = done * 4 // total
                log(f"  {label}: {reported * 25}% of {total / 1e9:.1f} GB")
    return dest


def _wheel_fits_here(filename: str) -> bool:
    machine = platform.machine().lower()
    if sys.platform == "win32":
        return filename.endswith("win_arm64.whl" if machine in ("arm64", "aarch64") else "win_amd64.whl")
    if sys.platform == "linux":
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        return "manylinux" in filename and filename.endswith(f"{arch}.whl")
    return False


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def pypi_wheel(name: str, major: int) -> tuple[str, str]:
    """URL and filename of the newest wheel of `name` in major version `major` for this platform."""
    with _open(f"https://pypi.org/pypi/{name}/json") as response:
        releases = json.load(response)["releases"]
    candidates = []
    for version, files in releases.items():
        if not re.fullmatch(r"\d+(\.\d+)*", version) or _version_key(version)[0] != major:
            continue  # pre-releases and other major versions
        for file in files:
            if file.get("packagetype") == "bdist_wheel" and not file.get("yanked") and _wheel_fits_here(file["filename"]):
                candidates.append((_version_key(version), file["url"], file["filename"]))
    if not candidates:
        raise RuntimeError(f"PyPI has no {name} {major}.x build for this platform")
    _, url, filename = max(candidates)
    return url, filename


def unpack(archive: Path, target: Path, members: str | None = None) -> None:
    """Extract a zip (or wheel); with `members`, only paths starting with that prefix."""
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if members is None or n.startswith(members)]
        zf.extractall(target, names)


def install(sources: list[tuple[str | Path, str | None]], target: Path, log=print) -> None:
    """Unpack each (URL or local archive, member prefix) into `target`, atomically.

    The files land in a temporary folder that is renamed into place only once
    everything succeeded, so an interrupted download never leaves a broken install.
    """
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(prefix="ytsummarize-") as tmp:
            for source, members in sources:
                if isinstance(source, Path):
                    unpack(source, partial, members)
                    continue
                filename = source.rsplit("/", 1)[-1].split("?")[0]
                archive = fetch(source, Path(tmp) / filename, f"downloading {filename}", log)
                unpack(archive, partial, members)
                archive.unlink()
        shutil.rmtree(target, ignore_errors=True)
        partial.rename(target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
