"""Reading a Markdown list of videos.

Headings become folders (a `##` heading is a subfolder of the `#` heading above it)
and every bullet point with a link becomes a video or playlist to process:

    # Machine learning
    ## Foundations
    - https://www.youtube.com/watch?v=aircAruvnKk
    - [Backpropagation](https://youtu.be/Ilg3gGewQ5U)
    ## Transformers
    - https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

Finished bullets are ticked in the file (`- [x] link`) and skipped on later runs.
"""

from __future__ import annotations

import codecs
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from yt_dlp.utils import sanitize_filename

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:\s+(.*?))?(?:\s+#+)?\s*$")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d{1,9}[.)])\s+(.*)$")
_BULLET_PREFIX = re.compile(r"^(\s*(?:[-*+]|\d{1,9}[.)])\s+)(\[([ xX])\]\s+)?")
_CHECKBOX = re.compile(r"^\[([ xX])\]\s+")  # "[x] " - unlike "[x](link)", which is a link
# A Markdown link's target, an <autolink>, or a bare URL - whichever comes first.
_URL = re.compile(
    r"\[[^\]]*\]\(\s*<?(https?://[^)\s>]+)>?[^)]*\)"
    r"|<(https?://[^>\s]+)>"
    r"|(https?://[^\s<>()\[\]]+)"
)
_INLINE_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_EMPHASIS = re.compile(r"[*`~]+|(?<!\w)_+|_+(?!\w)")


@dataclass(frozen=True)
class ListItem:
    url: str
    folder: tuple[str, ...]  # folder names, outermost first
    line: int
    done: bool = False  # ticked: "- [x] link"


def first_url(text: str) -> str | None:
    match = _URL.search(text)
    if not match:
        return None
    return next(g for g in match.groups() if g).rstrip(".,;:!?'\"")


def folder_name(heading: str) -> str:
    """A heading's text as a safe folder name: Markdown removed, illegal characters replaced."""
    text = _EMPHASIS.sub("", _INLINE_LINK.sub(r"\1", heading)).strip()
    name = sanitize_filename(text)[:100].strip(" .")
    return name or "Untitled"


def parse_list(text: str) -> tuple[list[ListItem], list[str]]:
    """Return the list's items, plus warnings as "line N: message" strings."""
    items: list[ListItem] = []
    warnings: list[str] = []
    stack: list[tuple[int, str]] = []  # open headings as (level, folder name)
    seen: set[tuple[tuple[str, ...], str]] = set()
    lines = text.splitlines()

    start = 0
    if lines and lines[0].strip() == "---":  # YAML front matter
        start = next((i + 1 for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), 0)

    in_fence = in_comment = False
    for number, line in enumerate(lines[start:], start + 1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if in_comment:
            if "-->" not in line:
                continue
            line, in_comment = line.split("-->", 1)[1], False
        line = re.sub(r"<!--.*?-->", "", line)
        if "<!--" in line:
            line, in_comment = line.split("<!--", 1)[0], True

        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, folder_name(heading.group(2) or "")))
            continue

        bullet = _BULLET.match(line)
        url = first_url(bullet.group(1) if bullet else line)
        if not bullet:
            if url:
                warnings.append(f"line {number}: link outside a bullet point, ignored: {url}")
            continue
        if not url:
            if bullet.group(1).strip():
                warnings.append(f"line {number}: bullet point without a link, ignored")
            continue
        folder = tuple(name for _, name in stack)
        if (folder, url) in seen:
            warnings.append(f"line {number}: same link already listed under this heading, ignored")
            continue
        seen.add((folder, url))
        checkbox = _CHECKBOX.match(bullet.group(1))
        items.append(ListItem(url, folder, number, done=bool(checkbox and checkbox.group(1) in "xX")))
    return items, warnings


def tick(path: Path, item: ListItem) -> bool:
    """Tick an item's bullet in the list file: "- link" becomes "- [x] link".

    Only that line changes; everything else, line endings and encoding included, is
    kept byte for byte. The file is re-read first, so edits made while a batch runs
    are kept; if the item's line moved, it is found by its link. Returns False when
    the bullet cannot be found unambiguously or the file cannot be written.
    """
    try:
        raw = path.read_bytes()
        bom = raw.startswith(codecs.BOM_UTF8)
        lines = raw.decode("utf-8-sig").splitlines(keepends=True)
    except (OSError, UnicodeDecodeError):
        return False

    def is_item(line: str) -> bool:
        bullet = _BULLET.match(line.rstrip("\r\n"))
        return bool(bullet) and first_url(bullet.group(1)) == item.url

    index = item.line - 1
    if not (0 <= index < len(lines) and is_item(lines[index])):
        matches = [i for i, line in enumerate(lines) if is_item(line)]
        if len(matches) != 1:
            return False
        index = matches[0]

    line = lines[index]
    prefix = _BULLET_PREFIX.match(line)
    if prefix.group(3) in ("x", "X"):
        return True  # already ticked
    lines[index] = prefix.group(1) + "[x] " + line[prefix.end():]
    data = (codecs.BOM_UTF8 if bom else b"") + "".join(lines).encode("utf-8")

    temp = path.with_name(f".{path.name}.ytsummarize-tmp")
    try:  # write a copy, then swap it in, so the list is never left half-written
        temp.write_bytes(data)
        shutil.copymode(path, temp)
        os.replace(temp, path)
    except OSError:
        temp.unlink(missing_ok=True)
        return False
    return True
