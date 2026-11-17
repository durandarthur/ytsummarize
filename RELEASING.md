# Releasing

Releases are built by GitHub Actions (`.github/workflows/release.yml`): pushing a tag
such as `v0.2.0` builds the Windows and Linux programs on GitHub's machines, tests each
one, and publishes them with checksums on the repository's Releases page.

## One-time setup

1. Install Git and the GitHub CLI, then sign in:

   ```powershell
   winget install Git.Git GitHub.cli
   gh auth login
   ```

2. Put the project under Git and create the GitHub repository (use `--private` instead
   of `--public` if you prefer; releases of a private repository are only visible to
   people with access to it):

   ```powershell
   cd C:\path\to\ytsummarize
   git init -b main
   git add .
   git commit -m "Initial commit"
   gh repo create ytsummarize --public --source . --push
   ```

   `.gitignore` already keeps `.venv/`, `.cache/`, `notes/`, `build/` and `dist/` out.

## Each release

1. Set the new version in `src/ytsummarize/__init__.py` (the only place it is defined):

   ```python
   __version__ = "0.2.0"
   ```

2. Commit, tag with the same version prefixed by `v`, and push both:

   ```powershell
   git commit -am "Release 0.2.0"
   git tag v0.2.0
   git push origin main v0.2.0
   ```

3. Watch the build (about 10 minutes), from the repository's **Actions** tab or with:

   ```powershell
   gh run watch
   ```

   When it finishes, the release is at `https://github.com/<you>/ytsummarize/releases`
   with these files:

   - `ytsummarize-0.2.0-windows-x64.zip`
   - `ytsummarize-0.2.0-linux-x64.tar.gz`
   - `SHA256SUMS.txt`

   If the tag and `__version__` disagree, the build stops before doing anything.
   Fix the version, then delete and recreate the tag:
   `git tag -d v0.2.0 && git push origin :v0.2.0`, and repeat step 2.

## Testing a build without releasing

Actions tab -> **Release** -> **Run workflow**. It builds and tests both programs and
attaches them to that run under **Artifacts**, without creating a release.

To build only for the machine you are on:

```powershell
uv sync --group build
uv run python scripts/build.py
```

The archive appears in `dist/`. PyInstaller cannot cross-compile, so a Windows machine
builds the Windows version only. A Linux binary built locally needs a glibc at least as
new as the build machine's; the workflow builds on Ubuntu 22.04 for that reason.

## Things users will see

- **Windows SmartScreen** ("Windows protected your PC") checks downloaded programs only
  when they are started from Explorer, e.g. by double-clicking. Programs started from a
  terminal (cmd, PowerShell) are not checked, and this is a terminal program, so most
  users never see it. Those who double-click it see the warning and can choose
  *More info* -> *Run anyway*.
- **Smart App Control** (Windows 11; on for some new installations) blocks unsigned
  programs outright, from a terminal too, with no way to run them anyway. Those users
  need the from-source install, or a signed build.
- Both are removed by code-signing the program, which needs a paid certificate; the
  workflow does not set that up.
- On Linux, users unpack with `tar -xzf ytsummarize-*.tar.gz` and run
  `./ytsummarize/ytsummarize`.
