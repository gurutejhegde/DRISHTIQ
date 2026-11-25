# DrishtiQ Team Setup Guide (Cross-Platform)

Welcome to the **DrishtiQ** project! This guide ensures that all 4 team members have the exact same environment and dependencies with zero version conflicts.

> ### 🛑 CRITICAL RULE: DO NOT USE `pip install`!
> **Never run `pip install` anywhere in this project.**
> - `pip install` will bypass the lockfile, install unpinned packages, and break other teammates' code.
> - **To install dependencies:** Always use `uv sync --all-extras`.
> - **To run any code:** Always prefix with `uv run` (e.g., `uv run python ...` or `uv run streamlit ...`).
> - **To add a new library:** Always use `uv add <package>` (this updates `pyproject.toml` and locks it in `uv.lock` for everyone).

---

## 📋 Quick 3-Step Setup for Teammates

### Step 1: Install `uv` & `ffmpeg` (One-Time Only)

#### A. Install `uv` (Fast Python Package & Version Manager)
`uv` automatically manages Python 3.12 without touching your computer's default Python.
- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Linux (Ubuntu/Debian/Fedora)**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
*(Close and reopen your terminal after running this command).*

#### B. Install `ffmpeg` & `ffprobe` (Video Processing Engine)
- **Windows**:
  ```powershell
  winget install Gyan.FFmpeg
  ```
- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```

---

### Step 2: Clone & Sync the Project
Run this single command inside the repository root:

```bash
uv sync --extra dev
```

`uv` will automatically:
1. Download Python 3.12 (if not already installed).
2. Create the `.venv` virtual environment.
3. Install the exact same versions of all packages locked in `uv.lock`.

---

### Step 3: Verify Your Setup
Run our automated verification script:

```bash
uv run python scripts/check_env.py
```

If you see green checkmarks `✅` across all checks, your machine is 100% ready!

---

## 💻 Daily Development Commands

Always run commands via `uv run`:

| Goal | Command |
| :--- | :--- |
| **Run a script** | `uv run python path/to/script.py` |
| **Run the Streamlit Demo** | `uv run streamlit run src/drishtiq/demo/app.py` |
| **Run Tests** | `uv run pytest` |
| **Format & Lint Code** | `uv run ruff check .` and `uv run ruff format .` |
| **Add a new library** | `uv add <package>` (never use pip!) |

---

## 👥 Team Module Ownership

| Member | Module Directory | Responsibility |
| :--- | :--- | :--- |
| **Member 1** | `src/drishtiq/analysis/` | Scene cut detection, motion/texture complexity, ROI detection |
| **Member 2** | `src/drishtiq/encoding/` | FFmpeg pipelines, fixed vs. content-aware chunk ladders, SSIM |
| **Member 3** | `src/drishtiq/streaming/` | ABR baseline, budget-aware bitrate selection, network simulation |
| **Member 4** | `src/drishtiq/demo/` | Streamlit side-by-side dashboard, player, metrics & charts |
| **Shared** | `src/drishtiq/core/` | Shared data schemas (`schemas.py`), configuration (`config.py`) |
