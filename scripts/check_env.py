#!/usr/bin/env python3
"""Cross-platform environment diagnostic script for NazarBit.

Checks:
1. Python version (>= 3.12, < 3.13)
2. External binaries: ffmpeg and ffprobe
3. Core Python library imports for all 4 team members
4. Project skeleton directory structure
"""

import importlib
import shutil
import sys
from pathlib import Path


def print_status(component: str, ok: bool, details: str = ""):
    symbol = "✅" if ok else "❌"
    print(f"{symbol} {component:<30} {details}")


def check_python_version() -> bool:
    v = sys.version_info
    ok = (v.major == 3) and (v.minor >= 12) and (v.minor < 13)
    details = f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if not ok:
        details += " [Expected Python 3.12.x]"
    print_status("Python Version", ok, details)
    return ok


def check_binary(name: str) -> bool:
    path = shutil.which(name)
    ok = path is not None
    details = path if ok else f"'{name}' not found on PATH. Please install FFmpeg (see SETUP.md)."
    print_status(f"Binary: {name}", ok, details)
    return ok


def check_imports() -> bool:
    modules = [
        ("Core: Pydantic", "pydantic"),
        ("Core: NumPy", "numpy"),
        ("Core: Pandas", "pandas"),
        ("Member 1: OpenCV", "cv2"),
        ("Member 1: PySceneDetect", "scenedetect"),
        ("Member 1: Scikit-Image", "skimage"),
        ("Member 1: SciPy", "scipy"),
        ("Member 1: Pillow", "PIL"),
        ("Member 2: FFmpeg-Python", "ffmpeg"),
        ("Member 2: M3U8 Manifests", "m3u8"),
        ("Member 3: HTTPX Simulator", "httpx"),
        ("Member 4: Streamlit", "streamlit"),
        ("Member 4: Plotly", "plotly"),
        ("Member 4: Matplotlib", "matplotlib"),
    ]
    all_ok = True
    for label, mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print_status(label, True, f"v{ver}")
        except ImportError as e:
            print_status(label, False, f"ImportError: {e}")
            all_ok = False
    return all_ok


def check_directories() -> bool:
    root = Path(__file__).resolve().parent.parent
    expected_dirs = [
        root / "data" / "raw",
        root / "data" / "chunks",
        root / "data" / "manifests",
        root / "data" / "logs",
        root / "src" / "nazarbit" / "core",
        root / "src" / "nazarbit" / "analysis",
        root / "src" / "nazarbit" / "encoding",
        root / "src" / "nazarbit" / "streaming",
        root / "src" / "nazarbit" / "demo",
        root / "tests",
    ]
    all_ok = True
    for p in expected_dirs:
        rel = p.relative_to(root)
        exists = p.exists() and p.is_dir()
        if not exists:
            all_ok = False
        print_status(f"Directory: {rel}", exists)
    return all_ok


def main():
    print("=" * 70)
    print(" NazarBit Environment & Dependency Diagnostic")
    print("=" * 70)

    py_ok = check_python_version()
    ffmpeg_ok = check_binary("ffmpeg")
    ffprobe_ok = check_binary("ffprobe")
    dirs_ok = check_directories()

    print("-" * 70)
    print("Checking Python Packages Across All 4 Member Roles...")
    pkgs_ok = check_imports()

    print("=" * 70)
    if py_ok and ffmpeg_ok and ffprobe_ok and dirs_ok and pkgs_ok:
        print("🎉 SUCCESS: All environment requirements and libraries are satisfied!")
        sys.exit(0)
    else:
        print("⚠️  ACTION REQUIRED: Some components are missing or misconfigured.")
        print("    Run 'uv sync --all-extras' and refer to SETUP.md.")
        sys.exit(1)


if __name__ == "__main__":
    main()
