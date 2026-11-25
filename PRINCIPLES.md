# NazarBit: Core Principles & Project Goal

## 🎯 The Main Goal
> **Develop an intelligent video streaming system that identifies visually important regions, preserves their quality while reducing quality in less important regions, and intelligently allocates available bandwidth to deliver a better viewing experience with lower data consumption and minimal buffering.**

---

## 📐 Engineering Guidelines for All Members

1. **Structured & Readable Code**:
   - Write clean, understandable code with clear variable and function names.
   - Avoid bloated abstractions, nested boilerplate, and redundant wrapper classes.
   - Keep functions focused: each function should do one thing well.

2. **Data Contracts First**:
   - All inter-module communication must use strict schemas defined in `src/nazarbit/core/schemas.py`.
   - Never pass loose untyped dictionaries between modules.

3. **No `pip` Rule**:
   - All dependencies are managed with `uv`. Never run `pip install`. Use `uv add <pkg>`.

4. **Verify What You Build**:
   - Every module must have automated unit tests in `tests/` verifying accuracy and performance.
   - Run `uv run --no-sync ruff check .` and `uv run --no-sync pytest` before pushing.
