#!/usr/bin/env python3
"""
Compare conventional vs content-aware encodes (Member 2 report numbers).

Usage:
  uv run python scripts/compare.py \
      --conv data/chunks/conventional/manifest.json \
      --ca   data/chunks/content_aware/manifest.json \
      --out  data/logs/compare

Outputs: a table in the terminal, <out>.csv and <out>.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: str) -> dict[str, list[dict]]:
    """rendition name -> list of per-chunk renditions."""
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for c in man["chunks"]:
        for r in c["renditions"]:
            out.setdefault(r["name"], []).append({**r, "chunk": c["index"]})
    return out


def stats(rows: list[dict]) -> dict:
    sizes = [r["size_bytes"] for r in rows]
    ssims = [r["ssim"] for r in rows if r["ssim"] is not None]
    roi_ssims = [r["roi_ssim"] for r in rows if r.get("roi_ssim") is not None]
    total_mb = sum(sizes) / 1e6
    mean_ssim = sum(ssims) / len(ssims) if ssims else float("nan")
    mean_roi = sum(roi_ssims) / len(roi_ssims) if roi_ssims else float("nan")
    return {
        "mb": total_mb,
        "mean_ssim": mean_ssim,
        "mean_roi_ssim": mean_roi,
        "min_ssim": min(ssims) if ssims else float("nan"),
        "ssim_per_mb": mean_ssim / total_mb if total_mb else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", required=True)
    ap.add_argument("--ca", required=True)
    ap.add_argument("--out", default="data/logs/compare")
    a = ap.parse_args()

    conv, ca = load(a.conv), load(a.ca)
    rungs = [r for r in conv if r in ca]
    rows = []
    for r in rungs:
        c, k = stats(conv[r]), stats(ca[r])
        rows.append({
            "rendition": r,
            "conv_MB": round(c["mb"], 3), "ca_MB": round(k["mb"], 3),
            "bytes_change_%": round((k["mb"] / c["mb"] - 1) * 100, 1),
            "conv_mean_ssim": round(c["mean_ssim"], 5),
            "ca_mean_ssim": round(k["mean_ssim"], 5),
            "conv_roi_ssim": round(c["mean_roi_ssim"], 5),
            "ca_roi_ssim": round(k["mean_roi_ssim"], 5),
        })

    cols = list(rows[0].keys())
    print("  ".join(f"{c:>15}" for c in cols))
    for row in rows:
        print("  ".join(f"{str(row[c]):>15}" for c in cols))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    try:  # chart: quality-vs-bytes per rendition
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for r in rungs:
            c, k = stats(conv[r]), stats(ca[r])
            ax.scatter(c["mb"], c["mean_ssim"], marker="o", s=70, color="tab:blue")
            ax.scatter(k["mb"], k["mean_ssim"], marker="^", s=80, color="tab:orange")
            ax.annotate(r, (k["mb"], k["mean_ssim"]), textcoords="offset points", xytext=(6, 4))
        ax.scatter([], [], marker="o", color="tab:blue", label="Conventional")
        ax.scatter([], [], marker="^", color="tab:orange", label="Content-aware")
        ax.set_xlabel("Total size (MB)")
        ax.set_ylabel("Mean SSIM")
        ax.set_title("Quality vs bytes per rendition")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out.with_suffix(".png"), dpi=150)
        print(f"\nSaved {out.with_suffix('.csv')} and {out.with_suffix('.png')}")
    except ImportError:
        print(f"\nSaved {out.with_suffix('.csv')} (matplotlib not found, skipped chart)")


if __name__ == "__main__":
    main()
