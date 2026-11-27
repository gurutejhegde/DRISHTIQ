
"""Export DrishtiQ streaming results to JSON and CSV."""

import csv
import json
from pathlib import Path
from typing import Any


def save_report(
    results: dict[str, Any],
    output_dir: str | Path = "reports",
) -> tuple[Path, Path]:
    """Save streaming results as JSON and CSV files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mode = results.get("mode", "streaming")

    json_file = output_path / f"{mode}_report.json"
    csv_file = output_path / f"{mode}_report.csv"

    # Save complete summary and chunk logs in JSON.
    json_file.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    # Save individual chunk records in CSV.
    logs = results.get("logs", [])

    if logs:
        fieldnames = sorted({
            key
            for entry in logs
            for key in entry.keys()
        })

        with csv_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(logs)

    return json_file, csv_file
