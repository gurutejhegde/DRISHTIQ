"""Interactive synchronized split-screen live chunk player for Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).with_name("live_player.html")


def build_live_player_html(
    stem: str,
    conv_manifest_path: str | Path,
    ca_manifest_path: str | Path,
    analysis_json_path: str | Path,
    chunk_server_port: int,
    viewer_focus: str = "people",
    server_host: str = "127.0.0.1",
) -> str:
    """Populate the self-contained HTML/JS synchronized dual chunk player."""
    conv_man = json.loads(Path(conv_manifest_path).read_text(encoding="utf-8"))
    ca_man = json.loads(Path(ca_manifest_path).read_text(encoding="utf-8"))
    analysis = json.loads(Path(analysis_json_path).read_text(encoding="utf-8"))

    chunks_raw = analysis.get("chunks", [])
    hm_by_idx = {c["chunk_index"]: c.get("saliency_heatmap", []) for c in chunks_raw}
    payload = {
        "stem": stem,
        "port": chunk_server_port,
        "host": server_host,
        "viewerFocus": viewer_focus,
        "convChunks": conv_man["chunks"],
        "caChunks": [
            {**c, "heatmap": hm_by_idx.get(c["index"], [])} for c in ca_man["chunks"]
        ],
    }
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("__DATA_JSON__", json.dumps(payload))
