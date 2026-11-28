"""Member 4: DRISHTIQ Visual Demo & Verification App (Streamlit).

Run locally:
  uv run streamlit run src/drishtiq/demo/app.py

Run for 2-laptop / LAN presentation:
  uv run streamlit run src/drishtiq/demo/app.py --server.address 0.0.0.0 --server.port 8501
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from drishtiq.demo.helpers import (
    discover_videos,
    ensure_encoded_assets,
    extract_visual_inspection,
    start_chunk_server,
)
from drishtiq.demo.live_player import build_live_player_html
from drishtiq.streaming import run_stream, save_report

st.set_page_config(page_title="DRISHTIQ — ROI-Aware Streaming", page_icon="🎬", layout="wide")

PRESET_TRACES = {
    "Live Drop Simulation (3 Mbps ➔ 300 kbps)": [3000, 3000, 300, 300, 300, 300, 2000, 2500],
    "Slow Mobile (650–950 kbps)": [950, 750, 800, 650, 850, 700, 900, 950],
    "Very Slow 3G (350–550 kbps)": [500, 400, 350, 450, 550, 380, 420, 500],
    "Congested Wi-Fi (550–1200 kbps)": [1200, 600, 950, 550, 1100, 650, 900, 800],
}


@st.cache_resource
def get_chunk_server_port() -> int:
    return start_chunk_server("data/chunks", port=8502)


def main() -> None:
    st.title("DRISHTIQ — Viewer-Aware Adaptive Streaming")
    st.caption(
        "Preserving visually important regions (ROI) at lower bandwidth by compressing unimportant "
        "backgrounds harder and adapting smoothly during live network drops."
    )

    videos = discover_videos("data/raw")
    if not videos:
        st.error("No videos found in `data/raw/`. Add an `.mp4` file and refresh.")
        return

    with st.sidebar:
        st.header("Demo Controls")
        video_names = [v.name for v in videos]
        default_idx = (
            video_names.index("avengers_15s.mp4") if "avengers_15s.mp4" in video_names else 0
        )
        chosen_name = st.selectbox("Source video", video_names, index=default_idx)
        video_path = next(v for v in videos if v.name == chosen_name)

        preset_name = st.selectbox("Batch trace (Tab 2)", list(PRESET_TRACES.keys()), index=0)
        bw_scale = st.slider("Bandwidth multiplier", 0.5, 2.0, 1.0, 0.1)
        bw_trace = [int(x * bw_scale) for x in PRESET_TRACES[preset_name]]

        viewer_focus = st.selectbox(
            "Viewer refinement focus", ["people", "action", "text", "balanced"], index=0
        )
        budget_mb = st.slider("Data budget cap (MB, secondary)", 0.5, 45.0, 15.0, 0.5)
        draw_roi = st.checkbox("Highlight ROI box in inspector", value=True)
        show_heatmap_inspector = st.checkbox("Show saliency heatmap in inspector", value=True)

        st.divider()
        if st.button("Re-run M1 + M2 Pipeline (Rebuild Chunks)"):
            stem_dir = Path("data/chunks") / video_path.stem
            if stem_dir.exists():
                shutil.rmtree(stem_dir)
            for ext in ("json", "csv"):
                ana_file = Path("data/manifests") / f"{video_path.stem}_analysis.{ext}"
                if ana_file.exists():
                    ana_file.unlink()
            with st.spinner("Re-analyzing saliency & encoding 144p–720p chunks..."):
                ensure_encoded_assets(video_path, preset="veryfast", workers=4)
            st.rerun()

    stem = video_path.stem
    expected_conv = Path("data/chunks") / stem / "conventional" / "manifest.json"
    expected_ca = Path("data/chunks") / stem / "content_aware" / "manifest.json"
    expected_analysis = Path("data/manifests") / f"{stem}_analysis.json"

    if not (expected_conv.exists() and expected_ca.exists() and expected_analysis.exists()):
        st.info(f"`{chosen_name}` needs analysis & ROI encoding (using `veryfast` + `addroi`).")
        if st.button("Run Pipeline (Member 1 Analyze + Member 2 ROI Encode)", type="primary"):
            with st.spinner("Analyzing saliency maps & encoding Conventional vs DRISHTIQ..."):
                ensure_encoded_assets(video_path, preset="veryfast", workers=4)
            st.rerun()
        return

    analysis_data = json.loads(expected_analysis.read_text(encoding="utf-8"))
    ca_manifest = json.loads(expected_ca.read_text(encoding="utf-8"))
    has_144p = any(r["name"] == "144p" for r in ca_manifest.get("ladder", []))
    chunks_list = analysis_data.get("chunks", [])
    has_heatmap = bool(chunks_list and chunks_list[0].get("saliency_heatmap"))
    if not (has_144p and has_heatmap):
        st.warning(
            "This clip was encoded with an older pipeline version (missing 144p or saliency "
            "heatmaps). Click **Re-run M1 + M2 Pipeline (Rebuild Chunks)** in the sidebar."
        )

    budget_bytes = int(budget_mb * 1_000_000)
    conv_res = run_stream("conventional", expected_conv, bw_trace, budget_bytes)
    ca_res = run_stream(
        "budget_aware", expected_ca, bw_trace, budget_bytes, viewer_focus=viewer_focus
    )
    save_report(conv_res, Path("data/logs") / stem)
    save_report(ca_res, Path("data/logs") / stem)

    tab_stream, tab_compare, tab_raw = st.tabs(
        ["1. Live Stream & Network Drop", "2. Comparison & Proof", "3. How It Works & Raw Data"]
    )

    with tab_stream:
        st.subheader("Live Split-Screen Playback (Same Throttled Bandwidth)")
        st.caption(
            "Press **Drop to 300 kbps [D]** while playing to watch Conventional ABR stall/crash "
            "to 144p while DRISHTIQ adapts smoothly. Press **[H]** for the DRISHTIQ saliency "
            "heatmap and **[M]** (or ▶ Play / Restart) for audio."
        )
        port = get_chunk_server_port()
        req_host = (
            st.context.headers.get("Host", "127.0.0.1")
            if hasattr(st, "context")
            else "127.0.0.1"
        )
        server_host = req_host.split(":")[0] or "127.0.0.1"
        player_html = build_live_player_html(
            stem=stem,
            conv_manifest_path=expected_conv,
            ca_manifest_path=expected_ca,
            analysis_json_path=expected_analysis,
            chunk_server_port=port,
            viewer_focus=viewer_focus,
            server_host=server_host,
        )
        st.iframe(
            player_html,
            height=740,
            width="stretch",
            alt="DRISHTIQ live split-screen ABR player",
        )

        st.divider()
        st.markdown("### Static Frame & Zoomed ROI Inspector")
        conv_ok_ids = {
            c["chunk_index"] for c in conv_res["logs"] if c.get("status") == "downloaded"
        }
        downloaded_indices = [
            x["chunk_index"]
            for x in ca_res["logs"]
            if x.get("status") == "downloaded" and x["chunk_index"] in conv_ok_ids
        ]
        if downloaded_indices:
            pick_idx = st.select_slider(
                "Inspect chunk #", options=downloaded_indices, value=downloaded_indices[0]
            )
            conv_entry = next(x for x in conv_res["logs"] if x["chunk_index"] == pick_idx)
            ca_entry = next(x for x in ca_res["logs"] if x["chunk_index"] == pick_idx)
            chunk_hm = (
                analysis_data["chunks"][pick_idx].get("saliency_heatmap")
                if show_heatmap_inspector and pick_idx < len(analysis_data["chunks"])
                else None
            )
            vis = extract_visual_inspection(
                expected_conv,
                expected_ca,
                conv_entry,
                ca_entry,
                pick_idx,
                draw_roi,
                heatmap=chunk_hm,
            )

            f1, f2 = st.columns(2)
            conv_cap = (
                f"Conventional Chunk #{pick_idx} "
                f"({conv_entry['rendition']}, ROI SSIM {conv_entry['roi_ssim']:.4f})"
            )
            ca_cap = (
                f"DRISHTIQ Chunk #{pick_idx} "
                f"({ca_entry['rendition']}, ROI SSIM {ca_entry['roi_ssim']:.4f})"
            )
            f1.image(vis["conv_frame"], caption=conv_cap, width="stretch")
            f2.image(vis["ca_frame"], caption=ca_cap, width="stretch")

            st.markdown("#### Zoomed Side-by-Side: Important Region (ROI) vs. Background")
            z1, z2, z3, z4 = st.columns(4)
            z1.image(vis["conv_roi_crop"], caption="Normal: ROI", width="stretch")
            z2.image(vis["ca_roi_crop"], caption="DRISHTIQ: ROI", width="stretch")
            z3.image(vis["conv_bg_crop"], caption="Normal: BG", width="stretch")
            z4.image(vis["ca_bg_crop"], caption="DRISHTIQ: BG", width="stretch")

    with tab_compare:
        st.subheader("Measured Performance: Quality at Low Bandwidth")

        roi_gain = ca_res["average_roi_ssim"] - conv_res["average_roi_ssim"]
        conv_mb = conv_res["total_downloaded_bytes"] / 1e6
        ca_mb = ca_res["total_downloaded_bytes"] / 1e6
        byte_diff_pct = ((ca_mb / conv_mb) - 1) * 100 if conv_mb else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "DRISHTIQ Avg ROI SSIM",
            f"{ca_res['average_roi_ssim']:.4f}",
            f"{roi_gain:+.4f} vs Normal",
        )
        k2.metric("Normal Avg ROI SSIM", f"{conv_res['average_roi_ssim']:.4f}")
        k3.metric(
            "DRISHTIQ Stalls",
            f"{ca_res['total_stall_s']} s",
            f"Normal: {conv_res['total_stall_s']} s",
        )
        k4.metric(
            "Data Used (Secondary)",
            f"{ca_mb:.2f} MB",
            f"{byte_diff_pct:+.1f}% vs {conv_mb:.2f} MB",
        )

        conv_df = pd.DataFrame([x for x in conv_res["logs"] if x["status"] == "downloaded"])
        ca_df = pd.DataFrame([x for x in ca_res["logs"] if x["status"] == "downloaded"])

        g1, g2 = st.columns(2)
        with g1:
            fig_ssim = go.Figure()
            fig_ssim.add_trace(
                go.Scatter(
                    x=conv_df["chunk_index"],
                    y=conv_df["roi_ssim"],
                    name="Conventional ROI SSIM",
                    mode="lines+markers",
                )
            )
            fig_ssim.add_trace(
                go.Scatter(
                    x=ca_df["chunk_index"],
                    y=ca_df["roi_ssim"],
                    name="DRISHTIQ ROI SSIM",
                    mode="lines+markers",
                )
            )
            fig_ssim.update_layout(
                title="Important Region (ROI) Quality per Chunk (Display SSIM)",
                xaxis_title="Chunk #",
                yaxis_title="ROI SSIM",
            )
            st.plotly_chart(fig_ssim, width="stretch")

        with g2:
            fig_br = go.Figure()
            fig_br.add_trace(
                go.Bar(x=conv_df["chunk_index"], y=conv_df["bitrate_kbps"], name="Normal kbps")
            )
            fig_br.add_trace(
                go.Bar(x=ca_df["chunk_index"], y=ca_df["bitrate_kbps"], name="DRISHTIQ kbps")
            )
            fig_br.update_layout(
                title="Bitrate Allocation per Chunk",
                xaxis_title="Chunk #",
                yaxis_title="kbps",
                barmode="group",
            )
            st.plotly_chart(fig_br, width="stretch")

    with tab_raw:
        st.subheader("Pipeline Internals & Raw Output")
        m1, m2, m3 = st.columns(3)
        m1.markdown(
            "**1. Member 1 (Multi-Cue Saliency)**\n"
            "Fuses LAB color, motion, and edge density into a 2D heatmap + ROIs."
        )
        m2.markdown(
            "**2. Member 2 (Spatial ROI Encode)**\n"
            "Softens background (`gblur`) + applies `addroi` (`qoffset=-1/3`) on ROI."
        )
        m3.markdown(
            "**3. Member 3 (Viewer-Refined Stream)**\n"
            "Maximizes display `roi_ssim` smoothly during drops without crashing or stalling."
        )

        st.markdown("#### Streaming Decision Log (DRISHTIQ)")
        st.dataframe(pd.DataFrame(ca_res["logs"]), width="stretch")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("#### Member 1 Analysis JSON (Sample)")
            st.json(analysis_data["chunks"][:2])
        with r2:
            st.markdown("#### Member 2 ROI Manifest JSON (Sample)")
            st.json(ca_manifest["chunks"][:2])


if __name__ == "__main__":
    main()
