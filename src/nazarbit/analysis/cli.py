"""Command-line interface for Member 1: Content Analysis."""

import argparse
from pathlib import Path

from nazarbit.analysis.analyzer import ContentAnalyzer


def main():
    parser = argparse.ArgumentParser(description="NazarBit Content Analysis CLI (Member 1)")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=2.0,
        help="Chunk duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=4.0,
        help="Sampling rate in frames-per-second (default: 4.0)",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON report path")
    parser.add_argument("--output-csv", type=str, default=None, help="Output CSV report path")

    args = parser.parse_args()
    video_p = Path(args.video_path)
    if not video_p.exists():
        print(f"Error: Video file '{video_p}' does not exist.")
        raise SystemExit(1)

    print(f"🔍 Analyzing video: {video_p.name} (chunk size: {args.chunk_duration}s)...")
    analyzer = ContentAnalyzer(chunk_duration_s=args.chunk_duration, sample_fps=args.sample_fps)
    report = analyzer.analyze(video_p)

    print("📊 Analysis Complete:")
    print(f"   • Total Chunks: {report.total_chunks}")
    print(f"   • Duration:     {report.metadata.duration_s}s ({report.metadata.fps} fps)")
    print(f"   • Resolution:   {report.metadata.width}x{report.metadata.height}")

    # Default output paths if none provided
    stem = video_p.stem
    if args.output_json:
        output_json = Path(args.output_json)
    else:
        output_json = Path(f"data/manifests/{stem}_analysis.json")

    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        output_csv = Path(f"data/manifests/{stem}_analysis.csv")

    report.to_json(output_json)
    report.to_csv(output_csv)
    print(f"   • Saved JSON:   {output_json}")
    print(f"   • Saved CSV:    {output_csv}")


if __name__ == "__main__":
    main()
