#!/usr/bin/env python3
# 代码原理：
# 1. 读取 data/videos/video_index.csv 中的映射关系，其中 origin_file_name 是原始
#    视频文件名，index 是该视频对应的编号。
# 2. 对 CSV 中的每一行，用 origin_file_name 在 data/videos 下找到源视频，并用
#    index 创建输出目录 data/frames/{index}。
# 3. 通过 subprocess 调用 ffmpeg，把该视频的全部帧按 frame_000001.jpg、
#    frame_000002.jpg 等顺序写入对应编号目录。CSV 读取和路径拼接都在 Python
#    中完成，因此可以稳定处理中文文件名。
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

FRAME_PATTERN_DEFAULT = "frame_%06d.jpg"


def extract_all_frames(
    video_dir: str | Path = "data/videos",
    index_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    frame_pattern: str = FRAME_PATTERN_DEFAULT,
) -> int:
    video_dir = Path(video_dir)
    index_csv = (
        Path(index_csv) if index_csv is not None else video_dir / "video_index.csv"
    )
    output_root = (
        Path(output_dir) if output_dir is not None else video_dir.parent / "frames"
    )

    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg is not installed or not in PATH", file=sys.stderr)
        return 1

    if not video_dir.is_dir():
        print(f"error: video directory not found: {video_dir}", file=sys.stderr)
        return 1

    if not index_csv.is_file():
        print(f"error: index csv not found: {index_csv}", file=sys.stderr)
        return 1

    with index_csv.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            origin_file_name = row["origin_file_name"]
            index = row["index"]
            video_path = video_dir / origin_file_name
            video_output_dir = output_root / index

            if not video_path.is_file():
                print(f"warn: missing video, skip: {video_path}", file=sys.stderr)
                continue

            video_output_dir.mkdir(parents=True, exist_ok=True)
            output_pattern = video_output_dir / frame_pattern
            print(f"extract: {origin_file_name} -> {output_pattern}")
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video_path),
                    str(output_pattern),
                ],
                check=True,
            )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract all frames for videos listed in video_index.csv."
    )
    parser.add_argument(
        "video_dir",
        nargs="?",
        default="data/videos",
        help="Directory containing videos and video_index.csv.",
    )
    parser.add_argument(
        "index_csv",
        nargs="?",
        default=None,
        help="CSV mapping origin_file_name to index. Defaults to VIDEO_DIR/video_index.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store extracted frame folders. Defaults to VIDEO_DIR/../frames.",
    )
    parser.add_argument(
        "--frame-pattern",
        default=FRAME_PATTERN_DEFAULT,
        help="Output frame filename pattern for ffmpeg. Default: frame_%%06d.jpg.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return extract_all_frames(
        args.video_dir,
        args.index_csv,
        args.output_dir,
        frame_pattern=args.frame_pattern,
    )


if __name__ == "__main__":
    raise SystemExit(main())
