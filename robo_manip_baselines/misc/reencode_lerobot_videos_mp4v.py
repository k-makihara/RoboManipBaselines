#!/usr/bin/env python3
"""
Re-encode all LeRobot dataset videos under `videos/` to mp4v.

This keeps the directory layout intact, writes each file atomically through a
temporary file, and optionally updates `meta/info.json` so the stored codec
metadata matches the re-encoded videos.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from tqdm import tqdm


def load_dataset_fps(dataset_root: Path) -> float | None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return None
    with info_path.open() as file_handle:
        info = json.load(file_handle)
    fps = info.get("fps")
    return float(fps) if fps is not None else None


def update_info_json(dataset_root: Path) -> None:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open() as file_handle:
        info = json.load(file_handle)

    for feature in info.get("features", {}).values():
        if feature.get("dtype") != "video":
            continue
        feature_info = feature.setdefault("info", {})
        feature_info["video.codec"] = "mp4v"
        feature_info["video.pix_fmt"] = "yuv420p"
        feature_info["has_audio"] = False

    tmp_path = info_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as file_handle:
        json.dump(info, file_handle, indent=2)
    os.replace(tmp_path, info_path)


def probe_video_codec(video_path: Path) -> str | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=nw=1:nk=1",
        video_path.as_posix(),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    codec_name = result.stdout.strip().lower()
    return codec_name or None


def reencode_video(src_path: Path, fps_fallback: float | None, force: bool) -> bool:
    tmp_path = src_path.with_name(f"{src_path.stem}.tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()

    current_codec = probe_video_codec(src_path)
    if current_codec in {"mpeg4", "mp4v"} and not force:
        return False

    if fps_fallback is None:
        raise RuntimeError(f"Could not determine dataset FPS for {src_path}")

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        src_path.as_posix(),
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-sn",
        "-f",
        "mp4",
        tmp_path.as_posix(),
    ]
    subprocess.run(command, check=True)

    os.replace(tmp_path, src_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the LeRobot dataset root that contains meta/ and videos/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even when the source video already reports mp4v.",
    )
    parser.add_argument(
        "--no-update-info-json",
        action="store_true",
        help="Do not rewrite meta/info.json after re-encoding.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    videos_root = dataset_root / "videos"
    if not videos_root.exists():
        raise FileNotFoundError(f"Video directory not found: {videos_root}")

    fps_fallback = load_dataset_fps(dataset_root)
    video_paths = sorted(videos_root.rglob("*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No mp4 files found under {videos_root}")

    reencoded_count = 0
    skipped_count = 0
    for video_path in tqdm(video_paths, desc="Re-encoding videos"):
        changed = reencode_video(video_path, fps_fallback=fps_fallback, force=args.force)
        if changed:
            reencoded_count += 1
        else:
            skipped_count += 1

    if not args.no_update_info_json:
        update_info_json(dataset_root)

    print(
        f"Re-encoded {reencoded_count} videos to mp4v under {dataset_root} "
        f"(skipped {skipped_count} already-mp4v files)"
    )


if __name__ == "__main__":
    main()
