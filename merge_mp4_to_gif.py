#!/usr/bin/env python3
"""Merge all MP4 files in a directory into a single GIF.

Usage:
	python merge_mp4_to_gif.py /path/to/dir -o out.gif --fps 15 --scale 640:-1

The script will:
 - Find all `.mp4` files (non-recursive) in the given directory, sorted by name.
 - Create an ffmpeg concat list and concatenate them into a single temporary mp4.
 - Generate a palette and convert the temporary mp4 into a high-quality GIF.

Requires `ffmpeg` installed and on PATH.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


def find_mp4_files(directory: Path) -> List[Path]:
	files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"]
	return sorted(files)


def run(cmd: List[str]) -> None:
	print("Running:", " ".join(shlex.quote(c) for c in cmd))
	subprocess.check_call(cmd)


def concatenate_mp4s(mp4_files: List[Path], out_mp4: Path) -> None:
	# Create a concat list file for ffmpeg
	with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
		list_path = Path(f.name)
		for p in mp4_files:
			# ffmpeg concat demuxer requires paths like: file 'path'
			# Use absolute paths so ffmpeg (which resolves relative paths
			# relative to the list file) can always find the inputs.
			f.write(f"file '{p.resolve().as_posix()}'\n")

	try:
		cmd = [
			"ffmpeg",
			"-y",
			"-f",
			"concat",
			"-safe",
			"0",
			"-i",
			str(list_path),
			"-c",
			"copy",
			str(out_mp4),
		]
		run(cmd)
	finally:
		try:
			list_path.unlink()
		except Exception:
			pass


def mp4_to_gif(input_mp4: Path, out_gif: Path, fps: int = 15, scale: str = "-1:-1") -> None:
	# Generate palette
	with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as pal:
		palette_path = Path(pal.name)

	try:
		palette_cmd = [
			"ffmpeg",
			"-y",
			"-i",
			str(input_mp4),
			"-vf",
			f"fps={fps},scale={scale}:flags=lanczos,palettegen",
			str(palette_path),
		]
		run(palette_cmd)

		gif_cmd = [
			"ffmpeg",
			"-y",
			"-i",
			str(input_mp4),
			"-i",
			str(palette_path),
			"-filter_complex",
			f"fps={fps},scale={scale}:flags=lanczos[x];[x][1:v]paletteuse",
			str(out_gif),
		]
		run(gif_cmd)
	finally:
		try:
			palette_path.unlink()
		except Exception:
			pass


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Merge all mp4 files in a directory into a single GIF")
	p.add_argument("directory", type=Path, help="Directory containing mp4 files (non-recursive)")
	p.add_argument("-o", "--output", type=Path, default=Path("out.gif"), help="Output GIF path")
	p.add_argument("--fps", type=int, default=15, help="FPS for the output GIF")
	p.add_argument("--scale", type=str, default="-1:-1", help="Scale filter for ffmpeg, e.g. 640:-1")
	return p.parse_args()


def main() -> int:
	args = parse_args()
	directory: Path = args.directory

	if not directory.exists() or not directory.is_dir():
		print(f"Error: directory '{directory}' not found or is not a directory", file=sys.stderr)
		return 2

	mp4_files = find_mp4_files(directory)
	if not mp4_files:
		print(f"No mp4 files found in '{directory}'", file=sys.stderr)
		return 1

	with tempfile.TemporaryDirectory() as td:
		td_path = Path(td)
		concat_mp4 = td_path / "_merged_temp.mp4"

		# If there's only one file, skip concatenation (copy to temp)
		if len(mp4_files) == 1:
			print("Only one mp4 found, copying to temporary file before GIF conversion")
			run(["ffmpeg", "-y", "-i", str(mp4_files[0]), "-c", "copy", str(concat_mp4)])
		else:
			concatenate_mp4s(mp4_files, concat_mp4)

		mp4_to_gif(concat_mp4, args.output, fps=args.fps, scale=args.scale)

	print(f"Created GIF: {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
