"""
Extract frames from a video file into a directory of PNGs.
"""

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path: str, output_dir: str, size: int = 256):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path.name}")
    print(f"Frames: {total}, FPS: {fps:.1f}, Duration: {total/fps:.1f}s")

    count = 0
    for i in tqdm(range(total), desc="Extracting"):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (size, size))
        cv2.imwrite(str(output_dir / f"frame_{count:05d}.png"), frame)
        count += 1

    cap.release()
    print(f"Extracted {count} frames to {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output-dir", type=str, default="data/frames")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    # Support directory of videos
    video_path = Path(args.video)
    if video_path.is_dir():
        videos = list(video_path.glob("*.mp4")) + list(video_path.glob("*.mov")) + list(video_path.glob("*.MOV"))
        if not videos:
            raise ValueError(f"No video files found in {video_path}")
        print(f"Found {len(videos)} videos")
        for v in videos:
            extract_frames(str(v), args.output_dir, args.size)
    else:
        extract_frames(args.video, args.output_dir, args.size)


if __name__ == "__main__":
    main()
