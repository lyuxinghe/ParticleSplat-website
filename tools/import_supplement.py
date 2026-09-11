"""Import the supplied supplementary videos as browser-ready, on-demand media.

Run with the repository virtual environment (OpenCV and imageio-ffmpeg required):
  .venv/bin/python website/tools/import_supplement.py /path/to/supplement.zip

Existing outputs are protected unless --overwrite is explicitly supplied.
PDFs are intentionally excluded from the public website output.
The website itself does not require Python or these preparation dependencies.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from zipfile import ZipFile

import cv2
import imageio_ffmpeg


SITE = Path(__file__).resolve().parents[1]
TASKS = {
    "close_jar": ("close-jar", "Close jar"),
    "push_buttons": ("push-buttons", "Push buttons"),
    "meat_off_grill": ("meat-off-grill", "Meat off grill"),
    "slide_block": ("slide-block", "Slide block"),
    "read_and_drag": ("drag-stick", "Drag stick"),
    "sweep_to_dustpan": ("sweep-to-dustpan", "Sweep to dustpan"),
    "turn_tap": ("turn-tap", "Turn tap"),
    "open_drawer": ("open-drawer", "Open drawer"),
    "put_item_in_drawer": ("put-in-drawer", "Put in drawer"),
    "stack_blocks": ("stack-blocks", "Stack blocks"),
}
OUTPUTS = {
    "rgb": "Novel-view RGB reconstruction",
    "segmentation": "Object-centric segmentation",
    "depth": "Depth reconstruction",
    "foreground": "Foreground rendering",
    "background": "Background rendering",
    "keypoints_3d": "3D keypoints",
    "bbox_2d": "2D bounding boxes",
    "bbox_3d": "3D bounding boxes",
}
TRIALS = {"s1": ("success-1", "Success 1"), "s2": ("success-2", "Success 2")}


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(member):
    parts = PurePosixPath(member).parts
    if not parts or parts[0] == "__MACOSX" or any(part.startswith(".") for part in parts):
        return None
    if len(parts) == 4 and parts[1] == "videos" and parts[2] in {"rlbench_decomp", "realdworld_decomp"}:
        dataset = "rlbench" if parts[2] == "rlbench_decomp" else "real-world"
        prefix = "rlbench_sweep_" if dataset == "rlbench" else "sweep_"
        stem = PurePosixPath(parts[3]).stem
        if not stem.startswith(prefix) or not member.endswith(".mp4"):
            return None
        output = stem.removeprefix(prefix)
        if output not in OUTPUTS:
            raise ValueError(f"Unknown decomposition output: {member}")
        return {"group": "decomposition", "dataset": dataset, "output": output,
                "label": OUTPUTS[output], "relative": f"decomposition/{dataset}/{output.replace('_', '-')}.mp4"}
    if len(parts) == 5 and parts[1:3] == ("videos", "policy_rollout") and member.endswith(".mp4"):
        if PurePosixPath(parts[4]).stem == "f1":
            return None
        task, label = TASKS[parts[3]]
        trial, trial_label = TRIALS[PurePosixPath(parts[4]).stem]
        return {"group": "policy", "dataset": "rlbench", "task": task, "task_label": label,
                "trial": trial, "label": trial_label, "relative": f"policy/{task}/{trial}.mp4"}
    return None


def inspect(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        return {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": fps, "frames": frames, "duration": round(frames / fps, 3),
                "codec": "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4))}
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest_path = SITE / "static" / "videos" / "manifest.json"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    jobs = []

    with tempfile.TemporaryDirectory(prefix="particlesplat-video-import-") as temporary, ZipFile(args.archive) as archive:
        for member in sorted(archive.namelist()):
            entry = classify(member)
            if entry is None:
                continue
            source = Path(temporary) / f"{len(jobs)}.mp4"
            with archive.open(member) as incoming, source.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            entry["source_member"] = member
            entry["source_sha256"] = checksum(source)
            entry["source_codec"] = inspect(source)["codec"]
            entry["path"] = f"static/videos/{entry.pop('relative')}"
            entry["poster"] = str(PurePosixPath(entry["path"]).with_suffix('.jpg'))
            jobs.append((source, entry))
        if len(jobs) != 36:
            raise ValueError(f"Expected 16 decomposition videos and 20 successful rollouts; found {len(jobs)}")
        targets = [manifest_path] + [SITE / entry[key] for _, entry in jobs for key in ("path", "poster")]
        if not args.overwrite and any(path.exists() for path in targets):
            raise FileExistsError("Media outputs already exist; use --overwrite to regenerate these exact assets")

        def prepare(job):
            source, entry = job
            destination = SITE / entry["path"]
            poster = SITE / entry["poster"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoding = ["-c:v", "copy"] if entry["source_codec"] == "avc1" else [
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2"]
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(source),
                            "-map", "0:v:0", "-an", *encoding, "-movflags", "+faststart", str(destination)], check=True)
            original = inspect(source)
            prepared = inspect(destination)
            for key in ("width", "height", "frames", "fps"):
                if original[key] != prepared[key]:
                    raise ValueError(f"Import changed {key} for {entry['path']}")
            if prepared["codec"] != "avc1":
                raise ValueError(f"Expected browser-compatible H.264: {destination}")
            cap = cv2.VideoCapture(str(destination))
            try:
                # Use the same viewpoint for the related decomposition posters.
                fraction = 0.45 if entry["group"] == "decomposition" else 0.15
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((prepared["frames"] - 1) * fraction))
                ok, still = cap.read()
                if not ok or not cv2.imwrite(str(poster), still, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    raise ValueError(f"Could not extract poster for {destination}")
            finally:
                cap.release()
            entry.update(prepared)
            entry["bytes"] = destination.stat().st_size
            entry["sha256"] = checksum(destination)
            return entry

        with ThreadPoolExecutor(max_workers=4) as pool:
            entries = list(pool.map(prepare, jobs))
        manifest = {"source_archive": args.archive.name,
                    "notes": "H.264 MP4 with fast-start metadata. Original frame counts, frame rates, dimensions, and durations preserved. Successful policy examples follow source filenames s1/s2; failure clips are excluded. No camera-input setting is inferred for the policy clips.",
                    "videos": entries}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Prepared {len(entries)} videos ({sum(item['bytes'] for item in entries) / 1e6:.2f} MB) and posters. PDFs are not imported.")


if __name__ == "__main__":
    main()
