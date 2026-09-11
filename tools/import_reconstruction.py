"""Extract matched RGB panels from the supplied paper's PNG figures.

These are lossless panel crops, not new evaluations or standalone raw renders.
No resizing, color correction, retouching, or image synthesis is performed.
Requires Pillow. Existing outputs are protected unless --overwrite is given.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image


SITE = Path(__file__).resolve().parents[1]
PANELS = {
    "rlbench": {
        "member": "figs/rlbench_nvs_ext.png",
        "setting": "Single-view RGB-D",
        "size": (2158, 912),
        # x_GT, x_ParticleSplat, y, height; each RGB panel is 175 pixels wide.
        "rows": [(414, 882, 85, 175), (414, 883, 290, 175),
                 (414, 883, 495, 176), (414, 880, 701, 175)],
    },
    "real-world": {
        "member": "figs/realworld_nvs_ext.png",
        "setting": "Multi-view RGB (two context views)",
        "size": (1676, 932),
        "rows": [(517, 902, 92, 175), (517, 900, 297, 176),
                 (517, 903, 503, 175), (517, 903, 707, 176)],
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = SITE / "static/images/reconstruction"
    targets = [output / f"{dataset}-{index}-{kind}.png"
               for dataset in PANELS for index in range(1, 5)
               for kind in ("gt", "reconstruction")]
    manifest_path = output / "manifest.json"
    if not args.overwrite and any(path.exists() for path in targets + [manifest_path]):
        parser.error("Reconstruction assets already exist; use --overwrite to regenerate them.")
    manifest = {
        "source_archive": args.archive.name,
        "preparation": "Lossless RGB panel crops from the paper's assembled PNG figures; no resizing, retouching, or new evaluation. Rows retain paper order. Crop boxes are [left, top, right, bottom], right/bottom exclusive.",
        "pairs": [],
    }
    with zipfile.ZipFile(args.archive) as archive:
        # Verify both sources before creating any output.
        sources = {}
        for dataset, spec in PANELS.items():
            data = archive.read(spec["member"])
            source = Image.open(io.BytesIO(data))
            source.load()
            if source.size != spec["size"]:
                raise ValueError(f"Unexpected figure dimensions: {spec['member']}: {source.size}")
            sources[dataset] = (source, hashlib.sha256(data).hexdigest())
        output.mkdir(parents=True, exist_ok=True)
        for dataset, spec in PANELS.items():
            source, checksum = sources[dataset]
            for index, (gt_x, reconstruction_x, y, height) in enumerate(spec["rows"], 1):
                pair = {
                    "id": f"{dataset}-{index}", "dataset": dataset, "scene": index,
                    "setting": spec["setting"], "source_member": spec["member"],
                    "source_sha256": checksum, "width": 175, "height": height,
                }
                for kind, x in [("gt", gt_x), ("reconstruction", reconstruction_x)]:
                    box = (x, y, x + 175, y + height)
                    path = output / f"{dataset}-{index}-{kind}.png"
                    source.crop(box).save(path)
                    pair[kind] = {"path": str(path.relative_to(SITE)), "crop": list(box),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                manifest["pairs"].append(pair)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest['pairs'])} GT/reconstruction pairs from paper figures.")


if __name__ == "__main__":
    main()
