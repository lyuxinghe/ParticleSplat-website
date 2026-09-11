#!/usr/bin/env python3
"""Fetch pinned MIT-licensed viewer dependencies; no npm or runtime CDN needed."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

DEST = Path(__file__).resolve().parents[1] / "static/vendor"
THREE = "https://cdn.jsdelivr.net/npm/three@0.180.0/"
FILES = {
    "spark/spark.module.js": "https://sparkjs.dev/releases/spark/2.1.0/spark.module.js",
    "spark/LICENSE": "https://raw.githubusercontent.com/sparkjsdev/spark/v2.1.0/LICENSE",
    "three/three.module.min.js": THREE + "build/three.module.min.js",
    "three/three.core.min.js": THREE + "build/three.core.min.js",
    "three/addons/controls/OrbitControls.js": THREE + "examples/jsm/controls/OrbitControls.js",
    "three/addons/controls/TransformControls.js": THREE + "examples/jsm/controls/TransformControls.js",
    "three/addons/postprocessing/Pass.js": THREE + "examples/jsm/postprocessing/Pass.js",
    "three/LICENSE": THREE + "LICENSE",
}

if __name__ == "__main__":
    # Download all before writing, so a failed request does not create half an install.
    downloaded = {name: urlopen(url, timeout=45).read() for name, url in FILES.items()}
    manifest = {}
    for name, content in downloaded.items():
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest[name] = {"source": FILES[name], "bytes": len(content),
                          "sha256": hashlib.sha256(content).hexdigest()}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Vendored {len(manifest)} files ({sum(len(b) for b in downloaded.values()):,} bytes)")
