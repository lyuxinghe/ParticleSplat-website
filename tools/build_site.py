#!/usr/bin/env python3
"""Validate and assemble a runtime-only static site; never deploy or overwrite.

Only the Python standard library is required. Development tools, documentation,
Git history, and local audit outputs are deliberately outside the artifact.
"""

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit

SITE = Path(__file__).resolve().parents[1]
ROOT_FILES = ("index.html", "styles.css", ".nojekyll")
ASSET_TYPES = {".css", ".js", ".json", ".csv", ".svg", ".png", ".jpg", ".mp4", ".ply"}
TEXT_TYPES = {".html", ".css", ".js", ".json", ".csv", ".svg"}
MAX_FILE_BYTES = 25 * 1024 * 1024


class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                raise ValueError(f"Duplicate HTML id: {attrs['id']}")
            self.ids.add(attrs["id"])
        for name in ("src", "href", "poster"):
            if attrs.get(name):
                self.links.append(attrs[name])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_site(site):
    """Return an explicit artifact file list, rejecting unexpected static content."""
    site = site.resolve()
    files = [site / name for name in ROOT_FILES]
    if not (site / "static").is_dir() or (site / "static").is_symlink():
        raise ValueError("Expected a regular static asset directory")
    for path in sorted((site / "static").rglob("*")):
        relative = path.relative_to(site)
        if path.is_symlink() or any(part.startswith(".") for part in relative.parts):
            raise ValueError(f"Hidden file or symlink is not publishable: {relative}")
        if path.is_dir():
            continue
        license_file = path.name == "LICENSE" and relative.parts[:2] == ("static", "vendor")
        if path.suffix.lower() not in ASSET_TYPES and not license_file:
            raise ValueError(f"Unexpected public asset type: {relative}")
        files.append(path)
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe site file: {path}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"File exceeds the site's 25 MiB per-file budget: {path.name}")
        relative = path.relative_to(site)
        if path.suffix in TEXT_TYPES and relative.parts[:2] != ("static", "vendor"):
            content = path.read_text()
            if re.search(r"(?:/home/|/Users/|/tmp/|file://|-----BEGIN .*PRIVATE KEY-----)", content):
                raise ValueError(f"Local path or private-key material in public file: {relative}")
    relative_files = {path.relative_to(site).as_posix() for path in files}
    page = PageLinks()
    page.feed((site / "index.html").read_text())
    for link in page.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        path = unquote(url.path)
        if path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"Link is not safe for repository-subpath hosting: {link}")
        if path and path not in relative_files:
            raise ValueError(f"Missing page asset: {link}")
        if not path and url.fragment and unquote(url.fragment) not in page.ids:
            raise ValueError(f"Missing page section: {link}")

    def verify(relative, expected):
        if relative not in relative_files or sha256(site / relative) != expected:
            raise ValueError(f"Missing or changed asset: {relative}")

    vendor = json.loads((site / "static/vendor/manifest.json").read_text())
    for relative, record in vendor.items():
        verify(f"static/vendor/{relative}", record["sha256"])
    for required in ("three/LICENSE", "spark/LICENSE"):
        if required not in vendor:
            raise ValueError(f"Missing vendor license: {required}")
    scene = json.loads((site / "static/scenes/close-jar/scene.json").read_text())
    for record in scene["assets"]:
        verify(f"static/scenes/close-jar/{record['file']}", record["sha256"])
    for pair in json.loads((site / "static/images/reconstruction/manifest.json").read_text())["pairs"]:
        for kind in ("gt", "reconstruction"):
            verify(pair[kind]["path"], pair[kind]["sha256"])
    for record in json.loads((site / "static/videos/manifest.json").read_text())["videos"]:
        verify(record["path"], record["sha256"])
        if record["poster"] not in relative_files:
            raise ValueError(f"Missing video poster: {record['poster']}")
    return files


def build_site(site, output):
    site, output = site.resolve(), output.resolve()
    if output == site or site in output.parents or output in site.parents:
        raise ValueError("Output must be separate from the website source tree")
    if output.exists():
        raise ValueError("Output already exists; choose a new directory (nothing was overwritten)")
    files = validate_site(site)
    output.mkdir(parents=True)
    for source in files:
        target = output / source.relative_to(site)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SITE.parent / "dist/website")
    parser.add_argument("--check", action="store_true", help="Validate without writing an artifact")
    args = parser.parse_args()
    try:
        files = validate_site(SITE) if args.check else build_site(SITE, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Site validation failed: {error}\n")
    size = sum(path.stat().st_size for path in files) / 1024**2
    print(f"PASS: {len(files)} runtime files, {size:.2f} MiB; local links, provenance hashes and vendor licenses verified.")
    if not args.check:
        print(f"Static artifact: {args.output.resolve()}")
        print("Publish only this directory. No Git history, tools, papers, checkpoints or audit arrays are included.")


if __name__ == "__main__":
    main()
