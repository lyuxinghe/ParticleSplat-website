# ParticleSplat project website

A static research page with architecture, scene decomposition, interactive
Gaussian-splat editing, and quantitative/qualitative reconstruction and policy
results. The layout and interactions are original; no code or media from 3D-DLP's
website are included.

## Preview

From the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory .
```

Open http://localhost:8000. In VS Code Remote SSH, forward port 8000 in the Ports
panel and open its forwarded URL. **Simple Browser: Show** also works for the
page; use a regular WebGL2-capable browser for the interactive demo. Browser
modules require HTTP, not a local-file URL.

## Publishable artifact

```bash
python3 tools/build_site.py --check
python3 tools/build_site.py --output /tmp/particlesplat-site
python3 -m http.server 8000 --bind 127.0.0.1 --directory /tmp/particlesplat-site
```

The build copies only `index.html`, `styles.css`, `.nojekyll`, and validated
`static/` assets to the `--output` directory. It checks local page links, duplicate IDs,
asset hashes, vendor licenses, unexpected file types, local machine paths, and a
25 MiB per-file budget. It rejects symlinks, hidden assets, private archives/PDFs,
checkpoints and audit arrays. It never overwrites an existing output; choose a
new `--output` directory for another build.

Publish **the contents of this artifact**, not the source directory. GitHub Pages does
this automatically: `.github/workflows/pages.yml` runs the build on every push to `main`
and deploys the artifact (Pages source: GitHub Actions). Relative URLs work under repository subpaths. No
deployment or push is performed by the build.

The artifact preserves scene provenance and third-party license notices.
Development tools, documentation, caches, screenshots, and Git history are
excluded. Imported second-success rollout assets remain available for reproducible
media imports; the page displays exactly one successful rollout per task.

### Before publishing the source repository

The current website contains no papers or source archives. However, the removed
supplementary PDF exists in older Git history (introduced in `4b0f7cc`, removed in
`2b1ee46`). Pushing the source history can expose it. Publishing the clean static
artifact avoids this; publishing the full source requires a separately approved
history cleanup or a fresh public repository. Git ignore rules do not remove
previously committed material.

The public paper is linked in the page header: [arXiv:2609.19463](https://arxiv.org/abs/2609.19463).
The header lists all six authors in paper order with their shared affiliation:
Robotics Institute, Carnegie Mellon University. Author names link to personal
research pages where available; Daniel Guo and Elizabeth Terveen use LinkedIn
because their listed/discovered personal sites were not publicly accessible.
Additional bibliographic metadata and venue/acceptance claims are omitted.

## Interactive demo

At `#manipulation`, press **Explore in 3D**. Eight translucent, unnamed handles
select editable regions. Selection immediately displays labeled XYZ controls:

- **Move:** drag an arrow, or enter offsets in centimeters (±25 cm).
- **Scale:** drag an axis tip, or enter independent X/Y/Z factors (0.5–2×).
- **Uniform scale:** sets all three factors equally.
- **Reset selection:** restores its initial position and scale.

Dragging a ball moves its region in the camera plane. Close with × or Escape;
edits are preserved. Tab and Enter/Space provide keyboard selection; numeric
controls provide keyboard-accessible axis edits. The panel moves below the scene
on narrow screens.

Background dragging orbits within the input-camera arc, plus 5° on either side
and above/below the input elevations. Right-drag pans; scroll/pinch zooms.
**Focus here** centers the selected region without expanding these limits.
**Reset view**, **Reset scene**, and **Handles** control the camera, edits and markers.

The scene is RLBench test frame `close_jar_episode1_40`, decoded from views 6/14
using the final 50k-step checkpoint. Its initial camera is unseen view 10; native
resolution is 128×128. The eight UI regions are curated groupings, not predicted
semantic labels or individual learned keypoints. They include two jars, an
independent lid, and robot sections.

This qualitative demo includes a manually chosen color filter removing 672
table-colored Gaussians from jar/lid fields (42,336 retained). Retained values and
other fields are unchanged. The scene manifest records the exact rule, removed
indices and source hashes; its metrics explicitly refer to the unpruned checkpoint
output. Paper tables, reconstruction comparisons and videos are unchanged.
No scene-specific fitting or new training was performed. See the
[reproduction audit](../docs/particle-demo-reproduction.md) for full provenance,
checkpoint selection, measured quality and limitations.

Three.js 0.180.0 and Spark 2.1.0 are pinned locally, with checksums and licenses in
`static/vendor/`. The renderer loads only after opt-in (~10 MB), needs WebGL2, and
uses no backend, runtime CDN, checkpoint download or npm build. Extended storage
and full covariance transforms preserve per-axis scale changes. These are affine
edits of decoded fields, not live latent decoding or physics. Rendering pauses
offscreen and when idle. Static previews remain available without JavaScript/WebGL.

## Content and media

Reading order: Abstract → Architecture → Scene decomposition → Interactive demo →
Results → Related work. Results separates reconstruction and policy learning,
each with quantitative and qualitative subsections.

- Architecture: supplied figure with one overview caption.
- Scene decomposition: 16 visible, silent clips grouped by dataset.
- Reconstruction: reported PSNR/SSIM/LPIPS and eight GT/reconstruction sliders.
  Pairs are lossless crops from assembled paper figures, not new evaluations.
  Source image hashes and crop coordinates remain in their manifest.
- Policy learning: independent RLBench single-view, RLBench multi-view and MimicGen
  charts/tables. Reported aggregates are preserved, not recomputed from rounded
  per-task values. Purple charts place representation ablations above baselines,
  identify policy-network differences and separate groups with a divider.
- Policy examples: one successful clip per RLBench task, uniformly using the
  supplied first-success trial. Camera-input settings for these rollouts are
  unconfirmed, so they are not labeled single-/multi-view. No MimicGen policy
  clips were supplied. These examples do not replace aggregate success rates.

Videos preserve dimensions, duration, frame rate and frame count, using
browser-compatible H.264 and fast-start metadata. Nothing autoplays. Manifests
record source members/checksums. Importers exclude failure clips and PDFs.

## Static renderer and regeneration

The prepared PLY files, poster and scene manifest are already included. Serve or
publish them directly; no pruning script runs in the browser or during the site
build. Offline preparation utilities are kept local, outside the public source
tree. The rule and source hashes remain in the scene manifest for verification.

To reproduce the original inference and verify the prepared assets, use the
repository's CUDA environment and matching local data/checkpoint:

```bash
python tools/export_particle_scene.py audit \
  --run <log_dir>/rlbench_particlesplat/<run> \
  --scenes close_jar_episode1_40 --views 6 14 \
  --target-views 6 8 10 12 14 --output /tmp/particlesplat-scene-audit

python tools/export_particle_scene.py publish \
  --source /tmp/particlesplat-scene-audit/close_jar_episode1_40.npz \
  --groups tools/close-jar-scene.json \
  --output /tmp/particlesplat-scene-unpruned

python tools/check_particle_export.py \
  --source /tmp/particlesplat-scene-audit/close_jar_episode1_40.npz \
  --scene static/scenes/close-jar --output /tmp/particlesplat-scene-roundtrip
```

Intermediate arrays and diagnostic renders stay outside the website. Inference
strictly loads matching weights and deterministic settings; weights/configs
remain untouched. Retained PLY values and render round trips are independently
checked. Original assets are recoverable from `991dcbd` or an unpruned export.
The optional `publish` command above writes an unpruned diagnostic export to a
separate directory; it does not replace the prepared public demo.

Other development commands:

```bash
python tools/import_reconstruction.py /path/to/manuscript.zip
python tools/import_supplement.py /path/to/supplement.zip
python tools/generate_charts.py
python3 -m unittest discover -s tools -p 'test_build_site.py'
python tools/check_site.py
python tools/check_particle_demo.py
```

Importers protect existing outputs; use `--overwrite` only to regenerate known
assets. Media preparation needs Pillow/OpenCV/ffmpeg; browser checks need
Playwright and Chromium. These are development dependencies only. The demo check
tests real XYZ drags, independent transforms, resets, camera limits, keyboard/touch
editing, 320–1440px layouts, lazy loading, local requests, fallbacks and GitHub
Pages-style subpath hosting. The site check exercises all 26 displayed videos and
eight image sliders. Both browser checkers accept `--site dist/website` to test
the assembled release instead of the source directory.
