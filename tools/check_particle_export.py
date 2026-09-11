#!/usr/bin/env python3
"""Check PLY round trips and render native/edited quality controls with CUDA.

Audit inputs and rendered diagnostics stay outside the public website. Recorded
demo-only pruning is checked against source indices; retained values must match
the checkpoint. This checker does not optimize, recolor or modify public assets.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from particlesplat.model.renderer import GaussianSplattingRenderer

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    source = np.load(args.source)
    manifest = json.loads((args.scene / "scene.json").read_text())
    provenance = manifest["provenance"]
    fields = ["means", "covariances", "harmonics", "opacities"]
    original = [np.concatenate((source["fg_" + f][0], source["bg_" + f][0])) for f in fields]
    restored = [np.zeros_like(a) for a in original]
    fixed = np.ones(len(original[0]), dtype=bool)
    keep = np.ones(len(original[0]), dtype=bool)
    n = provenance["gaussians_per_particle"]
    for particle in manifest["particles"]:
        fixed[particle["id"] * n:(particle["id"] + 1) * n] = False
    cleanup = manifest.get('postprocessing')
    if cleanup:
        rule = cleanup['rule']
        assert rule['scene'] == provenance['scene']
        assert set(map(int, cleanup['removed_indices'])) == set(rule['particles'])
        for pid_text, indices in cleanup['removed_indices'].items():
            pid = int(pid_text)
            assert indices == sorted(set(indices)) and all(0 <= i < n for i in indices)
            # Independently recompute the recorded DC-color predicate.
            rgb = np.clip(source['fg_harmonics'][0, pid*n:(pid+1)*n, :, 0] * .28209479177387814 + .5, 0, 1)
            r, g, b = rgb.T
            expected = ((r-b > rule['min_red_minus_blue']) & (g-b > rule['min_green_minus_blue']) &
                        (r > rule['min_red']) & (g > rule['min_green']))
            assert indices == np.flatnonzero(expected).tolist()
            keep[pid*n + np.array(indices)] = False
        assert cleanup['source_gaussians'] == len(original[0])
        assert cleanup['removed_gaussians'] == int((~keep).sum())
        assert not np.any(fixed & ~keep), 'Pruning changed the fixed scene'
    assert int(keep.sum()) == manifest['retained_gaussians']
    for asset in manifest["assets"]:
        content = (args.scene / asset["file"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == asset["sha256"]
        values = np.frombuffer(content.split(b"end_header\n", 1)[1], dtype="<f4").reshape(-1, 23)
        assert len(values) == asset["gaussians"]
        rotation = Rotation.from_quat(values[:, [20, 21, 22, 19]]).as_matrix()
        covariance = np.einsum("nik,nk,njk->nij", rotation, np.exp(2 * values[:, 16:19]), rotation)
        harmonics = np.concatenate((values[:, 3:6, None], values[:, 6:15].reshape(-1, 3, 3)), axis=-1)
        decoded = [values[:, :3].copy(), covariance, harmonics, 1 / (1 + np.exp(-values[:, 15]))]
        if asset["file"] == manifest["fixed"]:
            index = fixed
        else:
            particle = next(p for p in manifest["particles"] if p["file"] == asset["file"])
            decoded[0] += particle["center"]
            index = np.arange(particle["id"] * n, (particle["id"] + 1) * n)
            index = index[keep[index]]
        for target, value in zip(restored, decoded):
            target[index] = value
    errors = {f: float(np.max(np.abs(a[keep] - b[keep]))) for f, a, b in zip(fields, original, restored)}
    assert all(error < 1e-6 for error in errors.values()), errors

    def tensor(a):
        return torch.from_numpy(np.asarray(a, dtype=np.float32).copy()).cuda()

    renderer = GaussianSplattingRenderer(image_shape=(128, 128),
                                         scene_bound=provenance["scene_bound"], sh_degree=1).cuda()
    near = torch.full((1, 1), 0.01, device="cuda")
    far = torch.full((1, 1), 10., device="cuda")
    view = source["target_index"][0].tolist().index(manifest["camera"]["view"])
    intr = tensor(source["target_intrinsics"][:, view:view + 1])
    extr = tensor(source["target_extrinsics"][:, view:view + 1])

    def render(arrays, size, name):
        renderer.image_shape = (size, size)
        with torch.inference_mode():
            rgb = renderer.render_gaussians(*[tensor(a)[None] for a in arrays], extr, intr,
                                            near, far, mode=["rgb"])["rgb"][0, 0]
        rgb = rgb.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        Image.fromarray(np.uint8(rgb * 255 + 0.5)).save(args.output / f"{name}.png")
        return rgb

    native = render(original, 128, "native-128")
    cleaned = render([a[keep] for a in original], 128, "pruned-native-128") if cleanup else native
    roundtrip = render([a[keep] for a in restored], 128, "ply-cuda-128")
    image_error = float(np.abs(cleaned - roundtrip).mean())
    assert image_error < 1e-5, image_error
    expected = source["target_rec_rgb"][0, view].clip(0, 1).transpose(1, 2, 0)
    assert float(np.abs(native - expected).mean()) < 1e-5
    if cleanup:
        poster = np.asarray(Image.open(args.scene / 'poster.png').convert('RGB'), dtype=np.float32) / 255
        assert float(np.abs(poster - cleaned).mean()) < 1/255, 'Poster is not the cleaned reconstruction'
    render(original, 512, "native-512")
    edited = [a.copy() for a in original]
    pids = manifest["groups"][0]["particles"]
    center = np.mean([p["center"] for p in manifest["particles"] if p["id"] in pids], axis=0)
    for pid in pids:
        selected = slice(pid * n, (pid + 1) * n)
        edited[0][selected] = center + (edited[0][selected] - center) * 1.3 + [0, 0.12, 0.03]
        edited[1][selected] *= 1.3 ** 2
    render([a[keep] for a in edited], 512, "group-edited-512")
    report = {"gaussians": int(keep.sum()), "source_gaussians": len(original[0]),
              "pruned_gaussians": int((~keep).sum()), "ply_max_errors": errors,
              "ply_native_image_mae": image_error, "overview_view": manifest["camera"]["view"],
              "pruned_unpruned_image_mae": float(np.abs(native - cleaned).mean()),
              "resolution_note": "512px render is a resolution diagnostic, not a 512px-trained prediction.",
              "edit": {"group": manifest["groups"][0]["id"], "translation": [0, 0.12, 0.03], "scale": 1.3}}
    (args.output / "roundtrip.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
