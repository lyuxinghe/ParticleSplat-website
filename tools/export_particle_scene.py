#!/usr/bin/env python3
"""Export authentic decoded particles for the static manipulation demo.

Run from the repository root with the training environment and CUDA. The audit
output is intentionally separate from the public website; never publish weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw


def load_run_config(run):
    """Read old run names without mutating their saved configuration or weights."""
    import yaml
    from particlesplat.utils.config import load_config

    saved = yaml.safe_load((run / "config.yaml").read_text())
    unsupported = {key: saved["model"][key] for key in
                   ("bg_mask_ungated", "density_leash", "opacity_compensation")
                   if saved["model"].get(key)}
    if unsupported:
        raise ValueError(f"Historical model variant needs its original code: {unsupported}")
    if saved["model"]["name"] == "dlpgs":  # checkpoints saved before the rename
        # Documented naming-only migration; other historical model variants are
        # intentionally not aliased because their forward semantics may differ.
        saved["model"]["name"] = "particlesplat"
    with tempfile.TemporaryDirectory(prefix="particlesplat-config-") as temporary:
        path = Path(temporary) / "config.yaml"
        path.write_text(yaml.safe_dump(saved))
        return load_config(experiment_config=str(path),
                           config_dir=str(Path(__file__).resolve().parents[2] / "configs"))


def rgb_image(tensor, size=256):
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(array * 255 + 0.5)).resize((size, size))


def audit(args):
    import torch
    from particlesplat.dataset import create_dataloader
    from particlesplat.model import create_model_with_renderer

    torch.manual_seed(42)
    torch.set_num_threads(4)
    config = load_run_config(args.run)
    config.dataset.val_num_workers = 0
    config.dataset.val_batch_size = 1
    config.dataset.shuffle_val = False
    config.dataset.augment = False
    config.dataset.sampler_name = "fixed"
    config.dataset.context_views = args.views
    config.dataset.target_views = args.target_views or args.views
    config.dataset.max_total_samples = None
    loader = create_dataloader("val", config.dataset, None)
    loader.dataset.data_paths = [p for p in loader.dataset.data_paths
                                 if p["key"] in args.scenes]
    if len(loader.dataset.data_paths) != len(args.scenes):
        raise ValueError("Some requested scene keys were not found")
    model = create_model_with_renderer(config.model, config.renderer).cuda().eval()
    weights = args.run / "checkpoints" / args.checkpoint
    state = torch.load(weights, map_location="cpu")
    state = {k: v for k, v in state.items() if not k.startswith("recon_loss_func")}
    model.load_state_dict(state, strict=True)
    assert model.n_kp_enc == model.n_kp_dec, "Export requires unchanged decoder particle order"
    args.output.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        scene = batch["scene"][0]
        context = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch["context"].items()}
        target = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch["target"].items()}
        with torch.inference_mode():
            out = model(x=context["image"], xdepth=context["depth"],
                        extr=context["extrinsics"], intr=context["intrinsics"],
                        near=context["near"], far=context["far"],
                        target_extr=target["extrinsics"], target_intr=target["intrinsics"],
                        target_near=target["near"], target_far=target["far"],
                        deterministic=True, with_loss=False, render_mode=["rgb"])
        arrays = {k: v.detach().cpu().numpy() for k, v in out.items()
                  if k.startswith(("fg_", "bg_")) and torch.is_tensor(v)}
        for key in ("z", "z_depth", "obj_on", "z_scale", "z_depth_scale"):
            arrays[key] = out[key].detach().cpu().numpy()
        for key in ("extrinsics", "intrinsics", "image", "index"):
            arrays[key] = context[key].cpu().numpy()
        arrays["rec_rgb"] = out["rec_rgb"].cpu().numpy()
        arrays["target_rec_rgb"] = out["target_rec_rgb"].cpu().numpy()
        for key in ("extrinsics", "intrinsics", "image", "index"):
            arrays["target_" + key] = target[key].cpu().numpy()
        np.savez_compressed(args.output / f"{scene}.npz", **arrays)
        print(scene, {k: v.shape for k, v in arrays.items()}, flush=True)
        count = model.n_kp_enc
        columns = 8
        tile = 128
        sheet = Image.new("RGB", (columns * tile, 2 * (tile + 20) +
                                  ((len(args.views) * count + columns - 1) // columns) * (tile + 20)), "#f0edf5")
        draw = ImageDraw.Draw(sheet)
        for view in range(len(args.views)):
            sheet.paste(rgb_image(context["image"][0, view], tile), (view * 2 * tile, 20))
            sheet.paste(rgb_image(out["rec_rgb"][0, view], tile), ((view * 2 + 1) * tile, 20))
            draw.text((view * 2 * tile + 4, 4), f"View {args.views[view]} GT / reconstruction", fill="black")
        gaussians_per_particle = model.obj_patch_size ** 2
        assert arrays["fg_means"].shape[1] == len(args.views) * count * gaussians_per_particle
        for particle in range(len(args.views) * count):
            start, end = particle * gaussians_per_particle, (particle + 1) * gaussians_per_particle
            with torch.inference_mode():
                render = model.renderer.render_gaussians(
                    *[out[f"fg_{k}"][:, start:end] for k in ("means", "covariances", "harmonics", "opacities")],
                    context["extrinsics"][:, :1], context["intrinsics"][:, :1],
                    context["near"][:, :1], context["far"][:, :1], mode=["rgb"])["rgb"]
            x = (particle % columns) * tile
            y = 2 * (tile + 20) + (particle // columns) * (tile + 20)
            opacity = arrays["obj_on"].reshape(-1)[particle]
            draw.text((x + 4, y + 3), f"P{particle}  on={opacity:.2f}", fill="black")
            sheet.paste(rgb_image(render[0, 0], tile), (x, y + 20))
        sheet.save(args.output / f"{scene}.jpg", quality=95)
        # A separate view sheet checks reconstruction at cameras NOT used as input.
        nt = len(config.dataset.target_views)
        target_sheet = Image.new("RGB", (4 * 256, ((nt + 3) // 4) * 148), "#f0edf5")
        target_draw = ImageDraw.Draw(target_sheet)
        from particlesplat.utils.metrics import compute_psnr, compute_ssim
        scores = compute_psnr(target["image"][0], out["target_rec_rgb"][0]).cpu().tolist()
        ssim = compute_ssim(target["image"][0], out["target_rec_rgb"][0]).cpu().tolist()
        for i, view in enumerate(config.dataset.target_views):
            x, y = (i % 4) * 256, (i // 4) * 148
            target_draw.text((x + 4, y + 3), f"View {view}: GT / decoded  {scores[i]:.2f} dB", fill="black")
            # Left GT, right prediction; retain both images' native aspect ratio.
            comparison = Image.new("RGB", (256, 128))
            comparison.paste(rgb_image(target["image"][0, i], 128), (0, 0))
            comparison.paste(rgb_image(out["target_rec_rgb"][0, i], 128), (128, 0))
            target_sheet.paste(comparison, (x, y + 20))
        target_sheet.save(args.output / f"{scene}-views.jpg", quality=95)
        print("Target PSNR:", dict(zip(config.dataset.target_views, scores)), flush=True)
        provenance = {"run": args.run.name, "experiment": args.run.parent.name,
                      "checkpoint": args.checkpoint,
                      "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                      "scene": scene, "context_views": args.views,
                      "scene_bound": config.renderer.scene_bound,
                      "target_views": config.dataset.target_views,
                      "target_psnr": scores, "target_ssim": ssim,
                      "source_split": "test", "deterministic": True,
                      "refinement": "none; deterministic feed-forward checkpoint output",
                      "training_image_size": list(config.renderer.image_shape),
                      "config_sha256": hashlib.sha256((args.run / "config.yaml").read_bytes()).hexdigest(),
                      "gaussians_per_particle": int(gaussians_per_particle)}
        (args.output / f"{scene}.json").write_text(json.dumps(provenance, indent=2) + "\n")


def publish(args):
    """Preserve Gaussian ownership, covariance, opacity and all degree-1 SH."""
    from scipy.spatial.transform import Rotation

    source = np.load(args.source)
    spec = json.loads(args.groups.read_text())
    provenance = json.loads(args.source.with_suffix(".json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "scene.json").exists() and not args.overwrite:
        raise FileExistsError("Scene exists; pass --overwrite to regenerate this scene")
    particles = []
    selected = [pid for group in spec["groups"] for pid in group["particles"]]
    assert len(selected) == len(set(selected)), "A particle may belong to only one group"
    per_particle = provenance["gaussians_per_particle"]
    count = source["z"].shape[2]
    total = source["z"].shape[0] * count
    assert all(0 <= pid < total for pid in selected)
    # Unproject the actual inferred keypoints, not the Gaussian centroids.
    uv = (source["z"][:, 0, :, ::-1] + 1) * 0.5
    rays = np.einsum("vij,vpj->vpi", np.linalg.inv(source["intrinsics"][0]),
                     np.concatenate((uv, np.ones_like(uv[..., :1])), axis=-1))
    near, far = provenance["scene_bound"]
    metric_depth = near + (far - near) / (1 + np.exp(-source["z_depth"][:, 0]))
    cam = rays * metric_depth
    centers = (np.einsum("vij,vpj->vpi", source["extrinsics"][0, :, :3, :3], cam)
               + source["extrinsics"][0, :, None, :3, 3]).reshape(-1, 3)
    files = []

    def write_ply(filename, arrays, origin):
        means, cov, sh, opacity = arrays
        assert sh.shape[1:] == (3, 4), "Exporter expects degree-1 spherical harmonics"
        assert all(np.isfinite(a).all() for a in arrays)
        # Retain every Gaussian, including low-opacity ones: no segmentation by color,
        # replacement geometry, manual clean-up, or ownership reassignment.
        eigval, eigvec = np.linalg.eigh(cov)
        assert eigval.min() > -1e-7
        eigvec[:, :, 0] *= np.where(np.linalg.det(eigvec) < 0, -1, 1)[:, None]
        quat = Rotation.from_matrix(eigvec).as_quat()[:, [3, 0, 1, 2]]
        scales = np.sqrt(np.maximum(eigval, 1e-16))
        rebuilt = np.einsum("nik,nk,njk->nij", eigvec, scales ** 2, eigvec)
        assert np.allclose(cov, rebuilt, atol=1e-6), "Covariance conversion mismatch"
        alpha = np.clip(opacity, 1e-8, 1 - 1e-7)
        values = np.column_stack((means - origin, sh[:, :, 0], sh[:, :, 1:].reshape(-1, 9),
                                  np.log(alpha / (1 - alpha)), np.log(scales), quat)).astype("<f4")
        props = ["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)]
        props += [f"f_rest_{i}" for i in range(9)] + ["opacity"]
        props += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {len(values)}\n"
        header += "".join(f"property float {p}\n" for p in props) + "end_header\n"
        content = header.encode() + values.tobytes()
        (args.output / filename).write_bytes(content)
        files.append({"file": filename, "gaussians": len(values), "bytes": len(content),
                      "sha256": hashlib.sha256(content).hexdigest()})

    fields = ("means", "covariances", "harmonics", "opacities")
    fixed_mask = np.ones(total * per_particle, dtype=bool)
    for pid in selected:
        first, last = pid * per_particle, (pid + 1) * per_particle
        fixed_mask[first:last] = False
        filename = f"particle-{pid}.ply"
        write_ply(filename, [source[f"fg_{f}"][0, first:last] for f in fields], centers[pid])
        particles.append({"id": pid, "file": filename, "center": centers[pid].tolist(),
                          "context_slot": int(source["index"][0, pid // count]),
                          "slot_particle": pid % count,
                          "presence": float(source["obj_on"].reshape(-1)[pid])})
    fixed = [np.concatenate((source[f"fg_{f}"][0, fixed_mask], source[f"bg_{f}"][0])) for f in fields]
    write_ply("fixed-scene.ply", fixed, np.zeros(3))
    assert sum(f["gaussians"] for f in files) == len(source["fg_means"][0]) + len(source["bg_means"][0])
    overview = spec.get("overview_view", provenance["context_views"][0])
    indices = source["target_index"][0].tolist()
    if overview not in indices:
        raise ValueError("Overview must be one of the audited target cameras")
    overview_index = indices.index(overview)
    for kind, field in (("reference", "target_image"), ("poster", "target_rec_rgb")):
        rgb = np.uint8(np.clip(source[field][0, overview_index].transpose(1, 2, 0), 0, 1) * 255 + 0.5)
        Image.fromarray(rgb).save(args.output / f"{kind}.png")
    # Least-squares intersection of the input camera axes: the calibrated rig's
    # look-at point, not a manually guessed orbit center.
    poses = source["extrinsics"][0].astype(np.float64)
    directions = poses[:, :3, 2]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    projection = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    target = np.linalg.solve(projection.sum(0), np.einsum("vij,vj->i", projection, poses[:, :3, 3]))
    manifest = {"version": 1, **spec, "provenance": provenance, "particles": particles,
                "camera": {"view": overview,
                           "c2w": source["target_extrinsics"][0, overview_index].tolist(),
                           "intrinsics": source["target_intrinsics"][0, overview_index].tolist(),
                           "orbit_target": target.tolist(),
                           "arc_c2w": poses.tolist()},
                "fixed": "fixed-scene.ply", "assets": files,
                "representation": "Decoded particle-aligned Gaussian fields, degree-1 SH; no neural inference in browser.",
                "grouping": "Manually curated particle groups, not model-predicted semantic object labels.",
                "retained_gaussians": sum(f["gaussians"] for f in files),
                "inferred_particles": total}
    (args.output / "scene.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {manifest['retained_gaussians']:,} Gaussians; {len(particles)} editable particles")
    print(json.dumps(particles, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--run", type=Path, required=True)
    audit_parser.add_argument("--scenes", nargs="+", required=True)
    audit_parser.add_argument("--views", nargs="+", type=int, default=[0, 7])
    audit_parser.add_argument("--target-views", nargs="+", type=int)
    audit_parser.add_argument("--checkpoint", default="model.pth", choices=["model.pth", "best_model.pth"])
    audit_parser.add_argument("--output", type=Path, required=True)
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--source", type=Path, required=True)
    publish_parser.add_argument("--groups", type=Path, required=True)
    publish_parser.add_argument("--output", type=Path, required=True)
    publish_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    (audit if args.command == "audit" else publish)(args)
