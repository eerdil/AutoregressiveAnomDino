import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
from models.dinov3_utils import extract_dino_tokens_2d
from ar.anomaly_maps import compute_anomaly_maps_2d

def visualize_anomaly_grid(
    dino_model,
    ar_model,
    imgs_vis,        # [B_vis, 3, H, W] on CPU
    labels_vis,      # [B_vis, 1, H, W] on CPU, values {0,1}
    device,
    img_size,
    output_dir,
    epoch=None,
):
    """
    Create a 5×N grid:

      Row 0: normal (healthy) images
      Row 1: predicted anomaly maps for normals
      Row 2: anomalous images
      Row 3: predicted anomaly maps for anomalies
      Row 4: GT anomaly masks (labels) for anomalies

    Up to 10 normals and 10 anomalous are shown (if available).
    """

    dino_model.eval()
    ar_model.eval()

    # Move images to device to compute anomaly maps
    imgs_vis_dev = imgs_vis.to(device)

    with torch.no_grad():
        # 1) anomaly maps at token resolution
        anomaly_maps = compute_anomaly_maps_2d(dino_model, ar_model, imgs_vis_dev, device)  # [B, H_tok, W_tok]

        # 2) upsample to image resolution
        B = imgs_vis_dev.size(0)
        anomaly_maps_up = F.interpolate(
            anomaly_maps.unsqueeze(1),           # [B,1,H_tok,W_tok]
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)                             # [B, img_size, img_size]

    # Bring everything to CPU for plotting
    imgs_vis_cpu    = imgs_vis_dev.detach().cpu()
    maps_vis_cpu    = anomaly_maps_up.detach().cpu()
    labels_vis_cpu  = labels_vis.detach().cpu()  # [B,1,H,W] with 0/1

    # ------------------------------------------------------------------
    # Split into normal vs anomalous based on label mask
    # (if any pixel is 1 → anomalous)
    # ------------------------------------------------------------------
    # [B]
    is_anom = (labels_vis_cpu.view(B, -1).sum(dim=1) > 0)

    normal_idxs = torch.where(~is_anom)[0]
    anom_idxs   = torch.where(is_anom)[0]

    n_norm = min(10, normal_idxs.numel())
    n_anom = min(10, anom_idxs.numel())

    if n_norm == 0:
        print("[Vis] Warning: no normal samples in visualization batch.")
    if n_anom == 0:
        print("[Vis] Warning: no anomalous samples in visualization batch.")

    normal_idxs = normal_idxs[:n_norm]
    anom_idxs   = anom_idxs[:n_anom]

    n_cols = max(n_norm, n_anom, 1)  # at least 1 to avoid matplotlib issues

    # ------------------------------------------------------------------
    # Build 5×N grid
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(5, n_cols, figsize=(2 * n_cols, 10))

    # Handle the case n_cols == 1 (axes is 1D per row)
    def get_ax(row, col):
        if n_cols == 1:
            return axes[row]
        else:
            return axes[row, col]

    # -------------------------
    # 1) Normal samples (rows 0–1)
    # -------------------------
    for j in range(n_cols):
        ax_img = get_ax(0, j)
        ax_map = get_ax(1, j)

        if j < n_norm:
            idx = normal_idxs[j].item()

            # --- original image (denormalized) ---
            img_denorm = denormalize_img(imgs_vis_cpu[idx])   # [3,H,W]
            img_np = img_denorm.permute(1, 2, 0).numpy()      # [H,W,3]
            ax_img.imshow(img_np)
            ax_img.set_title("Normal Img")
            ax_img.axis("off")

            # --- anomaly map ---
            amap = maps_vis_cpu[idx].numpy()                  # [H,W]
            amap_min, amap_max = amap.min(), amap.max()
            if amap_max > amap_min:
                amap_vis = (amap - amap_min) / (amap_max - amap_min)
            else:
                amap_vis = np.zeros_like(amap)
            ax_map.imshow(amap_vis, cmap="hot", vmin=0.0, vmax=1.0)
            ax_map.set_title("Anomaly Map")
            ax_map.axis("off")
        else:
            ax_img.axis("off")
            ax_map.axis("off")

    # -------------------------
    # 2) Anomalous samples (rows 2–4)
    # -------------------------
    for j in range(n_cols):
        ax_img  = get_ax(2, j)
        ax_map  = get_ax(3, j)
        ax_mask = get_ax(4, j)

        if j < n_anom:
            idx = anom_idxs[j].item()

            # --- anomalous image ---
            img_denorm = denormalize_img(imgs_vis_cpu[idx])   # [3,H,W]
            img_np = img_denorm.permute(1, 2, 0).numpy()
            ax_img.imshow(img_np)
            ax_img.set_title("Anomaly Img")
            ax_img.axis("off")

            # --- anomaly map (prediction) ---
            amap = maps_vis_cpu[idx].numpy()                  # [H,W]
            amap_min, amap_max = amap.min(), amap.max()
            if amap_max > amap_min:
                amap_vis = (amap - amap_min) / (amap_max - amap_min)
            else:
                amap_vis = np.zeros_like(amap)
            ax_map.imshow(amap_vis, cmap="hot", vmin=0.0, vmax=1.0)
            ax_map.set_title("Anomaly Map")
            ax_map.axis("off")

            # --- GT anomaly mask ---
            gt_mask = labels_vis_cpu[idx, 0].numpy()          # [H,W], 0/1
            ax_mask.imshow(gt_mask, cmap="gray", vmin=0.0, vmax=1.0)
            ax_mask.set_title("GT Mask")
            ax_mask.axis("off")
        else:
            ax_img.axis("off")
            ax_map.axis("off")
            ax_mask.axis("off")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    if epoch is None:
        fname = "anomaly_grid.png"
    else:
        fname = f"anomaly_grid_epoch_{epoch:04}.png"

    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"[Visualization] Saved anomaly grid to: {out_path}")

def denormalize_img(img_tensor):
    """
    img_tensor: [3, H, W] normalized with ImageNet stats.
    Returns: [3, H, W] in [0,1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp(0, 1)

def select_visualization_subset_from_loader(
    dataloader_valid,
    n_anom: int = 10,
    n_norm: int = 10,
):
    """
    Take the first batch from `dataloader_valid` and build a visualization subset:
      - first `n_anom` samples are anomalous (target == 1)
      - next  `n_norm` samples are normal (target == 0)

    If there are fewer than requested in either class, it will use as many as available.

    Returns:
        imgs_vis:   [B_vis, 3, H, W]
        labels_vis: [B_vis, 1, H, W]
    """
    # Grab first batch
    imgs, labels, targets, meta = next(iter(dataloader_valid))  # all on CPU by default
    B = imgs.size(0)

    anomaly_idxs = (targets == 1).nonzero(as_tuple=True)[0]
    normal_idxs  = (targets == 0).nonzero(as_tuple=True)[0]

    # How many we can actually take
    n_anom_eff = min(n_anom, anomaly_idxs.numel())
    n_norm_eff = min(n_norm, normal_idxs.numel())

    if n_anom_eff == 0:
        print(f"[Init] Warning: no anomalous samples in first val batch.")
    if n_norm_eff == 0:
        print(f"[Init] Warning: no normal samples in first val batch.")

    # Build index list: anomalies first, then normals
    selected_idxs = []
    if n_anom_eff > 0:
        selected_idxs.append(anomaly_idxs[:n_anom_eff])
    if n_norm_eff > 0:
        selected_idxs.append(normal_idxs[:n_norm_eff])

    if len(selected_idxs) == 0:
        # total fallback: nothing? just return first min(B, n_anom+n_norm)
        B_vis = min(B, n_anom + n_norm)
        vis_idxs = torch.arange(B_vis)
        print(f"[Init] No class-specific slices found, using first {B_vis} samples.")
    else:
        vis_idxs = torch.cat(selected_idxs, dim=0)

    # Subset tensors
    imgs_vis   = imgs[vis_idxs]    # [B_vis, 3, H, W]
    labels_vis = labels[vis_idxs]  # [B_vis, 1, H, W]

    print(
        f"[Init] Visualization set: "
        f"{min(n_anom, anomaly_idxs.numel())} anomalous + "
        f"{min(n_norm, normal_idxs.numel())} normal = {vis_idxs.numel()} total."
    )

    return imgs_vis, labels_vis