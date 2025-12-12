import os
import torch
import torch.nn as nn
from models.dinov3_utils import extract_dino_tokens_2d
import torch.optim as optim
from ar.visualization import visualize_anomaly_grid

def validate_ar2d_model(dino_model, ar_model, val_loader, device):
    dino_model.eval()
    ar_model.eval()

    criterion = nn.MSELoss()
    running_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for imgs, labels, targets, meta in val_loader:
            imgs = imgs.to(device)

            feats_2d = extract_dino_tokens_2d(dino_model, imgs, device)  # [B, C, H, W]
            preds = ar_model(feats_2d)                                   # [B, C, H, W]

            loss = criterion(preds, feats_2d)
            running_loss += loss.item()
            n_batches += 1

    avg_loss = running_loss / max(1, n_batches)
    return avg_loss

def train_ar2d_model(dino_model, 
                     ar_model, 
                     train_loader, 
                     val_loader,
                     device,
                     epochs=10, 
                     lr=1e-3,
                     output_dir=None,
                     val_interval=1,
                     img_size=None,
                     imgs_vis=None,
                     labels_vis=None):
    
    dino_model.eval()
    ar_model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(ar_model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_epoch = -1

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        ar_model.train()
        running_loss = 0.0
        n_batches = 0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)

            # 1) DINO features as 2D maps
            feats_2d = extract_dino_tokens_2d(dino_model, imgs, device)  # [B, C, H, W]

            # 2) AR forward
            preds = ar_model(feats_2d)  # [B, C, H, W]

            # 3) reconstruction loss (per-location feature prediction)
            loss = criterion(preds, feats_2d)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train = running_loss / max(1, n_batches)
        print(f"[Epoch {epoch}/{epochs}] AR2D train loss: {avg_train:.6f}")

        # ---- Validation step every val_interval epochs ----
        if (val_loader is not None) and (epoch % val_interval == 0):
            val_loss = validate_ar2d_model(dino_model, ar_model, val_loader, device)
            print(f"           -> val loss: {val_loss:.6f}")

            # Visualization on the same fixed images every val epoch
            if imgs_vis is not None:
                visualize_anomaly_grid(
                    dino_model=dino_model,
                    ar_model=ar_model,
                    imgs_vis=imgs_vis,      # fixed across epochs
                    labels_vis=labels_vis,
                    device=device,
                    img_size=img_size,
                    output_dir=os.path.join(output_dir, "val_visualizations"),
                    epoch=epoch,
                )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                if output_dir is not None:
                    ckpt_dir = os.path.join(output_dir, "ckpt")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(ckpt_dir, "model.pth")

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": ar_model.state_dict(),
                            "val_loss": val_loss,
                            "img_size": img_size,
                        },
                        ckpt_path,
                    )
                    print(f"           -> New best model saved to: {ckpt_path}")

    print(f"Training finished. Best val loss: {best_val_loss:.6f} (epoch {best_epoch})")
    return ar_model

def train_bidirectional_ar2d_model(
    dino_model,
    ar_model_fwd,
    ar_model_bwd,
    train_loader,
    val_loader,
    device,
    epochs=10,
    lr=1e-3,
    output_dir=None,
    val_interval=1,
    img_size=None,
    vis_imgs=None,
):
    """
    Train two AR2D models:
      - ar_model_fwd: normal orientation (top-left → bottom-right)
      - ar_model_bwd: trained on flipped features (bottom-right → top-left)

    Loss = average of MSE_fwd and MSE_bwd.
    """
    dino_model.eval()
    ar_model_fwd.train()
    ar_model_bwd.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(ar_model_fwd.parameters()) + list(ar_model_bwd.parameters()),
        lr=lr,
    )

    best_val_loss = float("inf")
    best_epoch = -1

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        ar_model_fwd.train()
        ar_model_bwd.train()
        running_loss = 0.0
        n_batches = 0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)

            # DINO features as 2D maps
            feats_2d = extract_dino_tokens_2d(dino_model, imgs, device)  # [B, C, H, W]

            # ---------- forward direction ----------
            preds_fwd = ar_model_fwd(feats_2d)                           # [B, C, H, W]
            loss_fwd = criterion(preds_fwd, feats_2d)

            # ---------- backward direction ----------
            # flip spatially so causal mask sees "past" in reversed scan
            feats_flip = torch.flip(feats_2d, dims=[2, 3])               # [B, C, H, W]
            preds_flip = ar_model_bwd(feats_flip)                        # [B, C, H, W]
            preds_bwd = torch.flip(preds_flip, dims=[2, 3])              # unflip prediction
            loss_bwd = criterion(preds_bwd, feats_2d)

            # total loss = average
            loss = 0.5 * (loss_fwd + loss_bwd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train = running_loss / max(1, n_batches)
        print(f"[Epoch {epoch}/{epochs}] Bi-AR2D train loss: {avg_train:.6f}")

        # ---- Validation every val_interval epochs ----
        if (val_loader is not None) and (epoch % val_interval == 0):
            val_loss = validate_ar2d_model(dino_model, ar_model_fwd, val_loader, device)
            print(f"           -> val loss (forward model only): {val_loss:.6f}")

            # (Optional) visualization with forward model only; you can extend to both
            if vis_imgs is not None:
                visualize_anomaly_grid(
                    dino_model=dino_model,
                    ar_model=ar_model_fwd,
                    imgs_vis=vis_imgs,
                    device=device,
                    img_size=img_size,
                    output_dir=output_dir,
                    epoch=epoch,
                )

            # Save best (based on forward val loss; you could average both if you want)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                if output_dir is not None:
                    ckpt_fwd = os.path.join(output_dir, f"best_bi_ar2d_fwd_{img_size}.pth")
                    ckpt_bwd = os.path.join(output_dir, f"best_bi_ar2d_bwd_{img_size}.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": ar_model_fwd.state_dict(),
                            "val_loss": val_loss,
                            "img_size": img_size,
                        },
                        ckpt_fwd,
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": ar_model_bwd.state_dict(),
                            "img_size": img_size,
                        },
                        ckpt_bwd,
                    )
                    print(f"           -> New best bi-directional models saved to:\n"
                          f"              {ckpt_fwd}\n"
                          f"              {ckpt_bwd}")

    print(f"Bi-directional training finished. Best val loss: {best_val_loss:.6f} (epoch {best_epoch})")
    return ar_model_fwd, ar_model_bwd