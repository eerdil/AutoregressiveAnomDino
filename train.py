
import argparse
import os
import torch
import random
import numpy as np

from datetime import datetime

from data.brats import get_anomaly_loader, get_train_loader
from models.autoregressive2d import AR2DModel
from models.dinov3_utils import load_dinov3_models, extract_dino_tokens_2d
from config.paths import PROJECT_PATH, DATASET_PATH, checkpoint_paths
from ar.train_loops import train_ar2d_model
from ar.visualization import select_visualization_subset_from_loader

def set_seed(seed: int = 42):
    print(f"[Seed] Setting seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # (Optional, stricter but can be slower / may error on some ops)
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def main():
    # ARGUMENT PARSING
    parser = argparse.ArgumentParser(description="Visualize DINOv3 features.")
    parser.add_argument("--model", type=str, default="dinov3_vits16", choices=checkpoint_paths.keys(), help="Type of DINOv3 model to use.")
    parser.add_argument("--img_size", type=int, default=448, help="Size to resize images to.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs for autoregressive model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for autoregressive model training.")
    parser.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs.")
    parser.add_argument("--non_causal", action="store_true", help="Use non-causal convs (see past and future pixels).")
    parser.add_argument("--center_masked_first", action="store_true", help="Use center-masked convs (see all neighbors except center pixel).")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for AR model convolution.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    model_type = args.model
    img_size = args.img_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    lr = args.lr
    val_interval = args.val_interval
    causal = not args.non_causal
    kernel_size = args.kernel_size
    center_masked_first = args.center_masked_first
    seed = args.seed
    
    set_seed(seed)

    mode_str = (
        "causal" if causal
        else ("cmask" if center_masked_first else "noncausal")
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{model_type}_{mode_str}_ks{kernel_size}_img{img_size}_{timestamp}"
        
    output_path = os.path.join(PROJECT_PATH, f"results/{exp_name}")

    # GET TRAINING DATA LOADER
    dataloader_train = get_train_loader(os.path.join(DATASET_PATH, "train/good"), 
                                        batch_size=batch_size, 
                                        img_size=img_size, 
                                        num_workers=num_workers)
    
    dataloader_valid = get_anomaly_loader(os.path.join(DATASET_PATH, "valid"),
                                          batch_size=batch_size,
                                          img_size=img_size,
                                          num_workers=num_workers)
    
    # ------------------------------------------------------------
    # Pick a fixed set of validation images for visualization
    # ------------------------------------------------------------
    imgs_vis, labels_vis = select_visualization_subset_from_loader(
        dataloader_valid,
        n_anom=10,
        n_norm=10,
    )

    # LOAD DINOV3
    dino = load_dinov3_models(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = dino.to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False
    
    # GET FEATURE DIMENSION C FOR ONE BATCH
    imgs_batch, _ = next(iter(dataloader_train))
    feats_2d = extract_dino_tokens_2d(dino, imgs_batch.to(device), device)
    C = feats_2d.shape[1]
    print(f"DINO feature channels: {C}")
    
    print(f"Using {'causal' if causal else 'non-causal'} single AR2D model.")
    ar2d = AR2DModel(
        in_channels=C,
        hidden_channels=256,
        n_layers=5,
        kernel_size=kernel_size,
        causal=causal,
        center_masked_first=center_masked_first,
    ).to(device)

    ar2d = train_ar2d_model(
        dino_model=dino,
        ar_model=ar2d,
        train_loader=dataloader_train,
        val_loader=dataloader_valid,
        device=device,
        epochs=epochs,
        lr=lr,
        val_interval=val_interval,
        output_dir=output_path,
        img_size=img_size,
        imgs_vis=imgs_vis,
        labels_vis=labels_vis,
    )


if __name__ == "__main__":
    main()