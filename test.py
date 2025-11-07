
import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

BMIC_DATA_PATH = "/usr/bmicnas02/data-biwi-01/fm_originalzoo/dinov3"
PROJECT_PATH = "/usr/bmicnas02/data-biwi-01/erdile_data/projects/loobesity/DinoV3/"

checkpoint_paths = {
    'dinov3_vits16':         os.path.join(BMIC_DATA_PATH, "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    'dinov3_vits16plus':     os.path.join(BMIC_DATA_PATH, "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"),
    'dinov3_vitb16':         os.path.join(BMIC_DATA_PATH, "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    'dinov3_vitl16':         os.path.join(BMIC_DATA_PATH, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    'dinov3_vith16plus':     os.path.join(BMIC_DATA_PATH, "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"),
    'dinov3_vit7b16':        os.path.join(BMIC_DATA_PATH, "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"),
    'dinov3_convnext_tiny':  os.path.join(BMIC_DATA_PATH, "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"),
    'dinov3_convnext_small': os.path.join(BMIC_DATA_PATH, "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"),
    'dinov3_convnext_base':  os.path.join(BMIC_DATA_PATH, "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"),
    'dinov3_convnext_large': os.path.join(BMIC_DATA_PATH, "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"),
    'dinov3_vitl16_sat':     os.path.join(BMIC_DATA_PATH, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
    'dinov3_vit7b16_sat':    os.path.join(BMIC_DATA_PATH, "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"),
}

def load_dinov3_models(model_type):
    # DINOv3 ViT models pretrained on web images
    model = torch.hub.load(repo_or_dir = os.path.join(PROJECT_PATH, "dinov3"), 
                           model = model_type, 
                           source='local', 
                           weights=checkpoint_paths[model_type])
    return model

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])

def create_pca_image(features, n_components=3, H=None, W=None):
    N, S, D = features.shape
    if S < n_components:
        raise ValueError(f"Number of tokens {S} is less than the number of PCA components {n_components}.")
    
    features = features.cpu().numpy()
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features.reshape(N*S, D))
    if H is None or W is None:
        H = W = int(S**0.5)
    pca_result = pca_result.reshape(N, H, W, n_components)
    return pca_result

def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv3 features.")
    parser.add_argument("--model", type=str, default="dinov3_vitl16", choices=checkpoint_paths.keys(), help="Type of DINOv3 model to use.")
    parser.add_argument("--img_size", type=int, default=1024, help="Size to resize images to.")
    args = parser.parse_args()

    model_type = args.model
    img_size = args.img_size

    transform = make_transform(img_size)
    img = Image.open(os.path.join(BMIC_DATA_PATH, "image.png")).convert("RGB")
    img = transform(img).unsqueeze(0)

    model = load_dinov3_models(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    img = img.to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        # warmup (GPU clocks & cudnn autotune)
        for _ in range(5):
            _ = model(img)

        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        iters = 30
        start.record()
        for _ in range(iters):
            token_features = model.get_intermediate_layers(img)[0]
        end.record()

        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters   # milliseconds per iteration
        print(f"Avg GPU time {model_type}: {ms:.3f} ms")

        cls_features = model(img)

    # breakpoint()
    pca_result = create_pca_image(token_features)
    pca_result = pca_result.squeeze(0)
    pca_result = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
    plt.figure()
    plt.imshow(pca_result)
    plt.axis('off')
    plt.title(f"{model_type}, {img_size}x{img_size}")
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_PATH, f"pca_result_{model_type}_{img_size}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    # # FORCING FLASH ATTENTION - RESULTS THE SAME PERFORMANCE AS ABOVE. SO ABOVE ALSO USES FLASH ATTENTION
    # with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
    #     # ensure parameters run in fp16 too (safer to call once outside the context)

    #     # prefer new API if your torch has it:
    #     from torch.nn.attention import sdpa_kernel, SDPBackend
    #     ctx = sdpa_kernel(SDPBackend.FLASH_ATTENTION)  # flash only
    #     # try:
    #     #     ctx = sdpa_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
    #     # except Exception:
    #     #     ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)

    #     with ctx:
    #         # warmup in fp16
    #         for _ in range(5):
    #             _ = model(img)

    #         torch.cuda.synchronize()
    #         start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    #         iters = 30
    #         start.record()
    #         for _ in range(iters):
    #             token_features = model.get_intermediate_layers(img)[0]
    #         end.record()
    #         torch.cuda.synchronize()
    #         print(f"Avg GPU time (flash, fp16): {start.elapsed_time(end)/iters:.3f} ms")

    # breakpoint()

if __name__ == "__main__":
    main()