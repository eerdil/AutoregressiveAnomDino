import os

BMIC_DATA_PATH = "/usr/bmicnas02/data-biwi-01/fm_originalzoo/dinov3"
PROJECT_PATH = "/usr/bmicnas02/data-biwi-01/erdile_data/projects/AR_Anom_Det_Dino/dinov3"
DATASET_PATH = "/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/BMAD/BraTS2021_slice"

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