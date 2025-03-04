import torch
import requests
import rasterio as rio
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from omnicloudmask import predict_from_array, predict_from_load_func, load_multiband
import omnicloudmask

from functools import partial

import glob

scene_paths = [
    "/explore/nobackup/projects/above/misc/ABoVE_Shrubs/toa/002m/WV02_20100918_M1BS_1030010007073E00-toa.tif"
]
ortho_subimage_cloud_dir = '/explore/nobackup/projects/ilab/scratch/jacaraba/vhr-cloudmask'

load_maxar_pgc_4band_10m = partial(
    load_multiband, resample_res=10.0, band_order=[3, 2, 4])

for scene_path in scene_paths:
    pred_path = predict_from_load_func(
        # scene_paths=scene_paths,
        scene_paths=[scene_path],
        load_func=load_maxar_pgc_4band_10m,
        inference_dtype="bf16",
        no_data_value=65535,
        output_dir=ortho_subimage_cloud_dir,
        inference_device='cuda',
        mosaic_device='cpu',
        overwrite=False,
        batch_size=1,
        # patch_size=256,
        # patch_overlap=128
    )
    print(f"Predicted {pred_path}")

    with torch.no_grad():
        torch.cuda.empty_cache()
