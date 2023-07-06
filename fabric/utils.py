import os
import math
import warnings
from tempfile import NamedTemporaryFile

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_free_gpu(min_mem=9000):
    try:
        with NamedTemporaryFile() as f:
            os.system(f"nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > {f.name}")
            memory_available = [int(x.split()[2]) for x in open(f.name, 'r').readlines()]
        if max(memory_available) < min_mem:
            warnings.warn("Not enough memory on GPU, using CPU")
            return torch.device("cpu")
        return torch.device("cuda", np.argmax(memory_available))
    except:
        warnings.warn("Could not get free GPU, using CPU")
        return torch.device("cpu")


def display_images(images, n_cols=4, size=4):
    n_rows = int(np.ceil(len(images) / n_cols))
    fig = plt.figure(figsize=(size * n_cols, size * n_rows))
    for i, img in enumerate(images):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(img)
        ax.axis("off")
    fig.tight_layout()
    return fig


def tile_images(images):
    size = images[0].size
    assert all(img.size == size for img in images), "All images must have the same size"

    grid_size_x = math.ceil(len(images) ** 0.5)
    grid_size_y = math.ceil(len(images) / grid_size_x)

    # Create a new blank image with the size of the tiled grid
    tiled_image = Image.new('RGB', (grid_size_x * size[0], grid_size_y * size[1]))

    # Paste the four images into the tiled image
    for x in range(grid_size_x):
        for y in range(grid_size_y):
            idx = x + grid_size_x * y
            if idx >= len(images):
                break
            img = images[idx]
            tiled_image.paste(img, (x * size[0], y * size[1]))
    return tiled_image