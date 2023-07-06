import os

import pandas as pd
from PIL import Image


def get_prompt_image_pairs(base_path: str = "resources/prompthero"):
    prompt_df = pd.read_csv(os.path.join(base_path, "prompts.csv"))
    prompts, images = [], []
    for _, row in prompt_df.iterrows():
        prompt = row["prompt"]
        image = Image.open(os.path.join(base_path, row["image"]))
        prompts.append(prompt)
        images.append(image)
    return prompts, images
