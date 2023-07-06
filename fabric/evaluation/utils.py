import glob
import os
from datetime import date

from PIL import Image

def make_out_folder(output_path: str = "outputs"):
    date_str = date.today().strftime("%Y-%m-%d")
    out_folder = os.path.join(output_path, date_str)
    experiment_paths = sorted(glob.glob(os.path.join(out_folder, 'experiment_*')))
    n_experiment = len(experiment_paths)
    out_folder = os.path.join(out_folder, "experiment_" + str(n_experiment))
    os.makedirs(out_folder, exist_ok=True)

def get_prompts(path_to_dir):
    num_prompts = len(glob.glob(os.path.join(path_to_dir, "*.txt")))
    prompts = []
    for i in range(num_prompts):
        with open(os.path.join(path_to_dir, f"{i+1}.txt"), "r") as f:
            prompt = f.read()
        prompts.append(prompt)
    return prompts

def get_images(path_to_dir):
    num_images = len(glob.glob(os.path.join(path_to_dir, "*.jpg")))
    images = []
    for i in range(num_images):
        image = Image.open(os.path.join(path_to_dir, f"{i+1}.jpg"))
        images.append(image)
    return images
