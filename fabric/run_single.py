import os
import re
from datetime import date

import hydra
import torch
from PIL import Image
from omegaconf import DictConfig

from fabric.generator import AttentionBasedGenerator
from fabric.utils import get_free_gpu


@hydra.main(config_path="configs", config_name="single_round", version_base=None)
def main(ctx: DictConfig):
    print("main")
    device = "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu"
    device = get_free_gpu() if torch.cuda.is_available() else device
    print(f"Using device: {device}")

    dtype = torch.float16
    print(f"Using dtype: {dtype}")

    generator = AttentionBasedGenerator(model_ckpt=ctx.model_ckpt if hasattr(ctx, "model_ckpt") else None,
        model_name=ctx.model_name if hasattr(ctx, "model_name") else None, stable_diffusion_version=ctx.model_version,
        torch_dtype=dtype, ).to(device)

    liked = [Image.open("square.jpg")][:ctx.topk]

    imgs = generator.generate(prompt=ctx.prompt, negative_prompt=ctx.negative_prompt, liked=liked, disliked=[],
        seed=ctx.seed, n_images=ctx.n_images, guidance_scale=ctx.guidance_scale, denoising_steps=ctx.denoising_steps,
        feedback_start=ctx.feedback.start, feedback_end=ctx.feedback.end, min_weight=ctx.feedback.min_weight,
        max_weight=ctx.feedback.max_weight, neg_scale=ctx.feedback.neg_scale, size=ctx.size,
                              warmup_power=ctx.feedback.warmup_power)

    date_str = date.today().strftime("%Y-%m-%d")
    out_folder = os.path.join("outputs", "images", date_str)
    os.makedirs(out_folder, exist_ok=True)

    n_files = max(
        [int(f.split(".")[0].split("_")[1]) for f in os.listdir(out_folder) if re.match(r"example_[0-9_]+\.png", f)],
        default=0) + 1
    for i, img in enumerate(imgs):
        # each image is of the form example_ID.png. Extract the max id
        out_path = os.path.join(out_folder, f"example_{n_files}_{i}.png")
    img.save(out_path)
    print(f"Saved image to {out_path}")


if __name__ == "__main__":
    main()
