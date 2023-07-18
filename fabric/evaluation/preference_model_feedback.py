import functools
import os

import hydra
import torch
import pandas as pd
import tqdm
import numpy as np
from PIL import Image
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from fabric.generator import AttentionBasedGenerator
from fabric.iterative import IterativeFeedbackGenerator
from fabric.utils import get_free_gpu, tile_images
from fabric.data.hps_prompts import sample_prompts
from fabric.metrics.image_similarity import ImageSimilarity
from fabric.metrics.image_diversity import ImageDiversity
from fabric.metrics.pick_score import PickScore
from fabric.evaluation.utils import make_out_folder, generate_rounds_with_automatic_feedback


@hydra.main(
    config_path="../configs", config_name="preference_model_feedback", version_base=None
)
def main(ctx: DictConfig):
    # set global torch seed
    if ctx.global_seed is not None:
        torch.manual_seed(ctx.global_seed)
        np.random.seed(ctx.global_seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = get_free_gpu() if torch.cuda.is_available() else device
    print(f"Using device: {device}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using dtype: {dtype}")

    base_generator = AttentionBasedGenerator(
        model_ckpt=ctx.model_ckpt if hasattr(ctx, "model_ckpt") else None,
        model_name=ctx.model_name if hasattr(ctx, "model_name") else None,
        stable_diffusion_version=ctx.model_version,
        lora_weights=to_absolute_path(ctx.lora_weights) if hasattr(ctx, "lora_weights") else None,
        torch_dtype=dtype,
    ).to(device)

    init_liked_paths = list(ctx.liked_images) if ctx.liked_images else []
    init_disliked_paths = list(ctx.disliked_images) if ctx.disliked_images else []
    init_liked = [Image.open(to_absolute_path(img_path)) for img_path in init_liked_paths]
    init_disliked = [Image.open(to_absolute_path(img_path)) for img_path in init_disliked_paths]

    generator = IterativeFeedbackGenerator(
        base_generator,
        init_liked=init_liked,
        init_disliked=init_disliked,
    )

    # If LoRA is specified, make sure the negative prompt is prefixed with "Weird image. "
    negative_prompt = ctx.negative_prompt if hasattr(ctx, "negative_prompt") else ""
    if ctx.lora_weights and "hps_lora" in ctx.lora_weights:
        if not negative_prompt.startswith("Weird image. "):
            print(
                "Using HPS LoRA but 'Weird image. ' was not in negative prompt. Adding it."
            )
            negative_prompt = "Weird image. " + negative_prompt

    out_folder = ctx.output_path if hasattr(ctx, "output_path") else make_out_folder()
    os.makedirs(out_folder, exist_ok=True)

    if ctx.sample_prompt:
        prompts = sample_prompts(max_num_prompts=ctx.num_prompts, seed=0)
    else:
        prompts = [ctx.prompt]

    pref_model = PickScore(device=device)
    img_similarity_model = ImageSimilarity(device=device)
    img_diversity_model = ImageDiversity(device=device)

    metrics = []
    with torch.inference_mode():
        for prompt_idx, prompt in enumerate(tqdm.tqdm(prompts, smoothing=0.01)):
            print(f"Prompt {prompt_idx + 1}/{len(prompts)}: {prompt}")

            generator.reset()

            feedback_fn = functools.partial(pref_model.compute, prompt)

            ms = generate_rounds_with_automatic_feedback(
                ctx,
                generator,
                prompt_idx=prompt_idx,
                prompt=prompt,
                neg_prompt=negative_prompt,
                feedback_fn=feedback_fn,
                out_folder=out_folder,
                img_similarity_model=img_similarity_model,
                img_diversity_model=img_diversity_model,
                init_liked_paths=init_liked_paths,
                init_disliked_paths=init_disliked_paths,
                feedback_key="pref_score",
            )
            metrics.extend(ms)

            pd.DataFrame(metrics).to_csv("metrics.csv", index=False)
            print(f"Saved metrics to {os.path.abspath('.')}/metrics.csv")


if __name__ == "__main__":
    main()
