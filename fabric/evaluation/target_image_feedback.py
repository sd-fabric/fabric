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
from fabric.utils import get_free_gpu
from fabric.metrics.image_similarity import ImageSimilarity
from fabric.metrics.image_diversity import ImageDiversity
from fabric.metrics.pick_score import PickScore
from fabric.evaluation.utils import make_out_folder, generate_rounds_with_automatic_feedback
from fabric.data.prompthero import get_prompt_image_pairs



def feedback_scores(imgs, ref_image, model):
    return model.compute([ref_image], imgs)[0]


@hydra.main(config_path="../configs", config_name="target_image_feedback", version_base=None)
def main(ctx: DictConfig):
    if ctx.global_seed is not None:
        np.random.seed(ctx.global_seed)
        torch.manual_seed(ctx.global_seed)

    device = "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu"
    device = get_free_gpu() if torch.cuda.is_available() else device
    print(f"Using device: {device}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using dtype: {dtype}")
    
    init_liked_paths = list(ctx.liked_images) if ctx.liked_images else []
    init_disliked_paths = list(ctx.disliked_images) if ctx.disliked_images else []
    init_liked = [Image.open(to_absolute_path(img_path)) for img_path in init_liked_paths]
    init_disliked = [Image.open(to_absolute_path(img_path)) for img_path in init_disliked_paths]

    base_generator = AttentionBasedGenerator(
        model_ckpt=ctx.model_ckpt if hasattr(ctx, "model_ckpt") else None,
        model_name=ctx.model_name if hasattr(ctx, "model_name") else None,
        stable_diffusion_version=ctx.model_version,
        lora_weights=ctx.lora_weights if hasattr(ctx, "lora_weights") else None,
        torch_dtype=dtype
    ).to(device)

    generator = IterativeFeedbackGenerator(
        base_generator,
        init_liked=init_liked,
        init_disliked=init_disliked,
    )
    
    out_folder = ctx.output_path if hasattr(ctx, "output_path") else make_out_folder()
    os.makedirs(out_folder, exist_ok=True)
    
    prompts, images = get_prompt_image_pairs(to_absolute_path(ctx.prompthero_path))
    print(f"Found {len(prompts)} prompts and {len(images)} images")
    
    img_similarity_model = ImageSimilarity(device=device)
    img_diversity_model = ImageDiversity(device=device)

    # If LoRA is specified, make sure the negative prompt is prefixed with "Weird image. "
    negative_prompt = ctx.negative_prompt if hasattr(ctx, "negative_prompt") else ""
    if ctx.lora_weights and "hps_lora" in ctx.lora_weights:
        if not negative_prompt.startswith("Weird image. "):
            print("Using HPS LoRA but 'Weird image. ' was not in negative prompt. Adding it.")
            negative_prompt = "Weird image. " + negative_prompt

    metrics = []
    with torch.inference_mode():
        for idx, prompt in enumerate(tqdm.tqdm(prompts, smoothing=0.01)):
            print(f"Prompt {idx + 1}/{len(prompts)}: {prompt}")
            
            reference_image = images[idx]
            generator.reset()

            feedback_fn = functools.partial(feedback_scores, ref_image=reference_image, model=img_similarity_model)

            ms = generate_rounds_with_automatic_feedback(
                ctx,
                generator, 
                prompt_idx=idx,
                prompt=prompt, 
                neg_prompt=negative_prompt, 
                feedback_fn=feedback_fn, 
                out_folder=out_folder,
                img_similarity_model=img_similarity_model, 
                img_diversity_model=img_diversity_model, 
                init_liked_paths=[],
                init_disliked_paths=[],
                feedback_key="target_img_sim",
            )
            metrics.extend(ms)
            
            pd.DataFrame(metrics).to_csv("metrics.csv", index=False)
            print(f"Saved metrics to {os.path.abspath('.')}/metrics.csv")


if __name__ == "__main__":
    main()
