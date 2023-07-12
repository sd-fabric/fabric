import os

import hydra
import torch
import pandas as pd
import tqdm
import numpy as np
from PIL import Image
from omegaconf import DictConfig

from fabric.generator import AttentionBasedGenerator
from fabric.iterative import IterativeFeedbackGenerator
from fabric.utils import get_free_gpu, tile_images
from fabric.metrics.image_similarity import ImageSimilarity
from fabric.metrics.image_diversity import ImageDiversity
from fabric.metrics.pick_score import PickScore
from fabric.evaluation.utils import make_out_folder
from fabric.data.prompthero import get_prompt_image_pairs


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
    init_liked = [Image.open(img_path) for img_path in init_liked_paths]
    init_disliked = [Image.open(img_path) for img_path in init_disliked_paths]

    base_generator = AttentionBasedGenerator(
        model_ckpt=ctx.model_ckpt if hasattr(ctx, "model_ckpt") else None,
        model_name=ctx.model_name if hasattr(ctx, "model_name") else None,
        stable_diffusion_version=ctx.model_version,
        lora_weights=ctx.lora_weights if hasattr(ctx, "lora_weights") else None,
        torch_dtype=dtype,
        device=device
    ).to(device)

    generator = IterativeFeedbackGenerator(
        base_generator,
        init_liked=init_liked,
        init_disliked=init_disliked,
    )
    
    out_folder = make_out_folder(ctx.output_path)
    
    prompts, images = get_prompt_image_pairs(ctx.prompthero_path)
    print(f"Found {len(prompts)} prompts and {len(images)} images")
    
    pref_model = PickScore(device=device)
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
            liked_paths = init_liked_paths.copy()
            disliked_paths = init_disliked_paths.copy()
            generator.reset()

            for i in range(ctx.n_rounds):
                imgs, params = generator.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=ctx.image_seed,
                    n_images=ctx.n_images,
                    denoising_steps=ctx.denoising_steps,
                    guidance_scale=ctx.guidance_scale,
                    feedback_start=ctx.feedback.start,
                    feedback_end=ctx.feedback.end,
                    min_weight=ctx.feedback.min_weight,
                    max_weight=ctx.feedback.max_weight,
                    neg_scale=ctx.feedback.neg_scale,
                    return_params=True,
                )
                
                img_sim_scores = img_similarity_model.compute([reference_image], imgs)[0]
                liked_idx = np.argmax(img_sim_scores)
                disliked_idx = np.argmin(img_sim_scores)
                generator.give_feedback([imgs[liked_idx]], [imgs[disliked_idx]])

                liked, disliked = params["liked"], params["disliked"]
                if len(liked) > 0:
                    pos_sims = img_similarity_model.compute(imgs, liked)
                    pos_sims = np.mean(pos_sims, axis=1)
                else:
                    pos_sims = [None] * len(imgs)
                
                if len(disliked) > 0:
                    neg_sims = img_similarity_model.compute(imgs, disliked)
                    neg_sims = np.mean(neg_sims, axis=1)
                else:
                    neg_sims = [None] * len(imgs)
                    
                pref_scores = pref_model.compute(prompt, imgs)                
                
                round_diversity = img_diversity_model.compute(imgs)

                out_paths = []
                for j, (p, img, pref_score, target_img_sim, pos_sim, neg_sim) in enumerate(
                    zip(params["prompts"], imgs, pref_scores, img_sim_scores, pos_sims, neg_sims)
                ):
                    out_path = os.path.join(out_folder, f"prompt_{idx}_round_{i}_image_{j}.png")
                    out_paths.append(out_path)
                    img.save(out_path)
                    print(f"Saved image to {out_path}")

                    metrics.append({
                        "round": i,
                        "prompt": p,
                        "prompt_idx": idx,
                        "image_idx": j,
                        "image": out_path,
                        "pref_score": pref_score,
                        "target_img_sim": target_img_sim,
                        "round_diversity": round_diversity,
                        "pos_sim": pos_sim,
                        "neg_sim": neg_sim,
                        "seed": params["seed"],
                        "liked": liked_paths,
                        "disliked": disliked_paths,
                    })

                if len(imgs) > 1:
                    tiled = tile_images(imgs)
                    tiled_path = os.path.join(out_folder, f"prompt_{idx}_tiled_round_{i}.png")
                    tiled.save(tiled_path)
                    print(f"Saved tile to {tiled_path}")

                liked_paths.append(out_paths[liked_idx])
                disliked_paths.append(out_paths[disliked_idx])

                print(f"Similarities to target image: {img_sim_scores}")
                print(f"Preference scores: {pref_scores}")
                print(f"In-batch similarity: {round_diversity}")
                print(f"Pos. similarities: {pos_sims}")
                print(f"Neg. similarities: {neg_sims}")

            pd.DataFrame(metrics).to_csv(os.path.join(out_folder, "metrics.csv"), index=False)
            print(f"Saved metrics to {out_folder}/metrics.csv")


if __name__ == "__main__":
    main()
