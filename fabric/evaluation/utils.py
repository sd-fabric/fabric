import glob
import os
from datetime import date

import torch
import numpy as np
from PIL import Image

from fabric.utils import tile_images


def make_out_folder(output_path: str = "outputs"):
    date_str = date.today().strftime("%Y-%m-%d")
    out_folder = os.path.join(output_path, date_str)
    experiment_paths = sorted(glob.glob(os.path.join(out_folder, "experiment_*")))
    n_experiment = len(experiment_paths)
    out_folder = os.path.join(out_folder, "experiment_" + str(n_experiment))
    os.makedirs(out_folder, exist_ok=True)
    return out_folder


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

@torch.no_grad()
def generate_rounds_with_automatic_feedback(
    ctx,
    generator, 
    prompt_idx,
    prompt, 
    neg_prompt, 
    feedback_fn, 
    out_folder="images",
    img_similarity_model=None, 
    img_diversity_model=None, 
    init_liked_paths=[],
    init_disliked_paths=[],
    feedback_key="feedback_score"
):
    liked_paths = init_liked_paths.copy()
    disliked_paths = init_disliked_paths.copy()

    metrics = []
    for i in range(ctx.n_rounds):
        imgs, params = generator.generate(
            prompt=prompt,
            negative_prompt=neg_prompt,
            prompt_dropout=ctx.prompt_dropout,
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
        
        feedback_scores = feedback_fn(imgs)
        round_diversity = img_diversity_model.compute(imgs)

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

        out_paths = []
        for j, (p, img, score, pos_sim, neg_sim) in enumerate(
            zip(params["prompts"], imgs, feedback_scores, pos_sims, neg_sims)
        ):
            out_path = os.path.join(out_folder, f"prompt_{prompt_idx}_round_{i}_image_{j}.png")
            out_paths.append(out_path)
            img.save(out_path)
            print(f"Saved image to {out_path}")

            metrics.append({
                "round": i,
                "prompt": p,
                "prompt_idx": prompt_idx,
                "image_idx": j,
                "image": out_path,
                feedback_key: score,
                "round_diversity": round_diversity,
                "pos_sim": pos_sim,
                "neg_sim": neg_sim,
                "seed": params["seed"],
                "liked": liked_paths.copy(),
                "disliked": disliked_paths.copy(),
            })

        if len(imgs) > 1:
            tiled = tile_images(imgs)
            tiled_path = os.path.join(out_folder, f"prompt_{prompt_idx}_tiled_round_{i}.png")
            tiled.save(tiled_path)
            print(f"Saved tile to {tiled_path}")

        if ctx.use_pos_feedback:
            liked_idx = np.argmax(feedback_scores)
            generator.give_feedback([imgs[liked_idx]], [])
            liked_paths.append(out_paths[liked_idx])
        if ctx.use_neg_feedback:
            disliked_idx = np.argmin(feedback_scores)
            generator.give_feedback([], [imgs[disliked_idx]])
            disliked_paths.append(out_paths[disliked_idx])

        print(f"{feedback_scores}: {feedback_scores}")
        print(f"In-batch similarity: {round_diversity}")
        print(f"Pos. similarities: {pos_sims}")
        print(f"Neg. similarities: {neg_sims}")