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
from fabric.data.hps_prompts import sample_prompts
from fabric.metrics.image_similarity import ImageSimilarity
from fabric.metrics.image_diversity import ImageDiversity
from fabric.metrics.pick_score import PickScore
from fabric.evaluation.utils import make_out_folder


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
        lora_weights=ctx.lora_weights if hasattr(ctx, "lora_weights") else None,
        torch_dtype=dtype,
    ).to(device)

    init_liked_paths = list(ctx.liked_images) if ctx.liked_images else []
    init_disliked_paths = list(ctx.disliked_images) if ctx.disliked_images else []
    init_liked = [Image.open(img_path) for img_path in init_liked_paths]
    init_disliked = [Image.open(img_path) for img_path in init_disliked_paths]

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

    out_folder = make_out_folder(ctx.output_path)

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

                pref_scores = pref_model.compute(prompt, imgs)
                liked_idx = np.argmax(pref_scores)
                disliked_idx = np.argmin(pref_scores)
                generator.give_feedback([imgs[liked_idx]], [imgs[disliked_idx]])

                liked, disliked = generator.give_feedback()
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

                round_diversity = img_diversity_model.compute(imgs)

                out_paths = []
                for j, (img, pref_score, pos_sim, neg_sim) in enumerate(
                    zip(imgs, pref_scores, pos_sims, neg_sims)
                ):
                    out_path = os.path.join(
                        out_folder, f"prompt_{prompt_idx}_round_{i}_image_{j}.png"
                    )
                    out_paths.append(out_path)
                    img.save(out_path)
                    print(f"Saved image to {out_path}")

                    metrics.append(
                        {
                            "round": i,
                            "prompt": prompt,
                            "prompt_idx": prompt_idx,
                            "image_idx": j,
                            "image": out_path,
                            "pref_score": pref_score,
                            "pos_sim": pos_sim,
                            "neg_sim": neg_sim,
                            "seed": params["seed"],
                            "liked": liked_paths.copy(),
                            "disliked": disliked_paths.copy(),
                        }
                    )
                if len(imgs) > 1:
                    tiled = tile_images(imgs)
                    tiled_path = os.path.join(
                        out_folder, f"prompt_{prompt_idx}_tiled_round_{i}.png"
                    )
                    tiled.save(tiled_path)
                    print(f"Saved tile to {tiled_path}")

                liked_paths.append(out_paths[liked_idx])
                disliked_paths.append(out_paths[disliked_idx])

                print(f"Preference scores: {pref_scores}")
                print(f"Round diversity: {round_diversity}")
                print(f"Pos. similarities: {pos_sims}")
                print(f"Neg. similarities: {neg_sims}")

            pd.DataFrame(metrics).to_csv(
                os.path.join(out_folder, "metrics.csv"), index=False
            )
            print(f"Saved metrics to {out_folder}/metrics.csv")


if __name__ == "__main__":
    main()
