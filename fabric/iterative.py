from typing import Optional, List

import torch
import numpy as np
from PIL import Image


def word_dropout(x, p=0.5):
    words = x.split()
    words = [word for word in words if np.random.rand() > p]
    return " ".join(words)


class IterativeFeedbackGenerator:
    def __init__(
        self,
        generator,
        init_liked: List[Image.Image] = [],
        init_disliked: List[Image.Image] = [],
    ):
        self.init_liked = init_liked.copy()
        self.init_disliked = init_disliked.copy()
        self.generator = generator
        self.reset()

    def reset(self):
        self.liked = self.init_liked.copy()
        self.disliked = self.init_disliked.copy()
        self.round = 0

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        n_images: int = 4,
        prompt_dropout: float = 0.0,
        return_params: bool = False,
        **kwargs,
    ):
        if seed is None:
            seed = torch.randint(0, 2 ** 32, (1,)).item()

        ps = [prompt] * n_images
        if prompt_dropout > 0:
            ps = [word_dropout(p, prompt_dropout) for p in ps]

        imgs = self.generator.generate(
            prompt=ps,
            liked=self.liked,
            disliked=self.disliked,
            seed=seed,
            n_images=n_images,
            **kwargs,
        )
        
        if return_params:
            curr_round = self.round
            self.round += 1
            return imgs, {
                "round": curr_round,
                "prompts": ps,
                "liked": self.liked.copy(),
                "disliked": self.disliked.copy(),
                "seed": seed,
                "n_images": n_images,
                **kwargs,
            }
        return imgs

    def give_feedback(
        self,
        liked: List[Image.Image] = [],
        disliked: List[Image.Image] = [],
    ):
        self.liked.extend(liked)
        self.disliked.extend(disliked)
