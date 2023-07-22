# FABRIC: Personalizing Diffusion Models with Iterative Feedback

[Paper](https://arxiv.org/abs/2307.10159) |
[Website](https://sd-fabric.github.io/) |
[Colab](https://colab.research.google.com/drive/1rWZ4jQHMvjc-l7xYAssa_OUOaAx3XDQT?usp=sharing) |
[Gradio](https://colab.research.google.com/drive/12pFi6WAKASG18uH3UcxGMVI37e1pIwAz)

FABRIC (Feedback via Attention-Based Reference Image Conditioning) is a technique to incorporate iterative feedback into the generative process of diffusion models based on StableDiffusion.
This is done by exploiting the self-attention mechanism in the U-Net in order to condition the diffusion process on a set of positive and negative reference images that are to be chosen based on human feedback.


## Setup

- Option 1:
Install the repository as a pip-package (does not install dependencies, check `requirements.txt` for required dependencies):
```bash
pip install git+https://github.com/sd-fabric/fabric.git
```

- Option 2:
Clone the repository, create virtual environment and install the required packages as follows:
```bash
python3 -m venv .venv  # create new virtual environment
source .venv/bin/activate  # activate it
pip install -r requirements.txt  # install requirements
pip install -e .  # install current repository in editable mode
```

## Usage

The `fabric/single_round.py` script can be used to run a single round of (optionally) feedback-conditioned generation as follows:
```bash
# 1st round (text-to-image w/o feedback)
python fabric/single_round.py prompt="photo of a dog running on grassland, masterpiece, best quality, fine details"
# 2nd round (text-to-image w/ feedback)
python fabric/run_single.py \
    prompt="photo of a dog running on grassland, masterpiece, best quality, fine details" \
    liked="[outputs/images/2023-07-06/example_1_1.png]" \
    disliked="[outputs/images/2023-07-06/example_1_3.png]"
```

Alternatively, the FABRIC generators can be used to incorporate iterative feedback in the generation process as follows:
```python
from PIL import Image

from fabric.generator import AttentionBasedGenerator
from fabric.iterative import IterativeFeedbackGenerator

def get_feedback(images) -> tuple[list[Image.Image], list[Image.Image]]:
    raise NotImplementedError("TODO: Implement your own function to select positive and negative feedback")

base_generator = AttentionBasedGenerator("dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16)
base_generator.to("cuda")

generator = IterativeFeedbackGenerator(base_generator)

prompt = "photo of a dog running on grassland, masterpiece, best quality, fine details"
negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

for _ in range(4):
    images: list[Image.Image] = generator.generate(prompt, negative_prompt=negative_prompt)
    liked, disliked = get_feedback(images)
    generator.give_feedback(liked, disliked)
generator.reset()
```

## Evaluation

To replicate the evaluation results, the provided evaluation scripts can be used as follows:
```bash
# Experiment 1: Preference model-based feedback selection
python fabric/evaluation/preference_model_feedback.py
# Experiment 2: Target image-based feedback selection
python fabric/evaluation/target_image_feedback.py
```

## Citation
```
@misc{vonrutte2023fabric,
      title={FABRIC: Personalizing Diffusion Models with Iterative Feedback}, 
      author={Dimitri von Rütte and Elisabetta Fedele and Jonathan Thomm and Lukas Wolf},
      year={2023},
      eprint={2307.10159},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
