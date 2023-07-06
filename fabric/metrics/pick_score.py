import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image


class PickScore:
    """
    Compute Pick-A-Pic score from text and images.
    """

    def __init__(
        self,
        processor_name_or_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_pretrained_name_or_path: str = "yuvalkirstain/PickScore_v1",
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = (
            AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
        )
        self.device = device

    def compute_from_paths(self, text, image_paths):
        """
        ::param text: str
        ::param image_paths: list of str
        ::return: list of float
        """
        images = [Image.open(image_path) for image_path in image_paths]
        return self.compute(text, images)

    def compute_from_tensor(self, text, images):
        """
        ::param text: str
        ::param images: torch.Tensor
        ::return: list of float
        """
        pil_images = [
            Image.fromarray(image.permute(1, 2, 0).cpu().numpy()) for image in images
        ]
        return self.compute(text, pil_images)

    @torch.no_grad()
    def compute(self, text, images):
        """
        ::param text: str
        ::param images: list of PIL.Image
        ::return: list of float
        """
        # preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=text,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        # embed
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = 100 * self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        return scores.cpu().tolist()
