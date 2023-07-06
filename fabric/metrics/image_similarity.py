import torch
import clip


class ImageSimilarity:
    def __init__(
        self,
        clip_model: str = "ViT-L/14@336px", #"ViT-B/32",
        device: str = None,
    ):
        if not device:
            # Use MPS if available, otherwise use CPU
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            # Use CUDA if available, otherwise use CPU or MPS
            self.device = "cuda" if torch.cuda.is_available() else self.device
        else:
            self.device = device
        self.model, self.preprocess = clip.load(clip_model, device=self.device)

    @torch.no_grad()
    def compute(self, imgs1, imgs2):
        # Preprocess image
        processed1, processed2 = [], []
        for img in imgs1:
            img = self.preprocess(img).unsqueeze(0).to(self.device)
            processed1.append(img)
        processed1 = torch.cat(processed1, dim=0)
        for img in imgs2:
            img = self.preprocess(img).unsqueeze(0).to(self.device)
            processed2.append(img)
        processed2 = torch.cat(processed2, dim=0)

        imgs1_features = self.model.encode_image(processed1)
        imgs2_features = self.model.encode_image(processed2)
        imgs1_features /= imgs1_features.norm(dim=-1, keepdim=True)
        imgs2_features /= imgs2_features.norm(dim=-1, keepdim=True)

        # Compute the CLIP score
        similarity = (100.0 * imgs1_features @ imgs2_features.T)
        return similarity.cpu().numpy()
