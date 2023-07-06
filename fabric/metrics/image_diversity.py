import torch
import numpy as np
import clip


class ImageDiversity:
    def __init__(
        self,
        clip_model: str = "ViT-L/14@336px", #"ViT-B/32",
        # open_clip_dataset: str = "laion2b_s39b_b160k",
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
    def compute(self, imgs):
        n_images = len(imgs)
        # Preprocess image
        processed = []
        for img in imgs:
            img = self.preprocess(img).unsqueeze(0).to(self.device)
            processed.append(img)
        processed = torch.cat(processed, dim=0)

        imgs_features = self.model.encode_image(processed)
        imgs_features /= imgs_features.norm(dim=-1, keepdim=True)

        # Compute the CLIP score
        similarity = (100.0 * imgs_features @ imgs_features.T).cpu().numpy()
        
        return similarity[np.triu_indices(n_images, k=1)].mean()