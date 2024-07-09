import warnings
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import xformers
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler as Scheduler
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from tqdm import tqdm

has_xformers = True


def apply_unet_lora_weights(pipeline, unet_path):
    model_weight = torch.load(unet_path, map_location="cpu")
    unet = pipeline.unet
    lora_attn_procs = {}
    lora_rank = list(set([v.size(0) for k, v in model_weight.items() if k.endswith("down.weight")]))
    assert len(lora_rank) == 1
    lora_rank = lora_rank[0]
    for name in unet.attn_processors.keys():
        cross_attention_dim = (None if name.endswith("attn1.processor") else unet.config.cross_attention_dim)
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                       rank=lora_rank, ).to(pipeline.device)
    unet.set_attn_processor(lora_attn_procs)
    unet.load_state_dict(model_weight, strict=False)


def attn_with_weights(attn: nn.Module, hidden_states, encoder_hidden_states=None, attention_mask=None, weights=None,
                      # shape: (batch_size, sequence_length)
                      lora_scale=1.0, batch: Optional[int] = None):
    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if isinstance(attn.processor, LoRACrossAttnProcessor):
        query = attn.to_q(hidden_states) + lora_scale * attn.processor.to_q_lora(hidden_states)
    else:
        query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    if isinstance(attn.processor, LoRACrossAttnProcessor):
        key = attn.to_k(encoder_hidden_states) + lora_scale * attn.processor.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + lora_scale * attn.processor.to_v_lora(encoder_hidden_states)
    else:
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

    if batch is None:
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask, op=None,
                                                                scale=attn.scale)
    else:
        ref_q, ref_k, ref_v = query[batch:], key[batch:], value[batch:]
        q, k, v = query[:batch], key[:batch], value[:batch]
        q_pos, q_neg = q.chunk(2, 0)
        k_pos, k_neg = k.chunk(2, 0)
        v_pos, v_neg = v.chunk(2, 0)

        ref_v2 = ref_v.view(1, -1, *ref_v.shape[2:]).expand(q_pos.size(0), -1, -1)
        ref_k2 = ref_k.view(1, -1, *ref_k.shape[2:]).expand(q_pos.size(0), -1, -1)
        k, v = torch.cat([k_pos, ref_k2], dim=1), torch.cat([v_pos, ref_v2], dim=1)

        if weights is not None:
            bias = torch.ones(k.size(1), device=k.device, dtype=k.dtype)
            bias[k_pos.size(1):] = weights
            bias = bias.log().view(1, 1, -1).expand(attn.heads, query.size(1), -1)
            if attention_mask is None:
                attention_mask = bias
            else:
                attention_mask += bias

        ref_q, ref_k, ref_v, q_pos, k, v, q_neg, k_neg, v_neg = [attn.head_to_batch_dim(x).contiguous() for x in
                                                                 [ref_q, ref_k, ref_v, q_pos, k, v, q_neg, k_neg,
                                                                  v_neg]]
        ref_hidden_states = xformers.ops.memory_efficient_attention(ref_q, ref_k, ref_v, op=None, scale=attn.scale)
        pos_hidden_states = xformers.ops.memory_efficient_attention(q_pos, k, v, attn_bias=attention_mask, op=None,
                                                                    scale=attn.scale)
        neg_hidden_states = xformers.ops.memory_efficient_attention(q_neg, k_neg, v_neg, op=None, scale=attn.scale)
        hidden_states = torch.cat([pos_hidden_states, neg_hidden_states, ref_hidden_states], dim=0)
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    if isinstance(attn.processor, LoRACrossAttnProcessor):
        hidden_states = attn.to_out[0](hidden_states) + lora_scale * attn.processor.to_out_lora(hidden_states)
    else:
        hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    return hidden_states


class AttentionBasedGenerator(nn.Module):
    def __init__(self, model_name: Optional[str] = None, model_ckpt: Optional[str] = None,
                 stable_diffusion_version: str = "1.5", lora_weights: Optional[str] = None, torch_dtype=torch.float32):
        super().__init__()

        if stable_diffusion_version == "2.1":
            warnings.warn("StableDiffusion v2.x is not supported and may give unexpected results.")

        if model_name is None:
            if stable_diffusion_version == "1.5":
                model_name = "runwayml/stable-diffusion-v1-5"
            elif stable_diffusion_version == "2.1":
                model_name = "stabilityai/stable-diffusion-2-1"
            else:
                raise ValueError(
                    f"Unknown stable diffusion version: {stable_diffusion_version}. Version must be either '1.5' or '2.1'")

        scheduler = Scheduler.from_pretrained(model_name, subfolder="scheduler")

        if model_ckpt is not None:
            pipe = StableDiffusionPipeline.from_ckpt(model_ckpt, scheduler=scheduler, torch_dtype=torch_dtype,
                                                     safety_checker=None, )
            pipe.scheduler = scheduler
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_name, scheduler=scheduler, torch_dtype=torch_dtype,
                                                           safety_checker=None, )

        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

        if lora_weights:
            print(f"Applying LoRA weights from {lora_weights}")
            apply_unet_lora_weights(pipeline=pipe, unet_path=lora_weights)

        self.pipeline = pipe
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler: Scheduler = scheduler
        self.dtype = torch_dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, device):
        self.pipeline.to(device)
        return super().to(device)

    def initialize_prompts(self, prompts: List[str]):
        prompt_tokens = self.tokenizer(prompts, return_tensors="pt", max_length=self.tokenizer.model_max_length,
                                       padding="max_length", truncation=True, )

        if (hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask):
            attention_mask = prompt_tokens.attention_mask.to(self.device)
        else:
            attention_mask = None

        prompt_embd = self.text_encoder(input_ids=prompt_tokens.input_ids.to(self.device),
                                        attention_mask=attention_mask, ).last_hidden_state

        return prompt_embd

    def get_unet_hidden_states(self, z_all, t, prompt_embd):
        cached_hidden_states = []
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                def new_forward(self, hidden_states, *args, **kwargs):
                    cached_hidden_states.append(hidden_states.clone().detach().cpu())
                    return self.old_forward(hidden_states, *args, **kwargs)

                module.attn1.old_forward = module.attn1.forward
                module.attn1.forward = new_forward.__get__(module.attn1)

        # run forward pass to cache hidden states, output can be discarded
        _ = self.unet(z_all, t, encoder_hidden_states=prompt_embd)

        # restore original forward pass
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = module.attn1.old_forward
                del module.attn1.old_forward

        return cached_hidden_states

    def unet_forward_with_cached_hidden_states(self, z_all, t, prompt_embd,
                                               cached_pos_hiddens: Optional[List[torch.Tensor]] = None,
                                               cached_neg_hiddens: Optional[List[torch.Tensor]] = None,
                                               pos_weights=(0.8, 0.8), neg_weights=(0.5, 0.5),
                                               batch: Optional[int] = None):
        if batch is None:
            return self.unet(z_all, t, encoder_hidden_states=prompt_embd)

        local_pos_weights = torch.linspace(*pos_weights, steps=len(self.unet.down_blocks) + 1)[:-1].tolist()
        local_neg_weights = torch.linspace(*neg_weights, steps=len(self.unet.down_blocks) + 1)[:-1].tolist()

        for block, pos_weight, neg_weight in zip(self.unet.down_blocks + [self.unet.mid_block] + self.unet.up_blocks,
                                                 local_pos_weights + [pos_weights[1]] + local_pos_weights[::-1],
                                                 local_neg_weights + [neg_weights[1]] + local_neg_weights[::-1], ):
            for module in block.modules():
                if isinstance(module, BasicTransformerBlock):
                    def new_forward(self, hidden_states, pos_weight=pos_weight, neg_weight=neg_weight, **kwargs, ):
                        return attn_with_weights(self, hidden_states, weights=pos_weight, batch=batch)

                    module.attn1.old_forward = module.attn1.forward
                    module.attn1.forward = new_forward.__get__(module.attn1)

        out = self.unet(z_all, t, encoder_hidden_states=prompt_embd)

        # restore original forward pass
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = module.attn1.old_forward
                del module.attn1.old_forward

        return out

    @torch.no_grad()
    def generate(self, prompt: Union[str, List[str]] = "a photo of an astronaut riding a horse on mars",
                 negative_prompt: Union[str, List[str]] = "", liked: List[Image.Image] = [],
                 disliked: List[Image.Image] = [], seed: int = 42, n_images: int = 1, guidance_scale: float = 8.0,
                 denoising_steps: int = 20, feedback_start: float = 0.33, feedback_end: float = 0.66,
                 min_weight: float = 0.1, max_weight: float = 1.0, neg_scale: float = 0.5,
                 pos_bottleneck_scale: float = 1.0, neg_bottleneck_scale: float = 1.0, size: int = 2048,
                 warmup_power: float = 1):
        """
        Generate a trajectory of images with binary feedback.
        The feedback can be given as a list of liked and disliked images.
        """
        if seed is not None:
            torch.manual_seed(seed)

        z = torch.randn(n_images, 4, size // 8, size // 8, device=self.device, dtype=self.dtype)

        if liked and len(liked) > 0:
            pos_images = [self.image_to_tensor(img, size) for img in liked]
            pos_images = torch.stack(pos_images).to(self.device, dtype=self.dtype)
            pos_latents = (self.vae.config.scaling_factor * self.vae.encode(pos_images).latent_dist.sample())
        else:
            pos_latents = torch.tensor([], device=self.device, dtype=self.dtype)

        if disliked and len(disliked) > 0:
            neg_images = [self.image_to_tensor(img, size) for img in disliked]
            neg_images = torch.stack(neg_images).to(self.device, dtype=self.dtype)
            neg_latents = (self.vae.config.scaling_factor * self.vae.encode(neg_images).latent_dist.sample())
        else:
            neg_latents = torch.tensor([], device=self.device, dtype=self.dtype)

        if isinstance(prompt, str):
            prompt = [prompt] * n_images
        else:
            assert len(prompt) == n_images
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * n_images
        else:
            assert len(negative_prompt) == n_images

        (cond_prompt_embs, uncond_prompt_embs, null_prompt_emb,) = self.initialize_prompts(
            prompt + negative_prompt + [""]).split([n_images, n_images, 1])
        batched_prompt_embd = torch.cat([cond_prompt_embs, uncond_prompt_embs], dim=0)

        self.scheduler.set_timesteps(denoising_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        z = z * self.scheduler.init_noise_sigma

        num_warmup_steps = len(timesteps) - denoising_steps * self.scheduler.order

        ref_start_idx = round(len(timesteps) * feedback_start)
        ref_end_idx = round(len(timesteps) * feedback_end)

        with tqdm(total=denoising_steps) as pbar:
            for i, t in enumerate(timesteps):
                if hasattr(self.scheduler, "sigma_t"):
                    sigma = self.scheduler.sigma_t[t]
                elif hasattr(self.scheduler, "sigmas"):
                    sigma = self.scheduler.sigmas[i]
                else:
                    sigma = 0
                alpha_hat = 1 / (sigma ** 2 + 1)

                z_single = self.scheduler.scale_model_input(z, t)

                do_cfg = i >= ref_start_idx and i <= ref_end_idx
                if do_cfg:
                    z_all = torch.cat([z_single] * 2, dim=0)
                    z_ref = torch.cat([pos_latents, neg_latents], dim=0)
                    scale = (ref_end_idx - i) / (ref_end_idx - ref_start_idx)
                    weight = (max_weight - min_weight) * scale ** warmup_power + min_weight
                    prompt_embd = batched_prompt_embd
                    pos_ws = (weight, weight * pos_bottleneck_scale)
                    neg_ws = (weight * neg_scale, weight * neg_scale * neg_bottleneck_scale)
                else:
                    z_all = z_single
                    weight = 0
                    prompt_embd = cond_prompt_embs
                    pos_ws = None
                    neg_ws = None

                do_fabric = do_cfg and z_ref.size(0) > 0 and weight > 0
                if do_fabric:
                    batch = z_all.size(0)
                    noise = torch.randn_like(z_ref)
                    z_ref_noised = (alpha_hat ** 0.5 * z_ref + (1 - alpha_hat) ** 0.5 * noise)

                    ref_prompt_embd = torch.cat([null_prompt_emb] * (pos_latents.size(0) + neg_latents.size(0)), dim=0)

                    z_all = torch.cat([z_all, z_ref_noised], dim=0)
                    prompt_embd = torch.cat([prompt_embd, ref_prompt_embd], dim=0)
                else:
                    batch = None

                unet_out = self.unet_forward_with_cached_hidden_states(z_all, t, prompt_embd=prompt_embd,  #
                                                                       pos_weights=pos_ws, neg_weights=neg_ws,
                                                                       batch=batch).sample
                if do_fabric:
                    unet_out = unet_out[:batch]
                if do_cfg:
                    noise_cond, noise_uncond = unet_out.chunk(2)
                    guidance = noise_cond - noise_uncond
                    noise_pred = noise_uncond + guidance_scale * guidance
                else:
                    noise_pred = unet_out
                z = self.scheduler.step(noise_pred, t, z).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    pbar.update()

        y = self.pipeline.decode_latents(z)
        imgs = self.pipeline.numpy_to_pil(y)

        return imgs

    @staticmethod
    def image_to_tensor(image: Union[str, Image.Image], size: int):
        """
        Convert a PIL image to a torch tensor.
        """
        if isinstance(image, str):
            image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((size, size))
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)
