from vendors.CFGpp.latent_diffusion import *

def tangent_plane_projection(main_point, uncond_points, guided_update):
    """
    Projects guided update onto the plane defined by 3 unconditional points.
    Args:
        main_point: torch.Tensor of shape [B, C, H, W] (the main latent)
        uncond_points: list of 3 torch.Tensor latents [p0, p1, p2]
        guided_update: torch.Tensor, the conditional (CFG) update
    Returns:
        projected_update: guided update projected onto tangent plane
    """
    p0, p1, p2 = uncond_points
    v1 = (p1 - p0).view(-1)
    v2 = (p2 - p0).view(-1)
    
    # Orthonormalize
    b1 = v1 / (v1.norm() + 1e-8)
    v2 = v2 - (v2 @ b1) * b1
    b2 = v2 / (v2.norm() + 1e-8)
    
    # Project the guided update onto the plane
    delta = (main_point - p0).view(-1) + guided_update.view(-1)
    proj = (delta @ b1) * b1 + (delta @ b2) * b2
    projected = p0.view(-1) + proj
    return projected.view_as(main_point)


@register_solver("ddim_cfg_tangent")
class TangentDiffusion(StableDiffusion):
    """
    DDIM solver for SD with CFG++ Tangential.
    Useful for text-to-image generation
    """
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        super().__init__(solver_config, model_key, device, **kwargs)

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               **kwargs):
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize main latent
        zt = self.initialize_latent().requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="TangentDiffusion")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                # Compute guided and unconditional noise
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                guided_update = cfg_guidance * (noise_c - noise_uc)

                # Sample 3 unconditional points to define tangent plane
                uncond_points = [self.initialize_latent(method="random").detach() for _ in range(3)]

                # Project guided update onto tangent plane
                noise_pred = tangent_plane_projection(zt, uncond_points, guided_update)

            # Tweedie update
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                   'zt': zt.detach(),
                                   'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # Decode final image
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

    