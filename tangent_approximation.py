from vendors.CFGpp.latent_diffusion import *


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
            eta: float = 0.7,
            project_mode: str = "z0",  # "z0" or "guidance"
            callback_fn=None,
            latent_dim=(1,4,64,64),
            device=None,
            diag_eps: float = 1e-6,
            **kwargs):
        device = device or self.device

        # embeddings
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # One shared initial latent z_T
        zT = self.initialize_latent(method='random', latent_dim=latent_dim).to(device)
        # 3 unconditional + 1 conditional
        zt = torch.cat([zT.clone() for _ in range(4)], dim=0)  # (4, C, H, W)

        n_uncond = 3
        cond_idx = 3

        pbar = tqdm(self.scheduler.timesteps, desc="SD-parallel")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            # stochastic DDIM sigma_t
            ratio = (1 - at_prev) / (1 - at + 1e-12)
            inner = 1 - (at / (at_prev + 1e-12))
            sigma_t = eta * (ratio ** 0.5) * (inner ** 0.5)

            # predict noises per trajectory
            noise_uc_list = []
            noise_c_list = []
            noise_pred_list = []
            eps_list = [torch.randn_like(zt[0]) for _ in range(4)]

            for i in range(4):
                z_i = zt[i].unsqueeze(0)
                if i < n_uncond:
                    noise_uc, noise_c = self.predict_noise(z_i, t, uc, None)
                    noise_uc = noise_uc.squeeze(0); noise_c = noise_c.squeeze(0)
                    noise_pred = noise_uc
                else:
                    noise_uc, noise_c = self.predict_noise(z_i, t, uc, c)
                    noise_uc = noise_uc.squeeze(0); noise_c = noise_c.squeeze(0)
                    noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

                noise_uc_list.append(noise_uc)
                noise_c_list.append(noise_c)
                noise_pred_list.append(noise_pred)

            # compute z0t for each trajectory
            sqrt1_at = (1 - at).sqrt()
            sqrt_at = at.sqrt()
            z0t_list = []
            for i in range(4):
                z0t = (zt[i] - sqrt1_at * noise_pred_list[i]) / (sqrt_at + 1e-12)
                z0t_list.append(z0t)

            # build plane from 3 unconditional z0s
            z0_unconds = torch.stack([z0t_list[i] for i in range(n_uncond)], dim=0)  # (3, C, H, W)
            p0 = z0_unconds.mean(dim=0)  # origin

            v1 = (z0_unconds[1] - z0_unconds[0]).flatten()  # (D,)
            v2 = (z0_unconds[2] - z0_unconds[0]).flatten()  # (D,)

            # V: (D, 2)
            V = torch.stack([v1, v2], dim=1)  # dtype likely float16 under autocast

            # Stable small-matrix operations in float32 (autocast disabled)
            with torch.cuda.amp.autocast(enabled=False):
                V_f = V.float()                                 # (D,2) float32
                VtV = V_f.t().matmul(V_f)                       # (2,2) float32
                VtV = VtV + diag_eps * torch.eye(2, device=VtV.device, dtype=VtV.dtype)
                # invert the tiny 2x2 in float32
                try:
                    invVtV = torch.linalg.inv(VtV)
                except Exception:
                    invVtV = torch.pinverse(VtV)

                def project_flat32(y_flat):
                    # y_flat may be float16 or float32; do ops in float32 and return float32 result
                    y_f = y_flat.float()
                    vt_y = V_f.t().matmul(y_f)     # (2,)
                    coeffs = invVtV.matmul(vt_y)   # (2,)
                    proj = V_f.matmul(coeffs)      # (D,)
                    return proj

                # compute projection results in float32
                cond_z0_flat = z0t_list[cond_idx].flatten()
                if project_mode == "z0":
                    y = (cond_z0_flat - p0.flatten()).to(cond_z0_flat.device)
                    proj32 = project_flat32(y)                       # (D,) float32
                    z0_cond_proj_flat32 = p0.flatten().float() + proj32
                    # map back to pipeline dtype (float16) and shape
                    z0_cond_proj = z0_cond_proj_flat32.view_as(z0t_list[0]).to(z0t_list[0].dtype)
                    # recompute noise_pred that would correspond to this projected z0
                    noise_pred_cond_proj = (zt[cond_idx] - sqrt_at * z0_cond_proj) / (sqrt1_at + 1e-12)
                    chosen_noise_pred_cond = noise_pred_cond_proj
                elif project_mode == "guidance":
                    g = (noise_c_list[cond_idx] - noise_uc_list[cond_idx]).flatten()
                    g_proj32 = project_flat32(g)                     # (D,) float32
                    g_proj = g_proj32.view_as(g).to(g.dtype)
                    chosen_noise_pred_cond = noise_uc_list[cond_idx] + cfg_guidance * g_proj
                else:
                    raise ValueError("project_mode must be 'z0' or 'guidance'")

            # step each trajectory to next timestep (stochastic DDIM)
            zt_next = []
            sqrt_at_prev = at_prev.sqrt()
            for i in range(4):
                if i < n_uncond:
                    noise_pred_i = noise_pred_list[i]
                else:
                    noise_pred_i = chosen_noise_pred_cond

                z0_i = z0t_list[i]
                middle_sq = (1 - at_prev - sigma_t ** 2).clamp(min=0.0)
                middle = middle_sq.sqrt()
                zt_i_next = sqrt_at_prev * z0_i + middle * noise_pred_i + sigma_t * eps_list[i]
                zt_next.append(zt_i_next)

            zt = torch.stack(zt_next, dim=0)

            if callback_fn is not None:
                callback_kwargs = {
                    'z0t_unconds': [zz.detach().cpu() for zz in z0t_list[:n_uncond]],
                    'z0t_cond': z0t_list[cond_idx].detach().cpu(),
                    'zt_unconds': [zz.detach().cpu() for zz in zt[:n_uncond]],
                    'zt_cond': zt[cond_idx].detach().cpu(),
                    'decode': self.decode
                }
                callback_fn(step, t, callback_kwargs)

        # final z0 recomputation & decode
        final_z0s = []
        # use last timestep 't' and 'at' from loop end
        for i in range(4):
            if i < n_uncond:
                noise_uc, _ = self.predict_noise(zt[i].unsqueeze(0), t, uc, None)
                noise_pred = noise_uc.squeeze(0)
            else:
                noise_uc, noise_c = self.predict_noise(zt[i].unsqueeze(0), t, uc, c)
                noise_uc = noise_uc.squeeze(0); noise_c = noise_c.squeeze(0)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0_final = (zt[i] - (1 - at).sqrt() * noise_pred) / (at.sqrt() + 1e-12)
            final_z0s.append(z0_final)

        imgs = [ (self.decode(z0).float() / 2 + 0.5).clamp(0,1).detach().cpu() for z0 in final_z0s ]

        # return conditional image (as in your previous version)
        return imgs[3]

    