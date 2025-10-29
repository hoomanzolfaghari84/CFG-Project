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
        """
        Stochastic DDIM sampling with sequentialized UNet calls for memory efficiency.
        Returns list of 4 images [uncond0, uncond1, uncond2, cond] as CPU tensors in [0,1].
        """
        device = device or self.device

        # text embeddings
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # initial shared latent z_T
        zT = self.initialize_latent(method='random', latent_dim=latent_dim).to(device)
        # 4 trajectories stacked (0..2 unconditional, 3 conditional)
        zt = torch.cat([zT.clone() for _ in range(4)], dim=0)  # (4, C, H, W)

        n_uncond = 3
        cond_idx = 3

        pbar = tqdm(self.scheduler.timesteps, desc="SD-parallel")

        # main loop
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            # stochastic DDIM sigma_t (eta controls stochasticity)
            ratio = (1 - at_prev) / (1 - at + 1e-12)
            inner = 1 - (at / (at_prev + 1e-12))
            sigma_t = eta * (ratio ** 0.5) * (inner ** 0.5)

            sqrt1_at = (1 - at).sqrt()
            sqrt_at = at.sqrt()
            sqrt_at_prev = (at_prev).sqrt()

            # Prepare container holders
            noise_uc_list = [None] * 4
            noise_c_list = [None] * 4
            noise_pred_list = [None] * 4
            z0t_list = [None] * 4
            eps_list = [torch.randn_like(zt[0]) for _ in range(4)]

            # Sequential UNet calls under autocast (low mem)
            # Unconditionals (i = 0..2): call predict_noise with c=None
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                for i in range(n_uncond):
                    z_i = zt[i].unsqueeze(0)
                    # this returns (noise_uc, noise_c) but with c=None noise_c==noise_uc
                    noise_uc, noise_c = self.predict_noise(z_i, t, uc, None)
                    noise_uc = noise_uc.squeeze(0)
                    noise_c = noise_c.squeeze(0)
                    noise_uc_list[i] = noise_uc
                    noise_c_list[i] = noise_c
                    noise_pred_list[i] = noise_uc          # unconditional prediction (no guidance)
                    # free small temporary memory if needed
                    torch.cuda.empty_cache()

                # Conditional: get both unconditional & conditional predictions in one call
                z_cond = zt[cond_idx].unsqueeze(0)
                noise_uc_cond, noise_c_cond = self.predict_noise(z_cond, t, uc, c)
                noise_uc_cond = noise_uc_cond.squeeze(0)
                noise_c_cond = noise_c_cond.squeeze(0)
                # default conditional noise_pred using CFG
                noise_pred_cond = noise_uc_cond + cfg_guidance * (noise_c_cond - noise_uc_cond)

                noise_uc_list[cond_idx] = noise_uc_cond
                noise_c_list[cond_idx] = noise_c_cond
                noise_pred_list[cond_idx] = noise_pred_cond
                torch.cuda.empty_cache()

            # compute z0t estimates for all trajectories (deterministic estimate from noise_pred)
            for i in range(4):
                z0t = (zt[i] - sqrt1_at * noise_pred_list[i]) / (sqrt_at + 1e-12)
                z0t_list[i] = z0t

            # Build tangent plane from three unconditional z0t's
            z0_unconds = torch.stack([z0t_list[i] for i in range(n_uncond)], dim=0)  # (3, C, H, W)
            p0 = z0_unconds.mean(dim=0)  # plane origin

            v1 = (z0_unconds[1] - z0_unconds[0]).flatten()  # (D,)
            v2 = (z0_unconds[2] - z0_unconds[0]).flatten()  # (D,)
            V = torch.stack([v1, v2], dim=1)  # (D, 2)

            # Project either z0_cond or guidance in robust float32 block
            # project_flat computes projection coefficients by inverting 2x2 V^T V in float32
            with torch.cuda.amp.autocast(enabled=False):
                V_f = V.float()  # (D,2) float32 for stable small-matrix ops
                VtV = V_f.t().matmul(V_f)                       # (2,2)
                VtV = VtV + diag_eps * torch.eye(2, device=VtV.device, dtype=VtV.dtype)
                try:
                    invVtV = torch.linalg.inv(VtV)
                except Exception:
                    invVtV = torch.pinverse(VtV)

                def project_flat32(y_flat):
                    # y_flat -> float32 -> compute coeffs -> project; returns float32 flattened projection
                    y_f = y_flat.float()
                    vt_y = V_f.t().matmul(y_f)        # (2,)
                    coeffs = invVtV.matmul(vt_y)      # (2,)
                    proj = V_f.matmul(coeffs)         # (D,)
                    return proj

                cond_z0_flat = z0t_list[cond_idx].flatten()  # (D,)
                if project_mode == "z0":
                    # project z0_cond onto affine plane with origin p0
                    y = (cond_z0_flat - p0.flatten()).to(cond_z0_flat.device)
                    proj_flat32 = project_flat32(y)          # (D,) float32
                    z0_cond_proj_flat32 = p0.flatten().float() + proj_flat32
                    z0_cond_proj = z0_cond_proj_flat32.view_as(z0t_list[0]).to(z0t_list[0].dtype)
                    # recompute noise_pred that would yield this z0 (back in pipeline dtype)
                    noise_pred_cond_proj = (zt[cond_idx] - sqrt_at * z0_cond_proj) / (sqrt1_at + 1e-12)
                    chosen_noise_pred_cond = noise_pred_cond_proj
                elif project_mode == "guidance":
                    # project guidance g = noise_c - noise_uc
                    g = (noise_c_list[cond_idx] - noise_uc_list[cond_idx]).flatten()  # (D,)
                    g_proj32 = project_flat32(g)                     # (D,) float32
                    g_proj = g_proj32.view_as(g).to(g.dtype)
                    chosen_noise_pred_cond = noise_uc_list[cond_idx] + cfg_guidance * g_proj
                else:
                    raise ValueError("project_mode must be 'z0' or 'guidance'")

            # Now perform stochastic DDIM step for each trajectory using chosen_noise_pred_cond for conditional
            zt_next = []
            for i in range(4):
                if i < n_uncond:
                    noise_pred_i = noise_pred_list[i]
                else:
                    noise_pred_i = chosen_noise_pred_cond

                z0_i = z0t_list[i]
                middle_sq = (1 - at_prev - sigma_t ** 2).clamp(min=0.0)
                middle = middle_sq.sqrt()
                # stochastic update
                zt_i_next = sqrt_at_prev * z0_i + middle * noise_pred_i + sigma_t * eps_list[i]
                zt_next.append(zt_i_next)

            zt = torch.stack(zt_next, dim=0)

            # callback (optional) - provide detached tensors to avoid big graphs
            if callback_fn is not None:
                callback_kwargs = {
                    'z0t_unconds': [zz.detach().cpu() for zz in z0t_list[:n_uncond]],
                    'z0t_cond': z0t_list[cond_idx].detach().cpu(),
                    'zt_unconds': [zz.detach().cpu() for zz in zt[:n_uncond]],
                    'zt_cond': zt[cond_idx].detach().cpu(),
                    'decode': self.decode
                }
                callback_fn(step, t, callback_kwargs)

        # end loop

        # Final recompute of z0 for each trajectory and decode
        final_z0s = []
        # Use last timestep's 't' and 'at' from loop above
        for i in range(4):
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
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

        # return list of 4 images: [uncond0, uncond1, uncond2, cond]
        return imgs


    