from vendors.CFGpp.latent_diffusion import *

def tangent_plane_projection(main_point, uncond_points, guided_update):
    """
    Projects guided update onto the plane defined by 3 unconditional points.
    """
    p0, p1, p2 = uncond_points
    V = torch.stack([p1 - p0, p2 - p0], dim=0)  # basis vectors of the plane

    # Compute projection matrix safely in float32
    VtV = (V @ V.T).to(torch.float32)
    try:
        invVtV = torch.inverse(VtV)
    except RuntimeError:
        invVtV = torch.pinverse(VtV)

    # Compute projection
    P = V.T @ invVtV @ V
    proj_update = guided_update - (guided_update - (P @ guided_update.T).T)

    return proj_update


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
        Parallel stochastic DDIM with projection of conditional onto plane formed by 3 unconditional trajectories.
        Returns 4 images: [uncond0, uncond1, uncond2, cond]
        """
        device = device or self.device

        # embeddings
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # One shared initial latent z_T
        zT = self.initialize_latent(method='random', latent_dim=latent_dim).to(device)
        # We'll copy it into 3 unconditional + 1 conditional (4 trajectories total)
        zt = torch.cat([zT.clone() for _ in range(4)], dim=0)  # shape (4, C, H, W)

        # We'll keep indices: 0..2 => unconditional, 3 => conditional
        n_uncond = 3
        cond_idx = 3

        pbar = tqdm(self.scheduler.timesteps, desc="SD-parallel")
        total_timesteps = len(self.scheduler.timesteps)

        # ensure scheduler indexing consistent
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            # compute sigma_t used for stochastic DDIM (eta-controlled)
            # avoid division by zero by clamping ratios if needed
            # sigma_t formula from DDIM (eta * ...)
            # If at == at_prev (shouldn't happen normally) clamp a bit
            ratio = (1 - at_prev) / (1 - at + 1e-12)
            inner = 1 - (at / (at_prev + 1e-12))
            sigma_t = eta * (ratio ** 0.5) * (inner ** 0.5)

            # For each trajectory, predict noise.
            # We'll run the UNet in batch: for uncond we pass c=None (so unet gets uc), for cond we pass both
            # Build encoder_hidden_states for batch call:
            # For uncond entries (0..2) we want encoder_hidden_states = uc; for cond (3) we will pass concatenation later
            # But predict_noise already supports batching when uc and c are both provided: it duplicates z_in and t_in etc.
            # We'll call predict_noise individually (clear and simple).
            noise_uc_list = []
            noise_c_list = []
            noise_pred_list = []

            # sample eps for stochastic step (different for each trajectory)
            eps_list = [torch.randn_like(zt[0]) for _ in range(4)]

            for i in range(4):
                if i < n_uncond:
                    # unconditional: c=None, use uc only (predict_noise returns noise_uc, noise_c where noise_c==noise_uc)
                    noise_uc, noise_c = self.predict_noise(zt[i].unsqueeze(0), t, uc, None)
                    noise_uc = noise_uc.squeeze(0)
                    noise_c = noise_c.squeeze(0)
                    noise_pred = noise_uc  # no guidance
                else:
                    # conditional: send both uc and c then apply CFG
                    noise_uc, noise_c = self.predict_noise(zt[i].unsqueeze(0), t, uc, c)
                    noise_uc = noise_uc.squeeze(0)
                    noise_c = noise_c.squeeze(0)
                    noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

                noise_uc_list.append(noise_uc)
                noise_c_list.append(noise_c)
                noise_pred_list.append(noise_pred)

            # compute z0t for each trajectory using same formula as before:
            # z0t = (zt - sqrt(1-at) * noise_pred) / sqrt(at)
            sqrt1_at = (1 - at).sqrt()
            sqrt_at = at.sqrt()
            z0t_list = []
            for i in range(4):
                z0t = (zt[i] - sqrt1_at * noise_pred_list[i]) / (sqrt_at + 1e-12)
                z0t_list.append(z0t)

            # Now compute the affine plane from the three unconditional z0t's.
            # We'll use p0 = mean of three unconds as the plane origin, and v1,v2 as two basis vectors:
            z0_unconds = torch.stack([z0t_list[i] for i in range(n_uncond)], dim=0)  # (3, C, H, W)
            p0 = z0_unconds.mean(dim=0)  # origin
            # Choose two non-collinear directions:
            v1 = (z0_unconds[1] - z0_unconds[0]).flatten()
            v2 = (z0_unconds[2] - z0_unconds[0]).flatten()

            # build V matrix of shape (D, 2)
            D = v1.numel()
            V = torch.stack([v1, v2], dim=1)  # (D,2)

            # compute projection operator P = V (V^T V + eps I)^{-1} V^T
            # We'll do small 2x2 inversion for stability
            VtV = V.t().matmul(V)  # (2,2)
            VtV = VtV + diag_eps * torch.eye(2, device=VtV.device, dtype=VtV.dtype)
            try:
                invVtV = torch.inverse(VtV)
            except Exception:
                invVtV = torch.pinverse(VtV)  # fallback

            P = V.matmul(invVtV).matmul(V.t())  # (D,D) projection matrix onto span{v1,v2}

            # For memory reasons we won't materialize full (D,D) when D huge if avoidable.
            # But for clarity and correctness here we use P on flattened vectors (D size). If D is too large, compute using
            # coefficients approach: coeffs = inv(VtV) @ V.T @ y, etc. We'll implement the coefficients route to avoid D x D mat.
            # So compute projection of a vector y flatten -> p = V @ (invVtV @ (V.T @ y))
            def project_flat(y_flat):
                # y_flat shape (D,)
                vt_y = V.t().matmul(y_flat)            # (2,)
                coeffs = invVtV.matmul(vt_y)           # (2,)
                proj = V.matmul(coeffs)               # (D,)
                return proj

            # Decide whether to project z0_cond or guidance:
            cond_z0_flat = z0t_list[cond_idx].flatten()
            if project_mode == "z0":
                # project z0_cond onto affine plane with origin p0
                y = (cond_z0_flat - p0.flatten())  # (D,)
                proj_flat = project_flat(y)        # (D,)
                z0_cond_proj_flat = p0.flatten() + proj_flat
                z0_cond_proj = z0_cond_proj_flat.view_as(z0t_list[0])
                # recompute noise_pred that would give z0_cond_proj:
                # noise_pred = (zt - sqrt(at) * z0) / sqrt(1 - at)
                noise_pred_cond_proj = (zt[cond_idx] - sqrt_at * z0_cond_proj) / (sqrt1_at + 1e-12)

                # choose to use projected noise_pred for update of conditional trajectory
                chosen_noise_pred_cond = noise_pred_cond_proj
            elif project_mode == "guidance":
                # project guidance vector g = (noise_c - noise_uc)
                g = (noise_c_list[cond_idx] - noise_uc_list[cond_idx]).flatten()  # (D,)
                g_proj = project_flat(g)
                g_proj = g_proj.view_as(g).view_as(noise_uc_list[cond_idx])

                # reconstruct noise_pred using projected guidance:
                # noise_pred = noise_uc + cfg * g_proj
                chosen_noise_pred_cond = noise_uc_list[cond_idx] + cfg_guidance * g_proj
            else:
                raise ValueError("project_mode must be 'z0' or 'guidance'")

            # Now step each trajectory to next timestep using stochastic DDIM update:
            zt_next = []
            sqrt_at_prev = (at_prev).sqrt()
            sqrt_1_at_prev = (1 - at_prev).sqrt()

            # For unconditional trajectories we use their own noise_pred (which equals noise_uc_list[i]) and eps_i
            for i in range(4):
                if i < n_uncond:
                    noise_pred_i = noise_pred_list[i]  # equals noise_uc
                else:
                    # conditional: use chosen_noise_pred_cond (projected)
                    noise_pred_i = chosen_noise_pred_cond

                # compute the deterministic part:
                z0_i = z0t_list[i]
                # stochastic DDIM update formula (general)
                # z_{t-1} = sqrt(alpha_{t-1}) * z0 + sqrt(1 - alpha_{t-1} - sigma_t^2) * noise_pred + sigma_t * eps
                # compute the middle sqrt term safely (clamp)
                middle_sq = (1 - at_prev - sigma_t ** 2).clamp(min=0.0)
                middle = middle_sq.sqrt()

                zt_i_next = sqrt_at_prev * z0_i + middle * noise_pred_i + sigma_t * eps_list[i]
                zt_next.append(zt_i_next)

            zt = torch.stack(zt_next, dim=0)  # new stacked latents for next loop

            # optional callback: give user access to intermediate z0's and zt's and decode function
            if callback_fn is not None:
                # detach to prevent large graph
                callback_kwargs = {
                    'z0t_unconds': [zz.detach().cpu() for zz in z0t_list[:n_uncond]],
                    'z0t_cond': z0t_list[cond_idx].detach().cpu(),
                    'zt_unconds': [zz.detach().cpu() for zz in zt[:n_uncond]],
                    'zt_cond': zt[cond_idx].detach().cpu(),
                    'decode': self.decode
                }
                callback_fn(step, t, callback_kwargs)

        # after loop, decode the final z0t for each trajectory
        # We need final z0 for each. For the conditional we may want the last projected z0 (if project_mode == 'z0')
        final_z0s = []
        # recompute final z0s from last zt and last noise_pred (we have zt and noise_pred_list from last iteration not present now)
        # but simpler: compute z0 as (zt - sqrt1_at * noise_pred)/sqrt_at using last t=final step (we can reuse at from last iter)
        # For safety, recompute z0 using the zt currently. Use the last "at" value (we ended with at for last t).
        # For correct final z0, re-run noise prediction on final zt (cheap, single pass).
        # We'll compute each final noise_pred again and then z0.
        final_z0s = []
        for i in range(4):
            if i < n_uncond:
                noise_uc, noise_c = self.predict_noise(zt[i].unsqueeze(0), t, uc, None)
                noise_uc = noise_uc.squeeze(0)
                noise_pred = noise_uc
            else:
                noise_uc, noise_c = self.predict_noise(zt[i].unsqueeze(0), t, uc, c)
                noise_uc = noise_uc.squeeze(0); noise_c = noise_c.squeeze(0)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0_final = (zt[i] - (1 - at).sqrt() * noise_pred) / (at.sqrt() + 1e-12)
            final_z0s.append(z0_final)

        # decode all final z0s to images
        imgs = [ (self.decode(z0).float() / 2 + 0.5).clamp(0,1).detach().cpu() for z0 in final_z0s ]
        # return 4 images (tensor per image)
        return imgs[3]  # list of 4 tensors, ordering: uncond0, uncond1, uncond2, cond

    