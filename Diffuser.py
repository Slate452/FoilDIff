import torch
import torch.nn.functional as F

class Diffuser:
    def __init__(self, timesteps=300, start_beta=0.0001, end_beta=0.02, device="cpu", sample_trajectory_factor=1, beta_type="linear"):
        self.device = device
        self.T = timesteps
        self.skip_step = sample_trajectory_factor
        self.betas = self.beta_schedule(timesteps, start_beta, end_beta, beta_type=beta_type)

        # Pre-calculate terms for closed-form equations
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def beta_schedule(self, timesteps, start, end, beta_type="linear"):
        """Creates a beta schedule."""
        if beta_type == "linear":
            return torch.linspace(start, end, timesteps, device=self.device)
        
        elif beta_type == "cosine":
            steps = torch.linspace(0, 1, timesteps + 1, device=self.device)
            alphas_cumprod = torch.cos(steps * 0.5 * torch.pi) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        
        return torch.linspace(start, end, timesteps, device=self.device)

    def get_index_from_list(self, vals, t, x_shape):
        """Retrieve the value at a specific timestep index `t` and adjusts for batch dimensions."""
        t = t.to(vals.device)
        out = vals.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1))).to(self.device)

    def forward_diffusion(self, x_0, t):
        """Forward diffusion process."""
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def get_loss(self, model, x_0, t):
        """Computes the loss between the predicted noise and actual noise."""
        x_noisy, noise = self.forward_diffusion(x_0, t)
        noise_pred = model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    def ddpm_sample_timestep(self, x, t, model):
        """Samples a single timestep during reverse diffusion."""
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)

        noise = torch.randn_like(x)
        if (t == 0).all():
            return model_mean
        else:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def ddim_sample_timestep(self, x, t, model, eta=0.0):
        """DDIM sampling: deterministic reverse sampling."""
        alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod, t, x.shape)
        alphas_cumprod_t_prev = self.get_index_from_list(self.alphas_cumprod_prev, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        pred_noise = model(x, t)
        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * pred_noise) / torch.sqrt(alphas_cumprod_t)
        mean = torch.sqrt(alphas_cumprod_t_prev) * pred_x0 + torch.sqrt(1. - alphas_cumprod_t_prev) * pred_noise

        if t[0] == 0:
            return pred_x0
        else:
            noise = eta * torch.randn_like(x)
            return mean + noise

    def sample_from_noise(self, model, condition, Tech="ddim"): 
        """Generates a sample from noise using one of the two sampling processes."""
        sampler = self.ddpm_sample_timestep if Tech == "ddpm" else self.ddim_sample_timestep
        batch_size = condition.shape[0] if condition.dim() == 4 else 1  # Handle single (CHW) vs. batch (BCHW)
        condition_shape = condition.shape[-3:]  # Extract (C, H, W)
        starter = torch.randn((batch_size, *condition_shape), device=self.device)  # Noise tensor
        x_t = torch.cat([condition,starter], dim=1)  # Concatenate along channel dimension

        model.to(self.device)

        timesteps = range(self.T - 1, -1, -1) if sampler == self.ddpm_sample_timestep else range(self.T - 1, -1, -self.skip_step)
        
        for i in timesteps:
            t_batch = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_t = sampler(x_t, t_batch, model=model)

        x_t = torch.clamp(x_t, -1.0, 1.0)
        return x_t
