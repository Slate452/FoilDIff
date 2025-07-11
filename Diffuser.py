
#usr/bin/python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Diffuser():
    def __init__(self, steps, device):
        self.device = device
        self.steps = steps
        self.betas = torch.tensor([])
        self.beta_source = torch.tensor([])
        self.alphas = 1-self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        self.one_minus_alphas_bar = 1 - self.alphas_bar
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(self.one_minus_alphas_bar)

    def forward_diffusion(self, x0, t, noise):
        xt = self.sqrt_alphas_bar[t]*x0+ self.sqrt_one_minus_alphas_bar[t]*noise
        return xt
    def sample_from_noise(self, model, condition, show_progress=True, ddim=False, skip_steps= 2):
        with torch.no_grad():
            x_t = torch.randn_like(condition)
            t_now = torch.tensor([self.steps], device=self.device).repeat(x_t.shape[0])
            t_pre = t_now - (skip_steps if ddim else 1)  # Default to 1 step for DDPM
            
            if show_progress:
                if ddim:
                    p_bar = tqdm(range(0, self.steps, skip_steps))  # Skip steps for DDIM
                else:
                    p_bar = tqdm(range(self.steps))  # Standard range for DDPM
            else:
                if ddim:
                    p_bar = range(0, self.steps, skip_steps)
                else:
                    p_bar = range(self.steps)
            
            for t in p_bar:
                predicted_noise = model(x_t, t_now, condition)
                
                if ddim:
                    x_t, x_0 = self.DDIM_sample_step(x_t, t_now, t_pre, predicted_noise)

                    # Handle final steps for DDIM
                    if t == p_bar[-1]:
                        x_t = x_0
    
                else:
                    x_t = self.DDPM_sample_step(x_t, t_now, t_pre, predicted_noise)
                
                t_now = t_pre
                t_pre = t_pre - (skip_steps if ddim else 1)
            
            return x_t

    def DDPM_sample_step(self, x_t, t, t_pre, noise):
        coef1 = 1/self.sqrt_alphas[t]
        coef2 = self.betas[t]/self.sqrt_one_minus_alphas_bar[t]
        sig = torch.sqrt(self.betas[t])*self.sqrt_one_minus_alphas_bar[t_pre]/self.sqrt_one_minus_alphas_bar[t]
        return coef1*(x_t-coef2*noise)+sig*torch.randn_like(x_t)

    def DDIM_sample_step(self, x_t,t, t_pre, noise):
        coef1 = self.sqrt_alphas_bar[t_pre]
        coef2 = self.sqrt_one_minus_alphas_bar[t]
        coef3 = 1/self.sqrt_alphas_bar[t]
        #sig = stochacity * ( torch.sqrt(self.one_minus_alphas[t_pre]/self.one_minus_alphas[t]) *  torch.sqrt(self.one_minus_alphas[t]/self.alphas[t_pre]))
        #sig_sqr = torch.square(sig)
        coef4 = self.sqrt_one_minus_alphas_bar[t_pre] #+sig_sqr)
        x_0_pred = coef3 * (x_t-coef2*noise)
        x_t_dir = (coef4*noise)
        x_t_pre = coef1*x_0_pred + x_t_dir  #+ sig*torch.randn_like(x_t)
        return  x_t_pre, x_0_pred

    def show_paras(self):
        plt.plot(self.betas[:,0,0,0].cpu(),label=r'$\beta$')
        plt.plot(self.alphas_bar[:,0,0,0].cpu(),label=r'$\bar{\alpha}$')
        plt.legend()
        plt.xlabel('$t$')
        plt.show()
        
    def change_device(self, device):
        self.device = device
        self._generate_parameters_from_beta()

    def generate_parameters_from_beta(self):
        self._generate_parameters_from_beta()
        #print('The sqrt_alphas_bar at the last step is {} , be careful if this value is not close to 0!'.format(
        #    self.sqrt_alphas_bar[-1].item()))

    def _generate_parameters_from_beta(self):
        self.betas = torch.cat(
            (torch.tensor([0]), self.beta_source), dim=0) 
        self.betas = self.betas.view(self.steps+1, 1, 1, 1)
        self.betas = self.betas.to(self.device)

        self.alphas = 1-self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        self.one_minus_alphas_bar = 1 - self.alphas_bar
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(self.one_minus_alphas_bar)


class LinearParamsDiffuser(Diffuser):

    def __init__(self, steps, beta_min, beta_max, device):
        super().__init__(steps, device)
        self.name = "LinearParamsDiffuser"
        self.beta_source = torch.linspace(
            0, 1, self.steps)*(beta_max-beta_min)+beta_min
        self.generate_parameters_from_beta()


class sigParamsDiffuser(Diffuser):

    def __init__(self, steps, beta_min, beta_max, device):
        super().__init__(steps, device)
        self.name = "sigParamsDiffuser"
        self.beta_source = torch.sigmoid(
            torch.linspace(-6, 6, self.steps))*(beta_max-beta_min)+beta_min
        self.generate_parameters_from_beta()

class Cos2ParamsDiffuser(Diffuser):

    def __init__(self, steps, device):
        super().__init__(steps, device)
        self.name = "Cos2ParamsDiffuser"
        s = 0.008
        tlist = torch.arange(1, self.steps+1, 1)
        temp1 = torch.cos((tlist/self.steps+s)/(1+s)*np.pi/2)
        temp1 = temp1*temp1
        temp2 = np.cos(((tlist-1)/self.steps+s)/(1+s)*np.pi/2)
        temp2 = temp2*temp2
        self.beta_source = 1-(temp1/temp2)
        self.beta_source[self.beta_source > 0.999] = 0.999
        self.generate_parameters_from_beta()
