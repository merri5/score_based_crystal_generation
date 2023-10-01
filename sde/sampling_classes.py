from sde.sde_lib import VESDE, VPSDE, subVPSDE
import torch

class Predictor():
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, sde_atom_types, sde_frac_coords, sde_lattice, model, probability_flow=False):
    super().__init__()
    self.sde = sde
    self.sde_atom_types = sde_atom_types
    self.sde_frac_coords = sde_frac_coords
    self.sde_lattice = sde_lattice
    self.model = model
    self.probability_flow = probability_flow
    # Compute the reverse SDE/ODE
    self.rsde_atom_types = sde_atom_types.reverse(model, probability_flow)
    self.rsde_frac_coords = sde_frac_coords.reverse(model, probability_flow)
    self.rsde_lattice = sde_lattice.reverse(model, probability_flow)

    # self.score_fn = score_fn

  def update_fn(self, cur_atom_types, cur_frac_coords, cur_lattice, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass

class Corrector():
    """The abstract class for a corrector algorithm."""

    # def __init__(self, sde, sde_atom_types, sde_frac_coords, sde_lattice, model, n_steps):
    def __init__(self, sde, sde_atom_types, sde_frac_coords, sde_lattice, model, n_steps):
        super().__init__()
        self.sde = sde
        self.sde_atom_types = sde_atom_types
        self.sde_frac_coords = sde_frac_coords
        self.sde_lattice = sde_lattice
        self.model = model
        # self.snr = snr
        # self.snr = snr
        self.n_steps = n_steps

    def update_fn(self, cur_atom_types, cur_frac_coords, cur_lattice, batch, 
                  t_atom_types, t_frac_coords, t_lattice):
        """One update of the corrector.

        Args:
        x: A PyTorch tensor representing the current state
        t: A PyTorch tensor representing the current time step.

        Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass



# @register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, sde_atom_types, sde_frac_coords, sde_lattice,
                    model, probability_flow=False):
        super().__init__(sde, sde_atom_types, sde_frac_coords, sde_lattice,
                    model, probability_flow)
    def discretize(self, cur_atom_types, cur_frac_coords, cur_lattice, num_atoms, t, probability_flow):
        f_atom_types, G_atom_types = self.sde_atom_types.discretize(cur_atom_types, t)
        f_frac_coords, G_frac_coords = self.sde_frac_coords.discretize(cur_frac_coords, t)
        f_lattice, G_lattice = self.sde_lattice.discretize(cur_lattice, t)

        G_atom_types = G_atom_types.repeat_interleave(num_atoms)
        G_frac_coords = G_frac_coords.repeat_interleave(num_atoms)

        cur_atom_types_diff, cur_frac_coords_diff, cur_lattice_diff = self.model(cur_atom_types, cur_frac_coords, num_atoms, cur_lattice, t)
        rev_f_atom_types = f_atom_types - G_atom_types[:, None] ** 2 * cur_atom_types_diff * (0.5 if probability_flow else 1.)
        rev_G_atom_types = torch.zeros_like(G_atom_types) if probability_flow else G_atom_types
        rev_f_frac_coords = f_frac_coords - G_frac_coords[:, None] ** 2 * cur_frac_coords_diff * (0.5 if probability_flow else 1.)
        rev_G_frac_coords = torch.zeros_like(G_frac_coords) if probability_flow else G_frac_coords
        rev_f_lattice = f_lattice - G_lattice[:, None] ** 2 * cur_lattice_diff * (0.5 if probability_flow else 1.)
        rev_G_lattice = torch.zeros_like(G_lattice) if probability_flow else G_lattice
        return rev_f_atom_types, rev_G_atom_types, \
                rev_f_frac_coords, rev_G_frac_coords, \
                rev_f_lattice, rev_G_lattice

    # def update_fn(self, cur_atom_types, cur_frac_coords, cur_lattice, num_atoms, t,
    #                 clamp_atom_types_01, clamp_atom_types_m11, wrap_coords):
    def update_fn(self, cur_atom_types, cur_frac_coords, cur_lattice, num_atoms, t,
                    clamp_atom_types_01, clamp_atom_types_eucledian, wrap_coords):
        rev_f_atom_types, rev_G_atom_types, \
        rev_f_frac_coords, rev_G_frac_coords, \
        rev_f_lattice, rev_G_lattice = self.discretize(cur_atom_types, cur_frac_coords, cur_lattice, num_atoms, t, self.probability_flow)


        noise_atom_types = torch.randn_like(cur_atom_types)
        cur_atom_types_mean = cur_atom_types - rev_f_atom_types
        cur_atom_types = cur_atom_types_mean + rev_G_atom_types[:, None] * noise_atom_types

        
        noise_frac_coords = torch.randn_like(cur_frac_coords)
        cur_frac_coords_mean = cur_frac_coords - rev_f_frac_coords
        cur_frac_coords = cur_frac_coords_mean + rev_G_frac_coords[:, None] * noise_frac_coords

        
        noise_lattice = torch.randn_like(cur_lattice)
        cur_lattice_mean = cur_lattice - rev_f_lattice
        cur_lattice = cur_lattice_mean + rev_G_lattice[:, None] * noise_lattice


        if clamp_atom_types_01:
                # cur_mean_atom_types = torch.clamp(cur_mean_atom_types, min=0.0, max=1.0)
                cur_atom_types = torch.clamp(cur_atom_types, min=0.0, max=1.0)
        if clamp_atom_types_eucledian:
                norm = cur_atom_types.norm(dim = -1, keepdim=True)
                # print(cur_atom_types.shape, norm.shape)
                norm[norm < 1] = 1
                cur_atom_types = cur_atom_types / norm #torch.clamp(cur_atom_types, min=-1.0, max=1.0)


        if wrap_coords:
            cur_frac_coords = cur_frac_coords % 1.

        return cur_atom_types, cur_atom_types_mean, \
                cur_frac_coords, cur_frac_coords_mean, \
                cur_lattice, cur_lattice_mean


class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

    We include this corrector only for completeness. It was not directly used in our paper.
    """

    def __init__(self, sde, sde_atom_types, sde_frac_coords, sde_lattice,
                  model, n_steps):
        # super().__init__(sde, sde_atom_types, sde_frac_coords, sde_lattice, model, n_steps)
        #           model, n_steps):
        super().__init__(sde, sde_atom_types, sde_frac_coords, sde_lattice, model, n_steps)
        if not isinstance(sde_atom_types, VPSDE) \
                    and not isinstance(sde_atom_types, VESDE) \
                    and not isinstance(sde_atom_types, subVPSDE):
            raise NotImplementedError(f"SDE class {sde_atom_types.__class__.__name__} not yet supported.")
        if not isinstance(sde_frac_coords, VPSDE) \
                    and not isinstance(sde_frac_coords, VESDE) \
                    and not isinstance(sde_frac_coords, subVPSDE):
                    raise NotImplementedError(f"SDE class {sde_frac_coords.__class__.__name__} not yet supported.")
        if not isinstance(sde_lattice, VPSDE) \
                    and not isinstance(sde_lattice, VESDE) \
                    and not isinstance(sde_lattice, subVPSDE):
                    raise NotImplementedError(f"SDE class {sde_lattice.__class__.__name__} not yet supported.")


    def get_labels(self, sde, x, t):
        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            if sde.continuous or isinstance(sde, subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
            return labels

        elif isinstance(sde, VESDE):
            if sde.continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()
            return labels

        else:
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        

    def update_fn(self, cur_atom_types, cur_frac_coords, cur_lattice, 
                  #sigma_atom_types, sigma_frac_coords, sigma_lattice, 
                  num_atoms, t, 
                  target_snr_atom_types, target_snr_frac_coords, target_snr_lattice,
                  clamp_atom_types_01, clamp_atom_types_eucledian, wrap_coords):

        # target_snr = self.snr
        t = (t * (self.sde.num_scales - 1) / self.sde.T).long()

        if isinstance(self.sde_atom_types, VPSDE) or isinstance(self.sde_atom_types, subVPSDE):
            alpha_atom_types = self.sde_atom_types.alphas.to(t.device)[t].repeat_interleave(num_atoms)
        else:
            alpha_atom_types = torch.ones_like(t).repeat_interleave(num_atoms)
        
        if isinstance(self.sde_frac_coords, VPSDE) or isinstance(self.sde_frac_coords, subVPSDE):
            alpha_frac_coords = self.sde_frac_coords.alphas.to(t.device)[t].repeat_interleave(num_atoms)
        else:
            alpha_frac_coords = torch.ones_like(t).repeat_interleave(num_atoms)
        
        if isinstance(self.sde_lattice, VPSDE) or isinstance(self.sde_lattice, subVPSDE):
            alpha_lattice = self.sde_lattice.alphas.to(t.device)[t]
        else:
            alpha_lattice = torch.ones_like(t)

    
        for i in range(self.n_steps):
            
            
            pred_atom_types_diff, pred_frac_coords_diff, pred_lattice_diff = self.model(
                            cur_atom_types, cur_frac_coords, num_atoms, cur_lattice,
                            t)
            
            noise_atom_types = torch.randn_like(cur_atom_types) 
            noise_frac_coords = torch.randn_like(cur_frac_coords) 
            noise_lattice = torch.randn_like(cur_lattice)

            
            self.sde_atom_types.discrete_sigmas = self.sde_atom_types.discrete_sigmas.to(pred_atom_types_diff.device)
            self.sde_frac_coords.discrete_sigmas = self.sde_frac_coords.discrete_sigmas.to(pred_frac_coords_diff.device)
            self.sde_lattice.discrete_sigmas = self.sde_lattice.discrete_sigmas.to(pred_lattice_diff.device)

            labels_atom_types = self.get_labels(self.sde_atom_types, pred_atom_types_diff, t)
            labels_frac_coords = self.get_labels(self.sde_frac_coords, pred_frac_coords_diff, t)
            labels_lattice = self.get_labels(self.sde_lattice, pred_lattice_diff, t)
          

            if isinstance(self.sde_atom_types, VPSDE) or isinstance(self.sde_atom_types, subVPSDE):
                pass #pred_atom_types_diff = -pred_atom_types_diff / std_atom_types[:, None, None, None]
            elif isinstance(self.sde_atom_types, VESDE):
                if self.sde_atom_types.continuous:
                    used_sigmas_atom_types = labels_atom_types.repeat_interleave(num_atoms, dim=0)
                else:
                    used_sigmas_atom_types = self.sde_atom_types.discrete_sigmas[labels_atom_types].repeat_interleave(num_atoms, dim=0)
                pred_atom_types_diff = pred_atom_types_diff / used_sigmas_atom_types[:, None]
            else:
                raise NotImplementedError(f"SDE class {self.sde_atom_types.__class__.__name__} not yet supported.")
            
            if isinstance(self.sde_frac_coords, VPSDE) or isinstance(self.sde_frac_coords, subVPSDE):
                pass #pred_frac_coords_diff = -pred_frac_coords_diff / std_frac_coords[:, None, None, None]
            elif isinstance(self.sde_frac_coords, VESDE):
                if self.sde_frac_coords.continuous:
                    used_sigmas_frac_coords = labels_frac_coords.repeat_interleave(num_atoms, dim=0)
                else:
                    used_sigmas_frac_coords = self.sde_frac_coords.discrete_sigmas[labels_frac_coords].repeat_interleave(num_atoms, dim=0)                
                    
                pred_frac_coords_diff = pred_frac_coords_diff / used_sigmas_frac_coords[:, None]
            else:
                raise NotImplementedError(f"SDE class {self.sde_lattice.__class__.__name__} not yet supported.")

            if isinstance(self.sde_lattice, VPSDE) or isinstance(self.sde_lattice, subVPSDE):
                pass #pred_lattice_diff = -pred_lattice_diff / std_lattice[:, None, None, None]
            elif isinstance(self.sde_lattice, VESDE):
                if self.sde_lattice.continuous:
                    used_sigmas_lattice = labels_lattice
                else:
                    used_sigmas_lattice = self.sde_lattice.discrete_sigmas[labels_lattice] #.repeat_interleave(num_atoms, dim=0)                
                    
            
                pred_lattice_diff = pred_lattice_diff / used_sigmas_lattice[:,None]
            else:
                raise NotImplementedError(f"SDE class {self.sde_lattice.__class__.__name__} not yet supported.")

            step_size_atom_types = target_snr_atom_types * self.sde_atom_types.marginal_prob(pred_atom_types_diff, t)[1].repeat_interleave(num_atoms) ** 2 * 2 * alpha_atom_types
            step_size_frac_coords = target_snr_frac_coords * self.sde_frac_coords.marginal_prob(pred_frac_coords_diff, t)[1].repeat_interleave(num_atoms) ** 2 * 2 * alpha_frac_coords
            step_size_lattice = target_snr_lattice * self.sde_lattice.marginal_prob(pred_lattice_diff, t)[1] ** 2 * 2 * alpha_lattice

            cur_mean_atom_types = cur_atom_types + step_size_atom_types[:, None] * pred_atom_types_diff
            cur_atom_types = cur_mean_atom_types + torch.sqrt(step_size_atom_types * 2)[:, None] * noise_atom_types
            cur_mean_frac_coords = cur_frac_coords + step_size_frac_coords[:, None] * pred_frac_coords_diff
            cur_frac_coords = cur_mean_frac_coords + torch.sqrt(step_size_frac_coords * 2)[:, None] * noise_frac_coords
            cur_mean_lattice = cur_lattice + step_size_lattice[:, None] * pred_lattice_diff
            cur_lattice = cur_mean_lattice + torch.sqrt(step_size_lattice * 2)[:, None] * noise_lattice

            # cur_mean_atom_types = cur_atom_types + step_size_atom_types[:, None] * pred_atom_types_diff
            # cur_atom_types = cur_mean_atom_types + noise_atom_types * torch.sqrt(step_size_atom_types * 2)[:, None]
            # cur_mean_frac_coords = cur_frac_coords + step_size_frac_coords[:, None] * pred_frac_coords_diff
            # cur_frac_coords = cur_mean_frac_coords + noise_frac_coords * torch.sqrt(step_size_frac_coords * 2)[:, None]
            # cur_mean_lattice = cur_lattice + step_size_lattice[:, None] * pred_lattice_diff
            # cur_lattice = cur_mean_lattice + noise_lattice * torch.sqrt(step_size_lattice * 2)[:, None]

            if i%50 == 0: 
                print(cur_lattice[0,:])
                print(cur_frac_coords[0,:])
                print(cur_atom_types[0,:])
                print('\n') 
            if clamp_atom_types_01:
                cur_atom_types = torch.clamp(cur_atom_types, min=0.0, max=1.0)
            if clamp_atom_types_eucledian:
                norm = cur_atom_types.norm(dim = -1, keepdim=True)
                norm[norm < 1] = 1
                cur_atom_types = cur_atom_types / norm 


            if wrap_coords:
                cur_frac_coords = cur_frac_coords % 1.

        return cur_atom_types, cur_mean_atom_types,\
                cur_frac_coords, cur_mean_frac_coords,\
                cur_lattice, cur_mean_lattice
