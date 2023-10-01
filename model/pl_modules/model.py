import functools
import pickle
from typing import Any, Dict

import hydra
from matplotlib import pyplot as plt
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from model.common.utils import PROJECT_ROOT
from model.common.data_utils import (
    EPSILON, cart_to_frac_coords, compute_volume, mard, lengths_angles_to_volume, 
    lattice_params_to_matrix_torch, lattice_matrix_to_params_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)
from model.pl_modules.embeddings import MAX_ATOMIC_NUM
from model.pl_modules.embeddings import KHOT_EMBEDDINGS
from model.common.data_utils import chemical_symbols

from sde.sampling_classes import ReverseDiffusionPredictor, AnnealedLangevinDynamics
from sde.sde_lib import VESDE, VPSDE, subVPSDE

import os

def plot_output_dist(atoms, atoms_gt, coords, coords_gt, lattices, lattices_gt,
                     sigma_begin_atom_types, sigma_end_atom_types,
                     sigma_begin_coords, sigma_end_coords,
                     sigma_begin_lattice, sigma_end_lattice,):
    atoms = atoms.to('cpu').detach().numpy()
    coords = coords.to('cpu').detach().numpy()
    lattices = lattices.to('cpu').detach().numpy()
    atoms_gt = atoms_gt.to('cpu').detach().numpy()
    coords_gt = coords_gt.to('cpu').detach().numpy()
    lattices_gt = lattices_gt.to('cpu').detach().numpy()

    fig = plt.figure(figsize=(15,12),layout="constrained")
    
    ax = fig.subplot_mosaic("""AAATTT
                            BBCCDD
                            EFGHIJ""")
    
    fig.suptitle(f'Atom types noise N(0,{str(int(sigma_end_atom_types))})-N(0,{str(int(sigma_begin_atom_types))})\n'\
                 f'Coords noise N(0,{str(int(sigma_end_coords))})-N(0,{str(int(sigma_begin_coords))})\n'\
                 f'Lattice noise N(0,{str(int(sigma_end_lattice))})-N(0,{str(int(sigma_begin_lattice))})')

    ax['A'].hist([atoms[:,0], atoms_gt[:,0]], bins = 100, alpha =0.5, label = ["Noisy atoms dim 1", "GT atoms dim 1"], density=True)#, histtype="step")
    ax['T'].hist([atoms[:,1], atoms_gt[:,1]], bins = 100, alpha =0.5, label = ["Noisy atoms dim 2", "GT atoms dim 2"], density=True)#, histtype="step")
    
    ax['B'].hist([coords[:,0], coords_gt[:,0]], bins = 50, alpha =0.5, label = ["Noisy coords x", "GT coords x"], density=True)#, histtype="step")
    ax['C'].hist([coords[:,1], coords_gt[:,1]], bins = 50, alpha =0.5, label = ["Noisy coords y", "GT coords y"], density=True)#, histtype="step")
    ax['D'].hist([coords[:,2], coords_gt[:,2]], bins = 50, alpha =0.5, label = ["Noisy coords z", "GT coords z"], density=True)#, histtype="step")

    ax['E'].hist([lattices[:,0], lattices_gt[:,0]], bins = 50, alpha =0.5, label = ["Noisy lattices x", "GT lattices x"], density=True)#, histtype="step")
    ax['F'].hist([lattices[:,1], lattices_gt[:,1]], bins = 50, alpha =0.5, label = ["Noisy lattices y", "GT lattices y"], density=True)#, histtype="step")
    ax['G'].hist([lattices[:,2], lattices_gt[:,2]], bins = 50, alpha =0.5, label = ["Noisy lattices z", "GT lattices z"], density=True)#, histtype="step")
    ax['H'].hist([lattices[:,3], lattices_gt[:,3]], bins = 50, alpha =0.5, label = ["Noisy lattices x", "GT lattices x"], density=True)#, histtype="step")
    ax['I'].hist([lattices[:,4], lattices_gt[:,4]], bins = 50, alpha =0.5, label = ["Noisy lattices y", "GT lattices y"], density=True)#, histtype="step")
    ax['J'].hist([lattices[:,5], lattices_gt[:,5]], bins = 50, alpha =0.5, label = ["Noisy lattices z", "GT lattices z"], density=True)#, histtype="step")
    
    
    ax['A'].legend()
    ax['T'].legend()
    ax['B'].legend()
    ax['C'].legend()
    ax['D'].legend()

    ax['E'].legend()
    ax['F'].legend()
    ax['G'].legend()
    ax['H'].legend()
    ax['I'].legend()
    ax['J'].legend()

    plt.savefig('gt_data_and_noisy_data.png')
    
    plt.show()
    print('a')

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)

def lattice_remove_zeros(L):
    L[:,0,1] = 0
    L[:,2,0] = 0
    L[:,2,1] = 0
    L = torch.flatten(L, start_dim=1)
    L = torch.cat((L[:, :1], L[:, 1 + 1:]), dim=1)
    L = torch.cat((L[:, :5], L[:, 5 + 1:]), dim=1)
    L = torch.cat((L[:, :5], L[:, 5 + 1:]), dim=1)
    return L

def lattice_add_zeros(L):

    zero = torch.zeros((L.size(0),1), device=L.device)
    L = torch.concat([L[:, :1], zero, L[:, 1:5], zero, zero, L[:, 5:6]], dim=1)
    L = L.reshape(-1,3,3)
    return L

def transform_to_batch(frac_coords, atom_type, lattice, num_atoms):
    max = torch.max(num_atoms)
    print(max)

    frac_coords_tensor = torch.empty(0).to(frac_coords.device)
    atom_type_tensor = torch.empty(0).to(atom_type.device)
    lattice_tensor = lattice.unsqueeze(0)

    sum = 0
    for i in range(len(num_atoms)):
        value = int(num_atoms[i])
        frac_coords_tensor = torch.concat(
            [frac_coords_tensor, F.pad(frac_coords[sum:sum + value , :,], (0,0,0,max-num_atoms[i]), mode='constant',value=0).unsqueeze(0)])
        atom_type_tensor = torch.concat(
            [atom_type_tensor, F.pad(atom_type[sum:sum + value , :,], (0,0,0,max-num_atoms[i]), mode='constant',value=0).unsqueeze(0)])
        sum += value
    return frac_coords_tensor, atom_type_tensor, lattice_tensor

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        params = list(self.named_parameters())
        for n, p in params:
            print(n)
        def is_ae(n): return 'atom_ae' in n
        def is_vae(n): return 'atom_vae' in n
        def is_complex_attention(n): return 'decoder.attention_sincos' in n
        grouped_parameters = [
            {"params": [p for n, p in params if is_vae(n) or is_ae(n)], 'lr': self.hparams.optim.special_lr.lr_atom_type_latent}, #/10},
            # {"params": [p for n, p in params if is_ae(n)], 'lr': self.hparams.optim.special_lr.lr_atom_type_latent}, #/10},
            {"params": [p for n, p in params if is_complex_attention(n)], 'lr': self.hparams.optim.special_lr.lr_complex}, #/10},
            {"params": [p for n, p in params if not is_vae(n) and not is_ae(n) and not is_complex_attention(n)], 'lr': self.hparams.optim.optimizer.lr},
    ]
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=grouped_parameters, _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        print(opt.state_dict()['param_groups'])
        
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": self.hparams.optim.frequency,
                "monitor": "val_loss",
            },
        }


class AtomTypeVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, embedding_dim, hidden_dim, output_dim):
        super(AtomTypeVAE, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.e_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.mean  = nn.Linear(hidden_dim, latent_dim)
        self.var   = nn.Linear (hidden_dim, latent_dim)

        self.d_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.d_fc2= nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def reparameterization(self, mean, log_var):
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var).to(mean.device)    
        z = mean + var*epsilon
        return z
        
    def encode(self, x):
        x = self.embedding(x - 1)
        x = self.e_fc1(x)
        x = self.LeakyReLU(x)
        mean = self.mean(x)
        log_var = self.var(x)

        return mean, log_var
                
    def decode(self, z):
        z = self.d_fc1(z)
        z = self.LeakyReLU(z)
        z = self.d_fc2(z)
        return z
    
    def predict_atom(self, z):
        z = self.decode(z)
        preds = torch.argmax(z, dim=1) + 1
        return preds


    def forward(self, x):
        mean, log_var = self.encode(x)

        z = self.reparameterization(mean, log_var)
        
        preds = self.decode(z)
        
        return preds, mean, log_var


class AtomTypeAE(nn.Module):
    def __init__(self, input_dim, latent_dim, embedding_dim, hidden_dim, output_dim):
        super(AtomTypeAE, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # nn.init.xavier_uniform_(self.embedding.weight, gain=1)
        nn.init.kaiming_uniform_(self.embedding.weight, a=0.2, nonlinearity='leaky_relu')
        self.e_fc1 = nn.Linear(embedding_dim, hidden_dim)
        # nn.init.xavier_uniform_(self.e_fc1.weight, gain=1)
        nn.init.kaiming_uniform_(self.e_fc1.weight, a=0.2, nonlinearity='leaky_relu')
        self.final  = nn.Linear(hidden_dim, latent_dim)
        nn.init.xavier_uniform_(self.final.weight, gain=1)
        # self.maxs   = nn.Linear (hidden_dim, latent_dim)
        # nn.init.xavier_uniform_(self.maxs.weight, gain=1)

        self.d_fc1 = nn.Linear(latent_dim, hidden_dim)
        # nn.init.xavier_uniform_(self.d_fc1.weight, gain=1)
        nn.init.kaiming_uniform_(self.d_fc1.weight, a=0.2, nonlinearity='leaky_relu')
        self.d_fc2= nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.d_fc2.weight, gain=1)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Sigmoid = nn.Sigmoid()
        
    def reparameterization(self, mins, maxs):
        epsilon = torch.rand_like(maxs).to(maxs.device)  
    
        z = mins + epsilon * (maxs - mins)
        return z
        
    def encode(self, x):
        x = self.embedding(x - 1)
        x = self.LeakyReLU(x)
        x = self.e_fc1(x)
        x = self.LeakyReLU(x)
        x = self.final(x)

        x = self.Sigmoid(x)

        return x 
                
    def decode(self, z):
        z = self.d_fc1(z)
        z = self.LeakyReLU(z)
        z = self.d_fc2(z)
        return z
    
    def predict_atom(self, z):
        z = self.decode(z)
        preds = torch.argmax(z, dim=1) + 1
        return preds


    def forward(self, x):
        z = self.encode(x)
        preds = self.decode(z)
        
        return preds


class AtomTypeAEPenalty(nn.Module):
    def __init__(self, input_dim, latent_dim, embedding_dim, hidden_dim, output_dim):
        super(AtomTypeAEPenalty, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # nn.init.xavier_uniform_(self.embedding.weight, gain=1)
        nn.init.kaiming_uniform_(self.embedding.weight, a=0.2, nonlinearity='leaky_relu')
        self.e_fc1 = nn.Linear(embedding_dim, hidden_dim)
        # nn.init.xavier_uniform_(self.e_fc1.weight, gain=1)
        nn.init.kaiming_uniform_(self.e_fc1.weight, a=0.2, nonlinearity='leaky_relu')
        self.final  = nn.Linear(hidden_dim, latent_dim)
        nn.init.xavier_uniform_(self.final.weight, gain=1)
        # self.maxs   = nn.Linear (hidden_dim, latent_dim)
        # nn.init.xavier_uniform_(self.maxs.weight, gain=1)

        self.d_fc1 = nn.Linear(latent_dim, hidden_dim)
        # nn.init.xavier_uniform_(self.d_fc1.weight, gain=1)
        nn.init.kaiming_uniform_(self.d_fc1.weight, a=0.2, nonlinearity='leaky_relu')
        self.d_fc2= nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.d_fc2.weight, gain=1)
        
        # self.LayerNorm = nn.LayerNorm(latent_dim)
        self.activation = nn.LeakyReLU(0.2)
        
    def encode(self, x):
        x = self.embedding(x - 1)
        x = self.activation(x)
        x = self.e_fc1(x)
        x = self.activation(x)
        x = self.final(x)
        # x = self.LayerNorm(x)
        x = self.activation(x)

        return x
                
    def decode(self, z):
        z = self.d_fc1(z)
        z = self.activation(z)
        z = self.d_fc2(z)
        return z
    
    def predict_atom(self, z):
        z = self.decode(z)
        preds = torch.argmax(z, dim=1) + 1
        return preds


    def forward(self, x):
        z = self.encode(x)
        preds = self.decode(z)
        
        return preds


class GridEncoderDecoder(nn.Module):
    def __init__(self):
        super(GridEncoderDecoder, self).__init__()

        
    def encode(self, x):
        x = x - 1
        row = (x//10)/10 + 0.05
        column = (x%10)/10 + 0.05
        return torch.concat([row.unsqueeze(1), column.unsqueeze(1)], dim=1)
        
                
    def decode(self, z):
        z = 10*torch.floor(10*z[:,0]) + torch.floor(10*z[:,1]) + 1
        return z.to(torch.int)
    
    def predict_atom(self, z):
        z = self.decode(z)
        return z


    def forward(self, x):
        z = self.encode(x)
        preds = self.decode(z)
        
        return preds
    
class Architecture(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.sigma_lattice_from_data = self.hparams.sigma_lattice_from_data

        self.scale_cord_loss_sigma = self.hparams.scale_cord_loss_sigma

        self.periodic_coords = self.hparams.periodic_coords
        self.periodic_coords_modified = self.hparams.periodic_coords_modified
        self.periodic_loss_square = self.hparams.periodic_loss_square

        self.CE_loss = nn.CrossEntropyLoss()

        self.use_vae = self.hparams.use_vae
        self.train_vae = self.hparams.train_vae
        self.train_vae_gt = self.hparams.train_vae_gt

        if not os.path.exists('ae'):
            os.makedirs('ae')

        if self.use_vae:
            self.atom_vae = AtomTypeVAE(input_dim=MAX_ATOMIC_NUM, 
                                        latent_dim=self.hparams.atom_type_vae.latent_dim, 
                                        embedding_dim = self.hparams.atom_type_vae.embedding_dim,
                                        hidden_dim=self.hparams.atom_type_vae.hidden_dim, 
                                        output_dim=MAX_ATOMIC_NUM)
            self.atom_vae.load_state_dict(torch.load(self.hparams.atom_type_vae.location))
            # self.atom_vae.eval()
            self.atom_vae.train()

            self.atom_type_latent_dim = self.hparams.atom_type_vae.latent_dim
            print("VAE")

        self.use_ae = self.hparams.use_ae
        self.use_ae_penalty = self.hparams.use_ae_penalty
        self.train_ae = self.hparams.train_ae
        self.train_ae_gt = self.hparams.train_ae_gt
        # self.train_ae_penalty = self.hparams.train_ae_penalty
        if self.use_ae:
            self.atom_ae = AtomTypeAE(input_dim=MAX_ATOMIC_NUM, 
                                        latent_dim=self.hparams.atom_type_ae.latent_dim, 
                                        embedding_dim = self.hparams.atom_type_ae.embedding_dim,
                                        hidden_dim=self.hparams.atom_type_ae.hidden_dim, 
                                        output_dim=MAX_ATOMIC_NUM)
            self.atom_ae.load_state_dict(torch.load(self.hparams.atom_type_ae.location))
            self.atom_ae.train()
            self.atom_type_latent_dim = self.hparams.atom_type_ae.latent_dim
            print("AE")
        if self.use_ae_penalty:
            self.atom_ae = AtomTypeAEPenalty(input_dim=MAX_ATOMIC_NUM, 
                                    latent_dim=self.hparams.atom_type_ae.latent_dim, 
                                    embedding_dim = self.hparams.atom_type_ae.embedding_dim,
                                    hidden_dim=self.hparams.atom_type_ae.hidden_dim, 
                                    output_dim=MAX_ATOMIC_NUM)
            # self.atom_ae.load_state_dict(torch.load(self.hparams.atom_type_ae.location))
            # self.atom_ae.eval()
            self.atom_ae.train()
            
            self.atom_type_latent_dim = self.hparams.atom_type_ae.latent_dim
            print("AE Penalty")
            self.ReLU_penalty = nn.ReLU()


        self.use_grid = self.hparams.use_grid

        if self.use_grid:
            self.atom_grid = GridEncoderDecoder()
            self.atom_grid.eval()
            self.atom_type_latent_dim = 2
            print("GRID")

        #########
        #########   SDE
        #########
        self.use_sde = self.hparams.use_sde
        self.sde = self.hparams.sde

        self.sde_atom_types = self.setup_sde(self.hparams.sde.sde_atom_types)
        self.sde_frac_coords = self.setup_sde(self.hparams.sde.sde_frac_coords)
        self.sde_lattice = self.setup_sde(self.hparams.sde.sde_lattice)
                



        self.num_atoms_prior = self.hparams.num_atoms_prior

        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        sigmas_lattice = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_lattice_begin),
            np.log(self.hparams.sigma_lattice_end),
            self.hparams.num_noise_level)), dtype=torch.float32)
        
        self.sigmas_lattice = nn.Parameter(sigmas_lattice, requires_grad=False)


        sigmas_atom_type = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_atom_type_begin),
            np.log(self.hparams.sigma_atom_type_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.sigmas_atom_type = nn.Parameter(sigmas_atom_type, requires_grad=False)


        sigmas_coords = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_coords_begin),
            np.log(self.hparams.sigma_coords_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.sigmas_coords = nn.Parameter(sigmas_coords, requires_grad=False)

        # obtain from datamodule.
        self.lattice_scaler = None
        self.scaler = None

        self.lattice_min_max_scaler = None

        self.diversity_batch = torch.tensor([i for i in range(1,MAX_ATOMIC_NUM+1)]) #, device=pred_atom_types.device)

    def setup_sde(self, sde_config):
        # Setup SDEs
        if sde_config.type == 'vpsde':
            sde = VPSDE(beta_min=sde_config.beta_min, beta_max=sde_config.beta_max, 
                        N=self.sde.num_scales, continuous=sde_config.continuous)
            
        elif sde_config.type == 'subvpsde':
            sde = subVPSDE(beta_min=sde_config.beta_min, beta_max=sde_config.beta_max, 
                           N=self.sde.num_scales, continuous=sde_config.continuous)
            
        elif sde_config.type == 'vesde':
            sde = VESDE(sigma_min=sde_config.sigma_min, sigma_max=sde_config.sigma_max, 
                        N=self.sde.num_scales, continuous=sde_config.continuous)
            
        else:
            raise NotImplementedError(f"SDE {sde_config.type} unknown.")

        # sde.discrete_sigmas = sde.discrete_sigmas.to(self.device)
        return sde     
        

    def setup_lattice_sigma(self):

        if self.sigma_lattice_from_data:
            max_sigma = self.lattice_min_max_scaler.maxs - self.lattice_min_max_scaler.mins
            print("max_sigma shape", max_sigma.shape,  max(max_sigma.flatten()).shape)
            max_sigma = max(max_sigma.flatten())
            print(max_sigma)
            sigmas_lattice = torch.tensor(np.linspace(
                max_sigma,
                max_sigma/1000.,
                self.hparams.num_noise_level), dtype=torch.float32)
            
            self.sigmas_lattice = nn.Parameter(sigmas_lattice, requires_grad=False)




    @torch.no_grad()
    def langevin_dynamics(self, num_samples, ld_kwargs, gt_num_atoms=None, gt_atom_types=None):
        """
        decode crystral structure from latent embeddings.
        ld_kwargs: args for doing annealed langevin dynamics sampling:
            n_step_each:  number of steps for each sigma level.
            step_lr:      step size param.
            min_sigma:    minimum sigma to use in annealed langevin dynamics.
            save_traj:    if <True>, save the entire LD trajectory.
            disable_bar:  disable the progress bar of langevin dynamics.
        gt_num_atoms: if not <None>, use the ground truth number of atoms.
        gt_atom_types: if not <None>, use the ground truth atom types.
        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            # all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []
            all_lattices = []

        num_atoms_bin_count = pickle.load(open(self.num_atoms_prior, 'rb'))
        num_atoms = torch.multinomial(num_atoms_bin_count,num_samples=num_samples,replacement=True)


        clamp_atom_types_01 = (self.use_ae or self.use_grid)
        clamp_atom_types_eucledian = (self.use_ae_penalty or self.use_vae)
        wrap_coords = True
        

        if not self.use_sde:


            cur_atom_types = torch.randn(num_atoms.sum(), self.atom_type_latent_dim, device=self.device)
            cur_frac_coords = torch.randn(num_atoms.sum(), 3, device=self.device)
            cur_lattice = torch.randn((num_atoms.size(0), 6), device=self.device)

            
            cur_atom_types = torch.randn(num_atoms.sum(), self.atom_type_latent_dim, device=self.device)
            cur_frac_coords = torch.rand(num_atoms.sum(), 3, device=self.device)
            cur_lattice = torch.randn((num_atoms.size(0), 6), device=self.device)


            if ld_kwargs.langevin == 'annealed':

                for sigma_atom_types, sigma_coords, sigma_lattice in tqdm(
                    zip(self.sigmas_atom_type, self.sigmas_coords, self.sigmas_lattice), total=self.sigmas_atom_type.size(0), disable=ld_kwargs.disable_bar):
                    
                    step_size_atom_types = ld_kwargs.step_lr_atom_types * (sigma_atom_types / self.sigmas_atom_type[-1]) ** 2
                    step_size_coords = ld_kwargs.step_lr_coords * (sigma_coords / self.sigmas_coords[-1]) ** 2
                    step_size_lattice = ld_kwargs.step_lr_lattice * (sigma_lattice / self.sigmas_lattice[-1]) ** 2


                    for step in range(ld_kwargs.n_step_each):
                        noise_atom_types = torch.randn_like(
                            cur_atom_types) * torch.sqrt(step_size_atom_types*2)
                        noise_cart = torch.randn_like(
                            cur_frac_coords) * torch.sqrt(step_size_coords*2) #step_size_coords * 2)
                        noise_lattice = torch.randn_like(
                            cur_lattice) * torch.sqrt(step_size_lattice*2) #step_size_lattice * 2)

                    
                        pred_atom_types_diff, pred_frac_coord_diff, pred_lattice_diff = self.decoder(
                            cur_atom_types, cur_frac_coords, num_atoms, cur_lattice, None)
                        
                        cur_atom_types = cur_atom_types + step_size_atom_types * pred_atom_types_diff/sigma_atom_types + noise_atom_types
                        

                        if self.scale_cord_loss_sigma:
                            cur_frac_coords = cur_frac_coords + step_size_coords * pred_frac_coord_diff/sigma_coords + noise_cart
                        else:
                            cur_frac_coords = cur_frac_coords + step_size_coords * pred_frac_coord_diff + noise_cart

                        cur_lattice = cur_lattice + step_size_lattice * pred_lattice_diff/sigma_lattice + noise_lattice
                        
                        if clamp_atom_types_01:
                            cur_atom_types = torch.clamp(cur_atom_types, min=0.0, max=1.0)
                        if clamp_atom_types_eucledian:
                            norm = cur_atom_types.norm(dim = -1, keepdim=True)
                            norm[norm < 1] = 1
                            cur_atom_types = cur_atom_types / norm


                        if wrap_coords:
                            cur_frac_coords = (cur_frac_coords % 1.)

                        if step%50 == 0: 
                            print(cur_lattice[0,:])
                            print(cur_frac_coords[0,:])
                            print(cur_atom_types[0,:])
                            print('\n') 
                           
                        if ld_kwargs.save_traj:
                            all_frac_coords.append(cur_frac_coords)
                            all_atom_types.append(cur_atom_types)
                            all_lattices.append(cur_lattice)


        if self.use_sde:

            self.sde_atom_types.discrete_sigmas = self.sde_atom_types.discrete_sigmas.to(self.device)
            self.sde_frac_coords.discrete_sigmas = self.sde_frac_coords.discrete_sigmas.to(self.device)
            self.sde_lattice.discrete_sigmas = self.sde_lattice.discrete_sigmas.to(self.device)

            if ld_kwargs.use_predictor:
                predictor = ReverseDiffusionPredictor(
                self.sde, self.sde_atom_types, self.sde_frac_coords, self.sde_lattice,
                self.decoder
                )
            corrector = AnnealedLangevinDynamics(
                self.sde, self.sde_atom_types, self.sde_frac_coords, self.sde_lattice,
                # self.decoder, 0.15, self.sde.T
                self.decoder, self.sde.T
            )
            with torch.no_grad():
            # Initial sample
                cur_atom_types = self.sde_atom_types.prior_sampling(shape=(num_atoms.sum(), self.atom_type_latent_dim)).to(self.device)
                # cur_frac_coords = self.sde_frac_coords.prior_sampling(shape=(num_atoms.sum(), 3)).to(self.device)
                cur_lattice = self.sde_lattice.prior_sampling(shape=(num_atoms.size(0), 6)).to(self.device)
                
                cur_frac_coords = torch.rand(num_atoms.sum(), 3, device=self.device) * self.sde_frac_coords.sigma_max

                timesteps = torch.linspace(self.sde_atom_types.T, self.sde.eps, self.sde.num_scales, device=self.device)


                

                for i in range(self.sde.num_scales):
                    t = timesteps[i]
                    vec_t = torch.ones(num_atoms.shape[0], device=t.device) * t
                    
                    if ld_kwargs.use_predictor:
                        cur_atom_types, cur_mean_atom_types,\
                        cur_frac_coords, cur_mean_frac_coords,\
                        cur_lattice, cur_mean_lattice = predictor.update_fn(
                            cur_atom_types, cur_frac_coords, cur_lattice, num_atoms, vec_t,
                            clamp_atom_types_01=clamp_atom_types_01,
                            clamp_atom_types_eucledian =clamp_atom_types_eucledian,
                            wrap_coords = wrap_coords)

                        if ld_kwargs.save_traj:
                                all_frac_coords.append(cur_frac_coords)
                                all_atom_types.append(cur_atom_types)
                                all_lattices.append(cur_lattice)

                    cur_atom_types, cur_mean_atom_types,\
                    cur_frac_coords, cur_mean_frac_coords,\
                    cur_lattice, cur_mean_lattice = corrector.update_fn(cur_atom_types, cur_frac_coords, cur_lattice, \
                        num_atoms, vec_t,
                        ld_kwargs.target_snr_atom_types, ld_kwargs.target_snr_frac_coords, ld_kwargs.target_snr_lattice,
                        clamp_atom_types_01=clamp_atom_types_01,
                        clamp_atom_types_eucledian =clamp_atom_types_eucledian,
                        wrap_coords = wrap_coords)
                

                    if ld_kwargs.save_traj:
                            all_frac_coords.append(cur_frac_coords)
                            all_atom_types.append(cur_atom_types)
                            all_lattices.append(cur_lattice)

                #######
                ####### last prediction without noise
                #######
                cur_atom_types = cur_mean_atom_types
                cur_frac_coords = cur_mean_frac_coords
                cur_lattice = cur_mean_lattice

                if clamp_atom_types_01:
                    cur_atom_types = torch.clamp(cur_atom_types, min=0.0, max=1.0)
                if clamp_atom_types_eucledian:
                    norm = cur_atom_types.norm(dim = -1, keepdim=True)
                    norm[norm < 1] = 1
                    cur_atom_types = cur_atom_types / norm


        cur_frac_coords = (cur_frac_coords % 1.)

        if self.use_vae:
            self.atom_vae = self.atom_vae.to(self.device)
            self.atom_vae.eval()
            with torch.no_grad():
                cur_atom_types = self.atom_vae.predict_atom(cur_atom_types)
            
        if self.use_ae or self.use_ae_penalty:
            self.atom_ae = self.atom_ae.to(self.device)
            self.atom_ae.eval()
            with torch.no_grad():
                cur_atom_types = self.atom_ae.predict_atom(cur_atom_types)
        if self.use_grid:
            cur_atom_types = self.atom_grid.predict_atom(cur_atom_types)

        cur_lattice = lattice_add_zeros(cur_lattice)
        if self.hparams.data.lattice_scaler == 'min_max':
            self.lattice_min_max_scaler.match_device(cur_lattice)
            cur_lattice = self.lattice_min_max_scaler.inverse_transform(cur_lattice)
            # cur_lattice = torch.clamp(cur_lattice, min=0.0)
            
        elif self.hparams.data.lattice_scaler == 'standard':
            self.lattice_scaler.match_device(cur_lattice)
            cur_lattice = self.lattice_scaler.inverse_transform(cur_lattice)


        output_dict = {'num_atoms': num_atoms, 'lattice': cur_lattice,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}


        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_lattices=torch.stack(all_lattices, dim=0),
                is_traj=True))

        return output_dict



    def sample(self, num_samples, ld_kwargs):
        samples = self.langevin_dynamics(num_samples, ld_kwargs)
        return samples
    
    def add_noise(self, atom_types, frac_coords, lattice, num_atoms):

        if not self.use_sde:
            #sample noise levels.
            noise_level = torch.randint(0, self.sigmas_atom_type.size(0),
                                        (num_atoms.size(0),),
                                        device=self.device)

            ### add noise to atom types
            
            rand_noise_atoms = torch.randn_like(atom_types)

            used_sigmas_per_atom_types = self.sigmas_atom_type[noise_level].repeat_interleave(
                num_atoms, dim=0)
            noises_per_atom = (
                rand_noise_atoms *
                used_sigmas_per_atom_types[:, None])
            noisy_atom_types_z = atom_types + noises_per_atom


            
            # add noise to the frac coords
            rand_noise_coords = torch.randn_like(frac_coords)

            used_sigmas_per_atom_coords = self.sigmas_coords[noise_level].repeat_interleave(
                num_atoms, dim=0)
            frac_noises_per_atom = (
                rand_noise_coords *
                used_sigmas_per_atom_coords[:, None])
            noisy_frac_coords = frac_coords + frac_noises_per_atom
            # noisy_frac_coords = noisy_frac_coords % 1.

            used_sigmas_per_lattice = self.sigmas_lattice[noise_level]

            ### add the noise that is computed on the scaled lattice to the scaled lattice
            rand_noise_lattice  = torch.randn_like(lattice)
            lattice_noise = (rand_noise_lattice * used_sigmas_per_lattice[:,None])

            noisy_lattice = lattice + lattice_noise


            return noisy_atom_types_z, noisy_frac_coords, noisy_lattice, \
            rand_noise_atoms, rand_noise_coords, rand_noise_lattice, \
            used_sigmas_per_atom_types, used_sigmas_per_atom_coords, used_sigmas_per_lattice, \
            frac_noises_per_atom, lattice_noise, None
        

        if self.use_sde:
            timestep = torch.rand(num_atoms.size(0), device=self.device)
            t = timestep * (self.sde_atom_types.T - self.sde.eps) + self.sde.eps

            z_atom_types = torch.randn_like(atom_types)
            mean_atom_types, std_atom_types = self.sde_atom_types.marginal_prob(atom_types, t)
            std_atom_types = std_atom_types.repeat_interleave(num_atoms, dim=0)
            noisy_atom_types = mean_atom_types + std_atom_types[:, None] * z_atom_types

            z_frac_coords = torch.randn_like(frac_coords)
            mean_frac_coords, std_frac_coords = self.sde_frac_coords.marginal_prob(frac_coords, t)
            std_frac_coords = std_frac_coords.repeat_interleave(num_atoms, dim=0)
            noisy_frac_coords = mean_frac_coords + std_frac_coords[:, None] * z_frac_coords
            frac_noises_per_atom = std_frac_coords[:, None] * z_frac_coords
            
            z_lattice = torch.randn_like(lattice)
            mean_lattice, std_lattice = self.sde_lattice.marginal_prob(lattice, t)
            noisy_lattice = mean_lattice + std_lattice[:, None] * z_lattice
            lattice_noise = std_lattice[:, None] * z_lattice
            

            return noisy_atom_types, noisy_frac_coords, noisy_lattice, \
            z_atom_types, z_frac_coords, z_lattice, \
            std_atom_types, std_frac_coords, std_lattice, \
            frac_noises_per_atom, lattice_noise, t

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
        

    def get_model(self, noisy_atom_types_z, noisy_frac_coords, noisy_lattice, 
                  std_atom_types, std_frac_coords, std_lattice,
                  num_atoms, t):
        
        if not self.use_sde:
            pred_atom_types_diff, pred_frac_coords_diff, pred_lattice_diff = self.decoder(
                            noisy_atom_types_z, noisy_frac_coords, num_atoms, noisy_lattice,
                            None)
        
            used_sigmas_atom_types = std_atom_types
            used_sigmas_frac_coords = std_frac_coords
            used_sigmas_lattice = std_lattice
            
        if self.use_sde:
            
            pred_atom_types_diff, pred_frac_coords_diff, pred_lattice_diff = self.decoder(
                            noisy_atom_types_z, noisy_frac_coords, num_atoms, noisy_lattice,
                            t)

            self.sde_atom_types.discrete_sigmas = self.sde_atom_types.discrete_sigmas.to(pred_atom_types_diff.device)
            self.sde_frac_coords.discrete_sigmas = self.sde_frac_coords.discrete_sigmas.to(pred_frac_coords_diff.device)
            self.sde_lattice.discrete_sigmas = self.sde_lattice.discrete_sigmas.to(pred_lattice_diff.device)

            labels_atom_types = self.get_labels(self.sde_atom_types, pred_atom_types_diff, t)
            labels_frac_coords = self.get_labels(self.sde_frac_coords, pred_frac_coords_diff, t)
            labels_lattice = self.get_labels(self.sde_lattice, pred_lattice_diff, t)
          
            if isinstance(self.sde_atom_types, VPSDE) or isinstance(self.sde_atom_types, subVPSDE):
                pred_atom_types_diff = -pred_atom_types_diff / std_atom_types[:, None, None, None]
            elif isinstance(self.sde_atom_types, VESDE):
                if self.sde_atom_types.continuous:
                    used_sigmas_atom_types = labels_atom_types.repeat_interleave(num_atoms, dim=0)
                else:
                    used_sigmas_atom_types = self.sde_atom_types.discrete_sigmas[labels_atom_types].repeat_interleave(num_atoms, dim=0)
                pred_atom_types_diff = pred_atom_types_diff / used_sigmas_atom_types[:, None]
            else:
                raise NotImplementedError(f"SDE class {self.sde_atom_types.__class__.__name__} not yet supported.")
            
            if isinstance(self.sde_frac_coords, VPSDE) or isinstance(self.sde_frac_coords, subVPSDE):
                pred_frac_coords_diff = -pred_frac_coords_diff / std_frac_coords[:, None, None, None]
            elif isinstance(self.sde_frac_coords, VESDE):
                if self.sde_frac_coords.continuous:
                    used_sigmas_frac_coords = labels_frac_coords.repeat_interleave(num_atoms, dim=0)
                else:
                    used_sigmas_frac_coords = self.sde_frac_coords.discrete_sigmas[labels_frac_coords].repeat_interleave(num_atoms, dim=0)                
                    
                pred_frac_coords_diff = pred_frac_coords_diff / used_sigmas_frac_coords[:, None]
            else:
                raise NotImplementedError(f"SDE class {self.sde_lattice.__class__.__name__} not yet supported.")

            if isinstance(self.sde_lattice, VPSDE) or isinstance(self.sde_lattice, subVPSDE):
                pred_lattice_diff = -pred_lattice_diff / std_lattice[:, None, None, None]
            elif isinstance(self.sde_lattice, VESDE):
                if self.sde_lattice.continuous:
                    used_sigmas_lattice = labels_lattice
                else:
                    used_sigmas_lattice = self.sde_lattice.discrete_sigmas[labels_lattice].repeat_interleave(num_atoms, dim=0)                
                    
            
                pred_lattice_diff = pred_lattice_diff / used_sigmas_lattice[:,None]
            else:
                raise NotImplementedError(f"SDE class {self.sde_lattice.__class__.__name__} not yet supported.")

        return pred_atom_types_diff, pred_frac_coords_diff, pred_lattice_diff, \
        used_sigmas_atom_types, used_sigmas_frac_coords, used_sigmas_lattice

    def compute_losses(self, 
            pred_atom_types_diff, pred_frac_coords_diff, pred_lattice_diff, \
            # noisy_atom_types_z, noisy_frac_coords, noisy_lattice, \
            rand_noise_atoms, rand_noise_coords, rand_noise_lattice, \
            used_sigma_atoms, used_sigma_coords, used_sigma_lattice,\
            frac_noises_per_atom, t, batch):
        
        if not self.use_sde:
            atom_type_loss = self.atom_type_loss_sigma(
                pred_atom_types_diff, rand_noise_atoms, used_sigma_atoms, batch)
            if self.periodic_coords:
                if self.scale_cord_loss_sigma:
                    coord_loss = self.coord_loss_periodic_sigma(
                        pred_frac_coords_diff, rand_noise_coords, used_sigma_coords, batch)
                else:
                    coord_loss = self.coord_loss_periodic_no_scale(
                        pred_frac_coords_diff, frac_noises_per_atom, used_sigma_coords, batch)
            else:
                coord_loss = self.coord_loss_NONperiodic_sigma(
                    pred_frac_coords_diff, rand_noise_coords, used_sigma_coords, batch
                )

            lattice_loss = self.noisy_lattice_loss_dim6_sigma(
                pred_lattice_diff, rand_noise_lattice, used_sigma_lattice, batch)
        if self.use_sde:
            if not self.sde.likelihood_weighting:
                loss_per_atom = torch.square(pred_atom_types_diff * used_sigma_atoms[:, None] + rand_noise_atoms)
                loss_per_atom = torch.sum(loss_per_atom, dim=1)
                loss_per_atom = 0.5 * loss_per_atom
                atom_type_loss = scatter(loss_per_atom, batch.batch, reduce='mean').mean()

                if self.periodic_coords:
                    rand_noise_coords = - rand_noise_coords
                    if self.scale_cord_loss_sigma:
                        pred_frac_coords_diff = pred_frac_coords_diff * used_sigma_coords[:, None]
                    else:
                        pred_frac_coords_diff = pred_frac_coords_diff
                    f1_ = 2*torch.pi*pred_frac_coords_diff
                    f2_ = 2*torch.pi*rand_noise_coords
                    zero = torch.tensor([0.], device=f1_.device)
                    distance = torch.real(torch.exp(torch.complex(zero,f1_)) * torch.exp(torch.complex(zero,-f2_) ))
                    distance_vec = (-distance + 1)/2 
                    if self.periodic_loss_square:
                        distance_vec = distance_vec**2
                    coord_loss_per_atom = torch.sum(distance_vec, dim=1)
                    coord_loss = scatter(coord_loss_per_atom, batch.batch, reduce='mean').mean()
                
                
                elif self.periodic_coords_modified:
                    # print("periodic MSE")
                    if self.scale_cord_loss_sigma:
                        pred_frac_coords_diff = pred_frac_coords_diff * used_sigma_coords[:, None]
                    else:
                        pred_frac_coords_diff = pred_frac_coords_diff
                    loss_per_atom = torch.square((pred_frac_coords_diff + rand_noise_coords + 0.5)%1. - 0.5)
                    loss_per_atom = torch.sum(loss_per_atom, dim=1)
                    loss_per_atom = 0.5 * loss_per_atom
                    coord_loss = scatter(loss_per_atom, batch.batch, reduce='mean').mean()


                else:
                    if self.scale_cord_loss_sigma:
                        pred_frac_coords_diff = pred_frac_coords_diff * used_sigma_coords[:, None]
                    else:
                        pred_frac_coords_diff = pred_frac_coords_diff

                    # rand_noise_coords = (rand_noise_coords + 0.5)%1. - 0.5

                    loss_per_atom = torch.square(pred_frac_coords_diff + rand_noise_coords)
                    loss_per_atom = torch.sum(loss_per_atom, dim=1)
                    loss_per_atom = 0.5 * loss_per_atom
                    coord_loss = scatter(loss_per_atom, batch.batch, reduce='mean').mean()

                lattice_loss = torch.square(pred_lattice_diff * used_sigma_lattice[:, None] + rand_noise_lattice)
                lattice_loss = 0.5 * torch.sum(lattice_loss, dim=1) 
                lattice_loss = lattice_loss.mean(dim=0)
                
            else:
                g2_atom_types = self.sde_atom_types.sde(torch.zeros_like(pred_atom_types_diff), t)[1] ** 2
                loss_per_atom = torch.square(pred_atom_types_diff + rand_noise_atoms /used_sigma_atoms[:, None] )
                loss_per_atom = torch.sum(loss_per_atom, dim=1)
                loss_per_atom = 0.5 * loss_per_atom * g2_atom_types.repeat_interleave(batch.num_atoms, dim=0)
                atom_type_loss = scatter(loss_per_atom, batch.batch, reduce='mean').mean()
                
                if self.periodic_coords:
                    g2_frac_coords = self.sde_frac_coords.sde(torch.zeros_like(pred_frac_coords_diff), t)[1] ** 2
                    
                    # pred_frac_coords_diff = pred_frac_coords_diff * used_sigma_coords[:, None]
                    rand_noise_coords = - rand_noise_coords
                    f1_ = 2*torch.pi*pred_frac_coords_diff
                    f2_ = 2*torch.pi*(rand_noise_coords / used_sigma_coords[:,None])
                    zero = torch.tensor([0.], device=f1_.device)
                    distance = torch.real(torch.exp(torch.complex(zero,f1_)) * torch.exp(torch.complex(zero,-f2_) ))
                    distance_vec = (-distance + 1)/2 
                    coord_loss_per_atom = torch.sum(distance_vec, dim=1) * g2_frac_coords.repeat_interleave(batch.num_atoms, dim=0) * 0.5
                    coord_loss = scatter(coord_loss_per_atom, batch.batch, reduce='mean').mean()
                
                elif self.periodic_coords_modified:
                    if self.scale_cord_loss_sigma:
                        pred_frac_coords_diff = pred_frac_coords_diff * used_sigma_coords[:, None]
                    else:
                        pred_frac_coords_diff = pred_frac_coords_diff

                    g2_atom_types = self.sde_atom_types.sde(torch.zeros_like(pred_frac_coords_diff), t)[1] ** 2
                    loss_per_atom = torch.square((pred_frac_coords_diff + rand_noise_coords /used_sigma_coords[:, None] + 0.5)%1. - 0.5 )
                    loss_per_atom = torch.sum(loss_per_atom, dim=1)
                    loss_per_atom = 0.5 * loss_per_atom * g2_atom_types.repeat_interleave(batch.num_atoms, dim=0)
                    coord_loss = scatter(loss_per_atom, batch.batch, reduce='mean').mean()


                else:
                    g2_atom_types = self.sde_atom_types.sde(torch.zeros_like(pred_frac_coords_diff), t)[1] ** 2
                    loss_per_atom = torch.square(pred_frac_coords_diff + rand_noise_coords /used_sigma_coords[:, None] )
                    loss_per_atom = torch.sum(loss_per_atom, dim=1)
                    loss_per_atom = 0.5 * loss_per_atom * g2_atom_types.repeat_interleave(batch.num_atoms, dim=0)
                    coord_loss = scatter(loss_per_atom, batch.batch, reduce='mean').mean()

                
                g2_lattice = self.sde_lattice.sde(torch.zeros_like(pred_lattice_diff), t)[1] ** 2
                lattice_loss = torch.square(pred_lattice_diff + rand_noise_lattice /used_sigma_lattice[:, None] )
                lattice_loss = 0.5 * torch.sum(lattice_loss, dim=1) * g2_lattice
                lattice_loss = lattice_loss.mean(dim=0)

        return atom_type_loss, coord_loss, lattice_loss




    def predict_crystal(self, noisy_atom_types_z, noisy_frac_coords, noisy_lattice,\
                    pred_atom_types_diff, pred_frac_coords_diff, pred_lattice_diff, \
                    rand_noise_atoms, rand_noise_coords, rand_noise_lattice, \
                    used_sigmas_atom_types, used_sigmas_frac_coords, used_sigmas_lattice):

        if not self.use_sde:


            pred_atom_type_noise = pred_atom_types_diff*used_sigmas_atom_types[:, None]
            pred_atom_types_z = noisy_atom_types_z + pred_atom_type_noise

            if self.scale_cord_loss_sigma:
                pred_frac_coords_noise = pred_frac_coords_diff*used_sigmas_frac_coords[:, None]
            else:
                pred_frac_coords_noise = pred_frac_coords_diff
            pred_frac_coords = noisy_frac_coords + pred_frac_coords_noise

            pred_lattice_noise = pred_lattice_diff*used_sigmas_lattice[:, None]
            pred_lattice = noisy_lattice + pred_lattice_noise


            return pred_atom_types_z, pred_frac_coords, pred_lattice



        if self.use_sde:

            if not self.sde.likelihood_weighting:

                pred_atom_type_noise = pred_atom_types_diff*used_sigmas_atom_types[:, None]
                pred_atom_types_z = noisy_atom_types_z + pred_atom_type_noise

                if self.scale_cord_loss_sigma:
                    pred_frac_coords_noise = pred_frac_coords_diff*used_sigmas_frac_coords[:, None]
                else:
                    pred_frac_coords_noise = pred_frac_coords_diff
                pred_frac_coords = noisy_frac_coords - pred_frac_coords_noise

                pred_lattice_noise = pred_lattice_diff*used_sigmas_lattice[:, None]
                pred_lattice = noisy_lattice + pred_lattice_noise


                return pred_atom_types_z, pred_frac_coords, pred_lattice
            
            else:

                pred_atom_type_noise = pred_atom_types_diff*used_sigmas_atom_types[:, None]**2
                pred_atom_types_z = noisy_atom_types_z + pred_atom_type_noise

                pred_frac_coords_noise = pred_frac_coords_diff*used_sigmas_frac_coords[:, None]**2
                # pred_frac_coords = noisy_frac_coords + pred_frac_coords_noise
                pred_frac_coords = noisy_frac_coords + pred_frac_coords_noise

                pred_lattice_noise = pred_lattice_diff*used_sigmas_lattice[:, None]**2
                pred_lattice = noisy_lattice + pred_lattice_noise


                return pred_atom_types_z, pred_frac_coords, pred_lattice
                # raise NotImplementedError('Currently the following combination is not supported:\
                #                             - finetune the atom type latent representation on the denoised data \
                #                             - SDE with Likelihood weighting')



    def forward(self, batch, teacher_forcing, training):
        
        # self.diversity_batch = self.diversity_batch.to(self.device)
        atom_types_numpy = batch.atom_types.cpu().numpy()
        _, diversity_index = np.unique(atom_types_numpy, return_index=True, axis=0)
        # self.diversity_batch_dataset = torch.tensor(self.diversity_batch_dataset, device=self.device)

        if self.use_ae or self.use_ae_penalty:
            atom_types_kld = 0

            self.atom_ae = self.atom_ae.to(self.device)
            if self.train_ae:
                atom_types_z = self.atom_ae.encode(batch.atom_types)
                # diversity_batch_z = self.atom_ae.encode(self.diversity_batch)
                # diversity_batch_dataset_z = self.atom_ae.encode(self.diversity_batch_dataset)
            else:
                self.atom_ae.eval()
                with torch.no_grad():
                    atom_types_z = self.atom_ae.encode(batch.atom_types)
                    # diversity_batch_z = self.atom_ae.encode(self.diversity_batch)
                    # diversity_batch_dataset_z = self.atom_ae.encode(self.diversity_batch_dataset)


        elif self.use_grid:
            atom_types_kld = 0

            self.atom_grid = self.atom_grid.to(self.device)
            atom_types_z = self.atom_grid.encode(batch.atom_types)
            # diversity_batch_z = self.atom_grid.encode(self.diversity_batch)
        elif self.use_vae:
            self.atom_vae = self.atom_vae.to(self.device)

            if self.train_vae:
                atom_types_z, atom_types_z_log_var = self.atom_vae.encode(batch.atom_types)
                atom_types_kld = self.atom_types_kld(atom_types_z, atom_types_z_log_var)
                # diversity_batch_z = self.atom_vae.encode(self.diversity_batch)
            else:
                atom_types_kld = 0
                self.atom_vae.eval()
                with torch.no_grad():
                    atom_types_z, _ = self.atom_vae.encode(batch.atom_types)
                    # diversity_batch_z = self.atom_vae.encode(self.diversity_batch)
                
                # atom_types_z = atom_types_mean


        # add noise to lattice

        if self.hparams.data.lattice_scale_method == 'scale_length':
            batch_lengths_scaled = batch.lengths /  batch.num_atoms.view(-1, 1).float()**(1/3)

            self.lattice_scaler.match_device(batch.lengths)
            target_lengths_and_angles = torch.cat(
                [batch_lengths_scaled, batch.angles], dim=-1)
            
            lattice = lattice_params_to_matrix_torch(target_lengths_and_angles[:,:3], target_lengths_and_angles[:,3:])
           
        else:
            lattice = batch.lattice

        if self.hparams.data.lattice_scaler == 'min_max':
            self.lattice_min_max_scaler.match_device(batch.lattice)
            lattice = self.lattice_min_max_scaler.transform(lattice)

        elif self.hparams.data.lattice_scaler == 'standard':
            self.lattice_scaler.match_device(batch.lengths)
            lattice = self.lattice_scaler.transform(lattice)
            
        lattice = lattice_remove_zeros(lattice)
        



        noisy_atom_types_z, noisy_frac_coords, noisy_lattice, \
        rand_noise_atoms, rand_noise_coords, rand_noise_lattice, \
        std_atom_types, std_frac_coords, std_lattice, \
        frac_noises_per_atom, lattice_noise,  t = self.add_noise(
            atom_types_z, batch.frac_coords, lattice, batch.num_atoms)



        if self.hparams.data.lattice_scale_method == 'scale_length':
            lattice_params = torch.stack([lattice_matrix_to_params_torch(x_i) for x_i in torch.unbind(noisy_lattice, dim=0)], dim=0)
        
            # ### scale lattice up again
            lengths = lattice_params[:,:3].to(batch.frac_coords.device)
            angles = lattice_params[:,3:].to(batch.frac_coords.device)
            
            lengths = lengths * batch.num_atoms.view(-1, 1).float()**(1/3)


        pred_atom_types_diff, pred_frac_coords_diff, pred_lattice_diff, \
             used_sigmas_atom_types, used_sigmas_frac_coords, used_sigmas_lattice = self.get_model(
            noisy_atom_types_z, noisy_frac_coords, noisy_lattice, \
            std_atom_types, std_frac_coords, std_lattice, \
            batch.num_atoms, t
        )

        atom_type_loss, coord_loss, lattice_loss = self.compute_losses(
            pred_atom_types_diff, pred_frac_coords_diff, pred_lattice_diff, \
            # noisy_atom_types_z, noisy_frac_coords, noisy_lattice, \
            rand_noise_atoms, rand_noise_coords, rand_noise_lattice, \
            used_sigmas_atom_types, used_sigmas_frac_coords, used_sigmas_lattice,\
            frac_noises_per_atom,  t, batch
        )

        pred_atom_types_z, pred_frac_coords, pred_lattice = self.predict_crystal(
                        noisy_atom_types_z, noisy_frac_coords, noisy_lattice, \
                        pred_atom_types_diff, pred_frac_coords_diff, pred_lattice_diff,\
                        rand_noise_atoms, rand_noise_coords, rand_noise_lattice, \
                        std_atom_types, std_frac_coords, std_lattice
        )
        
            
        if self.use_vae:
            atom_types_ae_penalty = 0
            atom_types_ae_penalty_std = 0
            if self.train_vae:
                if self.train_vae_gt:
                    pred_atom_types = self.atom_vae.predict_atom(atom_types_z)
                    decoded_atom_types = self.atom_vae.decode(atom_types_z)
                    
                else:
                    pred_atom_types = self.atom_vae.predict_atom(pred_atom_types_z)
                    decoded_atom_types = self.atom_vae.decode(pred_atom_types_z)
                
                atom_types_accuracy = self.atom_types_accuracy(batch.atom_types, pred_atom_types)
                atom_types_cross_entropy = self.atom_type_cross_entropy(batch.atom_types, decoded_atom_types)

            else:
                atom_types_cross_entropy = 0
                self.atom_vae.eval()
                with torch.no_grad():
                    pred_atom_types = self.atom_vae.predict_atom(pred_atom_types_z)
                    atom_types_accuracy = self.atom_types_accuracy(batch.atom_types, pred_atom_types)
                    atom_types_cross_entropy = 0
           

        if self.use_grid:
            pred_atom_types = self.atom_grid.predict_atom(pred_atom_types_z)
            atom_types_accuracy = self.atom_types_accuracy(batch.atom_types, pred_atom_types)
            atom_types_cross_entropy = 0
            atom_types_ae_penalty = 0
            atom_types_ae_penalty_std = 0


        if self.use_ae or self.use_ae_penalty:
            if self.train_ae:
                if self.train_ae_gt:
                    
                    pred_atom_types = self.atom_ae.predict_atom(atom_types_z)
                    decoded_atom_types = self.atom_ae.decode(atom_types_z)

                    if self.use_ae_penalty:
                        # atom_types_ae_penalty = self.ReLU_penalty(atom_types_z.norm(dim=-1) - 1).sum()
                        # atom_types_ae_penalty_std = self.ReLU_penalty(2 - atom_types_z.std(dim=0)).sum()

                        
                        # atom_types_ae_penalty = self.ReLU_penalty(atom_types_z[diversity_index].norm(dim=-1) - 1).mean()
                        # atom_types_ae_penalty = self.ReLU_penalty(atom_types_z.norm(dim=-1) - 1).mean()
                        # atom_types_ae_penalty_std = self.ReLU_penalty(2 - pred_atom_types_z.std(dim=0)).mean()
                        atom_types_ae_penalty = self.ReLU_penalty(atom_types_z.norm(dim=-1) - 1).mean()
                        atom_types_ae_penalty_std = 0 #self.ReLU_penalty(1 - atom_types_z[diversity_index].std(dim=0)).sum()

                    else:
                        atom_types_ae_penalty = 0
                        # atom_types_ae_penalty_std = self.ReLU_penalty(2 - diversity_batch_z.std(dim=0)).sum()
                        # atom_types_ae_penalty_std = self.ReLU_penalty(2 - pred_atom_types_z.std(dim=0)).mean()

                        atom_types_ae_penalty_std = 0 #self.ReLU_penalty(1 - atom_types_z[diversity_index].std(dim=0)).sum()

                    
                else:
                    pred_atom_types = self.atom_ae.predict_atom(pred_atom_types_z)
                    decoded_atom_types = self.atom_ae.decode(pred_atom_types_z)
                    
                    if self.use_ae_penalty:
                        # atom_types_ae_penalty = self.ReLU_penalty(pred_atom_types_z.norm(dim=-1) - 1).sum()
                        # atom_types_ae_penalty_std = 0 # self.ReLU_penalty(2 - diversity_batch_z.std(dim=0)).sum()
                        # atom_types_kld = self.atom_types_kld(diversity_batch_z, torch.log(torch.ones_like(diversity_batch_z)))

                        # atom_types_ae_penalty = self.ReLU_penalty(pred_atom_types_z[diversity_index].norm(dim=-1) - 1).mean()
                        # atom_types_ae_penalty = self.ReLU_penalty(atom_types_z.norm(dim=-1) - 1).mean()
                        # atom_types_ae_penalty_std = self.ReLU_penalty(2 - pred_atom_types_z.std(dim=0)).mean()
                        atom_types_ae_penalty = self.ReLU_penalty(atom_types_z.norm(dim=-1) - 1).mean()
                        atom_types_ae_penalty_std = 0 #self.ReLU_penalty(1 - pred_atom_types_z[diversity_index].std(dim=0)).sum()
                        # atom_types_kld = self.atom_types_kld(diversity_batch_z, torch.log(torch.ones_like(diversity_batch_z)))
                
                    else:
                        atom_types_ae_penalty = 0
                        # atom_types_ae_penalty_std = self.ReLU_penalty(2 - atom_types_z.std(dim=0)).mean()
                        # atom_types_kld = self.atom_types_kld(diversity_batch_z, torch.log(torch.ones_like(diversity_batch_z)))
                        atom_types_ae_penalty_std = 0 #self.ReLU_penalty(1 - pred_atom_types_z[diversity_index].std(dim=0)).sum()
                

                # atom_types_accuracy = self.atom_types_accuracy(batch.atom_types[diversity_index], pred_atom_types[diversity_index])
                # # atom_types_cross_entropy = self.atom_type_cross_entropy(batch.atom_types, decoded_atom_types)
                # atom_types_cross_entropy = self.atom_type_cross_entropy(batch.atom_types[diversity_index], decoded_atom_types[diversity_index])


                atom_types_accuracy = self.atom_types_accuracy(batch.atom_types, pred_atom_types)
                # atom_types_cross_entropy = self.atom_type_cross_entropy(batch.atom_types, decoded_atom_types)
                atom_types_cross_entropy = self.atom_type_cross_entropy(batch.atom_types, decoded_atom_types)

            else:
                atom_types_cross_entropy = 0
                atom_types_ae_penalty = 0
                atom_types_ae_penalty_std = 0
                self.atom_ae.eval()
                with torch.no_grad():
                    pred_atom_types = self.atom_ae.predict_atom(pred_atom_types_z)
                    atom_types_accuracy = self.atom_types_accuracy(batch.atom_types, pred_atom_types)
                    # atom_types_accuracy = self.atom_types_accuracy(batch.atom_types[diversity_index], pred_atom_types[diversity_index])
                    # atom_types_encoding_diversity = 0

                    # noisy_atom_types_accuracy = self.atom_types_accuracy(batch.atom_types, self.atom_ae.predict_atom(noisy_atom_types_z))
                    # acc_diff = atom_types_accuracy - noisy_atom_types_accuracy
                    # print("Difference between noisy and denoised acc \t ", acc_diff)
         
        pred_lattice = lattice_add_zeros(pred_lattice)

        if self.hparams.data.lattice_scaler == 'min_max':
            self.lattice_min_max_scaler.match_device(pred_lattice)
            pred_lattice = self.lattice_min_max_scaler.inverse_transform(pred_lattice)

        if self.hparams.data.lattice_scaler == 'standard':
            self.lattice_min_max_scaler.match_device(pred_lattice)
            pred_lattice = self.lattice_scaler.inverse_transform(pred_lattice)


        
    

        return {
            'lattice_loss': lattice_loss,
            'coord_loss': coord_loss,
            'atom_type_loss': atom_type_loss,
            'atom_types_cross_entropy': atom_types_cross_entropy,
            'atom_types_kld': atom_types_kld,
            'atom_types_ae_penalty': atom_types_ae_penalty,
            'atom_types_ae_penalty_std': atom_types_ae_penalty_std,
            'atom_types_accuracy': atom_types_accuracy,
            'pred_lattice': pred_lattice,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,

        }
    def atom_types_accuracy(self, gt_atom_types, pred_atom_types):
        return torch.sum(pred_atom_types == gt_atom_types) / gt_atom_types.size(0)



    def atom_type_cross_entropy(self, gt_atom_types, pred_atom_types):
        return self.CE_loss(pred_atom_types, (gt_atom_types-1).long())


    def atom_types_kld(self, mean, log_var):
        temp = 1+ log_var - mean.pow(2) - log_var.exp()

        kld_loss = torch.mean(-0.5 * torch.sum(temp, dim=1), dim=0)

        return kld_loss




    
    def atom_type_loss_sigma(self, pred_atom_types_diff, z,
                   sigma, batch):

        z = - z / sigma[:, None]
        pred_atom_types_diff = pred_atom_types_diff / sigma[:, None]
    
        loss_per_atom = torch.sum(
            (pred_atom_types_diff - z)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * sigma**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()
    

    def coord_loss_NONperiodic_sigma(self, pred_frac_coords_diff, z,
                   sigma, batch):

        z = - z / sigma[:, None]
        pred_frac_coords_diff = pred_frac_coords_diff / sigma[:, None]
    
        loss_per_atom = torch.sum(
            (pred_frac_coords_diff - z)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * sigma**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()
    
    def coord_loss_periodic_sigma(self, pred_frac_coord_diff, z,
                            sigma, batch):
        
        z = - z / sigma[:,None]
        f1_ = 2*torch.pi*pred_frac_coord_diff
        f2_ = 2*torch.pi*z
        zero = torch.tensor([0.], device=f1_.device)
        distance = torch.real(torch.exp(torch.complex(zero,f1_)) * torch.exp(torch.complex(zero,-f2_) ))
        distance_vec = (-distance + 1)/2 
        if self.periodic_loss_square:
            distance_vec = distance_vec**2
        loss_per_atom = torch.sum(distance_vec, dim=1)
        loss_per_atom = 0.5 * loss_per_atom * sigma**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()


    def noisy_lattice_loss_dim6_sigma(self, pred_lattice_diff, z, sigma, batch):
        
        z = - z / sigma[:,None]
        pred_lattice_diff = pred_lattice_diff / sigma[:, None]

        loss = (pred_lattice_diff - z)**2
        loss = 0.5 * torch.sum(loss, dim=1) * sigma** 2
        loss = loss.mean(dim=0)

        return loss
    

    def noisy_lattice_loss_dim6(self, noise_lattice, pred_lattice_diff, used_sigmas_per_lattice, batch):
        
        target = noise_lattice / (used_sigmas_per_lattice[:,None] ** 2)
        if not self.embed_sigmas:
            pred_lattice_diff = pred_lattice_diff / used_sigmas_per_lattice[:,None]

        loss = 0.5 * (pred_lattice_diff - target)**2 * used_sigmas_per_lattice[:,None] ** 2
        loss = torch.sum(loss, dim=1)
        loss = loss.mean(dim=0)

        return loss
    
    def atom_type_loss(self, pred_atom_types_diff, noises_per_atom,
                   used_sigmas_per_atom, batch):

        noises_per_atom = noises_per_atom / \
            used_sigmas_per_atom[:, None]**2
        pred_atom_types_diff = pred_atom_types_diff / \
            used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (noises_per_atom - pred_atom_types_diff)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()
    
    def coord_loss_periodic_no_scale(self, pred_frac_coord_diff, target_frac_coord_diff,
                            used_sigmas_per_atom, batch):

        
        target_frac_coord_diff = - target_frac_coord_diff

        f1_ = 2*torch.pi*pred_frac_coord_diff
        f2_ = 2*torch.pi*target_frac_coord_diff
        zero = torch.tensor([0.], device=f1_.device)
        distance = torch.real(torch.exp(torch.complex(zero,f1_)) * torch.exp(torch.complex(zero,-f2_) ))
        distance_vec = (-distance + 1)/2 
        loss_per_atom = torch.sum(distance_vec, dim=1)
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def coord_loss_periodic(self, pred_frac_coord_diff, target_frac_coord_diff,
                            used_sigmas_per_atom, batch):
        
        #### think about whether it makes sense to scale the loss with sigma (linearly)

        target_frac_coord_diff = target_frac_coord_diff / \
            used_sigmas_per_atom[:, None]**2
        pred_frac_coord_diff = pred_frac_coord_diff / \
            used_sigmas_per_atom[:, None]
        
        f1_ = 2*torch.pi*pred_frac_coord_diff
        f2_ = 2*torch.pi*target_frac_coord_diff
        zero = torch.tensor([0.], device=f1_.device)
        distance = torch.real(torch.exp(torch.complex(zero,f1_)) * torch.exp(torch.complex(zero,-f2_) ))
        distance_vec = (-distance + 1)/2 
        loss_per_atom = 0.5 *  torch.sum(distance_vec, dim=1) * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()
    

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)

        if self.current_epoch >= self.hparams.epoch_stop_ae_finetuning:
            if self.use_vae:
                self.train_vae = False
            if self.use_ae_penalty or self.use_ae:
                self.train_ae = False
        outputs = self(batch, teacher_forcing, training=True)
        
        
        if (self.current_epoch % 50 == 0) and self.current_epoch < self.hparams.epoch_stop_ae_finetuning + 100:
            if self.use_vae:
                torch.save(self.atom_vae.state_dict(), os.path.join('ae', f'vae_{self.current_epoch}.pt'))
            if self.use_ae_penalty or self.use_ae:
                torch.save(self.atom_ae.state_dict(), os.path.join('ae', f'ae_{self.current_epoch}.pt'))
            if self.use_grid:
                torch.save(self.atom_grid.state_dict(), os.path.join('ae', f'grid_{self.current_epoch}.pt'))

        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        log_dict.update({
                f'atom_types_accuracy': outputs['atom_types_accuracy']
        })
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, batch, outputs, prefix):
        # num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        atom_type_loss = outputs['atom_type_loss']
        atom_types_cross_entropy = outputs['atom_types_cross_entropy']
        atom_types_kld = outputs['atom_types_kld']
        atom_types_ae_penalty = outputs['atom_types_ae_penalty']
        atom_types_ae_penalty_std = outputs['atom_types_ae_penalty_std']
 
        loss = (
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_atom_type * atom_type_loss +
            self.hparams.cost_atom_types_cross_entropy * atom_types_cross_entropy +
            self.hparams.cost_atom_types_kld * atom_types_kld +
            self.hparams.cost_atom_types_ae_penalty * atom_types_ae_penalty 
            )

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_atom_type_loss': atom_type_loss,
            f'{prefix}_atom_types_cross_entropy': atom_types_cross_entropy,
            f'{prefix}_atom_types_kld': atom_types_kld,
            f'{prefix}_atom_types_ae_penalty': atom_types_ae_penalty,
            f'{prefix}_atom_types_ae_penalty_std': atom_types_ae_penalty_std,

        }

        if prefix != 'train':
            
            loss = (
                self.hparams.cost_lattice * lattice_loss +
                self.hparams.cost_coord * coord_loss +
                self.hparams.cost_atom_type * atom_type_loss +
                self.hparams.cost_atom_types_cross_entropy * atom_types_cross_entropy +
                self.hparams.cost_atom_types_kld * atom_types_kld +
                self.hparams.cost_atom_types_ae_penalty * atom_types_ae_penalty
                )


            pred_volumes = compute_volume(outputs['pred_lattice'])
            true_volumes = compute_volume(batch.lattice)
            volumes_mard = mard(true_volumes, pred_volumes)

            log_dict.update({
                f'{prefix}_loss': loss,
                f'{prefix}_volumes_mard': volumes_mard,
            })

        return log_dict, loss


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    return model


if __name__ == "__main__":
    main()
