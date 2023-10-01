

import argparse
import pickle
from model.pl_modules.embeddings import MAX_ATOMIC_NUM
# from matplotlib import figure as fig
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib import cm

import glob
import os

from  model.common.data_utils import chemical_symbols

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
        # maxs = self.maxs(x)

        return x #mins, maxs
                
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
        # mins, maxs = self.encode(x)

        # z = self.reparameterization(mins, maxs)
        z = self.encode(x)
        preds = self.decode(z)
        
        return preds #, mins, maxs



# class AtomTypeAEPenalty(nn.Module):
#     def __init__(self, input_dim, latent_dim, embedding_dim, hidden_dim, output_dim):
#         super(AtomTypeAEPenalty, self).__init__()

#         self.embedding = nn.Embedding(input_dim, embedding_dim)
#         # nn.init.xavier_uniform_(self.embedding.weight, gain=1)
#         nn.init.kaiming_uniform_(self.embedding.weight, a=0.2, nonlinearity='leaky_relu')
#         self.layer_norm1 = nn.LayerNorm(embedding_dim)
#         self.e_fc1 = nn.Linear(embedding_dim, hidden_dim)
#         self.layer_norm2 = nn.LayerNorm(hidden_dim)
#         # nn.init.xavier_uniform_(self.e_fc1.weight, gain=1)
#         nn.init.kaiming_uniform_(self.e_fc1.weight, a=0.2, nonlinearity='leaky_relu')
#         self.final  = nn.Linear(hidden_dim, latent_dim)
#         self.layer_norm3 = nn.LayerNorm(latent_dim)
#         nn.init.xavier_uniform_(self.final.weight, gain=1)
#         # self.maxs   = nn.Linear (hidden_dim, latent_dim)
#         # nn.init.xavier_uniform_(self.maxs.weight, gain=1)

#         self.d_fc1 = nn.Linear(latent_dim, hidden_dim)
#         self.layer_norm4 = nn.LayerNorm(hidden_dim)
#         # nn.init.xavier_uniform_(self.d_fc1.weight, gain=1)
#         nn.init.kaiming_uniform_(self.d_fc1.weight, a=0.2, nonlinearity='leaky_relu')
#         self.d_fc2= nn.Linear(hidden_dim, output_dim)
#         nn.init.xavier_uniform_(self.d_fc2.weight, gain=1)
        
#         # self.LayerNorm = nn.LayerNorm(latent_dim)
#         self.activation = nn.LeakyReLU(0.2)
        
#     def encode(self, x):
#         x = self.embedding(x - 1)
#         x = self.layer_norm1(x)
#         x = self.activation(x)
#         x = self.e_fc1(x)
#         x = self.layer_norm2(x)
#         x = self.activation(x)
#         x = self.final(x)
#         x = self.layer_norm3(x)
#         # x = self.LayerNorm(x)
#         x = self.activation(x)
#         # maxs = self.maxs(x)

#         return x #mins, maxs
                
#     def decode(self, z):
#         z = self.d_fc1(z)
#         z = self.layer_norm4(z)
#         z = self.activation(z)
#         z = self.d_fc2(z)
#         return z
    
#     def predict_atom(self, z):
#         z = self.decode(z)
#         preds = torch.argmax(z, dim=1) + 1
#         return preds


#     def forward(self, x):
#         # mins, maxs = self.encode(x)

#         # z = self.reparameterization(mins, maxs)
#         z = self.encode(x)
#         preds = self.decode(z)
        
#         return preds #, mins, maxs



def get_z_ae_penalty(embedding_dim, hidden_dim, latent_dim, plot_atoms, location):

    atom_vae = AtomTypeAEPenalty(input_dim=MAX_ATOMIC_NUM, 
                                latent_dim=latent_dim, 
                                embedding_dim = embedding_dim,
                                hidden_dim=hidden_dim, 
                                output_dim=MAX_ATOMIC_NUM)
    atom_vae.load_state_dict(torch.load(location))
    atom_vae.eval()
    atom_vae = atom_vae.to('cuda:0')
    atom_vae.eval()
    with torch.no_grad():
        atom_types = atom_vae.encode(plot_atoms)
    return atom_types

def load_data(data_location):
        with open(data_location, 'rb') as f:
            crys_array_list_gt = pickle.load(f)

        atoms_gt = torch.tensor([])
        for crystal in crys_array_list_gt: #[::50]:
            # new = torch.tensor(crystal['lattice']).unsqueeze(0)
            # lattices = torch.concat([lattices, new], dim = 0)
            atoms_gt = torch.concat([atoms_gt, torch.tensor(crystal['atom_types'])])

        atoms_gt = atoms_gt.to(torch.int)


        plot_atoms = atoms_gt.to('cuda:0')
        return plot_atoms

def plot_data(data_location, ae_locations, save_location):
    # data = load_data(data_location)
    data = load_data(data_location)
    data = data.unique()

    fig, ax = plt.subplots(1, len(ae_locations), figsize=((len(ae_locations) - 1)*9, 8))
    c = np.linspace(0, 1, MAX_ATOMIC_NUM)
    for i, ae_location in enumerate(ae_locations):
        data_after_ae = get_z_ae_penalty(32, 16, 2, data, ae_location)
        ax[i].scatter(data_after_ae[:,0].to('cpu'), data_after_ae[:,1].to('cpu'), 
                cmap = 'hsv', c = data.to('cpu'), s=500, alpha = 0.4)
        ax[i].set_xlim(-1.2,1.2)
        ax[i].set_ylim(-1.2,1.2)
        ax[i].set_aspect('equal', 'box')
        ax[i].title.set_text(f'AE epoch {i*50} latent space')


        for i_atom, atom in enumerate(data):
            if (data_after_ae[i_atom,:] < 1).all() and (data_after_ae[i_atom,:] >-1).all():
                ax[i].text(data_after_ae[i_atom,0],
                            data_after_ae[i_atom,1], chemical_symbols[int(atom.item())], fontsize = 14,ha="center", va="center", alpha=1)


    plt.tight_layout()
    # colors = cm.rainbow(np.linspace(0, 1, len(data)))
    plt.savefig(save_location + '/ae_development.pdf')

    # plt.colorbar()

    # # Create a scatter plot
    # plt.figure(figsize=(8, 6))
    # for i, _ in enumerate(data):
        
    #     ax["A"].scatter(data_after_training[i,0].to('cpu'), data_after_training[i,1].to('cpu'), color=colors[i])
    #     ax["B"].scatter(data_before_training[i,0].to('cpu'), data_before_training[i,1].to('cpu'),color=colors[i])

    #     # plt.scatter(x, y, color=colors[i], label=f'Datapoint {i}')

    # plt.show()  
    # print('a')


def main(args):
    
    # data_location = os.path.join(args.model_path, args.gen_file)
    ae_locations = []
    for i in range(args.num_plots):
        ae_locations.append(args.ae_folder_location + f'ae_{i * 50}.pt')
    plot_data(args.data_location, ae_locations, args.ae_folder_location) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ae_folder_location')
    parser.add_argument('--data_location')
    parser.add_argument('--num_plots')



    args = parser.parse_args()

    args.data_location = '/home/m/Thesis/cdvae/cdvae/data/perov_5/perov_train.pickle' 
    
    # args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8ReworkedSumFracNotQIndividual_sde_5to01_aePenScratchNOGT_UniquePenNOStd_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAEXTRAL_MLPCEXTRAL_MLPLEXTRAL_costA1C1CE1STD1_aelr3_lr5/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8ReworkedSumFracNotQIndividual_FIXEDNOISE_sde_5to01_aePenScratchNOGT_Stop600_UniquePenNOStd_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C1CE1STD1_aelr3_lr5/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8ReworkedSumFracNotQIndividual_FIXEDNOISE_sde_5to01_aePenScratchNOGT_Stop600_UniquePenNOStd_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C1CE5NormPen5_aelr3_lr5/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8ReworkedSumFracNotQIndividual_FIXEDNOISE_sde_5to01_aePenScratchNOGT_Stop600_UniquePenNOStd_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C1CE1NormPen5_aelr53_lr5/'



    
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8ReworkedSumFracNotQIndividual_FIXEDNOISE_sde_5to01_aePenScratchNOGT_LossNOtOnUniqueDiversity_Stop600_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C1CE1NormPen5_aelr53_lr5/'
    
    
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8ReworkedSumFracNotQIndividual_FIXEDNOISE_sde_5255_1e3_aePenScratchNOGT_LossNOtOnUniqueDiversity_Stop600_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C1CE1NormPen5_aelr4_lr5/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8ReworkedSumFracNotQIndividual_FIXEDNOISE_sde_5255_1e3_aePenScratchNOGT_STD2_LossNOtOnUniqueDiversity_Stop600_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C1CE1STD6NormPen5_aelr4_lr5/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8RewSumFracNotQIndiv_sde_5255_1e3_aePenScratchNOGT_STD2Diversity_LossNOtOnUniqueDiversity_Stop600_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C1CE1STD6NormPen5_aelr4_lr5/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8RewSumFracNotQIndiv_sde_5255_1e3_aePenScratchNOGT_STD2Diversity_LossNOtOnUniqueDiversity_Stop600_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C1CE1STD6NormPen5_aelr3_lr4/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8RewSumFracNotQIndiv_sde_5255_1e3_aePenScratchNOGT_PenSTD1Diversity_CEAccFullData_Stop600_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C1CE1STD6NormPen5_aelr3_lr4/'
    
    
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8RewSumFracNotQIndiv_sde_5255_1e3_aePenScratchNOGT_PenDiversity_CEAccFullData_Stop1000Soft200_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C1CE1STD6NormPen5_aelr3_lr4/'
    # args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8RewSumFracNotQIndiv_sde_5255_1e3_aePenScratchNOGT_PenMeanNOTDiversity_CEAccFullData_Stop1000Soft200_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C1CE1STD6NormPen5_aelr3_lr4/'
    # args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8RewSumFracNotQIndiv_sde_5255_1e3_aePenScratchNOGT_PenMeanNOTDiversity_CEAccFullData_Stop1000Soft200_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C1CE1STD6NormPen5_ae52_lr4/'





    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8RewSumFracNotQIndiv_sde_5255_1e3_aePenScratchNOGT_PenSumNOTDiversity_CEAccFullData_Stop1000Soft200min01_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C5CE1STD6NormPen5_ae52_lr4/ae/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8RewSumFracNotQIndiv_sde_5255_1e3_aePenScratchNOGT_PenMeanNOTDiversity_CEAccFullData_Stop1000Soft200min01_PrimitiveStandard_PredCoordMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C5CE1STD6NormPen5_ae52_lr4/ae/'

    ### the frac coords loss is with - now so all langevin can be the same
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8RewSumFracNotQIndiv_sde_CoordLossMinus_5255_1e3_aePenScratchNOGT_PenMeanNOTDivers_CEAccFullData_Stop1000Soft200min01_PrimitiveStand_PredCMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C2CE1STD6NormPen5_ae52_lr4/ae/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_Separate_SinCos8RewSumFracNotQIndiv_sde_CoordLossMinus_5255_1e3_aePenScratchNOGT_PenMeanNOTDivers_CEAccFullData_Stop1000Soft200min01_PrimitiveStand_PredCMinus_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C5CE1NormPen5_ae52_lr4/ae/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_SeparateLA_SinCos8RewSumFracNotQIndiv_sde_CoordLossMinus_555_1e343_aePenScratchNOGT_PenMeanNOTDivers_CEAccFullData_Stop1000_PrimitiveStand_WOutParam_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C2CE1NormPen5_ae52_lr4/ae/'

    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_SeparateLA_SinCos8RewSumFracNotQIndiv_vpsde_CoordLossMinus_cVP001_1_aePenScratchNOGT_PenMeanNOTDivers_CEAccFullData_Stop1000_PrimitiveStand_WOutParam_tanh_SwiGLUTrue_3232_448_256_H4L8_MLPAE_MLPCE_MLPLE_costA1C2CE1NormPen5_ae52_lr4/ae/'
    
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_SeparateLA_SinCos8RewSumFracNotQIndiv_8_CoordLossMinus_555_1e343_aePenScratchNOGT_PenMeanNOTDivers_CEAccFullData_Stop600_PrimitiveStand_WOutLin_tanh_SwiGLUTrue_96_3232_448_256_H4L8_costA1C2CE1NormPen5_ae3_lr5_gradclip/ae/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_SeparateLA_SinCos8RewSumFracNotQIndiv_8_CoordLossMinus_555_1e343_aePenScratchNOGT_PenMeanNOTDivers_CEAccFullData_Stop600_PrimitiveStand_WOutLin_tanh_SwiGLUTrue_96_3232_448_256_H4L8_costA1C2CE1NormPen5_ae3_lr4_gradclip/ae/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_SeparateLA_SinCos8RewSumFracNotQIndiv_8_CoordLossMinus_555_1e343_aePenScratchNOGT_PenMeanNOTDivers_CEAccFullData_Stop600_PrimitiveStand_WOutLin_tanh_SwiGLUTrue_12_3232_448_256_H4L8_costA1C2CE1NormPen5_ae2_lr4_gradclip/ae/'
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_attention_server_haicore_random_seed1/perov_mini_SeparateLA_SinCos8RewSumFracNotQIndiv_8_CoordLossMinus_555_1e343_aePenScratchNOGT_PenMeanNOTDivers_CEAccFullData_Stop600_PrimitiveStand_WOutLin_tanh_SwiGLUTrue_96_3232_448_256_H4L8_costA1C2CE1NormPen5_ae2_lr4_gradclip/ae/'


    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/plot/perov_mini_SeparateLA_SinCos8RewSumFracNotQIndiv_8_centeredSwapTanhPerLossModified_tahn_555_1e343_aePenScratchNOGT_PenMeanNOTDivers_CEAccFullData_Stop600_PrimitiveStand_WOutLin_SwiGLUTrue_96_3232_448_256_H4L8_jointMLP4_costA1C2CE1NormPen5_ae2_lr5_gradclip/ae/'
    
    args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/plot/perov_mini_SeparateLA_SinCos8RewSumFracNotQIndiv_8_periodicMSE_tahn_555_1e343_aePenScratchNOGT_PenMeanNOTDivers_CEAccFullData_Stop600_PrimitiveStand_WOutLin_SwiGLUTrue_96_3232_448_256_H4L8_jointMLP4_costA1C10CE1NormPen5_ae2_lr4_gradclip/ae/'
    args.num_plots = len(glob.glob(os.path.join(args.ae_folder_location, '*.pt')))


    args.ae_folder_location = '/home/m/Thesis/cdvae/Best_models/ae/'

    args.data_location = '/home/m/Thesis/cdvae/cdvae/data/mp_20/mp_20_train.pickle' 
    
    args.num_plots = len(glob.glob(os.path.join(args.ae_folder_location, '*.pt'))) - 5



    # ###### UNIFIED
    # args.ae_folder_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/unified/mse/perov_mini_AttentionSinCosFix_Separate_8_MSE_AEScratchLayerNormAtt_10251_1e3RLY_IndividualLayerNormAtt_NoMLPNoRes_AttMLPBias_stop1000_H4L8_6464_448_256_jointMLP2_1051_1e343_a5c5CE1_alr2_l4/ae/'

    main(args)
