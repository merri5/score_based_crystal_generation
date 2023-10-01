import argparse
import pickle
# from matplotlib import figure as fig
import numpy as np
import torch

import matplotlib.pyplot as plt

def plot_data(pickle_location, data_location):
        
    with open(pickle_location, 'rb') as f:
        crys_array_list = pickle.load(f)
        # crys_array_list, _ = get_crystal_array_list(data_location)
        # gen_crys = p_map(lambda x: Crystal(x), crys_array_list)

        # for i in gen_crys[0:10]:
        #     print(i.lattice, i.atom_types, i.frac_coords)

        # crystal = gen_crys[sample_idx]

        
    for i in crys_array_list[0:20]:
        print(i['lattice'], i['atom_types'], i['frac_coords'])



    lattices = torch.tensor([])
    atoms = torch.tensor([])
    coords = torch.tensor([])
    for crystal in crys_array_list:
        new = torch.tensor(crystal['lattice']).unsqueeze(0)
        lattices = torch.concat([lattices, new], dim = 0)
        atoms = torch.concat([atoms, torch.tensor(crystal['atom_types'])])
        coords = torch.concat([coords, torch.tensor(crystal['frac_coords'])])

        # print(crystal['frac_coords'])
        # print(crystal['atom_types'])
        # print(crystal['lattice'])
        # print('\n\n')
    # print(lattices[0:5].shape)
    unique = torch.unique(lattices, dim=0) 
    # print(unique)
    print("Unique lattices: ", unique.shape[0], ' / ', len(crys_array_list))
    # print(atoms)


    # with open('/home/m/Thesis/explore_learning/perov/perov_train.pickle', 'rb') as f:
    #     crys_array_list_gt = pickle.load(f)

    # with open('/hkfs/work/workspace_haic/scratch/ga8707-thesis/cdvae/data/perov_5_minier1/perov_5_minier1_train.pickle', 'rb') as f:
    with open(data_location, 'rb') as f:
        crys_array_list_gt = pickle.load(f)

    atoms_gt = torch.tensor([])
    coords_gt = torch.tensor([])
    lattices_gt = torch.tensor([])
    for crystal in crys_array_list_gt:
        new = torch.tensor(crystal['lattice']).unsqueeze(0)
        lattices_gt = torch.concat([lattices_gt, new], dim = 0)
        atoms_gt = torch.concat([atoms_gt, torch.tensor(crystal['atom_types'])])
        coords_gt = torch.concat([coords_gt, torch.tensor(crystal['frac_coords'])])

    # print(coords_gt[:200])
    # fig, axs = plt.subplots(3, 3, tight_layout=True)
    fig = plt.figure(figsize=(15,10),layout="constrained")
    # ax = fig.subplot_mosaic("""AAA
    #                         BCD""")
    ax = fig.subplot_mosaic("""AAAAAA
                            BBCCDD
                            EFGHIJ""")
    
    # ax['A'].hist(atoms, bins=100, alpha = 0.5, lw=3,color= 'r', label = "Generated atoms")
    # ax['A'].hist(atoms_gt, bins=100, alpha = 0.5, lw=3, color= 'b', label = "GT atoms")

    fig.suptitle(pickle_location.split('/')[-1][:-7])

    ax['A'].hist([atoms, atoms_gt], bins = 100, alpha =0.5, label = ["Generated atoms", "GT atoms"], density=True)#, histtype="step")
    ax['B'].hist([coords[:,0], coords_gt[:,0]], bins = 50, alpha =0.5, label = ["Generated coords x", "GT coords x"], density=True)#, histtype="step")
    ax['C'].hist([coords[:,1], coords_gt[:,1]], bins = 50, alpha =0.5, label = ["Generated coords y", "GT coords y"], density=True)#, histtype="step")
    ax['D'].hist([coords[:,2], coords_gt[:,2]], bins = 50, alpha =0.5, label = ["Generated coords z", "GT coords z"], density=True)#, histtype="step")

    ax['E'].hist([lattices[:,0,0], lattices_gt[:,0,0]], bins = 50, alpha =0.5, label = ["Generated lattices x", "GT lattices x"], density=True)#, histtype="step")
    ax['F'].hist([lattices[:,0,2], lattices_gt[:,0,2]], bins = 50, alpha =0.5, label = ["Generated lattices y", "GT lattices y"], density=True)#, histtype="step")
    ax['G'].hist([lattices[:,1,0], lattices_gt[:,1,0]], bins = 50, alpha =0.5, label = ["Generated lattices z", "GT lattices z"], density=True)#, histtype="step")
    ax['H'].hist([lattices[:,1,1], lattices_gt[:,1,1]], bins = 50, alpha =0.5, label = ["Generated lattices x", "GT lattices x"], density=True)#, histtype="step")
    ax['I'].hist([lattices[:,1,2], lattices_gt[:,1,2]], bins = 50, alpha =0.5, label = ["Generated lattices y", "GT lattices y"], density=True)#, histtype="step")
    ax['J'].hist([lattices[:,2,2], lattices_gt[:,2,2]], bins = 50, alpha =0.5, label = ["Generated lattices z", "GT lattices z"], density=True)#, histtype="step")
    
    

    # plt.hist(atoms, color='lightblue', ec='black', bins=100)
    ax['A'].legend()
    ax['B'].legend()
    ax['C'].legend()
    ax['D'].legend()
    
    
    plt.savefig(pickle_location[:-7] + '_train_data.png')
    # plt.show()

    # fig, axs = plt.subplots(2, 2, tight_layout=True)


    # # We can set the number of bins with the *bins* keyword argument.
    # axs[0][0].hist(atoms, bins=100)
    # axs[0][0].set_title('Generated atom types')
    # axs[0][1].hist(atoms_gt, bins=100)
    # axs[0][1].set_title('GT atom types')

    # # plt.hist(atoms, color='lightblue', ec='black', bins=100)
    # plt.show()



def main(args):
    
    # data_location = os.path.join(args.model_path, args.gen_file)
    plot_data(args.pickle_location, args.data_location) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_location')

    parser.add_argument('--data_location')


    args = parser.parse_args()

        
    #####one datapoint
    args.pickle_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/one_datapoint_complex_clean_models_random_seed1/perov_minier1_notPeriodicJoint_NOsde_NotPediodicLoss_aeNotGT_NoNumAtomsEmb_Emb64H4L4_SwiGLUFalse_MLPATrue_MLPCTrue_MLPLTrue_aelr0001_lr0001/noclampFrac_divSigmaFrac_a4_c4_l4_aU_cU_lN_test_generated_crystals.pickle'
    args.pickle_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/one_datapoint_complex_clean_models_random_seed1/perov_minier1_notPeriodicJoint_NOsde_NotPediodicLoss_aeNotGT_NoNumAtomsEmb_Emb16H1L2_SwiGLUFalse_MLPATrue_MLPCTrue_MLPLTrue_aelr0001_lr0001/noclampFrac_divSigmaFrac_a6_c10_l6_aU_cU_lN_test_generated_crystals.pickle'


    
    ########## complex_periodic_clean_models_attention_server_random_seed1
    args.pickle_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_periodic_clean_models_attention_server_random_seed1/perov_mini_JointTanh_NOsde_aeGT_COORDSx10_periodicLossSigma_NoNumAtomsEmb_Emb64H1L2_HidC12HidAtt512HidMLP256_SwiGLUTrue_MLPATrue_MLPCTrue_MLPLTrue_costA1C5CE01_aelr0001_lr0001/epoch_2029_noclampFrac_divSigmaFrac_a8_c8_l8_aU_cU_lN_test_generated_crystals.pickle'
    args.pickle_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_periodic_clean_models_attention_server_random_seed1/perov_mini_JointTanh_NOsde_aeGT_COORDSx10_periodicLossSigma_NoNumAtomsEmb_Emb64H1L2_HidC12HidAtt512HidMLP256_SwiGLUTrue_MLPATrue_MLPCTrue_MLPLTrue_costA1C5CE01_aelr0001_lr0001/epoch_2029_noclampFrac_divSigmaFrac_a8_c8_l8_aU-11_cU010_lN_test_generated_crystals.pickle'
    args.pickle_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_periodic_clean_models_attention_server_random_seed1/perov_mini_JointTanh_NOsde_aeGT_COORDSx10_periodicLossSigma_NoNumAtomsEmb_Emb64H1L2_HidC12HidAtt512HidMLP256_SwiGLUTrue_MLPATrue_MLPCTrue_MLPLTrue_costA1C5CE01_aelr0001_lr0001/epoch_2029_wrapFrac10_divSigmaFrac_a5_c5_l5_aU-11_cU010_lN_test_generated_crystals.pickle'
    

    ########## Models showing potential, INVESTIGATE MORE
    args.pickle_location ='/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_hybrid/perov_mini_test_sde_continuous_new_FracScale_coordScaleCost2_ae2GTlr3_8_H256_Emb3232_MLPatc_Norm_noInit_Adam4/ATclampStep_SDEtest_divideSigma_snr222_newEvalCode_annealed_predictor_generated_crystals.pickle'
    args.pickle_location ='/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_hybrid/perov_mini_test_sde_continuous_new_FracScale_coordScaleCost2_aePenaltyGTlr3_8_H256H4L4_Emb3232_MLPatc_Norm_noInit_Adam4/ATclampStep_SDEtest_divideSigma_snr050505_newEvalCode_annealed_predictor_generated_crystals.pickle'
    
    
    args.pickle_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_hybrid/perov_mini_test_sde_continuous_new_FracScale_coordScaleCost2_aePenaltyGTlr3_8_H256H4L4_Emb3232_MLPatc_Norm_noInit_Adam4/ATclampStep_SDEtest_divideSigma_snr111_newEvalCode_annealed_predictor_generated_crystals.pickle'
    args.data_location = '/home/m/Thesis/cdvae/cdvae/data/perov_5_mini/perov_5_mini_train.pickle'
    

    args.pickle_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/complex_hybrid/perov_mini_test_sde_continuous_new_FracScale_coordScaleCost2_ae2GTlr3_8_H256_Emb3232_MLPatc_Norm_noInit_Adam4/ATclampStep_SDEtest_divideSigma_snr120101_newEvalCode_annealed_predictor_generated_crystals.pickle'
    args.data_location = '/home/m/Thesis/cdvae/cdvae/data/perov_5_mini/perov_5_mini_train.pickle'


    args.pickle_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/sde_attention/not_periodic_complex_clean_models_random_seed1/perov_minier1_NOsde_NotPeriodic_MSE_AENOTrain_tanh_201105_NoNumAtomsEmb_3232_512256_H4L4_SwiGLUTrue_MLPATrue_MLPCTrue_MLPLTrue_costA1C50CE01_aelr0001_lr0001_1/2000_noWrapFrac_scaleSigma_divSigmaFrac_a9_c9_l9_FROMDATAPOINT_generated_crystals.pickle'
    args.data_location = '/home/m/Thesis/cdvae/cdvae/data/perov_5_minier1/perov_5_minier1_train.pickle'




    main(args)
