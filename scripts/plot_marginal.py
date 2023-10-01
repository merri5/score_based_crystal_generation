import argparse
import pickle
from model.pl_modules.embeddings import MAX_ATOMIC_NUM
from  model.common.data_utils import chemical_symbols
# from matplotlib import figure as fig
import numpy as np
import torch

import matplotlib.pyplot as plt

def plot_data(pickle_location, data_location):

    

    # fig, axs = plt.subplots(3, 3, tight_layout=True)
    fig = plt.figure(figsize=(15,8),layout="constrained")
    # ax = fig.subplot_mosaic("""AAA
    #                         BCD""")
    ax = fig.subplot_mosaic("""AAAAAA
                            BBCCDD
                            EFGHIJ""")
    

    fig.suptitle("Marginal distributions", fontsize=14)
        
    with open(pickle_location, 'rb') as f:
        crys_array_list = pickle.load(f)
        
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


    atoms = np.array(atoms)
    unique, counts = np.unique(atoms, return_counts=True)
    counts = counts/ len(atoms)
    ax['A'].bar(unique-0.2, counts,width=0.4, color ="#B3D5F5",align='center', edgecolor=None, label='Generated atoms')
    
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

    
    atoms_gt = np.array(atoms_gt)
    unique_gt, counts_gt = np.unique(atoms_gt, return_counts=True)
    counts_gt = counts_gt/ len(atoms_gt)

    ax['A'].bar(unique_gt+0.2, counts_gt,width=0.4, color =  "#FDAB77",align='center', edgecolor=None, label='GT atoms')



    ax['A'].set_xticks([i for i in range(0,MAX_ATOMIC_NUM)])
    ax['A'].set_xticklabels(chemical_symbols[:MAX_ATOMIC_NUM ], rotation=90, ha='center')

    ax['B'].hist([coords[:,0], coords_gt[:,0]], bins = 50,color=["#B3D5F5","#FDAB77"], alpha =1, label = ["Generated coords $x$", "GT coords $x$"], density=True)#, histtype="step")
    ax['C'].hist([coords[:,1], coords_gt[:,1]], bins = 50,color=["#B3D5F5","#FDAB77"], alpha =1, label = ["Generated coords $y$", "GT coords $y$"], density=True)#, histtype="step")
    ax['D'].hist([coords[:,2], coords_gt[:,2]], bins = 50,color=["#B3D5F5","#FDAB77"], alpha =1, label = ["Generated coords $z$", "GT coords $z$"], density=True)#, histtype="step")
    ax['B'].set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax['B'].set_xticklabels([0, 0.25, 0.5, 0.75, 1], ha='center')
    ax['C'].set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax['C'].set_xticklabels([0, 0.25, 0.5, 0.75, 1], ha='center')
    ax['D'].set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax['D'].set_xticklabels([0, 0.25, 0.5, 0.75, 1], ha='center')

    ax['E'].hist([lattices[:,0,0], lattices_gt[:,0,0]],color=["#B3D5F5","#FDAB77"], bins = 25, alpha =1, label = ["Generated $l_1$", "GT $l_1$"], density=True)#, histtype="step")
    ax['F'].hist([lattices[:,0,2], lattices_gt[:,0,2]],color=["#B3D5F5","#FDAB77"], bins = 25, alpha =1, label = ["Generated $l_2$", "GT $l_2$"], density=True)#, histtype="step")
    ax['G'].hist([lattices[:,1,0], lattices_gt[:,1,0]],color=["#B3D5F5","#FDAB77"], bins = 25, alpha =1, label = ["Generated $l_3$", "GT $l_3$"], density=True)#, histtype="step")
    ax['H'].hist([lattices[:,1,1], lattices_gt[:,1,1]],color=["#B3D5F5","#FDAB77"], bins = 25, alpha =1, label = ["Generated $l_4$", "GT $l_4$"], density=True)#, histtype="step")
    ax['I'].hist([lattices[:,1,2], lattices_gt[:,1,2]],color=["#B3D5F5","#FDAB77"], bins = 25, alpha =1, label = ["Generated $l_5$", "GT $l_5$"], density=True)#, histtype="step")
    ax['J'].hist([lattices[:,2,2], lattices_gt[:,2,2]],color=["#B3D5F5","#FDAB77"], bins = 25, alpha =1, label = ["Generated $l_5$", "GT $l_5$"], density=True)#, histtype="step")
    
    

    ax['A'].legend()
    ax['B'].legend()
    ax['C'].legend()
    ax['D'].legend()
    ax['E'].legend()
    ax['F'].legend()
    ax['G'].legend()
    ax['H'].legend()
    ax['I'].legend()
    ax['J'].legend()
    
    
    plt.savefig(pickle_location[:-7] + '_train_data.pdf')
    plt.show()



def main(args):
    
    plot_data(args.pickle_location, args.data_location) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_location')

    parser.add_argument('--data_location')


    args = parser.parse_args()
    main(args)
