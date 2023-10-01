import argparse
from collections import Counter
import pickle
from model.pl_modules.embeddings import MAX_ATOMIC_NUM
from  model.common.data_utils import chemical_symbols
# from matplotlib import figure as fig
import numpy as np
import torch

import matplotlib.pyplot as plt

def plot_data(perov, mp):
     # print(coords_gt[:200])
    # fig, axs = plt.subplots(3, 3, tight_layout=True)
    fig = plt.figure(figsize=(15,5),layout="constrained")
    # ax = fig.subplot_mosaic("""AAA
    #                         BCD""")
    ax = fig.subplot_mosaic("""A""")


    with open(perov, 'rb') as f:
        crys_array_list_gt = pickle.load(f)

    atoms_gt = torch.tensor([])
    coords_gt = torch.tensor([])
    lattices_gt = torch.tensor([])
    for crystal in crys_array_list_gt:
        atoms_gt = torch.concat([atoms_gt, torch.tensor(crystal['atom_types'])])

    atoms_gt = np.array(atoms_gt)
    unique_perov, counts_perov = np.unique(atoms_gt, return_counts=True)
    counts_perov = counts_perov/len(atoms_gt)

    ax['A'].bar(unique_perov-0.2, counts_perov,width=0.4, color = "#B3D5F5",align='center', edgecolor=None,label='Perov-5')
 
    with open(mp, 'rb') as f:
        crys_array_list_gt = pickle.load(f)

    atoms_gt = torch.tensor([])
    coords_gt = torch.tensor([])
    lattices_gt = torch.tensor([])
    for crystal in crys_array_list_gt:
        atoms_gt = torch.concat([atoms_gt, torch.tensor(crystal['atom_types'])])

    atoms_gt = np.array(atoms_gt)
    unique_mp, counts_mp = np.unique(atoms_gt, return_counts=True)
    counts_mp = counts_mp/ len(atoms_gt)
    ax['A'].bar(unique_mp+0.2, counts_mp,width=0.4, color = "#FDAB77",align='center', edgecolor=None, label='MP-20')


    ax['A'].set_xticks([i for i in range(0,MAX_ATOMIC_NUM)])
    ax['A'].set_xticklabels(chemical_symbols[:MAX_ATOMIC_NUM ], rotation=90, ha='center')

    ax['A'].set_xlabel("Atom type distribution", fontsize=14)
    ax['A'].set_ylabel("Density", fontsize=14)
    ax['A'].set_title("Atom type density in the training data")

    ax['A'].legend( fontsize=14)

    plt.show()

def main(args):
    
    plot_data(args.perov, args.mp) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--perov')

    parser.add_argument('--mp')


    args = parser.parse_args()



    main(args)
