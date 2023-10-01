import argparse
import pickle
# from matplotlib import figure as fig
import numpy as np
import torch

import matplotlib.pyplot as plt

def plot_data(pickle_location, data_location):
        
    with open(pickle_location, 'rb') as f:
        crys_array_list = pickle.load(f)

    lattices = torch.tensor([])
    atoms = torch.tensor([])
    coords = torch.tensor([])
    for crystal in crys_array_list:
        new = torch.tensor(crystal['lattice']).unsqueeze(0)
        lattices = torch.concat([lattices, new], dim = 0)
        atoms = torch.concat([atoms, torch.tensor(crystal['atom_types'])])
        coords = torch.concat([coords, torch.tensor(crystal['frac_coords'])])

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

    
    fig = plt.figure(figsize=(15,10),layout="constrained")
    ax = fig.subplot_mosaic("""AAAAAA
                            BBCCDD
                            EFGHIJ""")
    

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
    
    
    plt.savefig(pickle_location[:-7] + '.png')

def main(args):
    
    # data_location = os.path.join(args.model_path, args.gen_file)
    plot_data(args.pickle_location, args.data_location) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_location')

    parser.add_argument('--data_location')

    args = parser.parse_args()


    main(args)
