import argparse
import os
import pickle
from model.common.data_utils import cart_to_frac_coords, frac_to_cart_coords_lattice_matrix, get_pbc_distances, lattice_params_to_matrix, lattice_params_to_matrix_torch, radius_graph_pbc, lattice_matrix_to_params_torch

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

from pymatgen.core.structure import Structure


from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from collections import Counter

matplotlib.use('TkAgg')
from  model.common.data_utils import chemical_symbols

from scripts.eval_utils import smact_validity, structure_validity

def get_validity(structure, atom_types):
    elem_counter = Counter(atom_types)
    composition = [(elem, elem_counter[elem])
                    for elem in sorted(elem_counter.keys())]
    elems, counts = list(zip(*composition))
    counts = np.array(counts)
    counts = counts / np.gcd.reduce(counts)
    elems = elems
    comps = tuple(counts.astype('int').tolist())
    
    comp_valid = smact_validity(elems, comps)
    struct_valid = structure_validity(structure)
    return comp_valid and struct_valid

def visualize_data(generated_data_location, gt_data_location, sample_idx):
    with open(generated_data_location, 'rb') as f:
        crys_array_list = pickle.load(f)

    fig = plt.figure(figsize=(12,16))
    ax_1 = fig.add_subplot(5,4,1, projection='3d')
    ax_2 = fig.add_subplot(5,4,2, projection='3d')
    ax_3 = fig.add_subplot(5,4,3, projection='3d')
    ax_4 = fig.add_subplot(5,4,4, projection='3d')

    ax_5 = fig.add_subplot(5,4,5, projection='3d')
    ax_6 = fig.add_subplot(5,4,6, projection='3d')
    ax_7 = fig.add_subplot(5,4,7, projection='3d')
    ax_8 = fig.add_subplot(5,4,8, projection='3d')

    ax_1_1 = fig.add_subplot(5,4,9, projection='3d')
    ax_2_1 = fig.add_subplot(5,4,10, projection='3d')
    ax_3_1 = fig.add_subplot(5,4,11, projection='3d')
    ax_4_1 = fig.add_subplot(5,4,12, projection='3d')

    ax_5_1 = fig.add_subplot(5,4,13, projection='3d')
    ax_6_1 = fig.add_subplot(5,4,14, projection='3d')
    ax_7_1 = fig.add_subplot(5,4,15, projection='3d')
    ax_8_1 = fig.add_subplot(5,4,16, projection='3d')

    
    
    ax_1_2 = fig.add_subplot(5,4,17, projection='3d')
    ax_2_2 = fig.add_subplot(5,4,18, projection='3d')
    ax_3_2 = fig.add_subplot(5,4,19, projection='3d')
    ax_4_2 = fig.add_subplot(5,4,20, projection='3d')

    

    ax = [ax_1, ax_2, ax_3, ax_4, 
          ax_5, ax_6, ax_7, ax_8,
          ax_1_1, ax_2_1, ax_3_1, ax_4_1,
          ax_5_1, ax_6_1, ax_7_1, ax_8_1,
          ax_1_2, ax_2_2, ax_3_2, ax_4_2]

    i = 0
    for crystal in crys_array_list:
        if i == 20:
            break
        try:
            # print(self.lattice)
            # print(self.atom_types)
            # print(self.frac_coords)
            structure = Structure(
                lattice=crystal['lattice'],
                species=crystal['atom_types'], 
                coords=crystal['frac_coords'], 
                coords_are_cartesian=False)
            
            if get_validity(structure, crystal['atom_types']):


                ### do it only for valid crystals!
                lattice = torch.Tensor(crystal['lattice'])
                frac_coords = torch.Tensor(crystal['frac_coords'])
                atom_types = torch.Tensor(crystal['atom_types'])


                pos = frac_to_cart_coords_lattice_matrix(frac_coords, lattice.unsqueeze(0), frac_coords.shape[0])


                draw_3d_pos(atom_types,pos, lattice, ax[i])
                i += 1
        except Exception:
            print("invalid crystal")
        
        

    plt.savefig(generated_data_location[:-7] + 'generation_perovNoWIn_3D.pdf')
    plt.tight_layout()

    # plt.show()
    print("")


def draw_3d_pos_single(atom_type, pos, lattice, ax,  color_nodes = 'r', color_lattice = 'b'):



    # Plot nodes
    colors = np.linspace(0, 1, len(chemical_symbols))

    ax.scatter(pos[0], pos[1], pos[2]);#, c=colors[atom_type], s=200);

    
    ax.text(pos[0], pos[1], pos[2], chemical_symbols[int(atom_type)], color='black', size=15);


    ax.quiver(0, 0, 0, lattice[0,0], lattice[0,1], lattice[0,2], color=color_lattice, linewidths=1, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, lattice[1,0], lattice[1,1], lattice[1,2], color=color_lattice, linewidths=1, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, lattice[2,0], lattice[2,1], lattice[2,2], color=color_lattice, linewidths=1, arrow_length_ratio=0.1)

    min_axis = torch.tensor([lattice.min(),pos.min()]).min()
    max_axis = torch.tensor([lattice.max(),pos.max()]).max()

    ax.set_xlim([min_axis,max_axis])
    ax.set_ylim([min_axis,max_axis])
    ax.set_zlim([min_axis,max_axis])
    

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

            


def draw_3d_pos(atom_types, nodes, lattice, ax,  color_nodes = 'r', color_lattice = 'b'):

    nodes = nodes.to('cpu')

    x = [node[0] for node in nodes]
    y = [node[1] for node in nodes]
    z = [node[2] for node in nodes]


    # Plot nodes
    colors = np.linspace(0, 1, len(chemical_symbols))

    ax.scatter(x, y, z, c=[colors[int(atom_type)] for atom_type in atom_types], s=200);

    
    # Enumerate and label each node
    for i, (node, atom_type) in enumerate(zip(nodes, atom_types)):
        ax.text(node[0], node[1], node[2], chemical_symbols[int(atom_type)], color='black', size=15);


    ax.quiver(0, 0, 0, lattice[0,0], lattice[0,1], lattice[0,2], color=color_lattice, linewidths=1, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, lattice[1,0], lattice[1,1], lattice[1,2], color=color_lattice, linewidths=1, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, lattice[2,0], lattice[2,1], lattice[2,2], color=color_lattice, linewidths=1, arrow_length_ratio=0.1)

    min_axis = torch.tensor([lattice.min(),nodes.min()]).min()
    max_axis = torch.tensor([lattice.max(),nodes.max()]).max()
    ## Set the limits of the plot
    ax.set_xlim([min_axis,max_axis])
    ax.set_ylim([min_axis,max_axis])
    ax.set_zlim([min_axis,max_axis])
    
    # Set labels and display the graph
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    
    # ax.legend()


def main(args):
    
    # data_location = os.path.join(args.model_path, args.gen_file)
    visualize_data(args.pickle_location, args.data_location, args.sample_idx) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')#, required=True)
    parser.add_argument('--gen_file')#, required=True)
    parser.add_argument('--pickle_location')

    parser.add_argument('--sample_idx')
    # parser.add_argument('--compute_energy', default=False, type=lambda z: z=='True')


    args = parser.parse_args()

    main(args)


