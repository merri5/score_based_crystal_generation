import argparse
import pickle
# from matplotlib import figure as fig
import numpy as np
import torch

import matplotlib.pyplot as plt

def plot_data(pickle_location):
        
    with open(pickle_location, 'rb') as f:
        crys_array_list = pickle.load(f)
        
    fig = plt.figure(figsize=(15,10),layout="constrained")
    ax = fig.subplot_mosaic("""AB
                            CC
                            DD
                            EE""")

    for i in range(0,10):
        ax['A'].plot(crys_array_list['all_atom_types_stack'][0, :, i, 0])
        ax['B'].plot(crys_array_list['all_atom_types_stack'][0, :, i, 1])


        ax['C'].plot(crys_array_list['all_frac_coords_stack'][0, :, i, 0], alpha=0.6)
        ax['D'].plot(crys_array_list['all_frac_coords_stack'][0, :, i, 1], alpha=0.6)
        ax['E'].plot(crys_array_list['all_frac_coords_stack'][0, :, i, 2], alpha=0.6)


    plt.savefig(pickle_location[:-7] + '.png')


def main(args):
    
    plot_data(args.pickle_location) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_location')


    args = parser.parse_args()

    main(args)
