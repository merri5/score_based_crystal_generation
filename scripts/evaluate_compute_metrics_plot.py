
import argparse

import subprocess
from model.common.utils import PROJECT_ROOT


import os

def main(args):
    
    generated_crystals_location = os.path.join(args.model_path, args.label + '_generated_crystals.pickle')

    traj_location = os.path.join(args.model_path, args.label + '_trajectories.pickle')

    ae_location = os.path.join(args.model_path, 'ae/')

    # command = f"python {PROJECT_ROOT}/scripts/evaluate_joint.py --model_path {args.model_path} --label {args.label} --target_snr_atom_types {args.target_snr_atom_types} --target_snr_frac_coords {args.target_snr_frac_coords} --target_snr_lattice {args.target_snr_lattice} --step_lr_atom_types {float(args.step_lr_atom_types)} --step_lr_coords {float(args.step_lr_coords)} --step_lr_lattice {float(args.step_lr_lattice)}"
    # subprocess.run(command, shell=True)

    # command = f"python {PROJECT_ROOT}/scripts/compute_metrics_joint.py --root_path {args.model_path} --label {args.label}"
    # subprocess.run(command, shell=True)

    command = f"python {PROJECT_ROOT}/scripts/read_generated_crystals_pickle_joint.py --pickle_location {generated_crystals_location} --data_location {args.data_location}"
    subprocess.run(command, shell=True)


    command = f"python {PROJECT_ROOT}/scripts/plot_traj_joint.py --pickle_location {traj_location}"
    subprocess.run(command, shell=True)
    
    command = f"python {PROJECT_ROOT}/scripts/plot_ae_traj.py --ae_folder_location {ae_location} --data_location {args.data_location}"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--label')
    parser.add_argument('--data_location')
    parser.add_argument('--scripts_path')


    parser.add_argument('--step_lr_atom_types', default=1e-8, type=float)
    parser.add_argument('--step_lr_coords', default=1e-8, type=float)
    parser.add_argument('--step_lr_lattice', default=1e-8, type=float)
    parser.add_argument('--target_snr_atom_types', default=0.1, type=float)
    parser.add_argument('--target_snr_frac_coords', default=0.1, type=float)
    parser.add_argument('--target_snr_lattice', default=0.1, type=float)

    args = parser.parse_args()


    args.data_location = '/home/m/Thesis/thesis/score_based_crystal_generation/data/perov_5/perov_5_train.pickle'

    args.target_snr_atom_types = 0.3
    args.target_snr_frac_coords = 0.3
    args.target_snr_lattice = 0.3

    args.model_path = '/home/m/Thesis/thesis/hydra/singlerun/sde_attention/w_fixed/perov'
    # args.scripts_path = '/home/m/Thesis/thesis/score_based_crystal_generation/scripts'

    
    args.label = 'test_gen'
    main(args)