import pickle
import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from scripts.eval_utils import load_model


def generation(model, test_loader, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    all_lattices_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []

    for idx, batch in enumerate(test_loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f'batch {idx} in {len(test_loader)}')
        
        # Save the ground truth structure
        input_data_list = input_data_list + batch.to_data_list()
    input_data_batch = Batch.from_data_list(input_data_list)

    for z_idx in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_all_lattices = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattice = []

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(batch_size, ld_kwargs)
            # samples = model.langevin_dynamics(z, ld_kwargs)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lattice.append(samples['lattice'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu())
                batch_all_lattices.append(
                    samples['all_lattices'][::down_sample_traj_step].detach().cpu())


        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattice, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))
            all_lattices_stack.append(
                torch.stack(batch_all_lattices, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
        all_lattices_stack = torch.cat(all_lattices_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lattices,
            all_frac_coords_stack, all_atom_types_stack, all_lattices_stack, input_data_batch)


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=('recon' in args.tasks) or ('gen' in args.tasks) or
        ('opt' in args.tasks and args.start_from == 'data'), min_max_scaler=True)
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr_atom_types=args.step_lr_atom_types,
                                step_lr_coords=args.step_lr_coords,
                                step_lr_lattice=args.step_lr_lattice,
                                langevin=args.langevin,
                                use_predictor=args.use_predictor,
                                target_snr_atom_types=args.target_snr_atom_types,
                                target_snr_frac_coords=args.target_snr_frac_coords,
                                target_snr_lattice=args.target_snr_lattice,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    if torch.cuda.is_available():
        model.to('cuda')

    if 'recon' in args.tasks:
        print('Evaluate model on the reconstruction task.')
        start_time = time.time()
        (frac_coords, num_atoms, atom_types, lattice, #lengths, angles,
         all_frac_coords_stack, all_atom_types_stack, input_data_batch) = reconstructon(
            test_loader, model, ld_kwargs, args.num_evals,
            args.force_num_atoms, args.force_atom_types, args.down_sample_traj_step)

        if args.label == '':
            recon_out_name = 'eval_recon.pt'
        else:
            recon_out_name = f'eval_recon_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lattice': lattice,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / recon_out_name)

    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        start_time = time.time()

        (frac_coords, num_atoms, atom_types, lattice,
         all_frac_coords_stack, all_atom_types_stack, all_lattices_stack, input_data_batch) = generation(
            model, test_loader, ld_kwargs, args.num_batches_to_samples, args.num_evals,
            args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_gen.pt'
        else:
            gen_out_name = f'eval_gen_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lattice': lattice,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'all_lattices_stack': all_lattices_stack,
            'time': time.time() - start_time
        }, model_path / gen_out_name)
        
        traj_name = f'{args.label}_trajectories.pickle'
        with open(model_path / traj_name, 'wb') as f:
            pickle.dump(
                {'all_frac_coords_stack': all_frac_coords_stack,
                'all_atom_types_stack': all_atom_types_stack,
                'all_lattices_stack': all_lattices_stack}, f)


    if 'opt' in args.tasks:
        print('Evaluate model on the property optimization task.')
        start_time = time.time()
        if args.start_from == 'data':
            loader = test_loader
        else:
            loader = None
        optimized_crystals = optimization(model, ld_kwargs, loader)
        optimized_crystals.update({'eval_setting': args,
                                   'time': time.time() - start_time})

        if args.label == '':
            gen_out_name = 'eval_opt.pt'
        else:
            gen_out_name = f'eval_opt_{args.label}.pt'
        torch.save(optimized_crystals, model_path / gen_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='') #required=True)
    parser.add_argument('--tasks', nargs='+', default=['gen'])
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr_atom_types', default=1e-8, type=float)
    parser.add_argument('--step_lr_coords', default=1e-8, type=float)
    parser.add_argument('--step_lr_lattice', default=1e-8, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=True, type=lambda z: z=='True')
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=1, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=1, type=int)
    parser.add_argument('--label', default='')
    parser.add_argument('--langevin', default='annealed')
    parser.add_argument('--use_predictor', default=True, type=lambda z: z=='True')
    parser.add_argument('--target_snr_atom_types', default=0.1, type=float)
    parser.add_argument('--target_snr_frac_coords', default=0.1, type=float)
    parser.add_argument('--target_snr_lattice', default=0.1, type=float)

    args = parser.parse_args()

    main(args)
