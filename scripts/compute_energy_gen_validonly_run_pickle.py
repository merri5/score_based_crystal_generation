import argparse
from collections import Counter
import pickle
import numpy as np
from p_tqdm import p_map
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

from m3gnet.models import M3GNetCalculator
from m3gnet.models import Potential
from m3gnet.models import M3GNet

import pprint

import matplotlib.pyplot as plt
import os

from scripts.compute_metrics import Crystal,Crystal_lengths_angles, get_crystal_array_list, get_crystal_array_list_lengths_angles
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
    # print(comp_valid and struct_valid)
    return comp_valid and struct_valid
    
def compute_energy(pickle_location):
    with open(pickle_location, 'rb') as f:
        gen_crys = pickle.load(f)
    # crys_array_list, _ = get_crystal_array_list(data_location)
    # gen_crys = p_map(lambda x: Crystal(x), crys_array_list)

    # crys_array_list, _ = get_crystal_array_list_lengths_angles(data_location)
    # gen_crys = p_map(lambda x: Crystal_lengths_angles(x), crys_array_list)
    
    energy_list = []
    i = 0

    while len(energy_list)<50 and i < len(gen_crys):
        print(i, len(energy_list), len(gen_crys))
        # print(crystal['atom_types'])
        # print(crystal['frac_coords'])
        # print(crystal['lattice'])
        crystal = gen_crys[i]
        try:
            # print(self.lattice)
            # print(self.atom_types)
            # print(self.frac_coords)
            structure = Structure(
                lattice=crystal['lattice'],
                species=crystal['atom_types'], 
                coords=crystal['frac_coords'], 
                coords_are_cartesian=False)
            
        except Exception:
            print("invalid crystal")
        
        if get_validity(structure, crystal['atom_types']):

        # structure = Structure(
        #         lattice=crystal['lattice'],
        #         species=crystal['atom_types'], 
        #         coords=crystal['frac_coords'], 
        #         coords_are_cartesian=False)
        # structure = Structure(
        #             lattice=Lattice.from_parameters(
        #                 *(crystal['lengths'].tolist() + crystal['angles'].tolist())), #  element['lattices'],
        #             species=crystal['atom_types'], coords=crystal['frac_coords'], coords_are_cartesian=False)
            ase_adaptor = AseAtomsAdaptor()

            atoms = ase_adaptor.get_atoms(structure)
            # print("\n\n\n ATOMS OLD \n\n\n",atoms)
            m3gnet = M3GNet.load() #.to('cuda:0')
            potential = Potential(m3gnet)
            atoms.calc = M3GNetCalculator(potential=potential)

            energy = atoms.get_potential_energy()
            print("\n energy \n",energy)
            energy_list.append(energy)

        i+=1

    return energy_list
    

def main(args):
    pickle_location = os.path.join(args.data_location, args.pickle_file)
    energy_location = os.path.join(args.data_location, args.pickle_file[:-7] + '_energy_valid_50.pickle') 

    print(args)
    if args.compute_energy:
        energy_list = compute_energy(pickle_location)
        pickle.dump(energy_list, open(energy_location, 'wb'))


    
    energy = pickle.load(open(energy_location, 'rb'))
    
    num_bins = int(len(energy))
    plt.hist(energy, num_bins, facecolor='blue', alpha=0.5)
    plt.title('Computed energy with M3GNet on Perov generation') # {}'.format(args.gen_file[:-3]))
    plt.xlabel('Energy')
    plt.savefig(os.path.join(args.data_location, args.pickle_file[:-7] + '_energy_valid_50.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path')#, required=True)
    # parser.add_argument('--gen_file')#, required=True)
    parser.add_argument('--compute_energy', default=False, type=lambda z: z=='True')

    parser.add_argument('--data_location')
    parser.add_argument('--pickle_file')

    args = parser.parse_args()

 
    args.data_location ='/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/clean_preprocess_attention/complex_frac/perov_mini_test_sde_continuous_new_S10105005_FracScale_coordScaleCost10_vae2noGT_8_H256_Emb3232_MLPatc_Norm_noInit_Adam4'
    args.pickle_file = 'ATclampStep_SDEtest_divideSigma_snr15_newEvalCode_annealed_generated_crystals.pickle'
    
    args.data_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/haicore/standard_primitive/perov_pc_SeparateLA_SinCos8RewSumFracNotQIndiv8_MSE_tanh_sde_CLossMinus_5255_1e3_Stop300_PrimitiveStand_wOutLin_ResComp_jointFlMLPEx2_costA1C2CE1P'
    args.pickle_file = 'no_wrap_denoise_snr333_predictor_aN_cU_lN_generated_crystals.pickle'
    

    ### Haicore unified no w in

    args.data_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/haicore/unified/perov_AttentionSinCosFixStacked_NOWin_Separate_8_IndividualLayerNormAtt_NoWrapGT_MSE_AEScratch_10251_1e3RLY_NoMLPNoRes_stop400_EmbNumAtomsLattice_H4L8_6464_44'
    args.pickle_file = 'wrap_denoise_snr333_predictor_aN_cU_lN_generated_crystals.pickle'
    args.compute_energy = True

    

    args.data_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/haicore/unified/perov_AttentionSinCosFixStacked_64_NoWIn_IndividualLayerNormAtt_NoWrapGT_MSE_AEScratch64164_5255_1e343_stop200_EmbNumAtomsLattice_H4L8_6464_448_256_NOMLPNoRes'
    args.pickle_file = 'wrap_denoise_snr333_predictor_aN_cN_lN_generated_crystals.pickle'



    # #### Baselines
    # args.data_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/baselines/perov'
    # args.pickle_file = '500_generated_crystals.pickle'


    # #### GT
    # args.data_location = '/home/m/Thesis/cdvae/cdvae/data/perov_5'
    # args.pickle_file = 'perov_train.pickle'



    # ### haicore W In
    # args.data_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/haicore/unified/perov_AttentionSinCosJointAtomsWithLinear_MSE_WInDiag_MLPTrue_SkipFalse_64_AE64Scratch_555_1343NoMLP_ResSinCosFalse_EmbNumAtomsTrue_Emb64_63_MLPJoint512_MLPLat51'
    # args.pickle_file = 'wrap_denoise_snr01101_predictor_aN_cU_lN_generated_crystals.pickle'


    # ##### gpu W in
    # args.data_location = '/home/m/Thesis/cdvae/cdvae-main-hydra/singlerun/gpu/perov_AttentionSinCosJointAtomsWithLinear_MSE_WInTrue_MLPTrue_SkipFalse_32_AE64Scratch_555_1343NoMLP_ResSinCosFalse_EmbNumAtomsTrue_Emb64_127_MLPJoint512_MLPLat512_jointM'
    # args.pickle_file ='wrap_denoise_snr010101_predictor_aN_cN_lN_generated_crystals.pickle'
    
    
    main(args)