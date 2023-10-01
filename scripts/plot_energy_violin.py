import matplotlib.pyplot as plt
import numpy as np

import pickle


perov = '/home/m/Thesis/cdvae/Energy/perov_5_train_energy_valid_500.pickle'
perov_cdvae = '/home/m/Thesis/cdvae/Energy/perov_1000_generated_crystals_energy_valid_500.pickle'
perov_generated_noWIn = '/home/m/Thesis/cdvae/Energy/perov_now_wrap_denoise_snr01301_predictor_aN_cUxSigmaMax_lN_1000_generated_crystals_energy_valid_500.pickle'
perov_generated_WIn = '/home/m/Thesis/cdvae/Energy/perov_win_wrap_denoise_snr01301_predictor_aN_cUxSigmaMax_lN_1000_generated_crystals_energy_valid_500.pickle'


perov_energy = pickle.load(open(perov, 'rb'))
perov_cdvae_energy = pickle.load(open(perov_cdvae, 'rb'))
perov_generated_noWIn_energy = pickle.load(open(perov_generated_noWIn, 'rb'))
perov_generated_WIn_energy = pickle.load(open(perov_generated_WIn, 'rb'))


mp_20 = '/home/m/Thesis/cdvae/Energy/mp_20_train_energy_valid_500.pickle'
mp_20_cdvae = '/home/m/Thesis/cdvae/Energy/mp_20_1000_generated_crystals_energy_valid_500.pickle'
mp_20_generated_noWIn = '/home/m/Thesis/cdvae/Energy/mp_now_wrap_denoise_snr333_predictor_aN_cUxSigmaMax_lN_1000_generated_crystals_energy_valid_500.pickle'
mp_20_generated_WIn = '/home/m/Thesis/cdvae/Energy/mp_win_wrap_denoise_snr0051005_predictor_aN_cUxSigmaMax_lN_1000_generated_crystals_energy_valid_500.pickle'

mp_20_energy = pickle.load(open(mp_20, 'rb'))
mp_20_cdvae_energy = pickle.load(open(mp_20_cdvae, 'rb'))
mp_20_generated_noWIn_energy = pickle.load(open(mp_20_generated_noWIn, 'rb'))
mp_20_generated_WIn_energy = pickle.load(open(mp_20_generated_WIn, 'rb'))



# Generate your data or load it from a source
data = {
    # 'Train\ndata': perov_energy,
    # 'CDVAE': perov_cdvae_energy,
    # 'Ours\n${W}_{fixed}$': perov_generated_noWIn_energy,
    # # 'Predefined\nfrequencies\nAE 4': perov_generated_noWIn_AE4_energy,
    #  'Ours\n${W}_{learn}$': perov_generated_WIn_energy,

    'Train\ndata': mp_20_energy,
    'CDVAE': mp_20_cdvae_energy,
    'Ours\n${W}_{fixed}$': mp_20_generated_noWIn_energy,
    # 'Predefined\nfrequencies\nAE 4': mp_20_generated_noWIn_AE4_energy,
    'Ours\n${W}_{learn}$': mp_20_generated_WIn_energy,
}


fig, ax = plt.subplots(figsize=(10, 8))

violin_data = [list(data[key]) for key in data.keys()]

vp = ax.violinplot(violin_data, showmeans=True, widths = 0.8) #, showextrema=True) #, showmedians=True)

colors = [ '#FDAB77','#B3D5F5', '#B16690', '#C5A0A0'] 
for patch, color in zip(vp['bodies'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black') 
    patch.set_alpha(1) 

for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'): #'cmedians', 
    vp[partname].set_edgecolor("black")
    vp[partname].set_linewidth(1)


ax.set_xticks(range(1, len(data) + 1))
ax.set_xticklabels(data.keys(), fontsize=20)

ax.set_ylabel("Energy", fontsize = 20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()  
plt.savefig('/home/m/Thesis/cdvae/Energy/mp_20_energy.pdf')
plt.show()