# 	Score-Based Generative Modeling for Crystal Structures

## Setup 

### Install environment
To install the environment, run
```bash
conda env create -f environment.yml
```

To build the project, run
```bash
pip install -e .
```


### Setting up environment variables
Make sure to full the paths in `.env_template` and rename it to `.env`

- `PROJECT_ROOT`: path to the folder that contains this repo
- `HYDRA_JOBS`: path to a folder to store hydra outputs
- `WABDB`: path to a folder to store wabdb outputs


## Configuration
For the configurations we use hydra.
THe configuration files are located in `conf/`

## Training
After setting up the desired configuration, training is can be started as:
```bash
python model/run.py
```


## Generation
To evaluare the generation, run the evaluation metrics and generate some analysis plots, one can run the command:
```bash
python scripts/evaluate_compute_metrics_plot.py --moodel_path <> --data_location <> 
```
**model_path**: path to the model dir, containing the model ckpt
**data_location**: location to the training data (it is used for the plots only)

##### Additional arguments:
**num_batches_to_samples**: number of batches, default 1
**batch_size**: number of samples per batch, default 1
**target_snr_atom_types**: the signal-to-noise ratio of the atom types for the generation, default 0.1
**target_snr_frac_coords**: the signal-to-noise ratio of the coordinates for the generation, default 0.1
**target_snr_lattice**: the signal-to-noise ratio of the lattice for the generation, default 0.1

