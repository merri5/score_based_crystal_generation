import pickle
import random
from typing import Optional, Sequence
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader

from model.common.utils import PROJECT_ROOT
from model.common.data_utils import get_scaler_from_data_list


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        # download only
        pass

    def get_scaler(self, scaler_path):
        # Load once to compute property scaler
        if scaler_path is None:
            train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key='lattice')
            self.scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key=train_dataset.prop)
            self.lattice_min_max_scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key='lattice',
                type='minmax')
        else:
            self.lattice_scaler = torch.load(
                Path(scaler_path) / 'lattice_scaler.pt')
            self.scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')
            self.lattice_min_max_scaler = torch.load(Path(scaler_path) / 'lattice_min_max_scaler.pt')

    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]

            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.lattice_min_max_scaler = self.lattice_min_max_scaler
            self.train_dataset.scaler = self.scaler
            for val_dataset in self.val_datasets:
                val_dataset.lattice_scaler = self.lattice_scaler
                val_dataset.lattice_min_max_scaler = self.lattice_min_max_scaler
                val_dataset.scaler = self.scaler

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]
            for test_dataset in self.test_datasets:
                test_dataset.lattice_scaler = self.lattice_scaler
                test_dataset.lattice_min_max_scaler = self.lattice_min_max_scaler
                test_dataset.scaler = self.scaler

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    # import pdb
    # pdb.set_trace()

    train_dataloader = datamodule.train_dataloader()
    data = train_dataloader.dataset.cached_data

    # train_data = []
    

    # for el in data:
    #     (frac_coords, atom_types, lengths, angles, edge_indices,
    #      to_jimages, num_atoms) = el['graph_arrays']
    #     # lattice = lattice_params_to_matrix_torch(torch.tensor(lengths), torch.tensor(angles))
    #     lattice = lattice_params_to_matrix(lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2])
    #     crystal = dict()
    #     crystal['lattice_vectors'] = lattice
    #     crystal['lengths'] = lengths
    #     crystal['angles'] = angles
    #     crystal['frac_coords'] = frac_coords
    #     crystal['atom_types'] = atom_types
    #     train_data.append(data)
    #     # print(train_data[-1])
       
    pickle.dump(data,open('/hkfs/work/workspace_haic/scratch/ga8707-thesis/cdvae/data/carbon_24_mini/carbon_24_mini_train.pickle', 'wb'))

    val_dataloaders = datamodule.val_dataloader()
    data_set = []
    for val_dataloader in val_dataloaders:
        
        data = val_dataloader.dataset.cached_data
        data_set.append(data)
    pickle.dump(data_set[0],open('/hkfs/work/workspace_haic/scratch/ga8707-thesis/cdvae/data/carbon_24_mini/carbon_24_mini_val.pickle', 'wb'))


    datamodule.setup('test')
    test_dataloaders = datamodule.test_dataloader()
    data_set = []
    for test_dataloader in test_dataloaders:
        
        data = test_dataloader.dataset.cached_data
        data_set.append(data)
    pickle.dump(data_set[0],open('/hkfs/work/workspace_haic/scratch/ga8707-thesis/cdvae/data/carbon_24_mini/carbon_24_mini_test.pickle', 'wb'))

    # test_dataloader = datamodule.test_dataloader()
    # data = test_dataloader.dataset.cached_data
    # pickle.dump(data,open('/home/mimi/Thesis/cdvae/data_pickles/perov_mini_test.pickle', 'wb'))


if __name__ == "__main__":
    main()