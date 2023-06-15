# Script to train Fourcastnet on ERA5
# Ref: https://arxiv.org/abs/2202.11214

import modulus

from modulus.hydra.config import ModulusConfig
from modulus.key import Key
from modulus.domain import Domain
from modulus.domain.constraint import SupervisedGridConstraint
from modulus.domain.validator import GridValidator
from modulus.solver import Solver
from modulus.utils.io import GridValidatorPlotter

from src.dataset import ERA5HDF5GridDataset
from src.fourcastnet import FourcastNetArch
from src.loss import LpLoss


@modulus.main(config_path="conf", config_name="config_FCN")
def run(cfg: ModulusConfig) -> None:

    # load training/ test data
    channels = list(range(cfg.custom.n_channels))
    train_dataset = ERA5HDF5GridDataset(
        cfg.custom.training_data_path,
        chans=channels,
        tstep=cfg.custom.tstep,
        n_tsteps=cfg.custom.n_tsteps,
        patch_size=cfg.arch.afno.patch_size,
    )
    test_dataset = ERA5HDF5GridDataset(
        cfg.custom.test_data_path,
        chans=channels,
        tstep=cfg.custom.tstep,
        n_tsteps=cfg.custom.n_tsteps,
        patch_size=cfg.arch.afno.patch_size,
        n_samples_per_year=20,
    )

    # define input/output keys
    input_keys = [Key(k, size=train_dataset.nchans) for k in train_dataset.invar_keys]
    output_keys = [Key(k, size=train_dataset.nchans) for k in train_dataset.outvar_keys]

    # make list of nodes to unroll graph on
    model = FourcastNetArch(
        input_keys=input_keys,
        output_keys=output_keys,
        img_shape=test_dataset.img_shape,
        patch_size=cfg.arch.afno.patch_size,
        embed_dim=cfg.arch.afno.embed_dim,
        depth=cfg.arch.afno.depth,
        num_blocks=cfg.arch.afno.num_blocks,
    )
    nodes = [model.make_node(name="FCN")]

    # make domain
    domain = Domain()

    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
        loss=LpLoss(),
        num_workers=cfg.custom.num_workers.grid,
    )
    domain.add_constraint(supervised, "supervised")

    # add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
        num_workers=cfg.custom.num_workers.validation,
    )
    domain.add_validator(val, "test")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
