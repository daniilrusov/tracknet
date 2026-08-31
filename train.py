import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "outputs", "mplconfig"))

import hydra
import logging
from argparse import ArgumentParser
from omegaconf import DictConfig
from hydra.utils import instantiate
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, IterableDataset

from src.tracknet.data.collate import collate_fn

# Configure argument parser
parser = ArgumentParser()
parser.add_argument("--loglevel", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Logging level")

logging.basicConfig()
logger = logging.getLogger("train")


def setup_training(cfg: DictConfig):
    """Set up training components based on config."""
    # Instantiate datasets with all components from config
    train_dataset = instantiate(cfg.dataset, split='train')
    val_dataset = instantiate(cfg.dataset, split='validation')
    is_prebatched = getattr(train_dataset, "prebatched", False)
    if is_prebatched:
        train_loader = DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=cfg.training.num_workers,
            persistent_workers=cfg.training.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=None,
            num_workers=cfg.training.num_workers,
            persistent_workers=cfg.training.num_workers > 0,
        )
    else:
        train_shuffle = cfg.training.shuffle
        if isinstance(train_dataset, IterableDataset):
            if train_shuffle:
                logger.warning("Disabling DataLoader shuffle for IterableDataset.")
            train_shuffle = False

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=train_shuffle,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn,
            persistent_workers=cfg.training.num_workers > 0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn,
            persistent_workers=cfg.training.num_workers > 0,
        )

    # Instantiate model from config
    model_overrides = {}
    if getattr(cfg.model, "num_tubes", None) == "auto":
        model_overrides["num_tubes"] = train_dataset.num_tubes
    if getattr(cfg.model, "station_tube_counts", None) == "auto":
        station_tube_counts = getattr(train_dataset, "station_tube_counts", None)
        if station_tube_counts is None:
            raise ValueError(
                "Model requests automatic station_tube_counts, but dataset "
                "metadata does not define detector station geometry."
            )
        model_overrides["station_tube_counts"] = station_tube_counts
    model = instantiate(cfg.model, **model_overrides)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model
    }


def train(cfg: DictConfig, components):
    """Run training loop."""
    tb_logger = TensorBoardLogger(
        save_dir=cfg.logging.output_dir,
        name=cfg.experiment.name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=3,
        save_last=True,
        mode="min"
    )
    callbacks = [checkpoint_callback]
    early_stopping_patience = cfg.training.get("early_stopping_patience")
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=int(early_stopping_patience),
                mode="min",
            )
        )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        #strategy='ddp_spawn',
        #distributed_backend='ddp',
        logger=tb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        precision=cfg.training.get("precision", "32-true"),
        log_every_n_steps=int(cfg.training.get("log_every_n_steps", 50)),
        check_val_every_n_epoch=int(
            cfg.training.get("check_val_every_n_epoch", 1)
        ),
    )

    trainer.fit(
        model=components["model"],
        train_dataloaders=components["train_loader"],
        val_dataloaders=components["val_loader"],
        ckpt_path=cfg.training.resume_from
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    args, _ = parser.parse_known_args()
    logging.basicConfig(level=args.loglevel)

    pl.seed_everything(cfg.training.seed, workers=True)

    if cfg.training.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.training.matmul_precision)

    components = setup_training(cfg)
    train(cfg, components)


if __name__ == "__main__":
    main()  # type: ignore
