import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import plotly
import io
import PIL.Image

from typing import Optional
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .data.schemas import BatchSample
from .model import StepAheadTrackNET, StrawTrackNET, TrackPrediction, StrawTubePrediction
from .loss import TrackNetLoss, StrawTrackNetLoss
from .data.transformations import MinMaxNormalizeXYZ
from .metrics import SearchAreaMetric, HitEfficiencyMetric, HitDensityMetric, StrawHitEfficiencyMetric
from .visualization import visualize_track_predictions


class TrackNETModule(pl.LightningModule):
    def __init__(
        self,
        input_features: int = 3,
        hidden_features: int = 32,
        output_features: int = 3,
        learning_rate: float = 1e-3,
        loss_alpha: float = 0.9,
        batch_first: bool = True,
        hit_density_stats_path: Optional[str] = None,
        hits_normalizer: Optional[MinMaxNormalizeXYZ] = None
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = StepAheadTrackNET(
            input_features=input_features,
            hidden_features=hidden_features,
            output_features=output_features,
            batch_first=batch_first
        )

        # Loss
        self.loss_fn = TrackNetLoss(alpha=loss_alpha)

        # Metrics
        self.train_search_area_t1 = SearchAreaMetric('t1')
        self.train_search_area_t2 = SearchAreaMetric('t2')
        self.train_hit_efficiency_t1 = HitEfficiencyMetric('t1')
        self.train_hit_efficiency_t2 = HitEfficiencyMetric('t2')

        self.val_search_area_t1 = SearchAreaMetric('t1')
        self.val_search_area_t2 = SearchAreaMetric('t2')
        self.val_hit_efficiency_t1 = HitEfficiencyMetric('t1')
        self.val_hit_efficiency_t2 = HitEfficiencyMetric('t2')

        if hit_density_stats_path is not None:
            # this metric has long calculation time,
            # so we only compute it during validation
            self.val_hit_density_t1 = HitDensityMetric(
                density_stats_path=hit_density_stats_path,
                time_step='t1',
                normalizer=hits_normalizer
            )
            self.val_hit_density_t2 = HitDensityMetric(
                density_stats_path=hit_density_stats_path,
                time_step='t2',
                normalizer=hits_normalizer
            )

        # Save last output for logging
        # optional is needed to initialize with None
        self.last_batch: Optional[BatchSample] = None
        self.last_output: Optional[TrackPrediction] = None

    def forward(self, batch):
        return self.model(batch['inputs'], batch['input_lengths'])

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss_fn(output, batch['targets'], batch['target_mask'])

        # Update metrics
        self.train_search_area_t1.update(output, batch['target_mask'])
        self.train_search_area_t2.update(output, batch['target_mask'])
        self.train_hit_efficiency_t1.update(
            output, batch['targets'], batch['target_mask'])
        self.train_hit_efficiency_t2.update(
            output, batch['targets'], batch['target_mask'])

        # Log metrics
        batch_size = batch['inputs'].size(0)
        self.log('train_loss', loss, prog_bar=True, batch_size=batch_size)
        self.log_dict({
            "train_search_area_t1": self.train_search_area_t1,
            "train_hit_efficiency_t1": self.train_hit_efficiency_t1,
            "train_search_area_t2": self.train_search_area_t2,
            "train_hit_efficiency_t2": self.train_hit_efficiency_t2,
        }, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss_fn(output, batch['targets'], batch['target_mask'])

        # Update metrics
        self.val_search_area_t1.update(output, batch['target_mask'])
        self.val_search_area_t2.update(output, batch['target_mask'])
        self.val_hit_efficiency_t1.update(
            output, batch['targets'], batch['target_mask'])
        self.val_hit_efficiency_t2.update(
            output, batch['targets'], batch['target_mask'])

        # Log metrics
        batch_size = batch['inputs'].size(0)
        self.log('val_loss', loss, prog_bar=True, batch_size=batch_size)
        metrics_dict = {
            "val_search_area_t1": self.val_search_area_t1,
            "val_search_area_t2": self.val_search_area_t2,
            "val_hit_efficiency_t1": self.val_hit_efficiency_t1,
            "val_hit_efficiency_t2": self.val_hit_efficiency_t2,
        }

        if hasattr(self, 'val_hit_density_t1'):
            self.val_hit_density_t1.update(output, batch['target_mask'])
            self.val_hit_density_t2.update(output, batch['target_mask'])
            metrics_dict.update({
                "val_hit_density_t1": self.val_hit_density_t1,
                "val_hit_density_t2": self.val_hit_density_t2,
            })

        self.log_dict(
            metrics_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # Save last batch for visualization
        if batch_idx == 0:
            # Convert tensors to CPU numpy arrays for visualization
            self.last_batch = {
                'inputs': batch['inputs'].detach().cpu().numpy(),
                'targets': batch['targets'].detach().cpu().numpy(),
                'input_lengths': batch['input_lengths'],
                'target_mask': batch['target_mask'].detach().cpu().numpy()
            }
            self.last_output = {
                k: v.detach().cpu().numpy() for k, v in output.items()
            }

        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate  # type: ignore
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.3,
            patience=2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def on_save_checkpoint(self, checkpoint):
        """Save hyperparameters and model state."""
        checkpoint['hyperparameters'] = self.hparams

    def on_load_checkpoint(self, checkpoint):
        """Load hyperparameters and model state."""
        self.hparams.update(checkpoint['hyperparameters'])

    def on_validation_epoch_end(self):
        """Log track visualization."""
        if self.last_batch is not None and self.last_output is not None:
            fig = visualize_track_predictions(
                self.last_batch, self.last_output, track_idx=0)

            # Convert Plotly figure to image bytes
            img_bytes = fig.to_image(format="png")

            # Create a buffer from the bytes
            buffer = io.BytesIO(img_bytes)

            # Create PIL Image from buffer
            image = PIL.Image.open(buffer)

            # Convert PIL image to matplotlib figure
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')

            # Log to tensorboard
            self.logger.experiment.add_figure(
                'Track Visualization',
                plt.gcf(),
                global_step=self.current_epoch
            )

            plt.close()  # Clean up the matplotlib figure

            if hasattr(self.logger, 'log_dir'):
                # save the last track visualization as an HTML file
                save_path = f"{self.logger.log_dir}/last_track_viz.html"
                plotly.offline.plot(fig, filename=save_path, auto_open=False)


class StrawTrackNETModule(pl.LightningModule):
    def __init__(
        self,
        input_features: int = 5,
        hidden_features: int = 128,
        num_tubes: int = 1456,
        output_features: Optional[int] = None,
        learning_rate: float = 1e-3,
        batch_first: bool = True,
        use_plane_embedding: bool = True,
        plane_embedding_dim: int = 8,
        continuous_feature_center=(0.0, 0.0, 120.0, 2.5),
        continuous_feature_scale=(750.0, 750.0, 120.0, 2.5),
        seed_hits: int = 2,
        ranking_loss_weight: float = 0.0,
        ranking_margin: float = 0.5,
        ranking_top_m: int = 1,
        station_tube_counts: Optional[list[int]] = None,
        lr_scheduler_factor: float = 0.3,
        lr_scheduler_patience: int = 2,
        lr_scheduler_threshold: float = 1e-4,
        lr_scheduler_threshold_mode: str = "rel",
        lr_scheduler_min_lr: float = 0.0,
        hit_density_stats_path: Optional[str] = None,
        hits_normalizer: Optional[MinMaxNormalizeXYZ] = None
    ):
        super().__init__()
        if seed_hits < 1:
            raise ValueError("seed_hits must be at least 1.")
        if not 0 < lr_scheduler_factor < 1:
            raise ValueError("lr_scheduler_factor must be between zero and one.")
        if lr_scheduler_patience < 0:
            raise ValueError("lr_scheduler_patience must be non-negative.")
        if lr_scheduler_threshold < 0:
            raise ValueError("lr_scheduler_threshold must be non-negative.")
        if lr_scheduler_threshold_mode not in {"rel", "abs"}:
            raise ValueError(
                "lr_scheduler_threshold_mode must be either 'rel' or 'abs'."
            )
        if lr_scheduler_min_lr < 0:
            raise ValueError("lr_scheduler_min_lr must be non-negative.")
        self.save_hyperparameters()
        self.seed_hits = int(seed_hits)

        # Model
        self.model = StrawTrackNET(
            input_features=input_features,
            hidden_features=hidden_features,
            num_tubes=num_tubes if output_features is None else output_features,
            batch_first=batch_first,
            use_plane_embedding=use_plane_embedding,
            plane_embedding_dim=plane_embedding_dim,
            continuous_feature_center=continuous_feature_center,
            continuous_feature_scale=continuous_feature_scale,
        )

        # Loss
        self.loss_fn = StrawTrackNetLoss(
            ranking_loss_weight=ranking_loss_weight,
            ranking_margin=ranking_margin,
            ranking_top_m=ranking_top_m,
            station_tube_counts=station_tube_counts,
        )

        # Metrics
        self.train_hit_efficiency_t1 = StrawHitEfficiencyMetric('t1')

        self.val_hit_efficiency_t1 = StrawHitEfficiencyMetric('t1')

        # Save last output for logging
        # optional is needed to initialize with None
        self.last_batch: Optional[BatchSample] = None
        self.last_output: Optional[StrawTubePrediction] = None

    def forward(self, batch):
        return self.model(batch['inputs'], batch['input_lengths'])

    def _supervised_view(self, output, batch):
        """Drop predictions made before all seed hits have been consumed."""
        first_step = self.seed_hits - 1
        if output['tube_logits_t1'].size(1) <= first_step:
            raise ValueError(
                f"A sequence needs at least {self.seed_hits + 1} hits to provide "
                f"a target after {self.seed_hits} seed hits."
            )
        supervised_output = {
            'tube_logits_t1': output['tube_logits_t1'][:, first_step:],
        }
        return (
            supervised_output,
            batch['targets'][:, first_step:],
            batch['target_mask'][:, first_step:],
        )

    def training_step(self, batch, batch_idx):
        output = self(batch)
        supervised_output, targets, target_mask = self._supervised_view(output, batch)
        loss_parts = self.loss_fn.loss_components(
            supervised_output, targets, target_mask
        )
        loss = loss_parts["loss"]

        # Update metrics
        self.train_hit_efficiency_t1.update(
            supervised_output, targets, target_mask)

        # Log metrics
        batch_size = batch['inputs'].size(0)
        self.log('train_loss', loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log(
            'train_cross_entropy',
            loss_parts['cross_entropy'],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            'train_ranking_loss',
            loss_parts['ranking_loss'],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            'train_hard_negative_gap',
            loss_parts['hard_negative_gap'],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            'train_top_m_negative_gap',
            loss_parts['top_m_negative_gap'],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train_hit_efficiency_t1",
            self.train_hit_efficiency_t1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        supervised_output, targets, target_mask = self._supervised_view(output, batch)
        loss_parts = self.loss_fn.loss_components(
            supervised_output, targets, target_mask
        )
        loss = loss_parts["loss"]

        # Update metrics
        self.val_hit_efficiency_t1.update(
            supervised_output, targets, target_mask)

        # Log metrics
        batch_size = batch['inputs'].size(0)
        self.log('val_loss', loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log(
            'val_cross_entropy',
            loss_parts['cross_entropy'],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            'val_ranking_loss',
            loss_parts['ranking_loss'],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            'val_hard_negative_gap',
            loss_parts['hard_negative_gap'],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            'val_top_m_negative_gap',
            loss_parts['top_m_negative_gap'],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val_hit_efficiency_t1",
            self.val_hit_efficiency_t1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # Save last batch for visualization
        if batch_idx == 0:
            # Convert tensors to CPU numpy arrays for visualization
            self.last_batch = {
                'inputs': batch['inputs'].detach().cpu().numpy(),
                'targets': targets.detach().cpu().numpy(),
                'input_lengths': batch['input_lengths'],
                'target_mask': target_mask.detach().cpu().numpy()
            }
            self.last_output = {
                k: v.detach().cpu().numpy() for k, v in supervised_output.items()
            }

        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate  # type: ignore
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(self.hparams.lr_scheduler_factor),
            patience=int(self.hparams.lr_scheduler_patience),
            threshold=float(self.hparams.lr_scheduler_threshold),
            threshold_mode=str(self.hparams.lr_scheduler_threshold_mode),
            min_lr=float(self.hparams.lr_scheduler_min_lr),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def on_save_checkpoint(self, checkpoint):
        """Save hyperparameters and model state."""
        checkpoint['hyperparameters'] = self.hparams

    def on_load_checkpoint(self, checkpoint):
        """Load hyperparameters and model state."""
        self.hparams.update(checkpoint['hyperparameters'])

    def on_validation_epoch_end(self):
        """Log track visualization."""
        if False and self.last_batch is not None and self.last_output is not None:
            fig = visualize_track_predictions(
                self.last_batch, self.last_output, track_idx=0)

            # Convert Plotly figure to image bytes
            img_bytes = fig.to_image(format="png")

            # Create a buffer from the bytes
            buffer = io.BytesIO(img_bytes)

            # Create PIL Image from buffer
            image = PIL.Image.open(buffer)

            # Convert PIL image to matplotlib figure
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')

            # Log to tensorboard
            self.logger.experiment.add_figure(
                'Track Visualization',
                plt.gcf(),
                global_step=self.current_epoch
            )

            plt.close()  # Clean up the matplotlib figure

            if hasattr(self.logger, 'log_dir'):
                # save the last track visualization as an HTML file
                save_path = f"{self.logger.log_dir}/last_track_viz.html"
                plotly.offline.plot(fig, filename=save_path, auto_open=False)
