import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('cache', type=Path)
parser.add_argument('residual', choices=('true', 'false'))
parser.add_argument('batch_size', type=int)
parser.add_argument('accumulation', type=int)
args = parser.parse_args()

sys.path.insert(0, str(args.source))
from train import setup_training

with initialize_config_dir(config_dir=str(args.source / 'configs'), version_base=None):
    cfg = compose(config_name='train_v3_1m', overrides=[
        f'dataset.cache_dir={args.cache}',
        f'model.gru_residual_norm={args.residual}',
        f'training.batch_size={args.batch_size}',
        f'training.accumulate_grad_batches={args.accumulation}',
        'training.num_workers=0',
    ])

pl.seed_everything(cfg.training.seed, workers=True)
torch.set_float32_matmul_precision(cfg.training.matmul_precision)
components = setup_training(cfg)
model = components['model']
assert model.model.rnn.num_layers == 5 and not model.model.predict_coordinates
assert model.hparams.gru_residual_norm == (args.residual == 'true')
assert torch.cuda.is_available()
torch.cuda.reset_peak_memory_stats()
trainer = pl.Trainer(
    accelerator='gpu', devices=1, precision=cfg.training.precision,
    max_epochs=1, limit_train_batches=args.accumulation, limit_val_batches=1,
    accumulate_grad_batches=args.accumulation, gradient_clip_val=cfg.training.gradient_clip_val,
    logger=False, enable_checkpointing=False, enable_progress_bar=False,
    num_sanity_val_steps=0,
)
trainer.fit(model, components['train_loader'], components['val_loader'])
assert trainer.global_step == 1
assert torch.isfinite(trainer.callback_metrics['val_loss'])
print(f'PREFLIGHT_OK residual={args.residual} optimizer_steps={trainer.global_step} '
      f'val_loss={trainer.callback_metrics["val_loss"].item():.6f} '
      f'peak_reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.2f}', flush=True)
