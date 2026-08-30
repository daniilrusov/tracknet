# TrackNET: Intelligent Particle Track Building

TrackNET is a deep learning-based approach for particle track reconstruction, implemented using PyTorch Lightning. This repository contains the implementation of the StepAhead TrackNET model, a recurrent neural network for sequential track building in high-energy physics experiments.

## Model Overview

StepAhead TrackNET operates as a trainable Kalman filter, using a GRU-based architecture to predict search regions for the next hits in a particle track. For each input hit sequence, the model can generate two predictions:

- t1: Immediate next hit location and search radius
- t2: Location and search radius for the hit after next

The corrected geometry is available in `scripts/driftsim_v3.py`. The original
V1 and V2 generators are retained unchanged for reproducibility of existing
datasets and checkpoints.

## Key Features

- Sequential track building using RNN architecture
- Drift simulation data generation for straw detector training
- Straw tube next-hit classification model
- PyTorch Lightning training loop
- TensorBoard logging and checkpointing

## Repository Setup

1. Clone the repository:

```bash
git clone https://github.com/daniilrusov/tracknet.git
cd tracknet
```

2. Create and activate the conda environment:

```bash
conda env update -f environment.yml
conda activate tracknet
```

3. Generate V3 drift simulation data:

```bash
python scripts/driftsim_v3.py -n 10000 -o outputs/drift_sim_v3 --seed 42
```

4. Preprocess drift simulation data into fast training shards:

```bash
python scripts/preprocess_drift_sim.py --schema-version v3 --input-dir outputs/drift_sim_v3 --output-dir outputs/drift_sim_v3_cache
```

5. Configure settings if needed:

- Edit `configs/user_settings/user_settings.yaml` for custom paths.
- Adjust model parameters in `configs/model/straw_model.yaml`.
- Modify training parameters in `configs/train.yaml`.

## Training

Start training on V3 data:

```bash
python train.py dataset=drift_sim_v3
```

For a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py dataset=drift_sim_v3
```

For a quick smoke test:

```bash
python scripts/driftsim_v3.py -n 1000 -o outputs/drift_sim_v3 --seed 42
python scripts/preprocess_drift_sim.py --schema-version v3 --input-dir outputs/drift_sim_v3 --output-dir outputs/drift_sim_v3_cache --chunk-size 50000 --shard-size 10000
python train.py dataset=drift_sim_v3 training.max_epochs=1 training.batch_size=256 training.num_workers=2 training.limit_train_batches=5 training.limit_val_batches=2
```

V3 has 1456 geometry-derived tube classes instead of the 1208 legacy output
classes. Existing V1/V2 checkpoints remain valid for legacy datasets but are
not shape-compatible with V3. To reproduce the old pipeline, keep using
`driftsim_v2.py`, the default legacy preprocessing mode, and `python train.py`.
The complete V3 design, compatibility rules, and acceptance criteria are in
[`docs/drift_sim_v3_plan.md`](docs/drift_sim_v3_plan.md).

For the production V3 pipeline (1M training events, preprocessing, training,
resume, independent 100K benchmark, and detailed metrics), see
[`docs/v3_remote_training.md`](docs/v3_remote_training.md). On a configured
Linux machine the complete generation-to-training sequence can be launched with:

```bash
bash scripts/run_v3_1m_pipeline.sh
```

Monitor training:

```bash
tensorboard --logdir outputs/
```

Evaluate a V3 checkpoint on the independent fixed benchmark:

```bash
python scripts/driftsim_v3.py --config configs/drift_sim_v3_test_100k.yaml
python scripts/evaluate_straw_checkpoint.py \
  --checkpoint outputs/<run>/logs/straw_tracknet_v3_1m/version_0/checkpoints/<checkpoint>.ckpt \
  --data outputs/test_v3_100k/output.tsv \
  --output-dir outputs/test_v3_100k/metrics
```

The evaluator reports cross-entropy, MRR, top-k hit recall, exact-track
survival, and detailed breakdowns by hit position, station transition, track
length, and tube class. It auto-detects legacy or V3 data, validates the V3
metadata and 1456-class geometry, and records dataset/checkpoint SHA-256 hashes.

## Results and Outputs

Training results are stored in `outputs/` with the following structure:

```text
outputs/YYYY-MM-DD/HH-MM-SS/
|-- logs
|   `-- straw_tracknet_spd
|       `-- version_0
|           |-- checkpoints/             # Top-3 model checkpoints
|           |-- events.out.tfevents.*    # TensorBoard logs
|           `-- hparams.yaml             # Hyperparameters
`-- train.log                            # Training log
```

Key outputs:

- Model checkpoints: `logs/*/checkpoints/`
- Training metrics: TensorBoard logs
- Configuration: `hparams.yaml`

## References

1. Rusov, D., Goncharov, P., Zhemchugov, A. et al. Deep Tracking for the SPD Experiment. Phys. Part. Nuclei Lett. 20, 1180-1182 (2023). https://doi.org/10.1134/S1547477123050655
2. Bakina, O., et al. Deep Learning for Track Recognition in Pixel and Strip-Based Particle Detectors. Journal of Instrumentation, vol. 17, no. 12, IOP Publishing, Dec. 2022, P12023, doi:10.1088/1748-0221/17/12/P12023.
3. Rusov, D. et al. (2023). Recurrent and Graph Neural Networks for Particle Tracking at the BM@N Experiment. Advances in Neural Computation, Machine Learning, and Cognitive Research VI. Springer, Cham. https://doi.org/10.1007/978-3-031-19032-2_32

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{tracknet2024,
  author = {Goncharov, Pavel and Rusov, Daniil},
  title = {TrackNET: Intelligent Particle Track Building},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/daniilrusov/tracknet}},
}
```
