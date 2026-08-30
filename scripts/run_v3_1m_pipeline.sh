#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs/v3_1m_pipeline_logs

python -u scripts/driftsim_v3.py \
  --config configs/drift_sim_v3_1m.yaml \
  2>&1 | tee outputs/v3_1m_pipeline_logs/01_generate_train.log

python -u scripts/preprocess_drift_sim.py \
  --schema-version v3 \
  --input-dir outputs/drift_sim_v3_1m \
  --output-dir outputs/drift_sim_v3_1m_cache \
  --validation-split 0.1 \
  --split-seed 42 \
  --chunk-size 1000000 \
  --shard-size 100000 \
  2>&1 | tee outputs/v3_1m_pipeline_logs/02_preprocess.log

python -u train.py --config-name=train_v3_1m \
  2>&1 | tee outputs/v3_1m_pipeline_logs/03_train.log
