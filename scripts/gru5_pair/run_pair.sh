#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# != 2 ]]; then
    printf 'Usage: bash %s <data checkout> <run directory>\n' "$0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_dir=$(cd -- "$script_dir/../.." && pwd -P)
repo=$(realpath -e -- "$1")
mkdir -p -- "$2"
run_dir=$(realpath -e -- "$2")
cache_dir=$repo/outputs/drift_sim_v3_1m_cache
python=$repo/.venv/bin/python
batch_size=8000
accumulation=4

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPLCONFIGDIR="$repo/outputs/mplconfig"

record() {
    printf '%s %s\n' "$2" "$(date --iso-8601=seconds)" > "$1.tmp"
    mv -- "$1.tmp" "$1"
}

stage=setup
on_error() {
    local result=$?
    if [[ "$stage" == training_* ]]; then
        record "$run_dir/${stage#training_}/status.txt" "FAILED exit=$result"
    fi
    record "$run_dir/status.txt" "FAILED stage=$stage exit=$result"
}
exec 9>"$repo/outputs/.tracknet_gru5_pair_gpu.lock"
flock -n 9
if [[ -e "$run_dir/status.txt" ]]; then
    printf 'Choose a new run directory: %s\n' "$run_dir" >&2
    exit 1
fi
trap on_error ERR
git -C "$source_dir" diff --quiet HEAD
mkdir -p "$run_dir/baseline" "$run_dir/residual_norm"
record "$run_dir/baseline/status.txt" QUEUED
record "$run_dir/residual_norm/status.txt" QUEUED
git -C "$source_dir" rev-parse HEAD > "$run_dir/git_commit.txt"

cd "$source_dir"
for mode in baseline residual_norm; do
    stage="preflight_$mode"
    residual=false
    [[ "$mode" != residual_norm ]] || residual=true
    record "$run_dir/status.txt" "PREFLIGHT $mode"
    "$python" -u "$script_dir/preflight.py" "$source_dir" "$cache_dir" \
        "$residual" "$batch_size" "$accumulation" > "$run_dir/$mode/preflight.log" 2>&1
done

for mode in baseline residual_norm; do
    stage="training_$mode"
    residual=false
    [[ "$mode" != residual_norm ]] || residual=true
    record "$run_dir/status.txt" "TRAINING $mode"
    record "$run_dir/$mode/status.txt" TRAINING
    "$python" -u train.py --config-name=train_v3_1m \
        "dataset.cache_dir=$cache_dir" "hydra.run.dir=$run_dir/$mode/training" \
        "experiment.name=straw_gru5_$mode" "model.gru_residual_norm=$residual" \
        "training.batch_size=$batch_size" "training.accumulate_grad_batches=$accumulation" \
        > "$run_dir/$mode/train.log" 2>&1
    record "$run_dir/$mode/status.txt" COMPLETED
done

record "$run_dir/status.txt" COMPLETED
