#!/usr/bin/env python
"""Generate DriftSim V4 events with eight endcap stations at z=1200..1440 mm."""

from pathlib import Path

if __package__:
    from . import driftsim_v3
else:
    import driftsim_v3


GENERATOR_VERSION = 4


def default_config() -> dict:
    config = driftsim_v3.default_config()
    config["generator_version"] = GENERATOR_VERSION
    config["output_dir"] = "outputs/drift_sim_v4"
    config["detector"].update(z_min_mm=1200.0, z_max_mm=1440.0)
    return config


def main() -> None:
    args = driftsim_v3.parse_args("configs/drift_sim_v4.yaml", description=__doc__)
    config = driftsim_v3.load_config(Path(args.config), defaults=default_config())
    config = driftsim_v3.normalize_config(config, args)
    config["generator_version"] = GENERATOR_VERSION
    driftsim_v3.generate(config, resample_misses=True)


if __name__ == "__main__":
    main()
