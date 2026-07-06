#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import random
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

xmin = -1500
ymin = -1500
umin = -1500
vmin = -1500
uvangle = 45 * 3.14159 / 180.  # deg to rad
width = 10


def LineExtrapToZ(x0, y0, theta, phi, Z):
    perp = Z * math.tan(theta)
    x = x0 + perp * math.cos(phi)
    y = y0 + perp * math.sin(phi)
    z = float(Z)
    return (x, y, z)


def GetWireDrift(x, xmin):
    wirenum = round((x - xmin) / width)
    dR = (x - xmin) - wirenum * width
    lr = 1
    if dR < 0:
        dR = -dR
        lr = -1
    return (wirenum, dR, lr)


def GetWirePos(wireid):
    station = round(wireid / 1000)
    wirenum = wireid % 1000

    zplane = np.linspace(0, 240, 9)  # mm
    pos = wirenum * width

    if (station == 1) or (station == 5):  # X
        x0 = xmin + pos
        y0 = ymin
        z0 = zplane[station - 1]
        xyuv = 1
    if (station == 2) or (station == 6):  # Y
        x0 = xmin
        y0 = ymin + pos
        z0 = zplane[station - 1]
        xyuv = 2
    if (station == 3) or (station == 7):  # U
        x0 = xmin + pos / math.cos(uvangle)
        y0 = ymin + pos // math.sin(uvangle)
        z0 = zplane[station - 1]
        xyuv = 3
    if (station == 4) or (station == 8):  # V
        x0 = xmin + pos / math.sin(uvangle)
        y0 = ymin + pos / math.cos(uvangle)
        z0 = zplane[station - 1]
        xyuv = 4

    return (x0, y0, station, z0, xyuv)


def GetWire(x, y, station):
    # wireid = station*1000+tubeid
    # tubeid [0,150]
    # 8 stations [1,8]: x1 y1 u1 v1 x2 y2 u2 v2

    uvangle = 45 * 3.14159 / 180.  # deg to rad

    if (station == 1) or (station == 5):  # X
        pos = x
        posmin = xmin
    if (station == 2) or (station == 6):  # Y
        pos = y
        posmin = ymin
    if (station == 3) or (station == 7):  # U
        pos = math.sqrt(x * x + y * y) * math.cos(math.atan2(x, y) - uvangle)
        posmin = math.sqrt(xmin * xmin + ymin * ymin)
    if (station == 4) or (station == 8):  # V
        pos = math.sqrt(x * x + y * y) * math.sin(math.atan2(x, y) - uvangle)
        posmin = math.sqrt(xmin * xmin + ymin * ymin)

    wirenum, dR, lr = GetWireDrift(pos, posmin)
    wireid = station * 1000 + wirenum

    return (wireid, dR, lr)


def generate(nevents, output_path):
    eff = 1.  # 0.98  # detector efficiency

    # dimensions xy: 1500 x 1500
    # tube diameter 10 mm
    # 8 planes: x1 y1 u1 v1 x2 y2 u2 v2
    zplane = np.linspace(0, 240, 8)  # mm

    with output_path.open("w") as f:
        for evt in tqdm(range(0, nevents)):
            pi = 3.14156

            vtxx = random.uniform(-700, 700)
            vtxy = random.uniform(-700, 700)

            ntrk = int(random.uniform(1, 10))

            for trk in range(0, ntrk):
                pt = random.uniform(100, 1000)  # MeV/c
                phi = random.uniform(0, 2 * pi)
                theta = math.acos(random.uniform(0, 1))  # formard tracks

                charge = 0

                # while charge == 0:
                #    charge = random.randint(-1,1)

                station = 1
                for Z in zplane:
                    x, y, z = LineExtrapToZ(vtxx, vtxy, theta, phi, Z)

                    #  if (x,y,z) == (0,0,0):
                    #      continue
                    if math.fabs(x) >= 750 or math.fabs(y) >= 750:
                        continue

                    wireid, dr, lr = GetWire(x, y, station)
                    x0, y0, station, z0, xyuv = GetWirePos(wireid)

                    #                   if random.uniform(0,1) < eff:
                    f.write(
                        "%d\t%d\t%f\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%d\n"
                        % (evt, wireid, dr, lr, station, trk, x, y, x0, y0, z0)
                    )
                    #                 print(evt,wireid,dr,lr,station,trk,x,y,x0,y0,z0)
                    station = station + 1

            #    # add noise hits
            #            nhit = int(random.uniform(ntrk * ntrk * 35/2, ntrk * ntrk * 35/1)) # up to 100 noise hits
            #            for ihit in range(0, nhit):
            #                sta = int(random.uniform(0,35))
            #                R = radii[sta]
            #                phi = random.uniform(0, 2*pi)
            #                z = random.uniform(-2386, 2386)
            #                x = R*math.cos(phi)
            #                y = R*math.sin(phi)
            #                f.write("%d\t%f\t%f\t%f\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (evt,x,y,z,sta,-1,0,0,0,0,0,0) )


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must contain a mapping at the top level.")
    return cfg


def normalize_config(cfg, args):
    normalized = {
        "events": cfg.get("events", 1),
        "seed": cfg.get("seed", 42),
        "output_dir": cfg.get("output_dir", "outputs/drift_sim"),
        "data_filename": cfg.get("data_filename", "output.tsv"),
        "config_filename": cfg.get("config_filename", "config.yaml"),
        "seed_lock_filename": cfg.get("seed_lock_filename", "seed.lock"),
    }

    if args.events is not None:
        normalized["events"] = args.events
    if args.seed is not None:
        normalized["seed"] = args.seed
    if args.output_dir is not None:
        normalized["output_dir"] = args.output_dir
    if args.data_filename is not None:
        normalized["data_filename"] = args.data_filename

    normalized["events"] = int(normalized["events"])
    if normalized["events"] < 0:
        raise ValueError("events must be greater than or equal to 0.")

    if normalized["seed"] is None:
        normalized["seed"] = random.SystemRandom().randint(0, 2**32 - 1)
    normalized["seed"] = int(normalized["seed"])

    return normalized


def save_effective_config(cfg, output_dir):
    config_path = output_dir / cfg["config_filename"]
    seed_lock_path = output_dir / cfg["seed_lock_filename"]

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    with seed_lock_path.open("w", encoding="utf-8") as f:
        f.write(f"{cfg['seed']}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate drift simulation data from a YAML config."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/drift_sim.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Override output directory from config.",
    )
    parser.add_argument(
        "-n",
        "--events",
        type=int,
        help="Override number of events from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override seed from config.",
    )
    parser.add_argument(
        "--data-filename",
        help="Override output data filename from config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    cfg = normalize_config(load_config(config_path), args)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg["seed"])
    output_path = output_dir / cfg["data_filename"]
    generate(cfg["events"], output_path)
    save_effective_config(cfg, output_dir)

    print(f"Data saved to: {output_path}")
    print(f"Config saved to: {output_dir / cfg['config_filename']}")
    print(f"Seed lock saved to: {output_dir / cfg['seed_lock_filename']}")


if __name__ == "__main__":
    main()
