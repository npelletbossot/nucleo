#!/usr/bin/env python3
# plot_occupancy.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# dossier de sortie par défaut
out_dir = "out_test"
if len(sys.argv) > 1:
    out_dir = sys.argv[1]

OUTDIR = Path(out_dir)
os.makedirs(OUTDIR, exist_ok=True)

RES = OUTDIR / "ESC_chr14_47450015_47750172.bed.result.bed"
OCC = OUTDIR / "parsed_positions.csv"

# le fichier commence par un header comment (#), puis deux colonnes: position, relative-occupancy
df = pd.read_csv(OCC, sep="\t", comment="#", header=None, names=["pos","occ"])
print("Occupancy rows:", len(df))

# plot occupancy (zoom initial)
plt.figure(figsize=(10,3))
plt.plot(df["pos"], df["occ"])
plt.xlabel("Position (bp)")
plt.ylabel("Relative occupancy")
plt.title("Occupancy (smoothed)")
plt.tight_layout()
plt.savefig(OUTDIR / "occupancy_full.png")
print("Saved", OUTDIR / "occupancy_full.png")

# zoom around start..end (useful)
# detect non-zero region
nz = df[df["occ"]>0]
if len(nz) > 0:
    s = nz["pos"].min()
    e = nz["pos"].max()
    rng = 2000  # flank
    sub = df[(df["pos"] >= s-rng) & (df["pos"] <= e+rng)]
    plt.figure(figsize=(10,3))
    plt.plot(sub["pos"], sub["occ"])
    plt.xlim(s-rng, e+rng)
    plt.xlabel("Position (bp)")
    plt.ylabel("Relative occupancy")
    plt.title("Occupancy zoom")
    plt.tight_layout()
    plt.savefig(OUTDIR / "occupancy_zoom.png")
    print("Saved", OUTDIR / "occupancy_zoom.png")
else:
    print("No non-zero occupancy values found.")
