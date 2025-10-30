#!/usr/bin/env python3
# analyse_nucpos.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack
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
OCSV = OUTDIR / "parsed_positions.csv"
CSV = OUTDIR / "parsed_positions.csv"
NUC_SIZE = 147

df = pd.read_csv(CSV)
print("Loaded positions:", len(df))

def compute_linkers(df):
    rows = []
    for chrom, g in df.groupby("chr"):
        pos = np.sort(g["pos"].values)
        if len(pos) < 2: 
            continue
        linkers = (pos[1:] - pos[:-1]) - NUC_SIZE
        rows.append(pd.DataFrame({"chr": chrom, "linker": linkers}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["chr","linker"])

linkers = compute_linkers(df)
print("Linker stats (bp): mean=%.2f  median=%.2f  n=%d" %
      (linkers["linker"].mean(), linkers["linker"].median(), len(linkers)))

# Histogramme des linkers
plt.figure(figsize=(7,4))
plt.hist(linkers["linker"].dropna(), bins=80)
plt.xlabel("Linker length (bp)")
plt.ylabel("Count")
plt.title("Distribution des linkers")
plt.tight_layout()
plt.savefig(OUTDIR / "linker_hist.png")
print("Saved", OUTDIR / "linker_hist.png")

# FFT périodicité — on construit un occupancy binaire sur la région
# choisir chromosome principal
chrom = df["chr"].iloc[0]
pos = df[df["chr"]==chrom]["pos"].values
start = int(pos.min() - 2000)
end   = int(pos.max() + 2000)
L = end - start + 1
occ = np.zeros(L, dtype=float)
for p in pos:
    left = max(0, p - NUC_SIZE//2 - start)
    right= min(L-1, p + NUC_SIZE//2 - start)
    occ[left:right+1] = 1.0

# FFT
f = fftpack.fft(occ - occ.mean())
power = np.abs(f)**2
freq = fftpack.fftfreq(L, d=1.0)
period_bp = np.where(freq!=0, 1.0/np.abs(freq), np.inf)

mask = (period_bp > 10) & (period_bp < 400)  # on regarde 10-400 bp
plt.figure(figsize=(7,4))
plt.plot(period_bp[mask], power[mask])
plt.xlim(0,250)
plt.xlabel("Period (bp)")
plt.ylabel("Power")
plt.title("Spectre périodique ({} ; {}..{})".format(chrom, start, end))
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(OUTDIR / "period_spectrum.png")
print("Saved", OUTDIR / "period_spectrum.png")

# Save linkers table
linkers.to_csv(OUTDIR / "linkers.csv", index=False)
print("Saved", OUTDIR / "linkers.csv")
