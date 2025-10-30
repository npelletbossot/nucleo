#!/usr/bin/env python3
import pandas as pd
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

if not RES.exists():
    raise FileNotFoundError(RES)

df = pd.read_csv(RES, sep="\t", header=None, names=["chr","start","end"], comment="#")
df["pos"] = ((df["start"].astype(int) + df["end"].astype(int)) // 2).astype(int)
df["score"] = 1.0
df = df[["chr","pos","start","end","score"]]
df.to_csv(OCSV, index=False)
print(f"Wrote {OCSV} ({len(df)} rows)")
print(df.head())

