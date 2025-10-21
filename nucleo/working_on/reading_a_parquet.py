# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
# import pyarrow as pa
# import pyarrow.parquet as pq
from pathlib import Path
import polars as pl


# ─────────────────────────────────────────────
# 2 : Reading a single file
# ─────────────────────────────────────────────

# root = "/home/nicolas/Documents/PhD/Workspace/nucleo/outputs/2025-10-21_PC/nucleo_access_0/alphachoice=ntrandom_s=35_l=100_bpmin=0_mu=180_theta=90_nt=10000"
# df = pl.read_parquet(root)
# print(df)
# print(df.columns)

# print(df["results_mean"][0].to_numpy())


# plt.figure(figsize=(8,6))
# # plt.plot(df["alpha_mean"][0][0:int(df["results_mean"][0].to_numpy().max())].to_numpy())
# plt.plot(df["results_mean"][0].to_numpy())
# plt.show()


# Scan tous les fichiers .parquet d'un coup 
root = Path("/home/nicolas/Documents/PhD/Workspace/nucleo/outputs/2025-10-21_PC/nucleo_access_0")
paths = [str(p) for p in root.glob("*/**/*.parquet")] or [str(p) for p in root.glob("*/*.parquet")]
df_merged = pl.scan_parquet(paths).collect()
print(df_merged.head)
print(df_merged.columns)

s = df_merged["s"]
l = df_merged["l"]
d = 1 / (s + l) 

plot = "density"
plt.figure(figsize=(8,6))
plt.title(f"mu={df_merged["mu"].to_numpy()[0]} - theta={df_merged["theta"].to_numpy()[0]}")
if plot == "density":
    plt.scatter(d, df_merged["vi_med"], label="vi_med")
    plt.scatter(d, df_merged["v_mean"], label="v_mean")
    plt.xlabel("density")
else : 
    plt.scatter(df_merged["l"], df_merged["vi_med"], label="vi_med")
    plt.scatter(df_merged["l"], df_merged["v_mean"], label="v_mean")
plt.grid(True, which="both")
# plt.loglog(True)
plt.legend()
plt.show()




# ─────────────────────────────────────────────
# 3 : Generating a linear scale of density
# ─────────────────────────────────────────────

# s = 35
# N = 10
# Lmin, Lmax = 0, 100  # bornes souhaitées pour l (entières)

# # d linéaire entre 1/(s+Lmax) et 1/(s+Lmin)
# d = np.linspace(1/(s+Lmax), 1/(s+Lmin), N)

# # l déduit de d
# l = 1/d - s

# # si tu veux l entier naturel :
# l_int = np.rint(l).astype(int)  # arrondi au plus proche
# d_int = 1/(s + l_int)           # d correspondant après arrondi

# print("d linéaire :", d)
# print("l (float)  :", l)
# print("l (entier) :", l_int)
# print("d après arrondi l :", d_int)

# plt.figure(figsize=(8,6))
# plt.plot(d, 'o-', label="d linéaire (théorique)")
# plt.plot(d_int, 's--', label="d avec l entier")
# plt.xlabel("index")
# plt.ylabel("d = 1/(s+l)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
