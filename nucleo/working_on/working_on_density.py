# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl


# ─────────────────────────────────────────────
# 2 : Generating a linear scale of density
# ─────────────────────────────────────────────

# Calculs
s = 35
N = 20
Lmin, Lmax = 1e-6, 100  # bornes souhaitées pour l (entières)
d = np.linspace(1/(s+Lmax), 1/(s+Lmin), N)
l = 1/d - s
l_int = np.rint(l).astype(int)  # arrondi au plus proche
d_int = 1/(s + l_int)           # d correspondant après arrondi

# Prints
print("d linéaire :", d)
print("l (float)  :", l)
print("l (entier) :", l_int)
print("d après arrondi l :", d_int)

# Plot
plot = False
if plot:
    plt.figure(figsize=(8,6))
    plt.plot(d, 'o-', label="d linéaire (théorique)")
    plt.plot(d_int, 's--', label="d avec l entier")
    plt.xlabel("index")
    plt.ylabel("d = 1/(s+l)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 3 : Speeds in function of linker and density
# ─────────────────────────────────────────────

# DataFrame
root = Path("/home/nicolas/Documents/PhD/Workspace/nucleo/outputs/2025-10-21_PC/nucleo_access_0")
paths = [str(p) for p in root.glob("*/**/*.parquet")] or [str(p) for p in root.glob("*/*.parquet")]
df_merged = pl.scan_parquet(paths).collect()
df_sorted = df_merged.sort(by="l", descending=False)

# print(df_sorted.head)
# print(df_sorted.columns)
# print(df_sorted.row(0))

# Values
vi_med = df_sorted["vi_med"].to_numpy()
v_mean = df_sorted["v_mean"].to_numpy()
mu     = df_sorted["mu"].to_numpy()[0]      # 1 seul mu
th     = df_sorted["theta"].to_numpy()[0]   # 1 seul theta
s      = df_sorted["s"].to_numpy()[0]       # 1 seul s
l      = df_sorted["l"].to_numpy()          # plusieurs l
d      = 1 / (s + l)                        # plusieurs d

# Plot
plt.figure(figsize=(8,6))
plt.plot(df_sorted["results_mean"][0], label=f"results_mean_0__l={l[0]}")
plt.plot(df_sorted["results_mean"][1], label=f"results_mean_1__l={l[1]}")
plt.plot(df_sorted["results_mean"][2], label=f"results_mean_2__l={l[2]}")
plt.plot(df_sorted["results_mean"][3], label=f"results_mean_3__l={l[3]}")
plt.legend()
plt.show()


print(df_sorted["l"])
print(df_sorted["v_mean"])
print(df_sorted["vi_med"])


for line in range(len(l)):
    print(np.count_nonzero(df_sorted["vi_distrib"][line].to_numpy()))


# Plot
plot = "density"
plt.figure(figsize=(8,6))
plt.title(f"mu={mu} - theta={th}")
if plot == "density":
    plt.scatter(d, vi_med, label="vi_med")
    plt.scatter(d, v_mean, label="v_mean")
    plt.xlabel("density")
elif plot == "linker": 
    plt.scatter(l, vi_med, label="vi_med")
    plt.scatter(l, v_mean, label="v_mean")
plt.grid(True, which="both")
plt.legend()
plt.show()


# ─────────────────────────────────────────────
# 4 : Speeds in function of linker and density
# ─────────────────────────────────────────────

# 700 - 300

# Values
tmax  = df_sorted["tmax"].to_numpy()[0]
dt    = df_sorted["dt"].to_numpy()[0]
results_list = df_sorted["results_mean"].to_list()

# Plot
cmap = plt.cm.jet
norm = plt.Normalize(vmin=min(l), vmax=max(l))
plt.figure(figsize=(8,6))
plt.title(f"mu={mu} - theta={th}")

for l_val, y in sorted(zip(l, results_list), key=lambda t: t[0]):
    d_val = 1 / (s + l_val)
    color = cmap(norm(l_val))
    x = np.linspace(0, tmax, num=len(y), endpoint=False)
    plt.plot(x, y, label=f"l={l_val} - d={np.round(d_val, 3)} model", ls="-", color=color)
    y_th = mu * l_val / (l_val + s)
    plt.plot(x, x*y_th, label=f"l={l_val} - d={np.round(d_val, 3)} theo", ls="--", color=color)

plt.xlabel("time")
plt.ylabel("position")
plt.grid(True, which="both")
plt.legend(title="linkersize - density", ncol=2, fontsize=8)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────
# 5 : Gaps betwwen theoretical and model values
# ─────────────────────────────────────────────

# Theoretical speed + extraction
v_th = mu * l / (l + s)
diff = v_th - v_mean

# Subplot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(d, diff, marker='o')
axes[0].set_title(f"mu={mu} - theta={th}")
axes[0].set_xlabel("density d")
axes[0].set_ylabel("v_th - v_mean")
axes[0].grid(True, which="both")
axes[1].plot(l, diff, marker='o')
axes[1].set_title(f"mu={mu} - theta={th}")
axes[1].set_xlabel("linker size l")
axes[1].set_ylabel("v_th - v_mean")
axes[1].grid(True, which="both")
plt.tight_layout()
plt.show()