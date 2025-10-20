# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tls.probabilities import proba_gamma

# ─────────────────────────────────────────────
# 2 : Datas
# ─────────────────────────────────────────────

data = np.array([1, 0, 0, 1, 2, 3, 2, 4, 5, 10, 10, 21, 28, 19, 6, 0])
y_data = data[::-1]
x_data = np.arange(0, 150 + 1, 10).astype(float)

# éviter x = 0 qui peut faire diverger la Gamma si alpha < 1
x_safe = x_data.copy()
x_safe[0] = 1e-6

# wrapper: on garde l’ordre attendu par curve_fit (x, params),
# on réutilise proba_gamma(mu, theta, L), puis on le convertit en comptes via A
def proba_gamma_fit_counts(x, mu, theta, A):
    y = proba_gamma(mu, theta, x)
    # sécurisation: remplace NaN/inf par 0 pour éviter les plantages de SVD
    y = np.where(np.isfinite(y), y, 0.0)
    return A * y

# guesses:
A0 = float(y_data.sum())
mu0, theta0 = 30.0, 20.0  # guesses raisonnables; ajuste si tu sais mieux

# Ajustement
popt, pcov = curve_fit(
    proba_gamma_fit_counts,
    x_safe,
    y_data,
    p0=[mu0, theta0, A0],
    bounds=([1e-6, 1e-6, 1e-6], [np.inf, np.inf, np.inf]),
)
print("popt [mu, theta, A]:", popt)
print("pcov:", pcov)

# ─────────────────────────────────────────────
# 3 : Plots
# ─────────────────────────────────────────────

plt.figure(figsize=(8, 6))
plt.plot(x_data, y_data, 'o', label="Data")
plt.plot(x_data, proba_gamma_fit_counts(x_safe, *popt), '--', label="Gamma fit (counts)")
plt.title("0.2 pN")
plt.xlabel("Step size (nm)")
plt.ylabel("Count")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
