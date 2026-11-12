"""
Étape 2 du projet : la campagne de calcul.
Plan d'expériences (hypercube latin) -> boucle de calculs XFOIL -> fichier de résultats.

Usage : python campagne.py
Produit : resultats_campagne.csv et fig2_campagne.png
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc

from chaine_xfoil import naca4, solveur

N_CALCULS = 80

# ---------------------------------------------------------------------------
# 1. L'espace de conception : 4 paramètres et leurs bornes
# ---------------------------------------------------------------------------
PARAMETRES = {                      # nom : (min, max)
    "cambrure":     (0.00, 0.06),   # 0 à 6 % de la corde
    "pos_cambrure": (0.20, 0.60),   # cambrure max entre 20 et 60 % de la corde
    "epaisseur":    (0.08, 0.18),   # 8 à 18 % de la corde
    "incidence":    (0.0, 8.0),     # en degrés
}
noms = list(PARAMETRES)
bornes_min = np.array([b[0] for b in PARAMETRES.values()])
bornes_max = np.array([b[1] for b in PARAMETRES.values()])

# ---------------------------------------------------------------------------
# 2. Plan d'expériences : hypercube latin
# ---------------------------------------------------------------------------
# tirage de N points dans le cube [0, 1]^4, un point par tranche sur chaque axe
plan_unitaire = qmc.LatinHypercube(d=len(noms), seed=0).random(N_CALCULS)
# mise à l'échelle vers les vraies bornes
plan = qmc.scale(plan_unitaire, bornes_min, bornes_max)

# ---------------------------------------------------------------------------
# 3. Boucle de calculs
# ---------------------------------------------------------------------------
lignes = []
debut = time.time()
for i, (m, p, t, alpha) in enumerate(plan):
    cl, cd, finesse = solveur(m, p, t, alpha)
    lignes.append({"cambrure": m, "pos_cambrure": p, "epaisseur": t,
                   "incidence": alpha, "CL": cl, "CD": cd, "finesse": finesse})
    etat = "échec" if np.isnan(finesse) else f"finesse = {finesse:6.1f}"
    print(f"calcul {i + 1:3d}/{N_CALCULS} : {etat}")

resultats = pd.DataFrame(lignes)
resultats.to_csv("resultats_campagne.csv", index=False)

n_echecs = resultats["finesse"].isna().sum()
print(f"\n{N_CALCULS} calculs en {time.time() - debut:.0f} s, "
      f"dont {n_echecs} échecs ({100 * n_echecs / N_CALCULS:.0f} %)")
print("meilleur profil de la campagne :")
print(resultats.loc[resultats["finesse"].idxmax()])

# ---------------------------------------------------------------------------
# 4. Figure : mosaïque des profils calculés, colorés par leur finesse
# ---------------------------------------------------------------------------
ordre = resultats.sort_values("finesse", ascending=False, na_position="last")
fmin, fmax = resultats["finesse"].min(), resultats["finesse"].max()
cmap = plt.get_cmap("viridis")

ncol = 10
nlig = int(np.ceil(N_CALCULS / ncol))
fig, axes = plt.subplots(nlig, ncol, figsize=(16, 1.3 * nlig))
for ax, (_, cas) in zip(axes.flat, ordre.iterrows()):
    X, Y = naca4(cas["cambrure"], cas["pos_cambrure"], cas["epaisseur"])
    # on tourne le profil de son incidence pour la voir
    a = -np.radians(cas["incidence"])
    Xr = X * np.cos(a) - Y * np.sin(a)
    Yr = X * np.sin(a) + Y * np.cos(a)
    if np.isnan(cas["finesse"]):
        couleur, texte = "lightgray", "échec"
    else:
        couleur = cmap((cas["finesse"] - fmin) / (fmax - fmin))
        texte = f"{cas['finesse']:.0f}"
    ax.fill(Xr, Yr, color=couleur)
    ax.set_title(texte, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.25, 0.15)
    ax.axis("off")
for ax in axes.flat[N_CALCULS:]:
    ax.axis("off")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(fmin, fmax))
fig.colorbar(sm, ax=axes, shrink=0.6, label="finesse CL/CD")
fig.suptitle(f"Plan d'expériences : {N_CALCULS} profils calculés par XFOIL, "
             "classés du plus fin au moins fin", fontweight="bold")
fig.savefig("fig2_campagne.png", dpi=150, bbox_inches="tight")
plt.show()
