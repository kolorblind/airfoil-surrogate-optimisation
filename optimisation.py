"""
Étape 4 du projet : l'optimisation par modèle de substitution.
On cherche le profil de meilleure finesse en utilisant le réseau de neurones
comme remplaçant d'XFOIL, puis on vérifie chaque proposition avec le vrai solveur.

Usage : python optimisation.py
Produit : resultats_optimisation.csv et fig4_optimisation.png
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chaine_xfoil import naca4, solveur

warnings.filterwarnings("ignore", category=RuntimeWarning)

N_ITERATIONS = 8          # budget : 8 calculs XFOIL supplémentaires
N_RESEAUX = 5             # taille de l'ensemble de réseaux
K_EXPLORATION = 1.0       # poids de l'incertitude dans le critère

noms = ["cambrure", "pos_cambrure", "epaisseur", "incidence"]
bornes = [(0.00, 0.06), (0.20, 0.60), (0.08, 0.18), (0.0, 8.0)]

# ---------------------------------------------------------------------------
# 1. Point de départ : les 80 calculs de la campagne
# ---------------------------------------------------------------------------
donnees = pd.read_csv("resultats_campagne.csv")
donnees["origine"] = "plan d'expériences"
# Un calcul qui a échoué reçoit la pire finesse observée : on dit ainsi au
# modèle « n'allez pas par là », au lieu de le laisser reproposer ce point.
pire = donnees["finesse"].min()
donnees["finesse_modele"] = donnees["finesse"].fillna(pire)
meilleur_initial = donnees["finesse"].max()
print(f"meilleure finesse du plan d'expériences : {meilleur_initial:.1f}")


def entraine_ensemble(X, y):
    """Entraîne N_RESEAUX réseaux identiques, seule la graine change.
    Là où ils sont d'accord, le modèle est sûr de lui ; là où ils divergent,
    il manque des données : c'est notre mesure d'incertitude."""
    moy, std = y.mean(), y.std()
    reseaux = []
    for graine in range(N_RESEAUX):
        r = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32, 32), solver="lbfgs",
                         alpha=1e-3, max_iter=5000, random_state=graine))
        r.fit(X, (y - moy) / std)
        reseaux.append(r)

    def predit(Xn):
        P = np.array([r.predict(Xn) for r in reseaux]) * std + moy
        return P.mean(axis=0), P.std(axis=0)
    return predit


# ---------------------------------------------------------------------------
# 2. Boucle : modèle -> optimisation sur le modèle -> vérification XFOIL
# ---------------------------------------------------------------------------
for it in range(N_ITERATIONS):
    X = donnees[noms].to_numpy()
    y = donnees["finesse_modele"].to_numpy()
    predit = entraine_ensemble(X, y)

    # Critère : prédiction + K x incertitude.
    # Le premier terme exploite (aller là où le modèle prédit bien),
    # le second explore (aller là où le modèle ne sait pas).
    def critere(x):                      # x : (4, n) avec vectorized=True
        m, s = predit(x.T)
        return -(m + K_EXPLORATION * s)  # on minimise l'opposé

    res = differential_evolution(critere, bornes, seed=it, maxiter=100,
                                 polish=False, vectorized=True,
                                 updating="deferred")
    x_new = res.x
    m_pred, s_pred = predit(x_new.reshape(1, -1))

    # vérification avec le vrai solveur
    cl, cd, f_vraie = solveur(*x_new)
    print(f"itération {it + 1} : prédit {m_pred[0]:6.1f} ± {s_pred[0]:4.1f}, "
          f"XFOIL {f_vraie:6.1f}   "
          f"(cambrure {x_new[0]*100:.1f} %, pos {x_new[1]:.2f}, "
          f"épaisseur {x_new[2]*100:.1f} %, incidence {x_new[3]:.1f}°)")

    nouvelle = dict(zip(noms, x_new), CL=cl, CD=cd, finesse=f_vraie,
                    origine="optimisation", prediction=m_pred[0],
                    incertitude=s_pred[0],
                    finesse_modele=pire if np.isnan(f_vraie) else f_vraie)
    donnees = pd.concat([donnees, pd.DataFrame([nouvelle])], ignore_index=True)

donnees.to_csv("resultats_optimisation.csv", index=False)
optim = donnees[donnees["origine"] == "optimisation"]
meilleur = donnees.loc[donnees["finesse"].idxmax()]
print(f"\nmeilleure finesse : {meilleur['finesse']:.1f} "
      f"(plan d'expériences seul : {meilleur_initial:.1f})")
print(meilleur[noms + ["CL", "CD", "finesse"]])

# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) progression : chaque calcul, et le meilleur trouvé jusque-là
ax = axes[0]
n0 = (donnees["origine"] == "plan d'expériences").sum()
ax.scatter(range(1, n0 + 1), donnees["finesse"][:n0], color="0.6", s=15,
           label="plan d'expériences (80 calculs)")
ax.scatter(range(n0 + 1, len(donnees) + 1), optim["finesse"], color="tab:red",
           s=40, zorder=3, label="propositions de l'optimiseur")
ax.plot(range(1, len(donnees) + 1), donnees["finesse"].cummax(), "k-",
        label="meilleur profil trouvé")
ax.axvline(n0 + 0.5, color="k", ls=":", lw=1)
ax.set_xlabel("numéro du calcul XFOIL")
ax.set_ylabel("finesse")
ax.set_title("Progression de la recherche")
ax.legend(fontsize=9)

# (b) prédiction du modèle contre vérification XFOIL, itération par itération
ax = axes[1]
iters = np.arange(1, len(optim) + 1)
ax.errorbar(iters, optim["prediction"], yerr=optim["incertitude"], fmt="o",
            color="tab:blue", capsize=4, label="prédiction ± incertitude")
ax.scatter(iters, optim["finesse"], color="tab:red", marker="s", s=50,
           zorder=3, label="vérification XFOIL")
ax.set_xlabel("itération d'optimisation")
ax.set_ylabel("finesse")
ax.set_title("Le modèle avait-il raison ?")
ax.legend(fontsize=9)

# (c) profil de départ contre profil optimal, inclinés de leur incidence
ax = axes[2]
def trace(m, p, t, alpha, couleur, etiquette):
    X, Y = naca4(m, p, t)
    a = -np.radians(alpha)
    ax.fill(X * np.cos(a) - Y * np.sin(a), X * np.sin(a) + Y * np.cos(a),
            color=couleur, alpha=0.6, label=etiquette)
_, _, f_ref = solveur(0.02, 0.4, 0.12, 4.0)
trace(0.02, 0.4, 0.12, 4.0, "0.5", f"NACA 2412 à 4° : finesse {f_ref:.0f}")
trace(meilleur["cambrure"], meilleur["pos_cambrure"], meilleur["epaisseur"],
      meilleur["incidence"], "tab:red",
      f"profil optimisé à {meilleur['incidence']:.1f}° : "
      f"finesse {meilleur['finesse']:.0f}")
ax.set_aspect("equal")
ax.set_xlim(-0.05, 1.05)
ax.axis("off")
ax.legend(loc="lower center", fontsize=9)
ax.set_title("Profil de départ et profil optimisé")

fig.suptitle("Optimisation assistée par modèle de substitution",
             fontweight="bold")
fig.tight_layout()
fig.savefig("fig4_optimisation.png", dpi=150)
plt.show()
