"""
Étape 3 du projet : les modèles de substitution.
Forêt aléatoire et réseau de neurones entraînés sur les 80 calculs XFOIL,
comparés en validation croisée.

Usage : python modeles.py
Produit : fig3_modeles.png
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore", category=RuntimeWarning)   # bug NumPy/Accelerate sur Mac

# ---------------------------------------------------------------------------
# 1. Les données : on retire les calculs qui ont échoué
# ---------------------------------------------------------------------------
donnees = pd.read_csv("resultats_campagne.csv").dropna(subset=["finesse"])
noms = ["cambrure", "pos_cambrure", "epaisseur", "incidence"]
X = donnees[noms].to_numpy()
y = donnees["finesse"].to_numpy()
print(f"{len(y)} calculs utilisables")

# ---------------------------------------------------------------------------
# 2. Les deux modèles
# ---------------------------------------------------------------------------
foret = RandomForestRegressor(n_estimators=300, random_state=0)

# Pour le réseau, les paramètres n'ont pas la même échelle (incidence jusqu'à 8,
# cambrure jusqu'à 0,06) : on les normalise. Le Pipeline refait la normalisation
# à l'intérieur de chaque pli de validation croisée, sans tricher.
reseau = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(32, 32), solver="lbfgs", alpha=1e-3,
                 max_iter=5000, random_state=0),
)
# On normalise aussi la sortie du réseau (finesse de l'ordre de 100)
y_moy, y_std = y.mean(), y.std()

# ---------------------------------------------------------------------------
# 3. Validation croisée : chaque point est prédit par un modèle qui ne l'a pas vu
# ---------------------------------------------------------------------------
cv = KFold(5, shuffle=True, random_state=0)
pred_foret = cross_val_predict(foret, X, y, cv=cv)
pred_reseau = cross_val_predict(reseau, X, (y - y_moy) / y_std, cv=cv) * y_std + y_moy

for nom, pred in [("forêt", pred_foret), ("réseau", pred_reseau)]:
    r2 = r2_score(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    print(f"{nom:7s} : R² = {r2:.3f}, erreur typique = {rmse:.1f} points de finesse")

# ---------------------------------------------------------------------------
# 4. Modèles finaux, entraînés sur toutes les données
# ---------------------------------------------------------------------------
foret.fit(X, y)
reseau.fit(X, (y - y_moy) / y_std)

def predit_reseau(Xn):
    return reseau.predict(Xn) * y_std + y_moy

# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))   # grille 2 x 2 : plus de place
axes = axes.ravel()                                  # liste à plat : axes[0] ... axes[3]

# (a) et (b) : prédiction contre vérité, en validation croisée
for ax, nom, pred, col in [(axes[0], "Forêt aléatoire", pred_foret, "tab:green"),
                           (axes[1], "Réseau de neurones", pred_reseau, "tab:blue")]:
    ax.scatter(y, pred, color=col, edgecolor="k", s=30)
    lim = [y.min() - 5, y.max() + 5]
    ax.plot(lim, lim, "k--", lw=1, label="prédiction parfaite")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("finesse calculée par XFOIL")
    ax.set_ylabel("finesse prédite (point jamais vu)")
    ax.set_title(f"{nom}\nR² = {r2_score(y, pred):.2f} en validation croisée")
    ax.legend(loc="upper left", fontsize=9)

# (c) importance des paramètres selon la forêt
ordre = np.argsort(foret.feature_importances_)
axes[2].barh(np.array(noms)[ordre], foret.feature_importances_[ordre], color="tab:green")
axes[2].set_xlabel("importance (part de la variance expliquée)")
axes[2].set_title("Quels paramètres comptent ?")

# (d) carte de la finesse prédite par le réseau : incidence x cambrure,
#     les deux autres paramètres fixés à leur valeur médiane
inc = np.linspace(0, 8, 120)
cam = np.linspace(0, 0.06, 120)
I, C = np.meshgrid(inc, cam)
grille = np.column_stack([C.ravel(),
                          np.full(I.size, np.median(X[:, 1])),
                          np.full(I.size, np.median(X[:, 2])),
                          I.ravel()])
carte = predit_reseau(grille).reshape(I.shape)
cs = axes[3].contourf(I, C * 100, carte, levels=20, cmap="viridis")
axes[3].scatter(X[:, 3], X[:, 0] * 100, c="w", edgecolor="k", s=15,
                label="calculs XFOIL")
fig.colorbar(cs, ax=axes[3], label="finesse prédite")
axes[3].set_xlabel("incidence (°)")
axes[3].set_ylabel("cambrure (% de la corde)")
axes[3].set_title("Le modèle comme carte : finesse prédite\n"
                  "(position cambrure et épaisseur médianes)")
axes[3].legend(loc="lower right", fontsize=8)

fig.suptitle("Modèles de substitution entraînés sur 80 calculs XFOIL",
             fontweight="bold")
fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=3.0)     # marges entre les graphiques
fig.savefig("fig3_modeles.png", dpi=150)
plt.show()
