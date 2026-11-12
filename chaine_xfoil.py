"""
Étape 1 du projet : la chaîne de calcul.
paramètres du profil -> géométrie -> XFOIL -> finesse CL/CD

Usage : python chaine_xfoil.py
"""
import os
import subprocess
import tempfile

import numpy as np
import matplotlib.pyplot as plt

REYNOLDS = 1e6          # nombre de Reynolds fixé pour toute l'étude


# ---------------------------------------------------------------------------
# 1. Géométrie : profil NACA 4 chiffres à paramètres continus
# ---------------------------------------------------------------------------
def naca4(cambrure, pos_cambrure, epaisseur, n=120):
    """Coordonnées (x, y) d'un profil NACA 4 chiffres.

    cambrure      m : ex. 0.02 pour 2 % de la corde
    pos_cambrure  p : position de la cambrure max, ex. 0.4 pour 40 %
    epaisseur     t : épaisseur max, ex. 0.12 pour 12 %
    Ordre des points : bord de fuite -> extrados -> bord d'attaque -> intrados
    -> bord de fuite (format attendu par XFOIL).
    """
    m, p, t = cambrure, pos_cambrure, epaisseur
    beta = np.linspace(0, np.pi, n)
    x = (1 - np.cos(beta)) / 2                      # points resserrés aux bords
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1015 * x**4)
    if m == 0:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x),
                       2 * m / (1 - p)**2 * (p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    X = np.concatenate([xu[::-1], xl[1:]])
    Y = np.concatenate([yu[::-1], yl[1:]])
    return X, Y


# ---------------------------------------------------------------------------
# 2. Le "solveur" : un appel à XFOIL piloté par Python
# ---------------------------------------------------------------------------
def solveur(cambrure, pos_cambrure, epaisseur, incidence, verbose=False):
    """Lance XFOIL et renvoie (CL, CD, finesse).

    Renvoie (nan, nan, nan) si XFOIL ne converge pas : c'est fréquent près du
    décrochage, et la chaîne doit le gérer proprement plutôt que planter.
    """
    with tempfile.TemporaryDirectory() as dossier:
        fichier_profil = os.path.join(dossier, "profil.dat")
        fichier_polaire = os.path.join(dossier, "polaire.txt")

        X, Y = naca4(cambrure, pos_cambrure, epaisseur)
        with open(fichier_profil, "w") as f:
            f.write("profil\n")
            for x, y in zip(X, Y):
                f.write(f"{x:.6f} {y:.6f}\n")

        commandes = "\n".join([
            "PLOP", "G F", "",                  # pas de graphiques
            "LOAD profil.dat",                 # chemin court : XFOIL tronque les noms longs
            "PANE",                             # rediscrétisation du contour
            "OPER",
            f"VISC {REYNOLDS:.0f}",             # calcul visqueux
            "ITER 200",
            "PACC", "polaire.txt", "",         # enregistrement des résultats
            f"ALFA {incidence:.3f}",
            "", "QUIT", "",
        ])
        try:
            sortie = subprocess.run(["xfoil"], input=commandes, text=True,
                                    capture_output=True, timeout=30,
                                    cwd=dossier)  # XFOIL travaille dans le dossier temporaire
        except subprocess.TimeoutExpired:
            return np.nan, np.nan, np.nan
        if verbose:
            print(sortie.stdout)

        # lecture de la polaire : les résultats sont sous la ligne de tirets
        if not os.path.exists(fichier_polaire):
            return np.nan, np.nan, np.nan
        with open(fichier_polaire) as f:
            lignes = [l.strip() for l in f if l.strip()]
        tirets = [i for i, l in enumerate(lignes) if l.startswith("---")]
        if not tirets or tirets[-1] == len(lignes) - 1:
            return np.nan, np.nan, np.nan       # pas de convergence
        valeurs = lignes[-1].split()
        cl, cd = float(valeurs[1]), float(valeurs[2])
        return cl, cd, cl / cd


# ---------------------------------------------------------------------------
# 3. Test de la chaîne et première figure
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # vérification sur le cas déjà calculé à la main : NACA 2412 à 4°
    cl, cd, f = solveur(0.02, 0.4, 0.12, 4.0)
    print(f"NACA 2412 à 4° : CL = {cl:.4f}, CD = {cd:.5f}, finesse = {f:.1f}")

    # quelques profils de l'espace de conception, avec leur finesse
    cas = [(0.00, 0.40, 0.12, "symétrique"),
           (0.02, 0.40, 0.12, "NACA 2412"),
           (0.05, 0.30, 0.10, "très cambré, fin"),
           (0.03, 0.55, 0.18, "épais, cambrure arrière")]
    fig, axes = plt.subplots(len(cas), 1, figsize=(8, 7), sharex=True)
    for ax, (m, p, t, nom) in zip(axes, cas):
        X, Y = naca4(m, p, t)
        _, _, f = solveur(m, p, t, 4.0)
        ax.fill(X, Y, color="steelblue")
        ax.set_aspect("equal")
        ax.set_ylim(-0.15, 0.2)
        ax.axis("off")
        ax.set_title(f"{nom} (m={m}, p={p}, t={t}) : finesse à 4° = {f:.0f}",
                     fontsize=10, loc="left")
    fig.suptitle("Espace de conception : la forme change, la finesse aussi",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig("fig1_profils.png", dpi=150)
    plt.show()
