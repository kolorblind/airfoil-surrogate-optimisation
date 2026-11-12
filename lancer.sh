#!/bin/sh
# Usage : lancer.sh [tout | test | campagne | modeles | optimisation | figures | shell]
set -e
cd /resultats
case "${1:-tout}" in
  test)         python /app/chaine_xfoil.py ;;
  campagne)     python /app/campagne.py ;;
  modeles)      python /app/modeles.py ;;
  optimisation) python /app/optimisation.py ;;
  figures)      python /app/figures_slides.py ;;
  tout)
    echo "== 1/4 Chaîne de calcul";        python /app/chaine_xfoil.py
    echo "== 2/4 Plan d'expériences";      python /app/campagne.py
    echo "== 3/4 Modèles de substitution"; python /app/modeles.py
    echo "== 4/4 Optimisation";            python /app/optimisation.py
    echo "== Terminé : résultats dans le dossier monté sur /resultats" ;;
  shell)        exec /bin/bash ;;
  *) echo "Étape inconnue : $1 (tout, test, campagne, modeles, optimisation, figures, shell)"; exit 1 ;;
esac
