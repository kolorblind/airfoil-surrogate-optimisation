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
    echo "== 1/5 Chaîne de calcul";        python /app/chaine_xfoil.py
    echo "== 2/5 Plan d'expériences";      python /app/campagne.py
    echo "== 3/5 Modèles de substitution"; python /app/modeles.py
    echo "== 4/5 Optimisation";            python /app/optimisation.py
    echo "== 5/5 Envoi vers S3";           python /app/envoi_s3.py
    echo "== Terminé" ;;
  shell)        exec /bin/bash ;;
  *) echo "Étape inconnue : $1 (tout, test, campagne, modeles, optimisation, figures, shell)"; exit 1 ;;
esac
