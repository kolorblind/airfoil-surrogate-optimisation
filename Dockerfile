# Chaîne d'optimisation d'un profil d'aile : XFOIL + plan d'expériences
# + modèles de substitution + optimisation, dans un conteneur reproductible.

# Étape 1 : compiler le petit correctif pour XFOIL
FROM ubuntu:24.04 AS correctif
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*
COPY docker/nofpe.c /tmp/nofpe.c
RUN gcc -shared -fPIC -O2 -o /tmp/libnofpe.so /tmp/nofpe.c

# Étape 2 : l'image finale
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app

# XFOIL est disponible en paquet dans Ubuntu (dépôt universe)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        xfoil python3 python3-venv fonts-ibm-plex \
 && rm -rf /var/lib/apt/lists/*

# Le XFOIL d'Ubuntu est compilé avec les exceptions flottantes activées :
# il plante juste après la convergence, avant d'écrire la polaire.
# Le correctif les désactive, et un script « xfoil » le charge à chaque appel.
COPY --from=correctif /tmp/libnofpe.so /usr/local/lib/libnofpe.so
COPY docker/xfoil /usr/local/bin/xfoil
RUN chmod +x /usr/local/bin/xfoil

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && pip install --no-cache-dir -r /tmp/requirements.txt

COPY *.py lancer.sh /app/
RUN chmod +x /app/lancer.sh

WORKDIR /resultats
ENTRYPOINT ["/app/lancer.sh"]
CMD ["tout"]
