# Airfoil optimisation with a surrogate model, containerised and run on Scaleway

A small but complete simulation workflow (solver, design of experiments,
surrogate models, optimisation), packaged as a Docker image, checked to give
the same results as a native run, and executed as a Scaleway Serverless Job
with its results written to Scaleway Object Storage.

    Dockerfile ──build──> image ──push──> Container Registry
                                                 │
                                   Serverless Job (1 vCPU, 2 GB)
                                   XFOIL campaign, models, optimisation
                                                 │
                                   Object Storage bucket (CSV + figures)

## The problem

Find the NACA 4-digit airfoil with the best lift-to-drag ratio (L/D), under a
limited compute budget, as if each solver call took hours. The problem is
deliberately small and cheap, so that every prediction can be checked against
the real solver.

| Parameter        | Range      |
|------------------|------------|
| Camber           | 0 to 6 %   |
| Camber position  | 20 to 60 % |
| Thickness        | 8 to 18 %  |
| Angle of attack  | 0 to 8°    |

Solver: XFOIL, viscous, Re = 10^6.

## The workflow

1. `chaine_xfoil.py`: parametric NACA geometry and an XFOIL call driven from
   Python. Validated on NACA 2412 at 4° (L/D = 102.4).
2. `campagne.py`: 80 runs on a Latin hypercube. Non-converged runs (2 %) are
   handled without stopping the campaign.
3. `modeles.py`: random forest vs neural network, compared with 5-fold
   cross-validation (R² 0.78 vs 0.93).
4. `optimisation.py`: an ensemble of 5 networks; the criterion mean + κ·std
   (prediction + uncertainty) is maximised by differential evolution, each
   proposal is checked with XFOIL and added to the data. 8 iterations raise
   the best L/D from 146 to about 207.
5. `envoi_s3.py`: optional upload of the results to an S3-compatible bucket,
   configured through environment variables only.

## Run it locally

With Docker only:

    docker build -t profil .
    docker run --rm -v "$PWD/resultats:/resultats" profil         # full workflow
    docker run --rm -v "$PWD/resultats:/resultats" profil test    # NACA 2412 check

Steps: `tout` (all), `test`, `campagne`, `modeles`, `optimisation`, `shell`.
CSV files and figures are written to `resultats/`.

## Run it on Scaleway

1. Container Registry: create a private namespace in fr-par, then build for
   amd64 (Serverless Jobs only run amd64 images) and push:

       docker login rg.fr-par.scw.cloud/<namespace> -u nologin
       docker build --platform linux/amd64 -t rg.fr-par.scw.cloud/<namespace>/profil:latest .
       docker push rg.fr-par.scw.cloud/<namespace>/profil:latest

2. Object Storage: create a private bucket in fr-par.
3. IAM: create an application with a single policy, ObjectStorageFullAccess
   on one project, and generate its API key. The job never uses a personal,
   organisation-wide key.
4. Secret Manager: store that key's access key and secret key as two secrets.
5. Serverless Jobs: create a job from the image, with the environment variable
   `S3_BUCKET=<bucket>` and the two secrets referenced as `AWS_ACCESS_KEY_ID`
   and `AWS_SECRET_ACCESS_KEY`. No command is needed: the image runs the full
   workflow by default.
6. Run the job: the results appear in the bucket under `run-<date>-<time>/`.

Cost: the run fits in the monthly free allowance of Serverless Jobs; the
private image and the results cost a few cents per month in storage.

## Reproducibility

- The XFOIL campaign is bit-identical between native macOS and the Linux
  container (ARM and x86), including the run on Scaleway Serverless Jobs:
  same 80 L/D values, same 2 failures.
- The optimisation path differs slightly across platforms, because network
  training depends on the linear algebra library (final best L/D 207 to 209).
- Ubuntu's XFOIL package is built with floating-point traps enabled: it
  crashed right after convergence, before writing its polar file. The
  container disables the traps with a two-line library loaded through
  `LD_PRELOAD` (`docker/nofpe.c`).

## Limitations and next steps

- XFOIL is a low-fidelity solver (2D, incompressible); the same approach
  applies to a RANS solver, where each run would need a multi-core node
  rather than a single vCPU.
- The 80 campaign runs are independent and run sequentially here; on an HPC
  cluster they would map naturally to a Slurm job array.
- Three parameters end up at their bounds: the optimum lies outside the
  domain, and an 8 % thick airfoil raises structural questions.
- The exploration weight κ is fixed; decreasing it over the iterations would
  favour exploitation at the end of the loop.
