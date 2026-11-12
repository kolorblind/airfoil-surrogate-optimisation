# Airfoil optimisation with a surrogate model

A complete, containerised simulation workflow: a solver (XFOIL), a design of
experiments, two surrogate models and an optimisation loop. The problem is
deliberately small and cheap, so that every prediction can be checked against
the real solver.

## The problem

Find the NACA 4-digit airfoil with the best lift-to-drag ratio (L/D), under a
limited compute budget, as if each solver call took hours.

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

## Run it

With Docker only:

    docker build -t profil .
    docker run --rm -v "$PWD/resultats:/resultats" profil         # full workflow
    docker run --rm -v "$PWD/resultats:/resultats" profil test    # NACA 2412 check

Steps: `tout` (all), `test`, `campagne`, `modeles`, `optimisation`, `shell`.
CSV files and figures are written to `resultats/`.

## Reproducibility

- The XFOIL campaign is bit-identical between native macOS and the Linux
  container (ARM and x86): same 80 L/D values, same 2 failures.
- The optimisation path differs slightly across platforms, because network
  training depends on the linear algebra library (final best L/D 207 to 209).
- Ubuntu's XFOIL package is built with floating-point traps enabled: it
  crashed right after convergence, before writing its polar file. The
  container disables the traps with a two-line library loaded through
  `LD_PRELOAD` (`docker/nofpe.c`).

## Limitations

- XFOIL is a low-fidelity solver (2D, incompressible); the same approach
  applies to a RANS solver.
- Three parameters end up at their bounds: the optimum lies outside the
  domain, and an 8 % thick airfoil raises structural questions.
- The exploration weight κ is fixed; decreasing it over the iterations would
  favour exploitation at the end of the loop.
