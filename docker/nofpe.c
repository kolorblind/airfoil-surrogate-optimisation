/* Le XFOIL d'Ubuntu est compilé avec -ffpe-trap : une opération flottante
 * bénigne après la convergence le tue (SIGFPE) avant qu'il écrive la polaire.
 * Chargée par LD_PRELOAD, cette fonction vide remplace celle de libgfortran. */
void _gfortran_set_fpe(int val) { (void)val; }
