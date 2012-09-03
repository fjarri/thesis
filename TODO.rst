- Create phase noise histogram for losses and see if it fits Gaussian
- Need to find criteria for fragmentation of 2-species BEC
  See Penrose-Onsager fragmentation
- Test fragmentation in 1D
- Update appendix on distributions for full 3D ellipsoid on a Bloch Sphere
  Need to find way to calculate squeezing along all directions
- Add theory for thermal modes
- Add description and results of noise tests (and implement them in the code)
- Find a way to calculate energy/chemical potential for Wigner representation
- Add a chapter on noise propagation algorithms
- Check first approximation beyond truncated Wigner - Polkovnikov 2003
- Add some explanation on how the loss operators are obtained

Examples and tests:
- Application to experiments: Swin's (Ramsey, spin echo), Riedel (spatial separation), Widera (1D)
- Noise balance test: check that if we start with 0 atoms, we will still have 0 atoms even with enabled losses
  (try different loss sources, different cutoffs etc)
- Correlations (atom bunching): check that <Psi1^+ Psi1^+ Psi1 Psi1> / <Psi1+ Psi1>^2 ~ 1 in condensate.
  May also check <Psi1^+ Psi2^+ Psi1 Psi2> / (<Psi1+ Psi1> <Psi2+ Psi2>) (should be ~1 too in condensate).
- Check correlations for thermal states (should be >1). See if we can reach theoretical predictions (3! for 3-body, 2! for 2-body)
- Gradual decrease of cutoff (results should not diverge even when we get down to 2-mode)
- Compare uniform/harmonic grid

Questions:
- Do we actually need the generalisation for |L| = \infty? This creates infinities when
  transforming master equation to FPE (in form of \delta(x, x)) and possibly makes
  proofs in Formalism chapter not entirely mathematically correct.
  Moreover, the validity criterion for truncation is n(x) >> \delta(x, x).
- (notation) Do we need explicit statement about the format of types for functions/functionals?
- (notation) Do we need explicit notation for "restricted" functions, like tilda over restricted operators?
- (feature) Do we need to proof that W transformation of Hermitian operator is a real-valued function?
- fix F[\lambda] -> F[\Lambda] in zero-delta-integrals lemma
