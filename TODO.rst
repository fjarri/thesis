Examples and tests:
- Application to experiments: Swin's (Ramsey, spin echo), Riedel (spatial separation), Widera (1D)
- Noise balance test: check that if we start with 0 atoms, we will still have 0 atoms even with enabled losses
  (try different loss sources, different cutoffs etc)
- Correlations (atom bunching): check that <Psi1^+ Psi1^+ Psi1 Psi1> / <Psi1+ Psi1>^2 ~ 1 in condensate.
  May also check <Psi1^+ Psi2^+ Psi1 Psi2> / (<Psi1+ Psi1> <Psi2+ Psi2>) (should be ~1 too in condensate).
- Check correlations for thermal states (should be >1). See if we can reach theoretical predictions (3! for 3-body, 2! for 2-body)
- Gradual decrease of cutoff (results should not diverge even when we get down to 2-mode)
- Compare uniform/harmonic grid


Checklist:
- remove \copypaste
- write acknowledgements to: Qiongyi, Margaret, Laura, Michael
- check a11,12,22 for exact one-well squeezing results
- add references to published papers in the beginning of the application chapters
- fix this section - we are only talking about a single well, no need to mention two wells
- if \cite is used as a part of speech (e.g. "... in~\cite{...}), mention actual names of the authors
- check that each abbreviation is introduced only once (or maybe also once in an appendix): BEC, GPE, SDE, FPE, LHV, QM, GPU
- check that --- have ~ in front of them
- more references?
- chapters/bec-squeezing/separation: can we use Riedel et al squeezing plot?
- chapters/bec-squeezing/theory: any other planar squeezing papers to reference besides He2011?
- chapters/wigner-bec/fpe-bec: it should be possible to proof that the formula for d<Psi+ Psi>/dt agrees with the classical one up to the $1/N$ order, same as we do for particular cases later on.
- eqn:wigner-bec:fpe-bec:ordering-transformation : do we need proof?
- chapters/wigner-bec/master-eqn: elaborate on why the theorem for d<Psi>/dt is important
- proofread
- spellcheck
- grammar check
