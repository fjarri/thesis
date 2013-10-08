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
- do we need to define abbrevs where they are actually used in addition to first appearance?
- more references?
- chapters/bec-squeezing/theory: any other planar squeezing papers to reference besides He2011?
- chapters/wigner-bec/fpe-bec: it should be possible to proof that the formula for d<Psi+ Psi>/dt agrees with the classical one up to the $1/N$ order, same as we do for particular cases later on.
- eqn:wigner-bec:fpe-bec:ordering-transformation : do we need proof?
- chapters/wigner-bec/master-eqn: elaborate on why the theorem for d<Psi>/dt is important. It's basically the Ehrenfest theorem anyway.
- proofread
- spellcheck
- grammar check
- do we need a page with supervisory committee?
- "non-" or "non"? Currently I have "nonlinear", but everything else is with "non-" (non-zero, non-trivial, non-negative etc).
- commas after introductory phrases? E.g.: "In this chapter,", "Here,", "Therefore,", "At the beginning of the simulation"


Peter's changes:
- PDF/A compatibility: need to find a way to solve current problems with metadata and zero witdh of some symbols.
  (Check Adobe Acrobat, it may be able to perform the conversion)
- Introduction, p.2 ("under which the Wigner function can be truncated") - what's the word there?
- Introduction, p.5 - do we still need "correctly" after "It shows how the truncated Wigner method can predict"?
- Introduction, the end - describe the code.




