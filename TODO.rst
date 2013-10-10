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
- chapters/wigner-bec/fpe-bec: it should be possible to proof that the formula for d<Psi+ Psi>/dt agrees with the classical one up to the $1/N$ order, same as we do for particular cases later on.
- eqn:wigner-bec:fpe-bec:ordering-transformation : do we need proof?

Peter's changes:
- PDF/A compatibility: need to find a way to solve current problems with metadata and zero witdh of some symbols.
  (Check Adobe Acrobat, it may be able to perform the conversion)
- Introduction, the end - describe the code.
- Chapter 2, p.8: add the derivation for eqn (2.2)?
- Chapter 5, p.45, bottom - "broad-band"?
- Chapter 6, p.57 - discuss
- Chapter 7, add more explanations to the discrepancies with the experiment
- Chapter 7, p.60, do I need to add component-dependent \omega's too?
- Chapter 7, p.62: merge with Andrei's comments. I'll be dropping this part anyway.
- Chapter 7, p.64: discuss. I can use K, probably.
- Chapter 7, p.74: underlined phrase?
- Chapter 7, p.79: can't post factum be used as an adjective? Or if it's "o", then "ex post facto".
- Chapter 8, is it enough to just reference the paper, or some explanation needed too?
