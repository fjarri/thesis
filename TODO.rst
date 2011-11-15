- Create phase noise histogram for losses and see if it fits Gaussian
- Need to find criteria for fragmentation of 2-species BEC
  See Penrose-Onsager fragmentation
- Test fragmentation in 1D
- Add master equation -> FPE chapter
- Add FPE -> SGPE chapter
- Add chapter on ground state (explain 0.5 particle per mode and so on)
- Update appendix on distributions for full 3D ellipsoid on a Bloch Sphere
  Need to find way to calculate squeezing along all directions
- Add theory for thermal modes
- Add description and results of noise tests (and implement them in the code)
- Find a way to calculate energy/chemical potential for Wigner representation
- Add a chapter on noise propagation algorithms
- Formalise 'N' as '|L|' (along with infinite |L|) in Wigner function definition and lemmas
- Check first approximation beyond truncated Wigner - Polkovnikov 2003

Questions:
- Do we actually need the generalisation for |L| = \infty? This creates infinities when
  transforming master equation to FPE (in form of \delta(x, x)) and possibly makes
  proofs in Formalism chapter not entirely mathematically correct.
  Moreover, the validity criterion for truncation is n(x) >> \delta(x, x).
- Which way to write delta is better - delta(x - x') or delta(x, x')?
  On the one hand, first way corresponds to the fact that this is one-dimensional delta
  On the other hand, the function itself depends on x and x' separately, not on their difference.
  And delta(x, x) written in a first way looks misleading (delta(0)? but it can depend on x)
