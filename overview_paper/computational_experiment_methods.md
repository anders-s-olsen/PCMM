# Supplementary Methods: directional sampling and model estimation

## Sampling procedure

### Experimental design and common concentration scale

We considered the wrapped normal, von Mises--Fisher (vMF), Watson, Bingham, Fisher--Bingham, angular central Gaussian (ACG), matrix Fisher, matrix Bingham, matrix Fisher--Bingham, and matrix angular central Gaussian (MACG) distributions. Ambient dimensions were \(p=4,8,16,32,64\); matrix observations had \(q=2\) orthonormal columns. For each distribution and dimension, 400 observations were generated at each of two concentration levels. The first 320 observations were used for estimation and the remaining 80 for held-out evaluation.

Raw concentration parameters have different meanings across these families, so equal numerical parameter values would not give a meaningful comparison. We instead calibrated every family to the same expected, normalized alignment with its mode: 0.20 for the low-concentration condition and 0.80 for the high-concentration condition. A common random orthogonal basis was used for all distributions at a given \(p\), giving a shared modal direction \(\mu\) and modal frame \(X_0=(b_1,b_2)\).

The alignment statistic was chosen to respect whether a distribution identifies a signed direction, an axis, a signed frame, or only a subspace:

\[
\begin{aligned}
C_{\rm WN}(\theta)&=p^{-1}\sum_j\cos\theta_j,\\
C_{\rm signed}(x)&=x^\mathsf{T}\mu,\\
C_{\rm axial}(x)&=\{p(x^\mathsf{T}\mu)^2-1\}/(p-1),\\
C_{\rm frame}(X)&=q^{-1}\operatorname{tr}(X_0^\mathsf{T}X),\\
C_{\rm subspace}(X)&=\{p\lVert X_0^\mathsf{T}X\rVert_F^2/q-q\}/(p-q).
\end{aligned}
\]

These quantities equal zero in expectation under the corresponding uniform distribution and one at the mode. Signed alignment was used for vMF and Fisher--Bingham; axial alignment for Watson, Bingham, and ACG; signed-frame alignment for matrix Fisher and matrix Fisher--Bingham; and subspace alignment for matrix Bingham and MACG.

Before numerical calibration, a dimensionless strength \(s\) was converted to each family's natural parameter in a way that accounted for the number of directions transverse to the mode. This gave the following templates.

| Distribution | Concentration template |
|---|---|
| Wrapped normal | Full-rank Gaussian precision \(\tau=sp\), with equal variance \(1/\tau\) in all wrapped coordinates |
| vMF | Scalar concentration \(\kappa=s(p-1)\) about \(\mu\) |
| Watson | Scalar axial concentration \(\kappa=s(p-1)/2\) |
| Bingham | \(p-1\) transverse precision eigenvalues with mean \(s(p-1)/2\), using fixed relative weights equally spaced from 0.5 to 1.5 |
| Fisher--Bingham | vMF linear term \(\kappa=s(p-1)\), plus a transverse trace-free quadratic perturbation |
| ACG | Two-level precision \(\Omega=I+(\rho-1)P_{\mu^\perp}\), where \(\rho=1+s(p-1)\) |
| Matrix Fisher | \(F=s(p-q)X_0\) |
| Matrix Bingham | \(p-q\) transverse precision eigenvalues with mean \(s(p-q)/2\), again using relative weights from 0.5 to 1.5 |
| Matrix Fisher--Bingham | \(F=s(p-q)X_0\), plus a transverse trace-free quadratic perturbation |
| MACG | Row precision \(\Omega=I+(\rho-1)P_{X_0^\perp}\), where \(\rho=1+s(p-q)/q\) |

Thus scalar-precision families used a single \(\kappa\); covariance families used a controlled eigenvalue ratio; full quadratic families used a fixed-shape spectrum whose overall scale was calibrated; and the intermediate Fisher--Bingham families combined a linear concentration with anisotropy. In the two combined families, the largest absolute trace-free quadratic deviation was one half of the Fisher curvature. A common shift was then applied where necessary to make the stored quadratic precision positive semidefinite. Such a shift is an unidentifiable gauge on a sphere or Stiefel manifold and does not change the distribution.

Wrapped-normal concentration was calibrated analytically using \(E\{\cos\theta\}=\exp(-\sigma^2/2)\). vMF concentration was found by a bracketed solution of the Bessel mean-ratio equation \(I_{p/2}(\kappa)/I_{p/2-1}(\kappa)\). Watson and ACG used 160-node Gauss--Jacobi integration of the appropriate tilted beta expectation. Bingham, Fisher--Bingham, matrix Fisher, matrix Bingham, matrix Fisher--Bingham, and MACG used bounded pilot simulation: 240 draws per proposed strength, an initial bracket from \(10^{-6}\) to 1 with at most eight doublings, followed by five bisection steps. Pilot MCMC calculations used three chains, 250 discarded sweeps, and thinning by ten. The retained candidate was the evaluated strength closest to the requested alignment.

### Sampling algorithms

The final sampler was selected for each distribution as follows.

| Distribution(s) | Sampling method |
|---|---|
| Wrapped normal | Independent multivariate Gaussian draws, wrapped coordinate-wise to \([-\pi,\pi)\) |
| vMF | Wood--Ulrich beta-envelope rejection sampler [1] |
| Watson and Bingham | Bingham angular-central-Gaussian (BACG) rejection sampler [2] |
| ACG | Independent Gaussian draws normalized to unit length [3] |
| MACG | Independent matrix-normal draws followed by their polar factor [4] |
| Matrix Bingham | Matrix-BACG rejection sampler with a MACG proposal |
| Fisher--Bingham, matrix Fisher, matrix Fisher--Bingham | Coordinate-wise geodesic-slice MCMC |

The Bingham implementation used the BACG envelope construction of Kent, Ganeiber, and Mardia [2] after removing the distributionally irrelevant smallest eigenvalue. Watson sampling used its equivalent Bingham representation. The matrix-BACG sampler is an implementation-specific extension: it combines the BACG envelope idea with the MACG polar construction [4], using a determinant-form acceptance ratio. Rejection samplers were allowed at most \(5000n\) proposals. If matrix-BACG exhausted this budget, the code could fall back to the geodesic-slice sampler.

For each geodesic-slice update, one column of the current frame was selected, a random tangent direction orthogonal to the complete frame was constructed, and the selected column was moved on the resulting great circle. A slice level was drawn at the current state and a full-\(2\pi\) angular bracket was shrunk until acceptance. A polar projection restored numerical orthonormality after each sweep. This follows the column-wise conditional organization used for matrix Bingham--von Mises--Fisher simulation [5], with the conditional update replaced by a geodesic-slice kernel [6].

Five chains were used for final MCMC sampling. One began at the mode and four at independent uniform frames. At least 250 sweeps per chain were discarded. Diagnostics were recomputed in blocks of 400 sweeps until the maximum split-\(\widehat R\) of modal alignment and log kernel was at most 1.01, subject to a maximum of 2250 discarded sweeps. The retained thinning lag was the first lag, up to 200, for which both diagnostic autocorrelations remained at most 0.05 in absolute value for three successive lags. Each chain then contributed 80 observations. Four complete chains formed the training set and the fifth formed the test set, so dependent observations from one chain were never split between training and testing. Retained-sample checks used split-\(\widehat R\leq1.05\) and effective sample size at least 200; failures were recorded as warnings rather than silently resampled.

## Estimation procedure

Each sample was fitted by maximum likelihood as a one-component PCMM model. Quadratic factor ranks matched the sampling construction: \(p-1\) for Bingham and Fisher--Bingham, \(p-q\) for matrix Bingham, \(p-1\) for matrix Fisher--Bingham after gauge fixing, rank one for ACG, and rank \(q\) for MACG. Matrix Fisher and the linear part of matrix Fisher--Bingham used a direct \(p\times q\) concentration matrix. Sphere densities were defined relative to surface-area measure, matrix densities relative to normalized Haar measure on \(V_q(\mathbb R^p)\), and wrapped-normal densities relative to Lebesgue measure on the torus.

### Normalizing constants

The likelihood used the following distribution-specific evaluations; plotting and other downstream calculations were not part of estimation.

| Method | Distributions |
|---|---|
| Scaled modified-Bessel functions | vMF |
| Direct log-space Kummer series | Watson |
| Continuous-Euler contour quadrature | Bingham, Fisher--Bingham |
| Uniform-calibrated second-order Stiefel saddlepoint approximation | Matrix Fisher, matrix Bingham, matrix Fisher--Bingham |
| Closed-form determinant identities | ACG, MACG |
| Finite Gaussian winding-lattice sum | Wrapped normal |

For vMF, with \(\nu=p/2-1\), the surface-measure log normalizer was evaluated from its standard Bessel form [7],

\[
\log Z(\kappa)=\frac p2\log(2\pi)+\log I_\nu(\kappa)-\nu\log\kappa.
\]

Exponentially scaled modified-Bessel functions were used to avoid overflow, and the analytic gradient \(I_{\nu+1}(\kappa)/I_\nu(\kappa)\) was supplied to automatic differentiation. The removable limit at \(\kappa=0\) was handled by its local series. Because this calculation calls SciPy, vMF estimation is explicitly restricted to CPU tensors; requesting a GPU for a vMF cell raises an error rather than silently transferring data.

For Watson, \(\log{}_1F_1(1/2;p/2;\kappa)\) was accumulated from its defining series in log space. Negative concentration used Kummer's transformation \({}_1F_1(a;c;\kappa)=e^\kappa{}_1F_1(c-a;c;-\kappa)\) [8]. The series limit was \(10^7\) terms, and its convergence tolerance was the same tol supplied for the overall model fit (here \(10^{-5}\)); it was no longer fixed independently at \(10^{-10}\).

Bingham and Fisher--Bingham used the continuous-Euler transformed Fourier integral of Chen and Tanaka [9]. The setting \(N=400\) produced indices \(-N-1,\ldots,N\), or 802 contour terms, with Euler-window parameters \(\omega_d=0.5\) and \(\omega_u=2.0\). Bingham used a symmetric eigendecomposition and removed its scalar eigenvalue gauge before quadrature. Fisher--Bingham used the current low-rank quadratic spectrum and linear term. Contour locations were selected numerically and treated as fixed during reverse-mode differentiation, while the complex quadrature itself remained differentiable.

Matrix Fisher, matrix Bingham, and matrix Fisher--Bingham all used the second-order Stiefel saddlepoint approximation of Kume, Preston, and Wood [10], calibrated by the corresponding uniform saddlepoint value so that \(\log Z(0)=0\) under normalized Haar measure. This replaces the former Matrix Fisher-only \(q=2\) Gauss--Jacobi reduction and makes the estimator used for matrix Fisher comparable to those used for the other matrix exponential families. For general \(q\), \(q(q+1)/2\) saddle variables enforce the column-norm and pairwise-orthogonality constraints; the pure Matrix Fisher implementation therefore supports arbitrary \(q<p\). The benchmark itself used \(q=2\). The structured low-rank quadratic calculation used for matrix Bingham and matrix Fisher--Bingham remains specialized to \(q=2\), which is the only case included in the experiment.

The saddle was found by damped Newton iteration with at most 20 steps, tolerance \(10^{-8}\), a \(10^{-8}\) Hessian ridge, and feasibility-preserving backtracking. A differentiable Newton refinement supplied the implicit parameter derivative. The second-order Kume--Preston--Wood correction used third- and fourth-order cumulants. For the benchmarked \(q=2\) models these derivatives were obtained from a centered \(5^3\) grid with five-point finite-difference stencils and base step \(0.005\sqrt{p/4}\); nested automatic differentiation was the fallback when a feasible grid could not be formed. For \(q>2\) Matrix Fisher models, nested automatic differentiation is used directly. No special \(q=2\) Matrix Fisher normalizer remains.

ACG and MACG normalizers were evaluated by their closed-form determinant expressions [3,4], using determinant-lemma and Woodbury identities for the fitted low-rank parameterization. The wrapped-normal likelihood summed Gaussian images over \(m\in\{-1,0,1\}^p\) by chunked log-sum-exp. This is a finite truncation of the infinite winding sum. A cap of 200,000 winding vectors made the radius-one calculation feasible for \(p=4\) and \(p=8\); larger wrapped-normal cells were designated structurally unavailable rather than fitted with an unnormalized central-image approximation.

### Optimization and computing environment

Parameters were optimized in 64-bit arithmetic by reverse-mode differentiation and Adam [11], with initial learning rate 0.05. Fits used a one-component model and the fixed 320/80 training/test division. Objective convergence was assessed from improvement in log likelihood per training observation, with tolerance \(10^{-5}\) over a 25-update window. Fits were bounded at 20,000 optimizer updates and 1000 s of optimizer time. Data-independent, near-isotropic initial natural parameters had total identifiable norm \(10^{-2}\); combined linear--quadratic models divided this norm equally between their two blocks.

The benchmark ran on an **Intel Xeon Gold 6126 CPU at 2.60 GHz**. PyTorch intra-operation and inter-operation parallelism and the BLAS/OpenMP thread pools were each restricted to one thread. GPU execution was not used. Reported fitting time included model initialization, optimizer updates, final likelihood evaluation, and restoration of the best parameter state; sample loading, held-out evaluation, and serialization were timed separately.

## References

1. Wood, A. T. A. (1994). Simulation of the von Mises Fisher distribution. *Communications in Statistics--Simulation and Computation*, 23, 157--164. [https://doi.org/10.1080/03610919408813161](https://doi.org/10.1080/03610919408813161)
2. Kent, J. T., Ganeiber, A. M., and Mardia, K. V. (2018). A new unified approach for the simulation of a wide class of directional distributions. *Journal of Computational and Graphical Statistics*, 27, 291--301. [https://doi.org/10.1080/10618600.2017.1390468](https://doi.org/10.1080/10618600.2017.1390468)
3. Tyler, D. E. (1987). Statistical analysis for the angular central Gaussian distribution on the sphere. *Biometrika*, 74, 579--589. [https://doi.org/10.1093/biomet/74.3.579](https://doi.org/10.1093/biomet/74.3.579)
4. Chikuse, Y. (1990). The matrix angular central Gaussian distribution. *Journal of Multivariate Analysis*, 33, 265--274. [https://doi.org/10.1016/0047-259X(90)90050-R](https://doi.org/10.1016/0047-259X(90)90050-R)
5. Hoff, P. D. (2009). Simulation of the matrix Bingham--von Mises--Fisher distribution, with applications to multivariate and relational data. *Journal of Computational and Graphical Statistics*, 18, 438--456. [https://doi.org/10.1198/jcgs.2009.07177](https://doi.org/10.1198/jcgs.2009.07177)
6. Habeck, M., Hasenpflug, M., Kodgirwar, S., and Rudolf, D. (2025). Geodesic slice sampling on the sphere. *Journal of Machine Learning Research*, 26(297), 1--38. [https://jmlr.org/papers/v26/23-1158.html](https://jmlr.org/papers/v26/23-1158.html)
7. Mardia, K. V., and Jupp, P. E. (2000). *Directional Statistics*. Wiley. [https://doi.org/10.1002/9780470316979](https://doi.org/10.1002/9780470316979)
8. NIST Digital Library of Mathematical Functions, Section 13.2, confluent hypergeometric functions and Kummer transformations. [https://dlmf.nist.gov/13.2](https://dlmf.nist.gov/13.2)
9. Chen, Y., and Tanaka, K. (2021). Maximum likelihood estimation of the Fisher--Bingham distribution via efficient calculation of its normalizing constant. *Statistics and Computing*, 31. [https://doi.org/10.1007/s11222-021-10015-9](https://doi.org/10.1007/s11222-021-10015-9)
10. Kume, A., Preston, S. P., and Wood, A. T. A. (2013). Saddlepoint approximations for the normalizing constant of Fisher--Bingham distributions on products of spheres and Stiefel manifolds. *Biometrika*, 100, 971--984. [https://doi.org/10.1093/biomet/ast021](https://doi.org/10.1093/biomet/ast021)
11. Kingma, D. P., and Ba, J. (2015). Adam: a method for stochastic optimization. *International Conference on Learning Representations*. [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
