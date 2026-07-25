import torch
import torch.nn as nn
from PCMM.PCMMtorchBaseModel import PCMMtorchBaseModel
import math

class Watson(PCMMtorchBaseModel):
    """
    Watson distribution on the (complex) projective hyperplane.
    The Watson distirbution is parameterized by a concentration parameter kappa and a mean direction mu.
    Oppositely the other distributions in this file, it is NOT parameterized by a covariance matrix.
    The Watson distribution will fail if the input data vectors are not normalized to unit length.
    Args:
        p (int): dimensionality of the data vectors
        K (int): number of clusters
        HMM (bool): whether to use the HMM variant of the model (default: False)
        complex (bool): whether to use the complex Watson distribution (default: False)
        samples_per_sequence (int): number of samples per sequence to be used by HMM (default: 0, meaning one long sequence)
        params (dict): dictionary containing the parameters of the model, if available (default: None
    """
    def __init__(self, p:int, K:int=1, HMM:bool=False, complex:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = p
        self.K = K
        self.HMM = HMM
        if samples_per_sequence is None:
            samples_per_sequence = 0
        self.samples_per_sequence = torch.as_tensor(samples_per_sequence)
        if complex:
            self.distribution = 'Complex_Watson'
            self.a = torch.as_tensor(1)
            self.c = torch.as_tensor(p)
        else:
            self.distribution = 'Watson'
            self.a = torch.as_tensor(0.5)
            self.c = torch.as_tensor(p/2)
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = torch.lgamma(self.c) - torch.log(torch.as_tensor(2)) - self.c* torch.log(torch.as_tensor(math.pi))

        self.flag_normalized_input_data = False

        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def kummer_log(self,kappa, n=1e7,tol=1e-10):
        """ 
        Logarithm of the Kummer function for each kappa value.
        Args:
            kappa (torch.Tensor): A tensor of shape (K,) containing the kappa values.
            n (int): The maximum number of terms to compute in the series.
            tol (float): The tolerance for convergence.
        Returns:
            torch.Tensor: A tensor of shape (K,) containing the logarithm of the Kummer function for each kappa value.
        """
        results = []
        for k in kappa:
            if k<0:
                a = (self.c-self.a).to(dtype=k.dtype, device=k.device)
            else:
                a = self.a.to(dtype=k.dtype, device=k.device)
            c = self.c.to(dtype=k.dtype, device=k.device)
            if k == 0:
                # log M(a,c,0) is zero and its derivative is a/c.
                results.append(k * a / c)
                continue
            logkum = k.new_zeros(())
            logkum_old = k.new_ones(())
            tmp = k.new_zeros(())
            j = 1
            # I modified this from somewhere but I cannot remember where :(
            while torch.abs(logkum - logkum_old) > tol and (j < n):
                logkum_old = logkum
                tmp = tmp + torch.log((a + j - 1) / (j * (c + j - 1)) * torch.abs(k))
                logkum = torch.logsumexp(torch.stack((logkum,tmp),dim=0),dim=0)
                j += 1
            # Kummer's transformation for negative concentration:
            # M(a,c,k) = exp(k) M(c-a,c,-k).
            results.append(logkum + torch.where(k < 0, k, k.new_zeros(())))
        return torch.stack(results)

    def log_norm_constant(self):
        return self.logSA_sphere.to(
            dtype=self.kappa.dtype, device=self.kappa.device
        ) - self.kummer_log(self.kappa)
    
    def log_pdf(self, X, recompute_statics=False):
        # check for normalized input data
        ones = torch.ones(X.shape[0], dtype=X.real.dtype, device=X.device)
        if not torch.allclose(torch.linalg.norm(X, dim=1), ones):
            raise ValueError("For the Watson distribution, the input data vectors should be normalized to unit length.")

        # reparameterize mu to be unit norm
        mu_unit = nn.functional.normalize(self.mu, dim=1)

        # compute logpdf of Watson distribution for each component
        logpdf = self.log_norm_constant().unsqueeze(-1) + self.kappa.unsqueeze(-1)*(torch.abs(X @ mu_unit.mH)**2).T
        return logpdf 

class Bingham(PCMMtorchBaseModel):
    """Low-rank Bingham distribution on a real or complex unit sphere.

    The concentration is parameterized as ``A = M @ M.H + I``.  Since an
    isotropic term does not change a Bingham density on the unit sphere, it is
    omitted in computations.  Thus ``M`` follows the same low-rank-plus-
    isotropic convention as ACG without introducing an unidentifiable scalar.

    The normalizer is evaluated with the one-dimensional Fourier integral and
    continuous Euler transform of Chen and Tanaka (2020).  Its cost is linear
    in ``p`` and it remains differentiable through PyTorch autograd.
    """
    def __init__(self, p:int, rank:int, K:int=1, HMM:bool=False,
                 complex:bool=False, samples_per_sequence=0, params:dict=None,
                 integration_points:int=400, contour_shift:float=None,
                 omega_d:float=0.5, omega_u:float=2.0,
                 quadrature_order:int=None):
        super().__init__()
        if not isinstance(rank, int) or rank < 1 or rank > p:
            raise ValueError('rank should satisfy 1 <= rank <= p.')
        if quadrature_order is not None:
            integration_points = quadrature_order
        if not isinstance(integration_points, int) or integration_points < 1:
            raise ValueError('integration_points should be a positive integer.')
        effective_dimension = 2 * p if complex else p
        if contour_shift is None:
            # The inversion contour is exact anywhere to the right of the
            # poles.  Placing it near the zero-concentration saddle point
            # avoids catastrophic cancellation as dimension grows.
            contour_shift = max(effective_dimension / 2, 2.0)
        if contour_shift <= 0:
            raise ValueError('contour_shift should be positive.')
        if not (0 < omega_d <= 1 <= omega_u and omega_d / omega_u <= 0.5):
            raise ValueError(
                'omega_d and omega_u should satisfy '
                '0 < omega_d <= 1 <= omega_u and omega_d / omega_u <= 1/2.'
            )

        self.p, self.r, self.K, self.HMM = p, rank, K, HMM
        self.complex = complex
        self.integration_points = integration_points
        self.contour_shift = float(contour_shift)
        self.omega_d = float(omega_d)
        self.omega_u = float(omega_u)
        if samples_per_sequence is None:
            samples_per_sequence = 0
        self.samples_per_sequence = torch.as_tensor(samples_per_sequence)
        self.distribution = ('Complex_' if complex else '') + 'Bingham_lowrank'
        self.flag_normalized_input_data = False

        # Chen--Tanaka equations (8)--(9), with d equal to the distance from
        # the integration contour to the closest pole.  Our non-positive
        # concentration eigenvalues imply theta=-lambda >= 0, so a fixed
        # positive contour shift has d=contour_shift.
        h = math.sqrt(
            2 * math.pi * self.contour_shift * (omega_d + omega_u)
            / (omega_d**2 * integration_points)
        )
        window_p = math.sqrt(integration_points * h / omega_d)
        window_q = math.sqrt(omega_d * integration_points * h / 4)
        indices = torch.arange(-integration_points - 1, integration_points + 1)
        nodes = indices * h
        weights = 0.5 * torch.erfc(nodes.abs() / window_p - window_q)
        self.register_buffer('integration_nodes', nodes)
        self.register_buffer('integration_weights', weights)
        self.integration_step = h

        if params is not None:
            self.unpack_params(params)

    def concentration_matrix(self):
        """Return Hermitian concentration matrices with largest eigenvalue zero."""
        A = self.M @ self.M.mH
        largest_eigenvalue = torch.linalg.eigvalsh(A)[:, -1]
        return A - largest_eigenvalue[:, None, None] * torch.eye(
            self.p, dtype=A.dtype, device=A.device
        )

    def log_norm_constant(self, A):
        """Compute log Z(A) using Chen--Tanaka continuous-Euler quadrature."""
        eigenvalues = torch.linalg.eigvalsh(A)
        theta = -eigenvalues
        if self.complex:
            # z.H A z = sum_i lambda_i(Re(z_i)^2 + Im(z_i)^2):
            # use the real S^(2p-1) formula with each eigenvalue duplicated.
            theta = torch.repeat_interleave(theta, 2, dim=1)

        real_dtype = theta.dtype
        complex_dtype = (
            torch.complex128 if real_dtype == torch.float64 else torch.complex64
        )
        nodes = self.integration_nodes.to(dtype=real_dtype, device=A.device)
        weights = self.integration_weights.to(dtype=real_dtype, device=A.device)
        denominator = (
            theta[:, None, :].to(complex_dtype)
            + self.contour_shift
            + 1j * nodes[None, :, None]
        )
        log_integrand = (
            -0.5 * torch.log(denominator).sum(dim=-1)
            + 1j * nodes[None, :]
        )
        # Scaling before exponentiation prevents avoidable underflow while
        # preserving gradients (the scale cancels algebraically).
        log_scale = log_integrand.real.amax(dim=1, keepdim=True)
        integral = (
            weights[None, :]
            * torch.exp(log_integrand - log_scale)
        ).sum(dim=1).real
        if torch.any(integral <= 0):
            raise RuntimeError(
                'Chen--Tanaka quadrature returned a non-positive normalizer. '
                'Increase integration_points or contour_shift.'
            )

        effective_dimension = theta.shape[1]
        return (
            (effective_dimension / 2 - 1) * math.log(math.pi)
            + self.contour_shift
            + math.log(self.integration_step)
            + log_scale.squeeze(1)
            + torch.log(integral)
        )

    def log_pdf(self, X, recompute_statics=False):
        ones = torch.ones(X.shape[0], dtype=X.real.dtype, device=X.device)
        if not torch.allclose(torch.linalg.norm(X, dim=1), ones):
            raise ValueError(
                'For the Bingham distribution, input vectors must have unit norm.'
            )
        if torch.is_complex(X) != self.complex:
            kind = 'complex' if self.complex else 'real'
            raise ValueError(f'The {kind} Bingham model received data of the wrong dtype.')

        A = self.concentration_matrix()
        quadratic = torch.einsum('np,kpq,nq->kn', X.conj(), A, X).real
        return quadratic - self.log_norm_constant(A).unsqueeze(-1)

class ACG(PCMMtorchBaseModel):
    """ ACG distribution on the (complex) projective hyperplane.
    The ACG distribution is normally parameterized by a covariance matrix. Here, we have constructed 
    a low-rank-plus-diagonal approximation of the covariance matrix, which is parameterized by a rank r matrix M.
    The ACG distribution will fail if the input data vectors are not normalized to unit length.
    Args:
        p (int): dimensionality of the data vectors
        rank (int): rank of the low-rank approximation of the covariance matrix
        K (int): number of clusters
        HMM (bool): whether to use the HMM variant of the model (default: False)
        complex (bool): whether to use the complex ACG distribution (default: False)
        samples_per_sequence (int): number of samples per sequence to be used by HMM (default: 0, meaning one long sequence)
        params (dict): dictionary containing the parameters of the model, if available (default: None)
    """
    def __init__(self, p:int, rank:int, K:int=1, HMM:bool=False, complex:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = p
        self.r = rank
        self.K = K
        self.HMM = HMM
        if samples_per_sequence is None:
            samples_per_sequence = 0
        self.samples_per_sequence = torch.as_tensor(samples_per_sequence)
        self.distribution = 'ACG_lowrank'
        
        self.complex = complex
        if complex:
            self.a = torch.as_tensor(1)
            self.c = torch.as_tensor(self.p)
            self.distribution = 'Complex_'+self.distribution
        else:
            self.a = torch.as_tensor(0.5)
            self.c = torch.tensor(self.p/2)
        
        # precompute log-surface area of the unit hypersphere
        self.logSA_sphere = torch.lgamma(self.c) - torch.log(torch.as_tensor(2)) -self.c* torch.log(torch.as_tensor(math.pi))

        self.flag_normalized_input_data = False

        # initialize parameters
        if params is not None:
            self.unpack_params(params)
            
    def log_pdf(self,X, recompute_statics=False):
        # check for normalized input data
        ones = torch.ones(X.shape[0], dtype=X.real.dtype, device=X.device)
        if not torch.allclose(torch.linalg.norm(X, dim=1), ones):
            raise ValueError("For the ACG distribution, the input data vectors should be normalized to unit length.")

        # see supplementary material for the derivation of the logpdf for low-rank ACG
        D = torch.eye(self.r, dtype=self.M.dtype, device=self.M.device) + self.M.mH @ self.M
        v = torch.zeros(self.K, X.shape[0], dtype=X.dtype, device=X.device)

        # loop over components (is faster than batch matrix multiplication)
        for k in range(self.K):
            XM = torch.conj(X) @ self.M[k]
            v[k] = 1 - torch.sum(XM @ torch.linalg.inv(D[k]) * torch.conj(XM), dim=-1)

        real_dtype = X.real.dtype
        log_pdf = (
            self.logSA_sphere.to(dtype=real_dtype, device=X.device)
            - self.a.to(dtype=real_dtype, device=X.device) * torch.logdet(D).real.unsqueeze(-1)
            - self.c.to(dtype=real_dtype, device=X.device) * torch.log(v.real)
        )
        return log_pdf

class MACG(PCMMtorchBaseModel):
    def __init__(self, p:int, q:int, rank:int, K:int=1, HMM:bool=False, samples_per_sequence=0, params:dict=None):
        super().__init__()

        self.p = p
        self.q = q
        self.r = rank
        self.K = K
        self.HMM = HMM
        if samples_per_sequence is None:
            samples_per_sequence = 0
        self.samples_per_sequence = torch.as_tensor(samples_per_sequence)
        self.distribution = 'MACG_lowrank'

        self.flag_normalized_input_data = False

        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf(self,X, recompute_statics=False):
        ones = torch.ones(X.shape[0], dtype=X.real.dtype, device=X.device)
        if not torch.allclose(torch.linalg.norm(X[:,:,0], dim=1), ones):
            raise ValueError("For the MACG distribution, the input data vectors should be normalized to unit length (and orthonormal, but this is not checked).")
        D = torch.swapaxes(self.M,-2,-1)@self.M + torch.eye(
            self.r, dtype=self.M.dtype, device=self.M.device
        )
        log_det_D = torch.logdet(D)
        
        v = torch.zeros(self.K, X.shape[0], dtype=X.real.dtype, device=X.device)
        for k in range(self.K):
            L, Q = torch.linalg.eigh(torch.linalg.inv(D[k]))
            D_sqrtinv = (Q * L.sqrt().unsqueeze(-2)) @ Q.mH
            XtM = X.mH@self.M[k].unsqueeze(0)
            S2 = torch.linalg.svdvals(XtM@D_sqrtinv.unsqueeze(0))
            v[k] = torch.sum(torch.log(1/(S2**2)-1),dim=-1)+2*torch.sum(torch.log(S2),dim=-1)

        log_pdf = - (self.q/2)*log_det_D.unsqueeze(-1) - self.p/2*v
        return log_pdf

class SingularWishart(PCMMtorchBaseModel):
    def __init__(self, p:int, q:int, rank:int, K:int=1, HMM:bool=False, samples_per_sequence=0, params:dict=None, force_gamma_same:bool=False):
        super().__init__()

        self.p = p
        self.q = q
        self.r = rank
        self.K = K
        self.HMM = HMM
        if samples_per_sequence is None:
            samples_per_sequence = 0
        self.samples_per_sequence = torch.as_tensor(samples_per_sequence)
        self.distribution = 'SingularWishart_lowrank'
        self.log_det_S11 = None
        self.force_gamma_same = force_gamma_same
        
        a = torch.tensor(self.q / 2, dtype=torch.float64)
        log_gamma_q = torch.special.multigammaln(a, self.q)
        # loggamma_q = (self.q*(self.q-1)/4)*torch.log(torch.as_tensor(math.pi))+torch.sum(torch.lgamma(torch.as_tensor(self.q/2)-torch.arange(self.q)/2))
        self.log_norm_constant = self.q*(self.q-self.p)/2*math.log(math.pi)-self.p*self.q/2*math.log(2.0)-log_gamma_q

        self.flag_normalized_input_data = False
        
        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf(self,X, recompute_statics=False):
        X_weights = torch.linalg.norm(X, dim=1)**2
        expected = torch.full((X.shape[0],), self.p, dtype=X.real.dtype, device=X.device)
        if not torch.allclose(torch.sum(X_weights, dim=1), expected):
            Warning("While not explicitly required, the scale of the input data vectors is expected to be equal to the square root of the eigenvalues. If the scale does not sum to the dimensionality, this warning is thrown")

        # while Q_q^T Q_q != U_q^T L U_q, their determinants are the same
        # log_det_S11 = torch.logdet(torch.swapaxes(X[:,:self.q,:],-2,-1) @ X[:,:self.q,:]).unsqueeze(0)
        gram = X.mH @ X  # (N, q, q)
        sign, log_pdet_S = torch.linalg.slogdet(gram)
        if torch.any(sign <= 0):
            raise ValueError("X must have full column rank.")
        log_det_term = log_pdet_S.unsqueeze(0)

        gamma = torch.nn.functional.softplus(self.gamma)

        M_tilde = self.M*torch.sqrt(1/gamma.unsqueeze(-1).unsqueeze(-1))

        D = torch.swapaxes(M_tilde,-2,-1)@M_tilde + torch.eye(self.r, dtype=M_tilde.dtype, device=M_tilde.device)
        log_det_D = self.p*torch.log(gamma)+torch.logdet(D)
        
        v = torch.zeros(self.K, X.shape[0], dtype=X.real.dtype, device=X.device)
        for k in range(self.K):
            L, Q = torch.linalg.eigh(torch.linalg.inv(D[k]))
            D_sqrtinv = (Q * L.sqrt().unsqueeze(-2)) @ Q.mT
            QtM_tilde = X.mT@M_tilde[k].unsqueeze(0)
            v[k] = 1/gamma[k]*(self.p - torch.linalg.norm(QtM_tilde@D_sqrtinv.unsqueeze(0),dim=(-2,-1))**2)
        
        log_pdf = self.log_norm_constant.to(dtype=X.real.dtype, device=X.device) - (self.q/2)*log_det_D.unsqueeze(-1) + (self.q-self.p-1)/2*log_det_term - 1/2*v
        return log_pdf

class Normal(PCMMtorchBaseModel):
    def __init__(self, p:int, rank:int, K:int=1, complex:bool=False, HMM:bool=False, samples_per_sequence=0, params:dict=None, force_gamma_same:bool=False):
        super().__init__()

        self.p = p
        self.r = rank
        self.K = K
        self.HMM = HMM
        if samples_per_sequence is None:
            samples_per_sequence = 0
        self.samples_per_sequence = torch.as_tensor(samples_per_sequence)
        self.distribution = 'Normal_lowrank'
        self.complex = complex
        self.force_gamma_same = force_gamma_same

        if complex:
            self.a = torch.as_tensor(1)
            self.c = torch.as_tensor(self.p)
            self.distribution = 'Complex_'+self.distribution
        else:
            self.a = torch.as_tensor(0.5)
            self.c = torch.as_tensor(self.p/2)
        
        self.log_norm_constant = -self.c*torch.log(1/self.a*torch.as_tensor(math.pi))
        self.norm_x = None

        # initialize parameters
        if params is not None:
            self.unpack_params(params)

    def log_pdf(self,X, recompute_statics=False):
        norm_x = (torch.linalg.norm(X,dim=1)**2).unsqueeze(0)


        gamma = torch.nn.functional.softplus(self.gamma)

        M_tilde = self.M*torch.sqrt(1/gamma.unsqueeze(-1).unsqueeze(-1))

        D = M_tilde.mH @ M_tilde + torch.eye(
            self.r, dtype=M_tilde.dtype, device=M_tilde.device
        )
        log_det_D = self.p*torch.log(gamma)+torch.logdet(D)
        
        v = torch.zeros(self.K, X.shape[0], dtype=X.real.dtype, device=X.device)
        for k in range(self.K):
            L, Q = torch.linalg.eigh(torch.linalg.inv(D[k]))
            D_sqrtinv = (Q * L.sqrt().unsqueeze(-2)) @ Q.mH
            XtM_tilde = torch.conj(X).unsqueeze(-2)@M_tilde[k].unsqueeze(0)
            v[k] = 1/gamma[k]*(norm_x - torch.linalg.norm(XtM_tilde@D_sqrtinv.unsqueeze(0),dim=(-2,-1))**2)
        
        a = self.a.to(dtype=X.real.dtype, device=X.device)
        log_pdf = self.log_norm_constant.to(
            dtype=X.real.dtype, device=X.device
        ) - a*log_det_D.real.unsqueeze(-1) - a*v
        return log_pdf

def _winding_vectors(
    dimension: int,
    radius: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Integer vectors in {-radius, ..., radius}**dimension.

    Returns
    -------
    Tensor with shape [number_of_windings, dimension].
    """
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    axis = torch.arange(
        -radius,
        radius + 1,
        device=device,
        dtype=dtype,
    )

    if dimension == 1:
        return axis[:, None]

    return torch.cartesian_prod(*([axis] * dimension))


class WrappedNormal_old(Normal):
    """
    Multivariate wrapped-normal version of PCMM's Normal model.

    The underlying unwrapped Gaussian, including its low-rank covariance
    parameterization, is inherited unchanged from Normal.

    The density is approximated by

        p_WN(x) = sum_{m in {-R,...,R}^p}
                  p_Normal(x + 2*pi*m).

    Parameters added to the Normal constructor
    ------------------------------------------
    winding_radius:
        R in the expression above. Start with 1.

    winding_chunk_size:
        Number of winding vectors evaluated simultaneously. Lower this if
        GPU memory is exhausted.

    max_winding_vectors:
        Safety limit against accidentally constructing an enormous lattice.
    """

    def __init__(
        self,
        *args,
        winding_radius: int = 1,
        winding_chunk_size: int = 32,
        max_winding_vectors: int = 200_000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if winding_radius < 0:
            raise ValueError("winding_radius must be non-negative")
        if winding_chunk_size < 1:
            raise ValueError("winding_chunk_size must be positive")

        self.winding_radius = winding_radius
        self.winding_chunk_size = winding_chunk_size
        self.max_winding_vectors = max_winding_vectors

    def log_pdf(
        self, data: torch.Tensor, recompute_statics: bool = False
    ) -> torch.Tensor:
        """
        Wrapped component log-densities.

        Parameters
        ----------
        data:
            Quotiented phase observations in radians, shape [n, p].

        Returns
        -------
        Same orientation as Normal.log_pdf():
            normally either [K, n] or [n, K].
        """
        if data.ndim != 2:
            raise ValueError(
                f"Expected data with shape [n, p], got {tuple(data.shape)}"
            )
        if torch.is_complex(data):
            raise TypeError(
                "WrappedNormal expects real-valued phase angles in radians."
            )

        n, p = data.shape
        if p != self.p:
            raise ValueError(
                f"Expected data dimension p={self.p}, got p={p}."
            )

        # Put observations in the standard principal interval.
        # The density is periodic, so (-pi, pi] versus [0, 2*pi) does not
        # matter as long as the convention is consistent.
        data = torch.atan2(torch.sin(data), torch.cos(data))

        number_of_windings = (2 * self.winding_radius + 1) ** p
        if number_of_windings > self.max_winding_vectors:
            raise RuntimeError(
                "The requested winding lattice contains "
                f"{number_of_windings:,} vectors. "
                "Decrease winding_radius, increase max_winding_vectors "
                "deliberately, or use a latent-factor approximation."
            )

        windings = _winding_vectors(
            dimension=p,
            radius=self.winding_radius,
            device=data.device,
            dtype=data.dtype,
        )

        accumulated_log_prob = None
        output_orientation = None

        for winding_chunk in windings.split(self.winding_chunk_size):
            number_in_chunk = winding_chunk.shape[0]

            # Shape: [number_in_chunk, n, p]
            shifted_data = (
                data[None, :, :]
                + 2.0
                * math.pi
                * winding_chunk[:, None, :]
            )

            # Let the existing Normal model evaluate all shifted copies.
            flat_shifted_data = shifted_data.reshape(
                number_in_chunk * n,
                p,
            )

            # Calling Normal.log_pdf directly avoids recursion into this
            # overridden WrappedNormal.log_pdf method.
            base_log_prob = Normal.log_pdf(
                self,
                flat_shifted_data,
            )

            # Support either convention used by the base class:
            # [K, observations] or [observations, K].
            if base_log_prob.ndim == 1:
                if base_log_prob.numel() != flat_shifted_data.shape[0]:
                    raise RuntimeError(
                        "Unexpected output shape from Normal.log_pdf."
                    )

                # One-component case.
                base_log_prob = base_log_prob[None, :]
                output_orientation = "K_by_n"

            if base_log_prob.shape[1] == flat_shifted_data.shape[0]:
                # [K, number_in_chunk * n]
                output_orientation = "K_by_n"

                base_log_prob = base_log_prob.reshape(
                    base_log_prob.shape[0],
                    number_in_chunk,
                    n,
                )

                # Sum the Gaussian densities over winding vectors.
                chunk_log_prob = torch.logsumexp(
                    base_log_prob,
                    dim=1,
                )  # [K, n]

            elif base_log_prob.shape[0] == flat_shifted_data.shape[0]:
                # [number_in_chunk * n, K]
                output_orientation = "n_by_K"

                base_log_prob = base_log_prob.reshape(
                    number_in_chunk,
                    n,
                    base_log_prob.shape[1],
                )

                chunk_log_prob = torch.logsumexp(
                    base_log_prob,
                    dim=0,
                ).transpose(0, 1)  # [K, n]

            else:
                raise RuntimeError(
                    "Could not determine the output orientation of "
                    "Normal.log_pdf. Received shape "
                    f"{tuple(base_log_prob.shape)} for "
                    f"{flat_shifted_data.shape[0]} observations."
                )

            # Add this chunk to previous chunks in log space.
            if accumulated_log_prob is None:
                accumulated_log_prob = chunk_log_prob
            else:
                accumulated_log_prob = torch.logaddexp(
                    accumulated_log_prob,
                    chunk_log_prob,
                )

        if accumulated_log_prob is None:
            raise RuntimeError("No winding vectors were generated.")

        if output_orientation == "K_by_n":
            return accumulated_log_prob

        return accumulated_log_prob.transpose(0, 1)


class WrappedNormal(PCMMtorchBaseModel):
    """Multivariate wrapped normal with component means and low-rank-plus-isotropic covariance.

    Each component is the wrapping of ``N(mu[k], M[k] @ M[k].T + gamma[k] * I)`` onto the
    p-dimensional torus. The infinite wrapping sum is truncated to a configurable symmetric
    lattice. Unlike ``WrappedNormal_old``, this model estimates circular component means and
    initializes both means and covariances using the geometry of the torus.
    """

    def __init__(self, p:int, rank:int, K:int=1, HMM:bool=False, samples_per_sequence=0, params:dict=None, force_gamma_same:bool=False, winding_radius:int=1, winding_chunk_size:int=64, max_winding_vectors:int=200_000):
        super().__init__()
        if p < 1 or rank < 1 or rank > p:
            raise ValueError("WrappedNormal requires p >= 1 and 1 <= rank <= p.")
        if winding_radius < 0 or winding_chunk_size < 1:
            raise ValueError("winding_radius must be non-negative and winding_chunk_size must be positive.")
        if (2*winding_radius+1)**p > max_winding_vectors:
            raise ValueError(f"The winding lattice contains {(2*winding_radius+1)**p:,} vectors, exceeding max_winding_vectors={max_winding_vectors:,}.")

        self.p, self.r, self.K, self.HMM = p, rank, K, HMM
        self.samples_per_sequence = torch.as_tensor(0 if samples_per_sequence is None else samples_per_sequence)
        self.distribution = 'WrappedNormal_lowrank'
        self.force_gamma_same = force_gamma_same
        self.winding_radius, self.winding_chunk_size, self.max_winding_vectors = winding_radius, winding_chunk_size, max_winding_vectors
        self.log_norm_constant = -self.p/2*math.log(2*math.pi)
        if params is not None:
            self.unpack_params(params)

    @staticmethod
    def _inverse_softplus(value):
        value = torch.clamp(value, min=torch.finfo(value.dtype).tiny)
        return value + torch.log(-torch.expm1(-value))

    def unpack_params(self, params):
        reference = next(iter(params.values()))
        dtype = reference.dtype if torch.is_tensor(reference) else torch.get_default_dtype()
        device = reference.device if torch.is_tensor(reference) else None
        mu = torch.as_tensor(params['mu'], dtype=dtype, device=device)
        M = torch.as_tensor(params['M'], dtype=dtype, device=device)
        gamma = torch.as_tensor(params.get('gamma', torch.ones(self.K)), dtype=dtype, device=device)
        if mu.shape != (self.K, self.p) or M.shape != (self.K, self.p, self.r) or gamma.shape != (self.K,):
            raise ValueError(f"Expected mu {(self.K,self.p)}, M {(self.K,self.p,self.r)}, and gamma {(self.K,)}, got {tuple(mu.shape)}, {tuple(M.shape)}, and {tuple(gamma.shape)}.")
        self.mu, self.M, self.gamma = nn.Parameter(mu), nn.Parameter(M), nn.Parameter(gamma)
        self.pi = nn.Parameter(torch.as_tensor(params.get('pi', torch.ones(self.K)/self.K), dtype=dtype, device=device))
        if self.HMM and 'T' in params:
            self.T = nn.Parameter(torch.as_tensor(params['T'], dtype=dtype, device=device))

    def get_params(self):
        params = {'mu':self.mu.detach(), 'M':self.M.detach(), 'gamma':self.gamma.detach(), 'pi':self.pi.detach()}
        if self.HMM:
            params['T'] = self.T.detach()
        return params

    def initialize(self, X, init_method=None):
        if X.ndim != 2 or X.shape[1] != self.p or torch.is_complex(X):
            raise ValueError(f"WrappedNormal expects real angles with shape (n, {self.p}).")
        from PCMM.phase_coherence_kmeans import torus_clustering
        init = {'tc':'++', 'torus_clustering':'++'}.get(init_method, init_method)
        centers, labels, _ = torus_clustering(X.detach().cpu().numpy(), K=self.K, init=init, num_repl=1, suppress_output=True)
        mu = torch.as_tensor(centers, dtype=X.dtype, device=X.device)
        M = torch.zeros((self.K, self.p, self.r), dtype=X.dtype, device=X.device)
        component_gamma = torch.zeros(self.K, dtype=X.dtype, device=X.device)
        minimum_variance = torch.as_tensor(1e-4, dtype=X.dtype, device=X.device)

        for k in range(self.K):
            residuals = torch.atan2(torch.sin(X[labels == k]-mu[k]), torch.cos(X[labels == k]-mu[k]))
            if residuals.shape[0] < 2:
                residuals = torch.atan2(torch.sin(X-mu[k]), torch.cos(X-mu[k]))
            covariance = residuals.mT@residuals/max(residuals.shape[0], 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            eigenvalues, eigenvectors = eigenvalues.flip(0), eigenvectors.flip(1)
            residual_variance = eigenvalues[self.r:].mean() if self.r < self.p else torch.clamp(eigenvalues[-1]*0.05, min=minimum_variance)
            component_gamma[k] = torch.clamp(residual_variance, min=minimum_variance)
            M[k] = eigenvectors[:,:self.r]*torch.sqrt(torch.clamp(eigenvalues[:self.r]-component_gamma[k], min=minimum_variance))[None,:]

        if self.force_gamma_same:
            component_gamma[:] = component_gamma.mean()
        proportions = torch.bincount(torch.as_tensor(labels, device=X.device), minlength=self.K).to(X.dtype)
        self.unpack_params({'mu':mu, 'M':M, 'gamma':self._inverse_softplus(component_gamma), 'pi':torch.clamp(proportions/proportions.sum(), min=torch.finfo(X.dtype).eps)})

    def log_pdf(self, X, recompute_statics=False):
        if X.ndim != 2 or X.shape[1] != self.p or torch.is_complex(X):
            raise ValueError(f"WrappedNormal expects real angles with shape (n, {self.p}).")

        residuals = torch.atan2(torch.sin(X[None,:,:]-self.mu[:,None,:]), torch.cos(X[None,:,:]-self.mu[:,None,:]))
        gamma = torch.nn.functional.softplus(self.gamma)
        if self.force_gamma_same:
            gamma = gamma.mean().expand_as(gamma)
        covariance_factor = self.M/torch.sqrt(gamma[:,None,None])
        woodbury = torch.eye(self.r, dtype=X.dtype, device=X.device)[None,:,:] + covariance_factor.mT@covariance_factor
        log_determinant = self.p*torch.log(gamma) + torch.linalg.slogdet(woodbury).logabsdet
        windings = _winding_vectors(self.p, self.winding_radius, X.device, X.dtype)
        accumulated_log_pdf = None

        for winding_chunk in windings.split(self.winding_chunk_size):
            shifted = residuals[:,None,:,:] + 2*math.pi*winding_chunk[None,:,None,:]
            projected = torch.einsum('kwnp,kpr->kwnr', shifted, covariance_factor)
            solved = torch.linalg.solve(woodbury, projected.reshape(self.K,-1,self.r).mT).mT.reshape_as(projected)
            mahalanobis = torch.clamp((shifted.square().sum(-1)-torch.sum(projected*solved, dim=-1))/gamma[:,None,None], min=0)
            chunk_log_pdf = torch.logsumexp(self.log_norm_constant-0.5*log_determinant[:,None,None]-0.5*mahalanobis, dim=1)
            accumulated_log_pdf = chunk_log_pdf if accumulated_log_pdf is None else torch.logaddexp(accumulated_log_pdf, chunk_log_pdf)

        return accumulated_log_pdf
