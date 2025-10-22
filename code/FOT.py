import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot

# -------------------------------
# Utility: kernels & distances
# -------------------------------

def kappa_decreasing_exp(t):
    """Strictly decreasing and bounded (to 1 at t=0): exp(-t**2)."""
    return np.exp(-t**2)

def kappa_increasing_logistic(t):
    """Strictly increasing and bounded in (0,1): exp(t)/(1+exp(t))."""
    # stable logistic
    out = np.empty_like(t, dtype=float)
    pos = t >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-t[pos]))
    out[~pos] = np.exp(t[~pos]) / (1.0 + np.exp(t[~pos]))
    return out

def make_distance_kernel_matrix(points, h=1.0, kappa="decreasing_exp", metric="euclidean"):
    """
    Build hat{D}_X^kappa where (D)_{ii'} = K_h(x_i, x_{i'}) = (1/h) * kappa( d(x_i,x_{i'})/h ).
    - points: (n, d) array of coordinates in S (or arbitrary features used only for distances)
    - h: bandwidth > 0
    - kappa: str or callable
    - metric: str or callable
    """
    if callable(kappa):
        kappa_fun = kappa
    else:
        if kappa == "decreasing_exp":
            kappa_fun = kappa_decreasing_exp
        elif kappa == "increasing_logistic":
            kappa_fun = kappa_increasing_logistic
        else:
            raise ValueError("Unknown kappa; provide callable or one of {'decreasing_exp','increasing_logistic'}")

    D = cdist(points, points, metric=metric)
    D = D / float(D.max())

    Kh = (1.0 / h) * kappa_fun(D / h)

    return Kh

# -------------------------------
# Objective & gradient
# -------------------------------

def fused_convex_objective(pi, C_f, DkX, DkY, lam):
    """
    L(pi) = <C_f, pi> + (lam / (2 nX nY)) * || nY DkX pi - nX pi DkY ||_F^2
    """
    nX, nY = pi.shape
    A = nY * DkX
    B = nX * DkY
    diff = A @ pi - pi @ B
    quad = np.sum(diff * diff)
    return np.sum(C_f * pi) + (lam / (2.0 * nX * nY)) * quad

def fused_convex_gradient(pi, C_f, DkX, DkY, lam):
    """
    ∇L(pi) = C_f + (lam / (nX nY)) * [ A^T (A pi - pi B) - (A pi - pi B) B^T ]
    where A = nY DkX, B = nX DkY
    """
    nX, nY = pi.shape
    A = nY * DkX
    B = nX * DkY
    Ap_minus_pB = A @ pi - pi @ B
    grad_quad = A.T @ Ap_minus_pB - Ap_minus_pB @ B.T
    return C_f + (lam / (nX * nY)) * grad_quad

# -------------------------------
# Linear Minimization Oracle (FW step) using EMD
# -------------------------------

def lmo_transport_on_polytope(grad, a, b, method="emd"):
    """
    Solve s = argmin_{π ∈ Pi(a,b)} <grad, π>.
    We call POT's EMD with cost = shifted 'grad' to ensure non-negativity.
    - a: (nX,) source histogram (sums to 1)
    - b: (nY,) target histogram (sums to 1)
    Returns an (nX,nY) transport plan s.
    """
    G = grad
    # Shift costs to be >= 0 (doesn't change argmin over Pi(a,b))
    minG = G.min()
    C = G - minG

    if method == "emd":
        # exact network simplex
        s = ot.emd(a, b, C)
    elif method == "sinkhorn":
        # entropic approx; faster but inexact LMO (usually still fine)
        s = ot.sinkhorn(a, b, C, reg=1e-3)
    else:
        raise ValueError("Unknown LMO method. Choose 'emd' or 'sinkhorn'.")
    return s

# -------------------------------
# LAP projection (optional)
# -------------------------------

def rectangular_hard_assignment_from_plan(pi):
    """
    Solve P = argmax_{P in {0,1}} Tr(P^T pi)
    with at most one 1 per row/column (rectangular LAP).
    Equivalent to min_{P} < -pi, P >; use Hungarian algorithm.
    Returns P with shape (nX, nY), dtype=int {0,1}.
    """
    # nX, nY = pi.shape
    # SciPy's linear_sum_assignment finds min-cost assignment.
    # We want to maximize Tr(P^T pi) ⇒ minimize -pi.
    row_ind, col_ind = linear_sum_assignment(-pi)
    P = np.zeros_like(pi, dtype=int)
    # linear_sum_assignment returns |row_ind| = min(nX, nY) pairs
    P[row_ind, col_ind] = 1
    return P

# -------------------------------
# Main estimator class
# -------------------------------

class ConvexFusedTransport:
    """
    Convex Quadratic Fused Transport via Frank–Wolfe and (optional) LAP projection.

    Problem:
      min_{π ∈ Pi(a,b)}  <C_f, π> + (λ/(2 nX nY)) || nY DkX π - nX π DkY ||_F^2
      where a = 1/nX, b = 1/nY (uniform empirical marginals by default).

    Parameters
    ----------
    lam : float
        Regularization λ ≥ 0.
    h : float
        Bandwidth for distance kernel K_h.
    kappa : {"decreasing_exp","increasing_logistic"} or callable
        Monotone bounded κ; K_h(x,x') = (1/h) κ(d(x,x')/h).
    metric : str options available in `cdist` (e.g., "euclidean","cityblock","cosine") or callable
        Metric for cdist (if distance matrices not precomputed).
    fw_max_iter : int
        Max Frank–Wolfe iterations.
    fw_stepsize : {"classic", "line-search"} or callable
        If "classic": γ_t = 2/(t+2).
        If callable: gamma = fw_stepsize(t, pi, s) -> float in (0,1].
        (Simple backtracking line-search could be plugged in here if desired.)
    tol : float
        Stop when FW dual gap <= tol.
    lmo_method : {"emd","sinkhorn"}
        Linear minimization oracle backend.
    random_state : int or None
        For reproducibility of any randomized choices (not used by default).
    """
    def __init__(self,
                 lam=1.0,
                 h=1.0,
                 kappa="decreasing_exp",
                 metric="euclidean",
                 fw_max_iter=200,
                 fw_stepsize="classic",
                 tol=1e-7,
                 lmo_method="emd",
                 random_state=None):
        self.lam = lam
        self.h = h
        self.kappa = kappa
        self.metric = metric
        self.fw_max_iter = fw_max_iter
        self.fw_stepsize = fw_stepsize
        self.tol = tol
        self.lmo_method = lmo_method
        self.random_state = random_state

        # learned attributes
        self.pi_ = None
        self.P_ = None
        self.obj_history_ = []
        self.gap_history_ = []
        self.info_ = {}

    def _build_uniform_marginals(self, nX, nY):
        a = np.full(nX, 1.0 / nX)
        b = np.full(nY, 1.0 / nY)
        return a, b

    def _exact_linesearch_gamma(self, pi, s, C_f, DkX, DkY):
        """
        Exact line-search along D = s - pi for
          L(pi) = <C_f, pi> + (lam/(2 nX nY)) || A pi - pi B ||_F^2
        with A = nY DkX, B = nX DkY.

        Returns gamma* in [0,1].
        """
        nX, nY = pi.shape
        lam = float(self.lam)

        # Direction
        D = s - pi

        # Linear term: <C_f, D>
        lin = float(np.sum(C_f * D))

        # If lam == 0, the problem is linear in pi → FW exact step is gamma=1 (since lin <= 0 by LMO).
        if lam == 0.0:
            return 1.0 if lin < 0.0 else 0.0

        # Quadratic part
        A = nY * DkX
        B = nX * DkY

        M = A @ pi - pi @ B  # current residual
        N = A @ D - D @ B  # residual change along D

        m = float(np.sum(M * N))  # <M, N>
        n = float(np.sum(N * N))  # ||N||_F^2

        # If N == 0, quadratic term doesn't change along D → reduce to linear
        if n <= 0.0:
            return 1.0 if lin < 0.0 else 0.0

        mu = nX * nY
        # phi'(gamma) = lin + (lam/mu) * ( m + gamma * n )
        # set to zero → gamma* = - ( lin + (lam/mu)*m ) / ( (lam/mu)*n )
        gamma_star = - (mu * lin / lam + m) / n

        # project to [0,1]
        if not np.isfinite(gamma_star):
            # super defensive fallback
            return 1.0 if lin < 0.0 else 0.0
        return float(np.clip(gamma_star, 0.0, 1.0))

    def _stepsize(self, t, pi, s, C_f, DkX, DkY):
        if self.fw_stepsize == "classic":
            return 2.0 / (t + 2.0)
        elif self.fw_stepsize == "line-search":
            return self._exact_linesearch_gamma(pi, s, C_f, DkX, DkY)
        elif callable(self.fw_stepsize):
            return float(self.fw_stepsize(t, pi, s, C_f, DkX, DkY))
        else:
            # Default fallback
            return 2.0 / (t + 2.0)

    def fit(self, X, Y, FX=None, FY=None,
            return_hard_assignment=False):
        """
        Fit the CQFT plan between samples X and Y.

        Inputs
        ------
        X : (nX, dS) array
            Coordinates in S for source points (used only for d_S).
        Y : (nY, dS) array
            Coordinates in S for target points (used only for d_S).
        FX : (nX, df) array or None
            Features f(X_i). If None, uses X (identity as features).
        FY : (nY, df) array or None
            Features f(Y_j). If None, uses Y (identity as features).

        Returns
        -------
        self
          with attributes:
          - pi_ : optimal soft coupling (nX, nY)
          - P_ : optional hard assignment (nX, nY) with 0/1 (rectangular), if return_hard_assignment=True
          - obj_history_, gap_history_, info_
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        nX, nY = X.shape[0], Y.shape[0]

        if FX is None: FX = X
        if FY is None: FY = Y
        FX = np.asarray(FX, dtype=float)
        FY = np.asarray(FY, dtype=float)

        # Cost C_f for feature alignment
        # (squared Euclidean in feature space; customize if needed)
        C_f = cdist(FX, FY, metric='sqeuclidean')
        C_f = C_f / float(C_f.max())

        # Distance-kernel matrices hat{D}_X^κ, hat{D}_Y^κ
        DkX = make_distance_kernel_matrix(
            X, h=self.h, kappa=self.kappa, metric=self.metric
        )
        DkY = make_distance_kernel_matrix(
            Y, h=self.h, kappa=self.kappa, metric=self.metric
        )

        # Uniform empirical marginals
        a, b = self._build_uniform_marginals(nX, nY)

        # Initialize π^(0) = a b^T (independent coupling)
        pi = np.outer(a, b)

        self.obj_history_.clear()
        self.gap_history_.clear()

        # Frank–Wolfe loop
        for t in range(1, self.fw_max_iter + 1):
            # Gradient at current pi
            grad = fused_convex_gradient(pi, C_f, DkX, DkY, self.lam)

            # Linear Minimization Oracle: s = argmin_{π∈U(a,b)} <grad, π>
            s = lmo_transport_on_polytope(grad, a, b, method=self.lmo_method)

            # Dual gap g_t = <∇L(pi), pi - s>
            gap = np.sum(grad * (pi - s))
            self.gap_history_.append(gap)

            # Evaluate objective (optional per-iter)
            obj = fused_convex_objective(pi, C_f, DkX, DkY, self.lam)
            self.obj_history_.append(obj)

            # Stopping criterion
            if gap <= self.tol:
                # one last objective at final pi
                break

            # Stepsize
            gamma = self._stepsize(t, pi, s, C_f, DkX, DkY)
            gamma = float(np.clip(gamma, 1e-12, 1.0))

            # Update
            pi = (1.0 - gamma) * pi + gamma * s

        self.pi_ = pi

        if return_hard_assignment:
            self.P_ = rectangular_hard_assignment_from_plan(self.pi_)
        else:
            self.P_ = None

        self.info_ = {
            "iterations": len(self.obj_history_),
            "final_objective": fused_convex_objective(self.pi_, C_f, DkX, DkY, self.lam),
            "final_gap": self.gap_history_[-1] if self.gap_history_ else None,
            "nX": nX, "nY": nY, "lam": self.lam, "h": self.h,
            "kappa": self.kappa if isinstance(self.kappa, str) else getattr(self.kappa, "__name__", "callable")
        }
        return self