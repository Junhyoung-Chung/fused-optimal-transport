import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot

# -------------------------------
# Utility: kernels & distances
# -------------------------------

def kappa_linear_cutoff(t, k=1.0):
    """
    Linearly decreasing kernel: kappa(t) = max(0, k - t).
    Vectorized version works for scalar or array t.
    """
    t = np.asarray(t)
    return np.maximum(0.0, k - t)

def kappa_decreasing_exp(t,p=2):
    """Strictly decreasing and bounded (to 1 at t=0): exp(-t**2)."""
    return np.exp(-t**p)

def kappa_increasing_logistic(t):
    """Strictly increasing and bounded in (0,1): exp(t)/(1+exp(t))."""
    # stable logistic
    out = np.empty_like(t, dtype=float)
    pos = t >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-t[pos]))
    out[~pos] = np.exp(t[~pos]) / (1.0 + np.exp(t[~pos]))
    return out

def make_distance_kernel_matrix(points, h=1.0, kappa="decreasing_exp", metric="euclidean",pre_D=None):
    """
    Build hat{D}_X^kappa where (D)_{ii'} = K_h(x_i, x_{i'}) = (1/h) * kappa( d(x_i,x_{i'})/h ).
    - points: (n, d) array of coordinates in S (or arbitrary features used only for distances)
    - h: bandwidth > 0
    - kappa: str or callable
    - metric: str or callable
    - pre_D: pre-computed distance matrix
    """
    if h <= 0:
        raise ValueError("h must be positive")

    if callable(kappa):
        kappa_fun = kappa
    else:
        if kappa == "decreasing_exp":
            kappa_fun = kappa_decreasing_exp
        elif kappa == "increasing_logistic":
            kappa_fun = kappa_increasing_logistic
        else:
            raise ValueError("Unknown kappa; provide callable or one of {'decreasing_exp','increasing_logistic'}")

    if pre_D is None:
        D = cdist(points, points, metric=metric)
        Dmax = float(D.max())
        if Dmax > 0:
            D = D / Dmax
    else:
        Dmax = float(pre_D.max())
        if Dmax > 0:
            D = pre_D / Dmax

    Kh = (1.0 / h) * kappa_fun(D / h)
    Khmax = float(Kh.max())
    if Khmax > 0:
        Kh = Kh / Khmax

    return Kh

# -------------------------------
# Objective & gradient
# -------------------------------

def fused_convex_objective(pi, C_f, DkX, DkY, alpha):
    """
    L(pi) = (1 - alpha) * <C_f, pi> + (alpha / (2 nX nY)) * || nY DkX pi - nX pi DkY ||_F^2
    """
    nX, nY = pi.shape
    A = nY * DkX
    B = nX * DkY
    diff = A @ pi - pi @ B
    quad = np.sum(diff * diff)
    return (1 - alpha) * np.sum(C_f * pi) , (alpha / (2.0 * nX * nY)) * quad

def fused_convex_gradient(pi, C_f, DkX, DkY, alpha):
    """
    ∇L(pi) = (1-alpha) * C_f + (alpha / (nX nY)) * [ A^T (A pi - pi B) - (A pi - pi B) B^T ]
    where A = nY DkX, B = nX DkY
    """
    nX, nY = pi.shape
    A = nY * DkX
    B = nX * DkY
    Ap_minus_pB = A @ pi - pi @ B
    grad_quad = A.T @ Ap_minus_pB - Ap_minus_pB @ B.T
    return (1 - alpha) * C_f + (alpha / (nX * nY)) * grad_quad

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
      min_{π ∈ Pi(a,b)}  (1 - alpha) * <C_f, π> + (alpha / (2 nX nY)) || nY DkX π - nX π DkY ||_F^2
      where a = 1/nX, b = 1/nY (uniform empirical marginals by default).

    Parameters
    ----------
    alpha : float
        Weight parameter 0 <= alpha <= 1
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
    pre_Cf : pre-computed feature cost.
    pre_DX : pre-computed distance matrix of X.
    pre_DY : pre-computed distance matrix of Y.
    verbose : int, default=0
        Verbosity level controlling optimization logging:
          - 0 : silent mode (no output)
          - 1 : progress printed every 10 iterations and at convergence
          - 2 : detailed log every iteration (gap, objective components)
    """
    def __init__(self,
                 alpha=0.5,
                 h=1.0,
                 kappa="decreasing_exp",
                 metric="euclidean",
                 fw_max_iter=200,
                 fw_stepsize="classic",
                 tol=1e-7,
                 lmo_method="emd",
                 random_state=None,
                 pre_Cf=None,
                 pre_DX=None,
                 pre_DY=None,
                 verbose=0):
        self.alpha = alpha
        self.h = h
        self.kappa = kappa
        self.metric = metric
        self.fw_max_iter = fw_max_iter
        self.fw_stepsize = fw_stepsize
        self.tol = tol
        self.lmo_method = lmo_method
        self.random_state = random_state
        self.pre_Cf = pre_Cf
        self.pre_DX = pre_DX
        self.pre_DY = pre_DY
        self.verbose = verbose

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
          L(pi) = (1 - alpha) * <C_f, pi> + (alpha/(2 nX nY)) || A pi - pi B ||_F^2
        with A = nY DkX, B = nX DkY.

        Returns gamma* in [0,1].
        """
        nX, nY = pi.shape
        alpha = float(self.alpha)

        # Direction
        D = s - pi

        # Linear term: <C_f, D>
        lin = float(np.sum(C_f * D))

        # If alpha == 0, the problem is linear in pi → FW exact step is gamma=1 (since lin <= 0 by LMO).
        if alpha == 0.0:
            return 1.0 if lin < 0.0 else 0.0

        # Quadratic part
        A = nY * DkX
        B = nX * DkY

        M = A @ pi - pi @ B  # current residual
        N = A @ D - D @ B  # residual change along D

        m = float(np.sum(M * N))  # <M, N>
        n = float(np.sum(N * N))  # ||N||_F^2

        # If N == 0, quadratic term doesn't change along D → reduce to linear
        if n <= 0.0 or not np.isfinite(n):
            # Along D, quadratic term doesn't change; reduce to linear part.
            # If alpha==1, linear part weight is 0 → any gamma works; choose 0 for stability.
            wlin = (1.0 - alpha) * lin
            if (1.0 - alpha) == 0.0:
                return 0.0
            return 1.0 if wlin < 0.0 else 0.0

        mu = nX * nY
        # φ'(γ) = (1-α)*lin + (α/μ)*(m + γ n) = 0
        gamma_star = - ((1.0 - alpha) * mu * lin / alpha + m) / n

        # Project to [0,1] with defensive fallback
        if not np.isfinite(gamma_star):
            wlin = (1.0 - alpha) * lin
            if (1.0 - alpha) == 0.0:
                return 0.0
            return 1.0 if wlin < 0.0 else 0.0

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
            return_hard_assignment=False, init=None):
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
        init : (nX, nY) array or None
            Initial coupling used to warm-start the optimization

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

        if self.pre_Cf is None:
            if FX is None: FX = X
            if FY is None: FY = Y
            FX = np.asarray(FX, dtype=float)
            FY = np.asarray(FY, dtype=float)

            # Cost C_f for feature alignment
            # (squared Euclidean in feature space; customize if needed)
            C_f = cdist(FX, FY, metric='sqeuclidean')
            Cmax = float(C_f.max())
            if Cmax > 0:
                C_f = C_f / Cmax
        else:
            C_f = self.pre_Cf
            Cmax = float(C_f.max())
            if Cmax > 0:
                C_f = C_f / Cmax

        # Distance-kernel matrices hat{D}_X^κ, hat{D}_Y^κ
        if self.pre_DX is None:
            DkX = make_distance_kernel_matrix(
                X, h=self.h, kappa=self.kappa, metric=self.metric
            )
        else:
            DkX = self.pre_DX
            Dmax = float(DkX.max())
            if Dmax > 0:
                DkX = DkX / Dmax

        if self.pre_DY is None:
            DkY = make_distance_kernel_matrix(
                Y, h=self.h, kappa=self.kappa, metric=self.metric
            )
        else:
            DkY = self.pre_DY
            Dmax = float(DkY.max())
            if Dmax > 0:
                DkY = DkY / Dmax

        # Uniform empirical marginals
        a, b = self._build_uniform_marginals(nX, nY)

        if init is None:
            # Initialize π^(0) = a b^T (independent coupling)
            pi = np.outer(a, b)
        else:
            pi = init

        self.obj_history_.clear()
        self.gap_history_.clear()

        # Frank–Wolfe loop
        for t in range(1, self.fw_max_iter + 1):
            grad = fused_convex_gradient(pi, C_f, DkX, DkY, self.alpha)
            s = lmo_transport_on_polytope(grad, a, b, method=self.lmo_method)

            gap = np.sum(grad * (pi - s))
            self.gap_history_.append(gap)

            obj_feat, obj_struct = fused_convex_objective(pi, C_f, DkX, DkY, self.alpha)
            self.obj_history_.append([obj_feat, obj_struct])

            # === Verbose logging ===
            if self.verbose >= 2:
                print(f"[Iter {t:03d}] gap={gap:.3e} "
                      f"obj_feat={obj_feat:.3e} obj_struct={obj_struct:.3e}")
            elif self.verbose == 1 and (t % 10 == 0 or gap <= self.tol):
                print(f"[Iter {t:03d}] gap={gap:.3e}")

            # stopping criterion
            if gap <= self.tol:
                if self.verbose:
                    print(f"Converged at iteration {t} with gap={gap:.3e}")
                break

            gamma = self._stepsize(t, pi, s, C_f, DkX, DkY)
            gamma = float(np.clip(gamma, 1e-12, 1.0))
            pi = (1.0 - gamma) * pi + gamma * s

        self.pi_ = pi

        if return_hard_assignment:
            self.P_ = rectangular_hard_assignment_from_plan(self.pi_)
        else:
            self.P_ = None

        self.info_ = {
            "iterations": len(self.obj_history_),
            "final_objective": fused_convex_objective(self.pi_, C_f, DkX, DkY, self.alpha),
            "final_gap": self.gap_history_[-1] if self.gap_history_ else None,
            "nX": nX, "nY": nY, "alpha": self.alpha, "h": self.h,
            "kappa": self.kappa if isinstance(self.kappa, str) else getattr(self.kappa, "__name__", "callable")
        }

        if self.verbose:
            print("Optimization finished.")
        return self