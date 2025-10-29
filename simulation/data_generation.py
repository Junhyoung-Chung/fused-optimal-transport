import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# ==========================================================
# Helper functions: lattice wiring within blocks
# ==========================================================

def _grid_shape(m: int):
    """
    Compute the most square-like (r, c) grid shape for m nodes.

    Returns integers (r, c) such that r * c >= m and |r - c| is minimized.
    """
    r = int(np.floor(np.sqrt(m)))
    c = int(np.ceil(m / r))
    while r * c < m:
        r += 1
    return r, c


def _wire_grid(indices, periodic=False):
    """
    Connect nodes on a 2D lattice (von Neumann neighborhood, 4-neighbors).

    Parameters
    ----------
    indices : list[int]
        Global indices of nodes in this block.
    periodic : bool, optional
        If True, wrap boundaries to form a torus (periodic lattice).

    Returns
    -------
    edges : list[tuple[int, int]]
        List of undirected edges (u, v).
    """
    m = len(indices)
    if m <= 1:
        return []

    r, c = _grid_shape(m)
    pad = r * c - m
    arr = np.array(indices + [-1] * pad).reshape(r, c)

    edges = []
    for i in range(r):
        for j in range(c):
            u = arr[i, j]
            if u < 0:
                continue
            # Right neighbor
            j2 = (j + 1) % c if periodic else j + 1
            if j2 < c:
                v = arr[i, j2]
                if v >= 0:
                    edges.append((u, v))
            # Down neighbor
            i2 = (i + 1) % r if periodic else i + 1
            if i2 < r:
                v = arr[i2, j]
                if v >= 0:
                    edges.append((u, v))
    return edges


def _wire_ring(indices, periodic=True):
    """
    Connect nodes on a 1D lattice (chain or ring).

    Parameters
    ----------
    indices : list[int]
        Global indices of nodes in this block.
    periodic : bool, optional
        If True, connect last to first (ring). Otherwise, open chain.

    Returns
    -------
    edges : list[tuple[int, int]]
    """
    m = len(indices)
    if m <= 1:
        return []
    edges = [(indices[k], indices[k + 1]) for k in range(m - 1)]
    if periodic and m >= 3:
        edges.append((indices[-1], indices[0]))
    return edges


# ==========================================================
# Generator: Degree-corrected SBM with lattice within blocks
# ==========================================================

def sample_dc_sbm(
    n, B, pin, pout, similar_pair=(0, 1), delta=0.0, rng=None,
    within_mode="erdos",        # "erdos" | "grid" | "ring"
    grid_periodic=False,        # Torus flag (used for within_mode="grid")
    add_within_random=0.0       # Probability of adding random intra-block edges
):
    """
    Generate a degree-corrected stochastic block model (DC-SBM) with structured
    intra-block connections.

    Parameters
    ----------
    n : int
        Total number of nodes.
    B : int
        Number of blocks.
    pin : float or list[float]
        Within-block base edge probability (scalar or block-specific).
    pout : float
        Across-block edge probability.
    similar_pair : tuple[int, int]
        Indices of two blocks to enforce structural similarity.
    delta : float
        Difference level between the two similar blocks.
    within_mode : str
        Intra-block topology type:
            - "erdos": random Erdos–Rényi subgraph
            - "grid":  2D lattice (von Neumann)
            - "ring":  1D ring (or chain)
    grid_periodic : bool
        Whether to wrap edges on grid boundaries (toroidal lattice).
    add_within_random : float
        Probability of adding random edges on top of lattice structure.
        (ignored when within_mode="erdos")

    Returns
    -------
    A : ndarray (n × n)
        Symmetric adjacency matrix.
    z : ndarray (n,)
        Block labels (0..B−1).
    theta : ndarray (n,)
        Node-specific degree-correction factors.
    pin_b : ndarray (B,)
        Effective within-block base probabilities.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Balanced block sizes
    sizes = [n // B] * B
    for i in range(n - sum(sizes)):
        sizes[i] += 1
    z = np.repeat(np.arange(B), sizes)

    if within_mode == 'erdos':
        # Per-block p_in
        pin_b = np.full(B, pin, dtype=float) if np.isscalar(pin) else np.array(pin, dtype=float)

        # Enforce structural similarity between a and b
        a, b = similar_pair
        base = (pin_b[a] + pin_b[b]) / 2.0
        pin_b[a] = base + delta / 2.0
        pin_b[b] = base - delta / 2.0

    # Node-specific degree correction
    theta = rng.gamma(shape=2.0, scale=0.7, size=n)
    theta /= np.mean(theta)

    A = np.zeros((n, n), dtype=int)

    # ---- 1) Within-block edges ----
    for blk in range(B):
        idx = np.where(z == blk)[0].tolist()
        if len(idx) <= 1:
            continue

        if within_mode == "grid":
            edges = _wire_grid(idx, periodic=grid_periodic)
        elif within_mode == "ring":
            edges = _wire_ring(idx, periodic=True)
        elif within_mode == "erdos":
            edges = []
        else:
            raise ValueError("within_mode must be one of {'erdos','grid','ring'}")

        # (a) Deterministic lattice edges
        for u, v in edges:
            A[u, v] = A[v, u] = 1

        # (b) Optional random edges within block
        if add_within_random > 0 or within_mode == "erdos":
            basep = pin_b[blk]
            for i_pos, u in enumerate(idx):
                for j_pos in range(i_pos + 1, len(idx)):
                    v = idx[j_pos]
                    if A[u, v] == 1:
                        continue
                    pij = (add_within_random if within_mode != "erdos" else 1.0) * basep * theta[u] * theta[v]
                    pij = min(pij, 0.9)
                    if rng.random() < pij:
                        A[u, v] = A[v, u] = 1

    # ---- 2) Across-block edges ----
    for i in range(n):
        for j in range(i + 1, n):
            if z[i] != z[j]:
                pij = pout * theta[i] * theta[j]
                pij = min(pij, 0.9)
                if rng.random() < pij:
                    A[i, j] = A[j, i] = 1

    np.fill_diagonal(A, 0)
    if within_mode == 'erdos':
        return A, z, theta, pin_b
    else:
        return A, z, theta, np.inf


# ==========================================================
# Utilities for permutations, kernels, and features
# ==========================================================

def blockwise_permutation(z, block_perm=None, rng=None,
                          shuffle_within_source=True,
                          shuffle_within_target=True):
    """
    Construct a permutation that permutes blocks (and optionally within-block nodes).

    Parameters
    ----------
    z : ndarray (n,)
        Block labels.
    block_perm : dict[int,int], optional
        Mapping from source block → target block.
        If None, identity mapping is used.
    shuffle_within_source : bool
        Whether to shuffle nodes within each source block.
    shuffle_within_target : bool
        Whether to shuffle target indices before assignment.

    Returns
    -------
    perm : ndarray (n,)
        Permutation vector such that new_order = old[perm].
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(z)
    uniq = np.unique(z)
    if block_perm is None:
        block_perm = {b: b for b in uniq}

    idx_by_block = {b: np.where(z == b)[0] for b in uniq}
    if shuffle_within_source:
        for b in uniq:
            idx_by_block[b] = rng.permutation(idx_by_block[b])

    perm = np.empty(n, dtype=int)
    for b in uniq:
        src = idx_by_block[b]
        dst_block = block_perm[b]
        dst = np.where(z == dst_block)[0]
        if shuffle_within_target:
            dst = rng.permutation(dst)
        assert len(src) == len(dst), "Block sizes must match for permutation."
        perm[src] = dst
    return perm


def perm_matrix(perm):
    """Convert a permutation vector to a permutation matrix."""
    n = len(perm)
    P = np.zeros((n, n))
    P[np.arange(n), perm] = 1.0
    return P


def heat_kernel_from_adj(A, t=1.0, normalize=True, order=2):
    """
    Approximate the heat kernel exp(-tL) using a truncated Taylor expansion.

    Parameters
    ----------
    A : ndarray
        Adjacency matrix.
    t : float
        Diffusion time parameter.
    normalize : bool
        If True, normalize rows/columns to ensure unit diagonal mass.
    order : int
        Approximation order (1 or 2).

    Returns
    -------
    K : ndarray
        Heat kernel matrix.
    """
    L = laplacian(csr_matrix(A), normed=False).toarray()
    I = np.eye(A.shape[0])
    if order == 2:
        K = I - t * L + 0.5 * (t ** 2) * (L @ L)
    else:
        K = I - t * L

    if normalize:
        d = K.sum(1, keepdims=True)
        d = np.clip(d, 1e-8, None)
        K = K / np.sqrt(d @ d.T)
    return K


def all_pairs_geodesic(A, weighted=False):
    """
    Compute all-pairs shortest-path (geodesic) distances.

    Parameters
    ----------
    A : ndarray
        Adjacency matrix (binary or weighted).
    weighted : bool
        If True, interpret entries as weights; otherwise, unweighted graph.

    Returns
    -------
    D : ndarray
        Pairwise geodesic distance matrix.
    """
    G = csr_matrix(A if weighted else (A > 0).astype(float))
    D = shortest_path(G, directed=False, unweighted=not weighted)
    return D


def make_block_features(z, d=6, margin=2.0, noise=0.5, rng=None):
    """
    Generate block-wise Gaussian feature vectors.

    Each node feature f_i ~ N(mu_{z_i}, noise^2 I_d),
    where block means mu_b are separated by `margin`.

    Parameters
    ----------
    z : ndarray (n,)
        Block labels.
    d : int
        Feature dimension.
    margin : float
        Separation distance between block means.
    noise : float
        Gaussian noise level.
    rng : np.random.Generator, optional

    Returns
    -------
    FX : ndarray (n, d)
        Feature matrix.
    mu : ndarray (B, d)
        Block mean vectors.
    """
    if rng is None:
        rng = np.random.default_rng()

    B = z.max() + 1
    mu = np.zeros((B, d))
    for b in range(B):
        mu[b, b % d] = margin
    FX = mu[z] + noise * rng.standard_normal((len(z), d))
    return FX, mu
