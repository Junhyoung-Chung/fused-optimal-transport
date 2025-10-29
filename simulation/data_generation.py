import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

# -------------------------------
# Helpers: lattice wiring per block
# -------------------------------

def _grid_shape(m: int):
    """m개 노드를 r x c 격자에 최대한 정사각형에 가깝게 배치."""
    r = int(np.floor(np.sqrt(m)))
    c = int(np.ceil(m / r))
    while r * c < m:
        r += 1
    # (r-1)*c >=? keep minimal r*c >= m
    return r, c

def _wire_grid(indices, periodic=False):
    """
    indices: 이 블록의 전역 인덱스 리스트 (길이 m)
    2D 격자에 배치하고 von Neumann(4-neighbor) 연결.
    periodic=True면 토러스(모든 경계 wrap).
    """
    m = len(indices)
    if m <= 1:
        return []
    r, c = _grid_shape(m)
    # fill row-major; 마지막 셀들 비울 수 있음
    pad = r * c - m
    arr = np.array(indices + [-1] * pad).reshape(r, c)

    edges = []
    for i in range(r):
        for j in range(c):
            u = arr[i, j]
            if u < 0:  # padding
                continue
            # right neighbor
            j2 = (j + 1) % c if periodic else j + 1
            if j2 < c:
                v = arr[i, j2]
                if v >= 0:
                    edges.append((u, v))
            # down neighbor
            i2 = (i + 1) % r if periodic else i + 1
            if i2 < r:
                v = arr[i2, j]
                if v >= 0:
                    edges.append((u, v))
    return edges

def _wire_ring(indices, periodic=True):
    """
    indices: 이 블록의 전역 인덱스 리스트 (길이 m)
    1D lattice (chain or ring). periodic=True면 링, False면 체인.
    """
    m = len(indices)
    if m <= 1:
        return []
    edges = []
    for k in range(m - 1):
        edges.append((indices[k], indices[k+1]))
    if periodic and m >= 3:
        edges.append((indices[-1], indices[0]))
    return edges

# -------------------------------
# Generator: SBM with lattice inside blocks
# -------------------------------

def sample_dc_sbm(n, B, pin, pout, similar_pair=(0,1), delta=0.0, rng=None,
                  within_mode="erdos",         # "erdos" | "grid" | "ring"
                  grid_periodic=False,          # 토러스 여부 (within_mode="grid"일 때)
                  add_within_random=0.0         # 격자 위에 같은 블록 랜덤 간선 추가 확률(0~1)
                  ):
    """
    Degree-corrected SBM.
    - within_mode="erdos": 기존과 동일(블록 내 무작위)
    - within_mode="grid":  블록 내 노드를 2D 격자에 깔고 4-이웃 연결
    - within_mode="ring":  블록 내 노드를 1D 링(또는 체인)으로 연결
    - add_within_random: lattice 위에 추가로 같은 블록 내 임의 간선을 뿌릴 확률 (0이면 순수 lattice)

    Blocks 'similar_pair'는 delta≈0일수록 내부 구조가 유사.
    pin: scalar 또는 길이 B
    pout: scalar (블록 간 확률)
    """
    if rng is None: rng = np.random.default_rng()

    # balanced sizes
    sizes = [n // B] * B
    for i in range(n - sum(sizes)): sizes[i] += 1
    z = np.repeat(np.arange(B), sizes)                 # block labels (0..B-1)

    # per-block p_in
    if np.isscalar(pin): pin_b = np.array([pin]*B, dtype=float)
    else:               pin_b = np.array(pin, dtype=float)

    # enforce similarity between a,b
    a, b = similar_pair
    base = (pin_b[a] + pin_b[b]) / 2.0
    pin_b[a] = base + delta/2.0
    pin_b[b] = base - delta/2.0

    # node degree parameters (for cross-block and optional within-random edges)
    theta = rng.gamma(shape=2.0, scale=0.7, size=n)
    theta = theta / np.mean(theta)

    A = np.zeros((n, n), dtype=int)

    # ---- 1) Within-block edges: lattice or Erdos ----
    for blk in range(B):
        idx = np.where(z == blk)[0].tolist()
        if len(idx) <= 1:
            continue

        if within_mode == "grid":
            edges = _wire_grid(idx, periodic=grid_periodic)
        elif within_mode == "ring":
            edges = _wire_ring(idx, periodic=True)
        elif within_mode == "erdos":
            edges = []  # 무작위에서 아래에서 처리
        else:
            raise ValueError("within_mode must be one of {'erdos','grid','ring'}")

        # (a) lattice 간선은 deterministic하게 추가
        for u, v in edges:
            A[u, v] = A[v, u] = 1

        # (b) 같은 블록 내 랜덤 간선(옵션): lattice 위에 추가
        #     확률 = add_within_random * pin_b[blk] * theta[u]*theta[v]
        if add_within_random > 0 or within_mode == "erdos":
            # 에르되시 르니(블록 내): base는 pin_b[blk]
            basep = pin_b[blk]
            # 모든 쌍 중 이미 lattice로 연결된 건 중복 추가 방지
            for i_pos in range(len(idx)):
                u = idx[i_pos]
                for j_pos in range(i_pos+1, len(idx)):
                    v = idx[j_pos]
                    if A[u, v] == 1:
                        # 이미 연결(격자)되어 있으면 skip (원하면 중복 허용 가능하지만 여기선 0/1 그래프)
                        continue
                    pij = (add_within_random if within_mode != "erdos" else 1.0) * basep * theta[u] * theta[v]
                    if pij > 0.9: pij = 0.9
                    if rng.random() < pij:
                        A[u, v] = A[v, u] = 1

    # ---- 2) Across-block edges: DC-SBM random ----
    # 블록이 다른 모든 (i<j) 쌍에 대해 pout * theta_i * theta_j
    for i in range(n):
        for j in range(i+1, n):
            if z[i] != z[j]:
                pij = pout * theta[i] * theta[j]
                if pij > 0.9: pij = 0.9
                if rng.random() < pij:
                    A[i, j] = A[j, i] = 1

    np.fill_diagonal(A, 0)
    return A, z, theta, pin_b

def blockwise_permutation(z, block_perm=None, rng=None,
                          shuffle_within_source=True,
                          shuffle_within_target=True):
    """
    z: (n,) block labels (0..B-1)
    block_perm: dict, old_block -> new_block (예: {0:2,1:0,2:1}); None이면 identity
    shuffle_within_source: True면 소스 블록 내에서도 순서를 무작위 셔플
    shuffle_within_target: True면 타깃 블록 내 후보도 무작위 셔플
    return: perm (길이 n) so that new_order = old[perm]
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(z)
    uniq = np.unique(z)
    if block_perm is None:
        block_perm = {b: b for b in uniq}

    # (선택) 블록별 인덱스 사전
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
        # 크기가 같다고 가정(SBM 균형 또는 완전 퍼뮤일 때)
        assert len(src) == len(dst), "block sizes must match for a pure permutation"
        perm[src] = dst
    return perm

def perm_matrix(perm):
    n = len(perm)
    P = np.zeros((n, n), dtype=float)
    P[np.arange(n), perm] = 1.0
    return P

def heat_kernel_from_adj(A, t=1.0, normalize=True, order=2):
    L = laplacian(csr_matrix(A), normed=False).toarray()
    I = np.eye(A.shape[0])
    if order == 2:
        K = I - t*L + 0.5*(t**2) * (L @ L)
    else:
        K = I - t*L
    if normalize:
        d = K.sum(1, keepdims=True); d = np.clip(d, 1e-8, None)
        K = K / np.sqrt(d @ d.T)
    return K

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def all_pairs_geodesic(A, weighted=False):
    G = csr_matrix(A if weighted else (A > 0).astype(float))
    D = shortest_path(G, directed=False, unweighted=not weighted)
    return D

def make_block_features(z, d=6, margin=2.0, noise=0.5, rng=None):
    """Block-wise Gaussian features: f_i ~ N(mu_{z_i}, noise^2 I)."""
    if rng is None: rng = np.random.default_rng()
    B = z.max() + 1
    mu = np.zeros((B, d))
    for b in range(B): mu[b, b % d] = margin
    FX = mu[z] + noise * rng.standard_normal((len(z), d))
    return FX, mu