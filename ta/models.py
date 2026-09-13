import numpy as np
from .representation import normalized_indicator
from .init import litekmeans_compat
from .solver import (
    build_counts,
    compute_W,
    update_y_tolerance_asym,
    tolerance_penalty_asym,
)


class TA:
    """Tolerance-aware size intervention for discrete ensemble consensus.

    The balance regularizer is

        lambda * sum_k [ eta * [L_delta - n_k]_+^2
                         + [n_k - U_delta]_+^2 ],

    where L_delta=(1-delta)n/K and U_delta=(1+delta)n/K.

    Special cases used in the paper:
      * eta=1: Sym-TA;
      * eta=0: CapQ;
      * delta=0 and eta=1: Quad-SB balance up to an additive constant.

    Normalized partition evidence and adaptive weights are optimized
    jointly with discrete consensus assignments.
    """

    def __init__(
        self,
        n_clusters,
        lambda_=1e-3,
        gamma=2.0,
        delta=0.25,
        eta=1.0,
        seed=2026,
        max_outer=10,
        max_inner=10,
        tol=1e-5,
    ):
        self.n_clusters = int(n_clusters)
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.eta = float(eta)
        self.seed = int(seed)
        self.max_outer = int(max_outer)
        self.max_inner = int(max_inner)
        self.tol = float(tol)
        if not (0.0 <= self.delta < 1.0):
            raise ValueError("TA requires 0 <= delta < 1.")
        if self.eta < 0.0:
            raise ValueError("TA requires eta >= 0.")

    def fit_predict(self, members):
        members = np.asarray(members)
        F, meta = normalized_indicator(members, None)
        y = litekmeans_compat(
            F, self.n_clusters, seed=self.seed, max_iter=100
        ).astype(np.int32)

        enc, kbs, offsets, feat_idx, inv_sizes, feature_base = meta
        n, m = enc.shape
        counts, sizes = build_counts(
            y.copy(), feat_idx, self.n_clusters, int(offsets[-1])
        )
        alpha = np.ones(m, dtype=float) / m
        objs = []
        avg_size = n / float(self.n_clusters)

        for _ in range(self.max_outer):
            apow = alpha ** self.gamma
            update_y_tolerance_asym(
                y, counts, sizes, feat_idx, inv_sizes, feature_base,
                apow, self.lambda_, self.delta, self.eta,
                self.max_inner, self.tol,
            )
            W = compute_W(
                counts, inv_sizes, offsets, self.n_clusters, m, n
            )
            r = 1.0 / (1.0 - self.gamma)
            tmp = (self.gamma * W) ** r
            alpha = tmp / tmp.sum()
            pen = float(
                tolerance_penalty_asym(
                    sizes, avg_size, self.delta, self.eta
                )
            )
            obj = float(
                np.sum((alpha ** self.gamma) * W) + self.lambda_ * pen
            )
            objs.append(obj)
            if len(objs) > 1 and abs(objs[-1] - objs[-2]) < self.tol:
                break

        self.labels_ = y.copy()
        self.alpha_ = alpha.copy()
        self.cluster_sizes_ = sizes.copy()
        self.objective_ = np.asarray(objs)
        self.balance_penalty_ = float(
            tolerance_penalty_asym(
                sizes, avg_size, self.delta, self.eta
            )
        )
        return self.labels_


def make_ta(
    n_clusters, lambda_=1e-3, gamma=2.0, delta=0.25, eta=1.0, seed=2026
):
    """Construct a TA model with the specified tolerance and asymmetry."""
    return TA(
        n_clusters=n_clusters,
        lambda_=lambda_,
        gamma=gamma,
        delta=delta,
        eta=eta,
        seed=seed,
    )


def make_sym_ta(n_clusters, lambda_=1e-3, gamma=2.0, delta=0.25, seed=2026):
    """Symmetric tolerance-aware ablation (eta=1)."""
    return make_ta(n_clusters, lambda_, gamma, delta, 1.0, seed)


def make_capq(n_clusters, lambda_=1e-3, gamma=2.0, delta=0.25, seed=2026):
    """Upper-cap-only ablation (eta=0)."""
    return make_ta(n_clusters, lambda_, gamma, delta, 0.0, seed)


def make_quad_sb(n_clusters, lambda_=1e-3, gamma=2.0, seed=2026):
    """Quad-SB balance special case (delta=0, eta=1)."""
    return make_ta(n_clusters, lambda_, gamma, 0.0, 1.0, seed)
