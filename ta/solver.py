
import numpy as np
from numba import njit

@njit
def build_counts(y, feat_idx, K, L):
    counts = np.zeros((K, L), np.int32)
    sizes = np.zeros(K, np.int32)
    n, m = feat_idx.shape
    for i in range(n):
        c = y[i]
        sizes[c] += 1
        for b in range(m):
            counts[c, feat_idx[i, b]] += 1
    return counts, sizes

@njit
def compute_W(counts, inv_sizes, offsets, K, m, n):
    W = np.empty(m, np.float64)
    for b in range(m):
        s = 0.0
        for g in range(offsets[b], offsets[b+1]):
            inv = inv_sizes[g]
            for c in range(K):
                x = counts[c, g]
                if x:
                    s += x*x*inv
        W[b] = max(n - s, 1e-15)
    return W

@njit
def consensus_similarity(counts, inv_sizes, feature_base, apow):
    K, L = counts.shape
    sim = 0.0
    for c in range(K):
        for g in range(L):
            x = counts[c, g]
            if x:
                sim += apow[feature_base[g]] * x*x * inv_sizes[g]
    return sim

@njit
def update_y_target_prior(
    y, counts, sizes, feat_idx, inv_sizes, feature_base,
    apow, targets, lam, max_inner=10, tol=1e-5
):
    """
    Coordinate descent for:
      const - sum_b alpha_b^gamma ||Y^T B_b||_F^2
      + lambda * sum_k (n_k - target_k)^2

    If target_k=n/K for all k, this differs from Quad-SB's
    lambda*sum n_k^2 only by a constant.
    """
    n, m = feat_idx.shape
    K = sizes.shape[0]
    prev = 1e300

    for it in range(max_inner):
        sim = consensus_similarity(counts, inv_sizes, feature_base, apow)
        prior = 0.0
        for c in range(K):
            d = sizes[c] - targets[c]
            prior += d*d
        obj = lam*prior - sim

        if it > 1 and abs(prev - obj) < tol:
            return obj
        prev = obj

        moved = 0
        for i in range(n):
            p = y[i]
            if sizes[p] <= 1:
                continue

            best_q = p
            best_delta = 0.0

            for q in range(K):
                if q == p:
                    continue

                # Delta of negative consensus similarity.
                dcons = 0.0
                for b in range(m):
                    g = feat_idx[i, b]
                    inv = inv_sizes[g]
                    xp = counts[p, g]
                    xq = counts[q, g]
                    sim_change = apow[b] * (
                        (-2.0*xp + 1.0) + (2.0*xq + 1.0)
                    ) * inv
                    dcons -= sim_change

                np0 = sizes[p]
                nq0 = sizes[q]
                tp = targets[p]
                tq = targets[q]

                dprior = (
                    (np0 - 1.0 - tp)**2 - (np0 - tp)**2
                    + (nq0 + 1.0 - tq)**2 - (nq0 - tq)**2
                )

                delta = dcons + lam*dprior
                if delta < best_delta - 1e-15:
                    best_q = q
                    best_delta = delta

            if best_q != p:
                q = best_q
                for b in range(m):
                    g = feat_idx[i, b]
                    counts[p, g] -= 1
                    counts[q, g] += 1
                sizes[p] -= 1
                sizes[q] += 1
                y[i] = q
                moved += 1

        if moved == 0:
            break

    return prev


@njit
def tolerance_phi(size, avg_size, delta):
    """Squared distance of a cluster size to the tolerance interval.

    phi_delta(s) = [(1-delta)*avg_size - s]_+^2
                 + [s - (1+delta)*avg_size]_+^2.
    """
    lo = (1.0 - delta) * avg_size
    hi = (1.0 + delta) * avg_size
    if size < lo:
        d = lo - size
        return d * d
    if size > hi:
        d = size - hi
        return d * d
    return 0.0

@njit
def tolerance_penalty(sizes, avg_size, delta):
    value = 0.0
    for c in range(sizes.shape[0]):
        value += tolerance_phi(float(sizes[c]), avg_size, delta)
    return value

@njit
def update_y_tolerance(
    y, counts, sizes, feat_idx, inv_sizes, feature_base,
    apow, lam, delta, max_inner=10, tol=1e-5
):
    """Coordinate descent for TA with symmetric tolerance balancing.

    The consensus term is unchanged from Quad-SB. Only the balance term is
    replaced by lambda * sum_k phi_delta(n_k).
    """
    n, m = feat_idx.shape
    K = sizes.shape[0]
    avg_size = n / float(K)
    prev = 1e300

    for it in range(max_inner):
        sim = consensus_similarity(counts, inv_sizes, feature_base, apow)
        pen = tolerance_penalty(sizes, avg_size, delta)
        obj = lam * pen - sim

        if it > 1 and abs(prev - obj) < tol:
            return obj
        prev = obj

        moved = 0
        for i in range(n):
            p = y[i]
            if sizes[p] <= 1:
                continue

            best_q = p
            best_delta = 0.0
            np0 = sizes[p]
            old_p = tolerance_phi(float(np0), avg_size, delta)
            new_p = tolerance_phi(float(np0 - 1), avg_size, delta)

            for q in range(K):
                if q == p:
                    continue

                dcons = 0.0
                for b in range(m):
                    g = feat_idx[i, b]
                    inv = inv_sizes[g]
                    xp = counts[p, g]
                    xq = counts[q, g]
                    sim_change = apow[b] * (
                        (-2.0 * xp + 1.0) + (2.0 * xq + 1.0)
                    ) * inv
                    dcons -= sim_change

                nq0 = sizes[q]
                dpen = (
                    (new_p - old_p)
                    + tolerance_phi(float(nq0 + 1), avg_size, delta)
                    - tolerance_phi(float(nq0), avg_size, delta)
                )
                candidate_delta = dcons + lam * dpen
                if candidate_delta < best_delta - 1e-15:
                    best_q = q
                    best_delta = candidate_delta

            if best_q != p:
                q = best_q
                for b in range(m):
                    g = feat_idx[i, b]
                    counts[p, g] -= 1
                    counts[q, g] += 1
                sizes[p] -= 1
                sizes[q] += 1
                y[i] = q
                moved += 1

        if moved == 0:
            break

    return prev


@njit
def tolerance_phi_asym(size, avg_size, delta, eta):
    """Weighted asymmetric tolerance penalty, upper-side weight normalized to 1."""
    lo = (1.0 - delta) * avg_size
    hi = (1.0 + delta) * avg_size
    if size < lo:
        d = lo - size
        return eta * d * d
    if size > hi:
        d = size - hi
        return d * d
    return 0.0

@njit
def tolerance_penalty_asym(sizes, avg_size, delta, eta):
    value = 0.0
    for c in range(sizes.shape[0]):
        value += tolerance_phi_asym(float(sizes[c]), avg_size, delta, eta)
    return value

@njit
def update_y_tolerance_asym(
    y, counts, sizes, feat_idx, inv_sizes, feature_base,
    apow, lam, delta, eta, max_inner=10, tol=1e-5
):
    n, m = feat_idx.shape
    K = sizes.shape[0]
    avg_size = n / float(K)
    prev = 1e300
    for it in range(max_inner):
        sim = consensus_similarity(counts, inv_sizes, feature_base, apow)
        pen = tolerance_penalty_asym(sizes, avg_size, delta, eta)
        obj = lam * pen - sim
        if it > 1 and abs(prev - obj) < tol:
            return obj
        prev = obj
        moved = 0
        for i in range(n):
            p = y[i]
            if sizes[p] <= 1:
                continue
            best_q = p
            best_delta = 0.0
            np0 = sizes[p]
            old_p = tolerance_phi_asym(float(np0), avg_size, delta, eta)
            new_p = tolerance_phi_asym(float(np0 - 1), avg_size, delta, eta)
            for q in range(K):
                if q == p:
                    continue
                dcons = 0.0
                for b in range(m):
                    g = feat_idx[i, b]
                    inv = inv_sizes[g]
                    xp = counts[p, g]
                    xq = counts[q, g]
                    sim_change = apow[b] * ((-2.0*xp + 1.0) + (2.0*xq + 1.0)) * inv
                    dcons -= sim_change
                nq0 = sizes[q]
                dpen = ((new_p - old_p)
                    + tolerance_phi_asym(float(nq0 + 1), avg_size, delta, eta)
                    - tolerance_phi_asym(float(nq0), avg_size, delta, eta))
                candidate_delta = dcons + lam * dpen
                if candidate_delta < best_delta - 1e-15:
                    best_q = q
                    best_delta = candidate_delta
            if best_q != p:
                q = best_q
                for b in range(m):
                    g = feat_idx[i, b]
                    counts[p, g] -= 1
                    counts[q, g] += 1
                sizes[p] -= 1
                sizes[q] += 1
                y[i] = q
                moved += 1
        if moved == 0:
            break
    return prev
