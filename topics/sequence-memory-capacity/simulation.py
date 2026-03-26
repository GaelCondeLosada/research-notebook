"""
Sequence Memory Capacity in Small Recurrent Networks
=====================================================
Compares different architectures and coding schemes for one-shot
sequence memorization in networks with N <= 100 neurons.

Three architectures:
  A) Linear pseudoinverse: x_{k+1} = W x_k, W = X2 @ pinv(X1)
  B) Echo state (random features + pseudoinverse):
     x_{k+1} = W2 @ sigma(W1 @ x_k), W2 = X2 @ pinv(sigma(W1 @ X1))
  C) Modern Hopfield (softmax retrieval):
     x_{k+1} = X2 @ softmax(beta * X1^T @ x_k)

Four coding schemes for generating patterns:
  - Gaussian (iid N(0,1))
  - Laplace (iid Laplace(0, 1/sqrt(2))) [matched variance to Gaussian]
  - Sparse (k-sparse with Gaussian nonzero entries)
  - Orthogonal (columns from random orthogonal matrix, with wrapping)

Metrics:
  - Maximum sequence length L for perfect recall (MSE < epsilon)
  - Normalized capacity L/N
  - Total information stored (L * entropy_per_pattern)
"""

import numpy as np
from scipy.special import softmax as scipy_softmax
from scipy.stats import laplace, norm
import json
import os

np.random.seed(42)

# ============================================================
# Pattern generation
# ============================================================

def generate_patterns(N, L, distribution='gaussian', sparsity=0.1):
    """Generate L patterns of dimension N from the given distribution.
    All distributions are normalized to have unit variance per component."""
    if distribution == 'gaussian':
        return np.random.randn(N, L)
    elif distribution == 'laplace':
        # Laplace with scale b=1/sqrt(2) has variance 1
        return np.random.laplace(loc=0, scale=1/np.sqrt(2), size=(N, L))
    elif distribution == 'sparse':
        # k-sparse patterns: each pattern has k = sparsity*N nonzero entries
        k = max(1, int(sparsity * N))
        X = np.zeros((N, L))
        for j in range(L):
            idx = np.random.choice(N, k, replace=False)
            X[idx, j] = np.random.randn(k) * np.sqrt(N / k)  # normalize energy
        return X
    elif distribution == 'orthogonal':
        # Generate orthonormal columns, wrapping if L > N
        patterns = []
        remaining = L
        while remaining > 0:
            batch = min(remaining, N)
            Q = np.linalg.qr(np.random.randn(N, N))[0][:, :batch]
            patterns.append(Q * np.sqrt(N))  # scale to match energy of gaussian
            remaining -= batch
        return np.hstack(patterns)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ============================================================
# Architecture A: Linear Pseudoinverse
# ============================================================

def linear_pseudoinverse_store(X):
    """One-shot storage: compute W = X2 @ pinv(X1)."""
    X1 = X[:, :-1]  # patterns 1..L-1
    X2 = X[:, 1:]   # patterns 2..L
    W = X2 @ np.linalg.pinv(X1)
    return W

def linear_pseudoinverse_recall(W, x_start, L):
    """Recall sequence of length L starting from x_start."""
    seq = [x_start.copy()]
    x = x_start.copy()
    for _ in range(L - 1):
        x = W @ x
        seq.append(x.copy())
    return np.array(seq).T  # shape (N, L)


# ============================================================
# Architecture B: Echo State (Random Features + Pseudoinverse)
# ============================================================

def echo_state_store(X, M, activation='tanh'):
    """One-shot storage with random feature expansion.
    W1 (fixed random): N -> M, W2 (learned): M -> N."""
    N, L = X.shape
    # Fixed random projection (scaled for stable dynamics)
    W1 = np.random.randn(M, N) / np.sqrt(N)

    X1 = X[:, :-1]
    X2 = X[:, 1:]

    # Feature expansion
    if activation == 'tanh':
        Phi = np.tanh(W1 @ X1)
    elif activation == 'relu':
        Phi = np.maximum(0, W1 @ X1)
    else:
        Phi = W1 @ X1

    W2 = X2 @ np.linalg.pinv(Phi)
    return W1, W2, activation

def echo_state_recall(W1, W2, activation, x_start, L):
    """Recall sequence using echo state network."""
    seq = [x_start.copy()]
    x = x_start.copy()
    for _ in range(L - 1):
        if activation == 'tanh':
            h = np.tanh(W1 @ x)
        elif activation == 'relu':
            h = np.maximum(0, W1 @ x)
        else:
            h = W1 @ x
        x = W2 @ h
        seq.append(x.copy())
    return np.array(seq).T


# ============================================================
# Architecture C: Modern Hopfield / Softmax Retrieval
# ============================================================

def hopfield_store(X, beta=10.0):
    """Store patterns for softmax retrieval. One-shot: just store them."""
    X1 = X[:, :-1]  # keys
    X2 = X[:, 1:]   # values
    return X1, X2, beta

def hopfield_recall(X1, X2, beta, x_start, L):
    """Recall using softmax attention over stored keys."""
    seq = [x_start.copy()]
    x = x_start.copy()
    for _ in range(L - 1):
        # Compute attention scores
        scores = beta * (X1.T @ x)
        # Softmax
        weights = np.exp(scores - np.max(scores))
        weights = weights / np.sum(weights)
        # Retrieve
        x = X2 @ weights
        seq.append(x.copy())
    return np.array(seq).T


# ============================================================
# Evaluation
# ============================================================

def evaluate_recall(X_true, X_recalled, epsilon=1e-6):
    """Check if recall is perfect (MSE < epsilon per pattern)."""
    mse_per_pattern = np.mean((X_true - X_recalled)**2, axis=0)
    # Count how many patterns are correctly recalled
    n_correct = np.sum(mse_per_pattern < epsilon)
    total_mse = np.mean(mse_per_pattern)
    return n_correct, total_mse

def find_max_capacity(N, architecture, distribution,
                      epsilon=1e-6, M_factor=None, beta=None,
                      sparsity=0.1, n_trials=5):
    """Binary search for maximum sequence length L that can be perfectly recalled."""

    # Set search range based on architecture
    if architecture == 'linear':
        L_max_search = N + 5  # theoretical max is N+1
    elif architecture == 'echo_state':
        M = int(M_factor * N)
        L_max_search = M + 5
    elif architecture == 'hopfield':
        L_max_search = 5 * N  # search wider
    else:
        L_max_search = 2 * N

    best_L = 0

    # Binary search
    lo, hi = 2, L_max_search
    while lo <= hi:
        L = (lo + hi) // 2
        successes = 0
        for trial in range(n_trials):
            np.random.seed(42 + trial * 1000 + L)
            X = generate_patterns(N, L, distribution, sparsity)

            try:
                if architecture == 'linear':
                    W = linear_pseudoinverse_store(X)
                    X_recalled = linear_pseudoinverse_recall(W, X[:, 0], L)
                elif architecture == 'echo_state':
                    W1, W2, act = echo_state_store(X, M)
                    X_recalled = echo_state_recall(W1, W2, act, X[:, 0], L)
                elif architecture == 'hopfield':
                    X1, X2, b = hopfield_store(X, beta)
                    X_recalled = hopfield_recall(X1, X2, b, X[:, 0], L)

                n_correct, total_mse = evaluate_recall(X, X_recalled, epsilon)
                if n_correct == L:
                    successes += 1
            except Exception:
                pass

        if successes >= n_trials // 2 + 1:  # majority of trials succeed
            best_L = L
            lo = L + 1
        else:
            hi = L - 1

    return best_L


def compute_entropy(distribution, N, sparsity=0.1):
    """Compute differential entropy per pattern (in nats) for matched-variance distributions."""
    if distribution == 'gaussian':
        # H = N/2 * ln(2*pi*e*sigma^2), sigma^2 = 1
        return N / 2 * np.log(2 * np.pi * np.e)
    elif distribution == 'laplace':
        # H = N * (1 + ln(2b)), b = 1/sqrt(2), variance = 2b^2 = 1
        b = 1 / np.sqrt(2)
        return N * (1 + np.log(2 * b))
    elif distribution == 'sparse':
        # Mixture: (1-s)*delta(0) + s*N(0, N/k)
        # H ≈ s * [N/2 * ln(2*pi*e*N/k)] (continuous part, rough approx)
        k = max(1, int(sparsity * N))
        s = k / N
        return k / 2 * np.log(2 * np.pi * np.e * N / k)
    elif distribution == 'orthogonal':
        # Approximately Gaussian for large N
        return N / 2 * np.log(2 * np.pi * np.e)
    return 0


# ============================================================
# Main experiment
# ============================================================

def run_experiments():
    results = {}

    neuron_counts = [10, 25, 50, 100]
    distributions = ['gaussian', 'laplace', 'sparse', 'orthogonal']
    epsilon = 1e-4  # tolerance for "perfect" recall

    print("=" * 70)
    print("EXPERIMENT 1: Linear Pseudoinverse — Capacity vs N and Distribution")
    print("=" * 70)

    results['linear'] = {}
    for N in neuron_counts:
        results['linear'][N] = {}
        for dist in distributions:
            L = find_max_capacity(N, 'linear', dist, epsilon=epsilon)
            H = compute_entropy(dist, N)
            info = L * H
            results['linear'][N][dist] = {
                'capacity': L,
                'normalized': L / N,
                'entropy_per_pattern': round(H, 2),
                'total_info': round(info, 2)
            }
            print(f"  N={N:3d}, dist={dist:12s}: L={L:4d}, L/N={L/N:.2f}, "
                  f"H={H:.1f} nats, total={info:.1f} nats")

    print()
    print("=" * 70)
    print("EXPERIMENT 2: Echo State Network — Capacity vs Feature Dimension M")
    print("=" * 70)

    results['echo_state'] = {}
    M_factors = [1, 2, 4, 8]  # M = factor * N

    for N in [25, 50, 100]:
        results['echo_state'][N] = {}
        for mf in M_factors:
            M = mf * N
            results['echo_state'][N][f'M={mf}N'] = {}
            for dist in ['gaussian', 'laplace']:
                L = find_max_capacity(N, 'echo_state', dist,
                                      epsilon=epsilon, M_factor=mf)
                results['echo_state'][N][f'M={mf}N'][dist] = {
                    'capacity': L,
                    'M': M,
                    'normalized_by_M': round(L / M, 2) if M > 0 else 0
                }
                print(f"  N={N:3d}, M={M:4d} ({mf}N), dist={dist:10s}: "
                      f"L={L:4d}, L/M={L/M:.2f}")

    print()
    print("=" * 70)
    print("EXPERIMENT 3: Modern Hopfield — Capacity vs Beta and Distribution")
    print("=" * 70)

    results['hopfield'] = {}
    betas = [1.0, 5.0, 10.0, 50.0, 100.0]

    for N in [25, 50, 100]:
        results['hopfield'][N] = {}
        for beta in betas:
            results['hopfield'][N][f'beta={beta}'] = {}
            for dist in ['gaussian', 'laplace', 'orthogonal']:
                L = find_max_capacity(N, 'hopfield', dist,
                                      epsilon=epsilon, beta=beta)
                results['hopfield'][N][f'beta={beta}'][dist] = {
                    'capacity': L
                }
                print(f"  N={N:3d}, beta={beta:6.1f}, dist={dist:12s}: L={L:4d}")

    print()
    print("=" * 70)
    print("EXPERIMENT 4: Error Accumulation Analysis")
    print("=" * 70)

    # For N=50, linear, show how MSE grows with sequence position
    N = 50
    results['error_analysis'] = {}
    for dist in distributions:
        Ls_to_test = [N//2, N, N+1, int(1.5*N), 2*N]
        results['error_analysis'][dist] = {}
        for L in Ls_to_test:
            np.random.seed(42)
            X = generate_patterns(N, L, dist)
            W = linear_pseudoinverse_store(X)
            X_recalled = linear_pseudoinverse_recall(W, X[:, 0], L)
            mse_per_step = np.mean((X - X_recalled)**2, axis=0)
            results['error_analysis'][dist][L] = {
                'mse_per_step': mse_per_step.tolist(),
                'max_mse': float(np.max(mse_per_step)),
                'mean_mse': float(np.mean(mse_per_step))
            }
            print(f"  N={N}, dist={dist:12s}, L={L:3d}: "
                  f"max_MSE={np.max(mse_per_step):.2e}, "
                  f"mean_MSE={np.mean(mse_per_step):.2e}")

    print()
    print("=" * 70)
    print("EXPERIMENT 5: Laplace Scale Parameter Sweep")
    print("=" * 70)

    # Does the scale/sparsity of Laplace affect capacity?
    N = 50
    results['laplace_sweep'] = {}
    scales = [0.1, 0.3, 0.5, 0.707, 1.0, 2.0, 5.0]
    for scale in scales:
        # Generate Laplace with different scales
        best_L = 0
        for L in range(2, N + 10):
            successes = 0
            for trial in range(5):
                np.random.seed(42 + trial * 1000 + L)
                X = np.random.laplace(0, scale, (N, L))
                try:
                    W = linear_pseudoinverse_store(X)
                    X_recalled = linear_pseudoinverse_recall(W, X[:, 0], L)
                    n_correct, _ = evaluate_recall(X, X_recalled, epsilon)
                    if n_correct == L:
                        successes += 1
                except:
                    pass
            if successes >= 3:
                best_L = L
            else:
                break

        var = 2 * scale**2
        H_per_component = 1 + np.log(2 * scale)
        results['laplace_sweep'][f'b={scale}'] = {
            'capacity': best_L,
            'variance': round(var, 3),
            'entropy_per_component': round(H_per_component, 3),
            'total_info': round(best_L * N * H_per_component, 1)
        }
        print(f"  scale={scale:.3f}, var={var:.3f}, H/comp={H_per_component:.3f}: "
              f"L={best_L}, total_info={best_L * N * H_per_component:.1f} nats")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'results.json')

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            return convert(obj)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == '__main__':
    results = run_experiments()
