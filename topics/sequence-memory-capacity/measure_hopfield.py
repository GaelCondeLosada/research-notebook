"""Quick measurement of Hopfield capacity for key configurations."""
import numpy as np
import time

def gen_norm(N, L, dist):
    if dist == 'gaussian':
        X = np.random.randn(N, L)
    elif dist == 'laplace':
        X = np.random.laplace(0, 1/np.sqrt(2), (N, L))
    elif dist == 'orthogonal':
        parts = []
        rem = L
        while rem > 0:
            b = min(rem, N)
            Q = np.linalg.qr(np.random.randn(N, N))[0][:, :b]
            parts.append(Q)
            rem -= b
        return np.hstack(parts)
    else:
        raise ValueError(dist)
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms[norms == 0] = 1
    return X / norms

def check(X, beta, eps=1e-4):
    N, L = X.shape
    X1, X2 = X[:, :-1], X[:, 1:]
    x = X[:, 0].copy()
    for k in range(L - 1):
        s = beta * (X1.T @ x)
        s -= s.max()
        w = np.exp(s); w /= w.sum()
        x = X2 @ w
        if np.sum((x - X[:, k+1])**2) / N > eps:
            return False
    return True

def search(N, dist, beta, maxL, trials=3):
    lo, hi, best = 2, maxL, 0
    while lo <= hi:
        L = (lo + hi) // 2
        ok = sum(1 for t in range(trials)
                 if (np.random.seed(42 + t*9973 + L*37) or True)
                 and check(gen_norm(N, L, dist), beta))
        if ok > trials // 2:
            best = L; lo = L + 1
        else:
            hi = L - 1
    return best

print("Hopfield capacity (normalized, beta=50)")
print("-" * 50)
for N in [10, 25, 50]:
    for dist in ['gaussian', 'laplace', 'orthogonal']:
        maxL = {10: 200, 25: 2000, 50: 3000}[N]
        t0 = time.time()
        L = search(N, dist, 50.0, maxL)
        dt = time.time() - t0
        print(f"N={N:3d} {dist:12s}: L={L:5d} (L/N={L/N:6.1f}) [{dt:.1f}s]")

print("\nCoherence analysis (N=50, L=200)")
print("-" * 50)
for dist in ['gaussian', 'laplace', 'orthogonal']:
    coh_max, coh_avg = [], []
    for t in range(10):
        np.random.seed(42 + t)
        X = gen_norm(50, 200, dist)
        G = X.T @ X
        np.fill_diagonal(G, 0)
        coh_max.append(np.max(np.abs(G)))
        coh_avg.append(np.mean(np.abs(G[np.triu_indices(200, 1)])))
    print(f"{dist:12s}: max_coh={np.mean(coh_max):.4f}, avg_coh={np.mean(coh_avg):.4f}")

print("\nNormalized vs raw patterns (N=50, beta=50)")
print("-" * 50)
for label, raw in [("Gaussian norm", False), ("Gaussian raw", True),
                    ("Laplace norm", False), ("Laplace raw", True)]:
    dist = 'gaussian' if 'Gaussian' in label else 'laplace'
    lo, hi, best = 2, 3000, 0
    while lo <= hi:
        L = (lo + hi) // 2
        ok = 0
        for t in range(3):
            np.random.seed(42 + t*9973 + L*37)
            if raw:
                if dist == 'gaussian':
                    X = np.random.randn(50, L)
                else:
                    X = np.random.laplace(0, 1/np.sqrt(2), (50, L))
            else:
                X = gen_norm(50, L, dist)
            if check(X, 50.0):
                ok += 1
        if ok > 1:
            best = L; lo = L + 1
        else:
            hi = L - 1
    print(f"{label:16s}: L={best:5d} (L/N={best/50:.1f})")
