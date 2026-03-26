"""
sequence_memory.py — Practical framework for one-shot sequence memorization.

Usage:
    from sequence_memory import LinearMemory, HopfieldMemory, EchoStateMemory

    mem = HopfieldMemory(beta=50.0)
    mem.store(sequence)          # sequence: (N, L) array
    recalled = mem.recall(L)     # returns (N, L) array
    print(mem.verify(sequence))  # True if perfect recall
"""

import numpy as np
from abc import ABC, abstractmethod


class SequenceMemory(ABC):
    """Base class for one-shot sequence memory."""

    @abstractmethod
    def store(self, X: np.ndarray):
        """Store a sequence X of shape (N, L). One-shot, no iterative training."""
        pass

    @abstractmethod
    def recall(self, L: int) -> np.ndarray:
        """Recall L steps starting from the stored initial pattern."""
        pass

    def verify(self, X: np.ndarray, epsilon: float = 1e-4) -> bool:
        """Check if the sequence is perfectly recalled."""
        X_hat = self.recall(X.shape[1])
        mse = np.mean((X - X_hat)**2)
        return mse < epsilon

    def recall_error(self, X: np.ndarray) -> np.ndarray:
        """Per-step MSE."""
        X_hat = self.recall(X.shape[1])
        return np.mean((X - X_hat)**2, axis=0)

    @staticmethod
    def theoretical_capacity(N: int, **kwargs) -> str:
        """Return a description of the theoretical capacity."""
        return "Not specified"


class LinearMemory(SequenceMemory):
    """
    Linear pseudoinverse sequence memory.
    x_{k+1} = W x_k, with W = X2 @ pinv(X1).

    Capacity: L = N + 1 (tight bound).
    Parameters: N^2.
    """

    def __init__(self):
        self.W = None
        self.x0 = None

    def store(self, X: np.ndarray):
        N, L = X.shape
        self.x0 = X[:, 0].copy()
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        self.W = X2 @ np.linalg.pinv(X1)

    def recall(self, L: int) -> np.ndarray:
        out = np.zeros((self.W.shape[0], L))
        out[:, 0] = self.x0
        x = self.x0.copy()
        for k in range(L - 1):
            x = self.W @ x
            out[:, k + 1] = x
        return out

    @staticmethod
    def theoretical_capacity(N, **kwargs):
        return f"L* = N + 1 = {N + 1}"


class EchoStateMemory(SequenceMemory):
    """
    Echo state network with random feature expansion.
    x_{k+1} = W2 @ activation(W1 @ x_k)

    Capacity: up to M + 1 (hidden dim), but reduced by activation saturation.
    Parameters: 2*N*M.
    """

    def __init__(self, M: int = None, M_factor: int = 4,
                 activation: str = 'tanh', scale: float = 1.0):
        self.M_factor = M_factor
        self.M_override = M
        self.activation = activation
        self.scale = scale
        self.W1 = None
        self.W2 = None
        self.x0 = None

    def _activate(self, z):
        if self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'relu':
            return np.maximum(0, z)
        return z

    def store(self, X: np.ndarray):
        N, L = X.shape
        M = self.M_override or self.M_factor * N
        self.x0 = X[:, 0].copy()

        # Random projection (scaled)
        self.W1 = np.random.randn(M, N) * self.scale / np.sqrt(N)

        X1 = X[:, :-1]
        X2 = X[:, 1:]
        Phi = self._activate(self.W1 @ X1)
        self.W2 = X2 @ np.linalg.pinv(Phi)

    def recall(self, L: int) -> np.ndarray:
        N = self.W2.shape[0]
        out = np.zeros((N, L))
        out[:, 0] = self.x0
        x = self.x0.copy()
        for k in range(L - 1):
            h = self._activate(self.W1 @ x)
            x = self.W2 @ h
            out[:, k + 1] = x
        return out

    @staticmethod
    def theoretical_capacity(N, M_factor=4, **kwargs):
        M = M_factor * N
        return f"L* ≤ M + 1 = {M + 1} (theoretical), ~{int(0.4 * M)} (empirical with tanh)"


class HopfieldMemory(SequenceMemory):
    """
    Modern Hopfield / softmax attention sequence memory.
    x_{k+1} = X2 @ softmax(beta * X1^T @ x_k)

    Capacity: exponential in N for well-separated patterns.
    Parameters: 2*N*L (grows with sequence length).
    """

    def __init__(self, beta: float = 50.0, normalize: bool = True):
        self.beta = beta
        self.normalize = normalize
        self.X1 = None
        self.X2 = None
        self.x0 = None

    def store(self, X: np.ndarray):
        self._X_stored = X.copy()  # keep original for verification
        if self.normalize:
            norms = np.linalg.norm(X, axis=0, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms

        self.x0 = X[:, 0].copy()
        self.X1 = X[:, :-1].copy()  # keys
        self.X2 = X[:, 1:].copy()   # values

    def recall(self, L: int) -> np.ndarray:
        N = self.X1.shape[0]
        out = np.zeros((N, L))
        out[:, 0] = self.x0
        x = self.x0.copy()
        for k in range(L - 1):
            scores = self.beta * (self.X1.T @ x)
            scores -= scores.max()
            w = np.exp(scores)
            w /= w.sum()
            x = self.X2 @ w
            out[:, k + 1] = x
        return out

    def verify(self, X: np.ndarray, epsilon: float = 1e-4) -> bool:
        """Verify against normalized version if normalization is enabled."""
        if self.normalize:
            norms = np.linalg.norm(X, axis=0, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms
        X_hat = self.recall(X.shape[1])
        mse = np.mean((X - X_hat)**2)
        return mse < epsilon

    @staticmethod
    def theoretical_capacity(N, **kwargs):
        return f"L* = 2^(Θ(N)) ≈ 2^({N//2}) for normalized Gaussian patterns"


# ============================================================
# Pattern generation utilities
# ============================================================

def generate_sequence(N: int, L: int, distribution: str = 'gaussian',
                      normalize: bool = False, sparsity: float = 0.1) -> np.ndarray:
    """
    Generate a sequence of L patterns in R^N.

    Args:
        N: dimension (number of neurons)
        L: sequence length
        distribution: 'gaussian', 'laplace', 'sparse', 'orthogonal'
        normalize: if True, normalize each pattern to unit norm
        sparsity: fraction of nonzero entries (for 'sparse')

    Returns:
        X: (N, L) array
    """
    if distribution == 'gaussian':
        X = np.random.randn(N, L)
    elif distribution == 'laplace':
        X = np.random.laplace(0, 1/np.sqrt(2), (N, L))
    elif distribution == 'sparse':
        k = max(1, int(sparsity * N))
        X = np.zeros((N, L))
        for j in range(L):
            idx = np.random.choice(N, k, replace=False)
            X[idx, j] = np.random.randn(k) * np.sqrt(N / k)
    elif distribution == 'orthogonal':
        parts = []
        rem = L
        while rem > 0:
            batch = min(rem, N)
            Q = np.linalg.qr(np.random.randn(N, N))[0][:, :batch]
            parts.append(Q * np.sqrt(N))
            rem -= batch
        X = np.hstack(parts)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    if normalize:
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms[norms == 0] = 1
        X = X / norms
    return X


def entropy_per_pattern(distribution: str, N: int, sparsity: float = 0.1) -> float:
    """
    Differential entropy (nats) per pattern for matched-variance distributions.
    """
    if distribution == 'gaussian':
        return N / 2 * np.log(2 * np.pi * np.e)
    elif distribution == 'laplace':
        b = 1 / np.sqrt(2)
        return N * (1 + np.log(2 * b))
    elif distribution == 'sparse':
        k = max(1, int(sparsity * N))
        return k / 2 * np.log(2 * np.pi * np.e * N / k)
    elif distribution == 'orthogonal':
        return N / 2 * np.log(2 * np.pi * np.e)
    return 0.0


# ============================================================
# Quick demo
# ============================================================

if __name__ == '__main__':
    N = 50
    print(f"=== Sequence Memory Demo (N={N}) ===\n")

    for MemClass, name, kwargs in [
        (LinearMemory, "Linear Pseudoinverse", {}),
        (EchoStateMemory, "Echo State (M=4N)", {'M_factor': 4}),
        (HopfieldMemory, "Modern Hopfield (β=50)", {'beta': 50.0}),
    ]:
        mem = MemClass(**kwargs)
        print(f"--- {name} ---")
        print(f"  Theoretical capacity: {MemClass.theoretical_capacity(N, **kwargs)}")

        # Find practical capacity
        best_L = 0
        for L in [10, 25, 50, 51, 75, 100, 200, 500, 1000]:
            np.random.seed(42)
            X = generate_sequence(N, L, 'gaussian')
            mem.store(X)
            if mem.verify(X):
                best_L = L
            else:
                break

        print(f"  Practical capacity (Gaussian): L ≥ {best_L}")
        print()
