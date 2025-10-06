import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# ---------- Funções de ativação / perda (manuais) ----------
def tanh(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            u = float(x[i, j])
            e1 = math.exp(u)
            e2 = math.exp(-u)
            out[i, j] = (e1 - e2) / (e1 + e2)
    return out

def dtanh_from_z(z):
    t = tanh(z)
    return 1.0 - t * t

def sigmoid_scalar(u: float) -> float:
    if u >= 0:
        e = math.exp(-u)
        return 1.0 / (1.0 + e)
    else:
        e = math.exp(u)
        return e / (1.0 + e)

def sigmoid(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = sigmoid_scalar(float(x[i, j]))
    return out

def bce_loss(y_true, y_prob):
    eps = 1e-12
    N = y_true.shape[0]
    s = 0.0
    for i in range(N):
        p = max(min(float(y_prob[i, 0]), 1.0 - eps), eps)
        y = float(y_true[i, 0])
        s += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return s / N

# ---------- Dataset binário com 1 cluster (classe 0) vs 2 clusters (classe 1) ----------
def make_binary_1v2_clusters(n_total=1000, random_state=42, class_sep=1.2):
    n_per_class = n_total // 2
    X0_list, X1_list = [], []
    rs = random_state
    # classe 0: 1 cluster
    while len(X0_list) < n_per_class:
        X, y = make_classification(
            n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
            n_clusters_per_class=1, n_classes=2, class_sep=class_sep,
            random_state=rs
        )
        Xi = X[y == 0]
        take = min(n_per_class - len(X0_list), Xi.shape[0])
        X0_list.extend(list(Xi[:take]))
        rs += 1
    # classe 1: 2 clusters
    while len(X1_list) < n_per_class:
        X, y = make_classification(
            n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
            n_clusters_per_class=2, n_classes=2, class_sep=class_sep,
            random_state=rs
        )
        Xi = X[y == 1]
        take = min(n_per_class - len(X1_list), Xi.shape[0])
        X1_list.extend(list(Xi[:take]))
        rs += 1
    X0 = np.array(X0_list)
    X1 = np.array(X1_list)
    X = np.vstack([X0, X1])
    y = np.vstack([np.zeros((n_per_class, 1)), np.ones((n_per_class, 1))])
    idx = np.random.RandomState(42).permutation(X.shape[0])
    return X[idx], y[idx]

def train_test_split(X, y, test_ratio=0.2, seed=7):
    N = X.shape[0]
    idx = np.random.RandomState(seed).permutation(N)
    ntest = int(N * test_ratio)
    return X[idx[ntest:]], y[idx[ntest:]], X[idx[:ntest]], y[idx[:ntest]]

# ---------- MLP de 1 camada oculta ----------
class MLPBinary:
    def __init__(self, in_dim, hidden_dim, lr=0.05, seed=1):
        rng = np.random.RandomState(seed)
        self.W1 = rng.normal(0, 1.0, (in_dim, hidden_dim)) * 0.5
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, 1.0, (hidden_dim, 1)) * 0.5
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.h1 = tanh(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.p  = sigmoid(self.z2)
        return self.p

    def loss(self, y_true):
        return bce_loss(y_true, self.p)

    def backward(self, X, y_true):
        N = X.shape[0]
        dz2 = (self.p - y_true) / N
        dW2 = self.h1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * dtanh_from_z(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=300, verbose=True):
        losses = []
        for e in range(1, epochs + 1):
            self.forward(X)
            L = self.loss(y); losses.append(L)
            self.backward(X, y)
            if verbose and (e % 50 == 0 or e == 1 or e == epochs):
                print(f"[epoch {e:03d}] loss={L:.4f}")
        return losses

    def predict(self, X):
        p = self.forward(X)
        return (p >= 0.5).astype(np.int64)

# ---------- Execução + gráficos ----------
if __name__ == "__main__":
    X, y = make_binary_1v2_clusters(n_total=1000, class_sep=1.2, random_state=42)
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_ratio=0.2, seed=7)

    model = MLPBinary(in_dim=2, hidden_dim=8, lr=0.05, seed=1)
    losses = model.fit(Xtr, ytr, epochs=300, verbose=True)

    # Gráfico 1: loss vs epochs
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy")
    plt.title("Ex2 - Training Loss")
    plt.tight_layout()
    plt.savefig("docs/roteiro3/graficos_ex2/loss.png", dpi=140)
    plt.close()

    # Gráfico 2: fronteira de decisão (2D)
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = model.predict(grid).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, pred, alpha=0.3)
    plt.scatter(Xte[:,0], Xte[:,1], c=yte.flatten(), s=10)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.title("Ex2 - Decision Boundary (test scatter)")
    plt.tight_layout()
    plt.savefig("docs/roteiro3/graficos_ex2/decision_boundary.png", dpi=140)
    plt.close()

    ypred = model.predict(Xte)
    acc = (ypred.flatten() == yte.flatten()).mean()
    print(f"Test accuracy: {acc:.3f}")
