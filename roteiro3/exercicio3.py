import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# ---------- Ativações e perdas (manuais) ----------
def relu(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = float(x[i, j]) if x[i, j] > 0 else 0.0
    return out

def drelu_from_z(z):
    out = np.empty_like(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            out[i, j] = 1.0 if z[i, j] > 0 else 0.0
    return out

def softmax_row(v):
    m = float(np.max(v))
    exps = [math.exp(float(v[k] - m)) for k in range(v.shape[0])]
    s = sum(exps)
    return np.array([e / s for e in exps], dtype=v.dtype)

def softmax(X):
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = softmax_row(X[i])
    return out

def cce_loss(y_true_onehot, y_prob):
    eps = 1e-12
    N = y_true_onehot.shape[0]
    total = 0.0
    for i in range(N):
        for c in range(y_true_onehot.shape[1]):
            y = float(y_true_onehot[i, c])
            if y > 0.0:
                p = max(min(float(y_prob[i, c]), 1.0 - eps), eps)
                total += -math.log(p)
    return total / N

def one_hot(y, n_classes):
    N = y.shape[0]
    Y = np.zeros((N, n_classes))
    for i in range(N):
        Y[i, int(y[i])] = 1.0
    return Y

# ---------- Dataset: 3 classes, 4 features; clusters assimétricos 2/3/4 ----------
def make_multiclass_asym_clusters(n_total=1500, random_state=42, class_sep=1.6):
    n_classes = 3
    per_class = n_total // n_classes
    want = {0: (2, per_class), 1: (3, per_class), 2: (4, per_class)}
    collected = {0: [], 1: [], 2: []}
    seed = random_state
    while any(len(collected[c]) < want[c][1] for c in range(n_classes)):
        for cls in range(n_classes):
            clusters, need = want[cls]
            if len(collected[cls]) >= need:
                continue
            X, y = make_classification(
                n_samples=2500, n_features=4, n_informative=4, n_redundant=0,
                n_classes=3, n_clusters_per_class=clusters, class_sep=class_sep,
                random_state=seed
            )
            Xi = X[y == cls]
            take = min(need - len(collected[cls]), Xi.shape[0])
            collected[cls].extend(list(Xi[:take]))
            seed += 1
    X = np.vstack([np.array(collected[0]),
                   np.array(collected[1]),
                   np.array(collected[2])])
    y = np.concatenate([
        np.zeros((per_class,), dtype=int),
        np.ones((per_class,), dtype=int),
        np.full((per_class,), 2, dtype=int)
    ])
    idx = np.random.RandomState(123).permutation(X.shape[0])
    return X[idx], y[idx]

def train_test_split(X, y, test_ratio=0.2, seed=77):
    N = X.shape[0]
    idx = np.random.RandomState(seed).permutation(N)
    ntest = int(N * test_ratio)
    return X[idx[ntest:]], y[idx[ntest:]], X[idx[:ntest]], y[idx[:ntest]]

# ---------- MLP (1 hidden, Softmax) ----------
class MLP3:
    def __init__(self, in_dim, hidden_dim, out_dim, lr=0.05, seed=1):
        rng = np.random.RandomState(seed)
        self.W1 = rng.normal(0, 1.0, (in_dim, hidden_dim)) * 0.5
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, 1.0, (hidden_dim, out_dim)) * 0.5
        self.b2 = np.zeros((1, out_dim))
        self.lr = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.P  = softmax(self.z2)
        return self.P

    def loss(self, Y):
        return cce_loss(Y, self.P)

    def backward(self, X, Y):
        N = X.shape[0]
        dz2 = (self.P - Y) / N
        dW2 = self.h1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * drelu_from_z(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=300, verbose=True):
        Y = one_hot(y, 3)
        losses = []
        for e in range(1, epochs + 1):
            self.forward(X)
            L = self.loss(Y); losses.append(L)
            self.backward(X, Y)
            if verbose and (e % 50 == 0 or e == 1 or e == epochs):
                print(f"[epoch {e:03d}] loss={L:.4f}")
        return losses

    def predict(self, X):
        P = self.forward(X)
        return np.argmax(P, axis=1)

# ---------- Execução + gráficos ----------
if __name__ == "__main__":
    X, y = make_multiclass_asym_clusters(n_total=1500, class_sep=1.6, random_state=42)
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_ratio=0.2, seed=77)

    model = MLP3(in_dim=4, hidden_dim=16, out_dim=3, lr=0.05, seed=1)
    losses = model.fit(Xtr, ytr, epochs=300, verbose=True)

    # Gráfico 1: loss vs epochs
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Categorical Cross-Entropy")
    plt.title("Ex3 - Training Loss")
    plt.tight_layout()
    plt.savefig("docs/roteiro3/graficos_ex3/loss.png", dpi=140)
    plt.close()

    # Gráfico 2: scatter das 2 primeiras features com predições
    ypred = model.predict(Xte)
    plt.figure()
    plt.scatter(Xte[:,0], Xte[:,1], c=ypred, s=10)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.title("Ex3 - Test scatter by predicted class (features 1 & 2)")
    plt.tight_layout()
    plt.savefig("docs/roteiro3/graficos_ex3/scatter_test.png", dpi=140)
    plt.close()

    acc = (ypred == yte).mean()
    print(f"Test accuracy: {acc:.3f}")
