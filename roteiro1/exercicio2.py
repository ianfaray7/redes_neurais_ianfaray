# Exercise 2 — Non-Linearity in Higher Dimensions
# Requisitos básicos: numpy, matplotlib
# (Opcional) classificar: scikit-learn

import numpy as np
import matplotlib.pyplot as plt

# ---------- 1) Parâmetros e geração dos dados (5D) ----------
rng = np.random.default_rng(7)

mu_A = np.zeros(5)
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])

mu_B = np.full(5, 1.5)
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

n_per_class = 500
A = rng.multivariate_normal(mu_A, Sigma_A, size=n_per_class)
B = rng.multivariate_normal(mu_B, Sigma_B, size=n_per_class)

X = np.vstack([A, B])                 # (1000, 5)
y = np.array([0]*n_per_class + [1]*n_per_class)  # 0=A, 1=B

# ---------- 2) PCA (NumPy): 5D -> 2D ----------
Xc = X - X.mean(axis=0, keepdims=True)
Cov = np.cov(Xc, rowvar=False)        # (5x5)
eigvals, eigvecs = np.linalg.eigh(Cov)
order = np.argsort(eigvals)[::-1]     # maiores primeiro
W = eigvecs[:, order[:2]]             # autovetores dos 2 maiores autovalores
X2 = Xc @ W                           # projeção 2D

explained_ratio = eigvals[order[:2]] / eigvals.sum()
print(f"Explained variance ratio PC1, PC2: {explained_ratio[0]:.4f}, {explained_ratio[1]:.4f}")
print(f"Explained variance (total 2D): {explained_ratio.sum():.4f}")

# ---------- 3) Visualização ----------
plt.figure(figsize=(7,6))
plt.scatter(X2[y==0,0], X2[y==0,1], s=12, alpha=0.7, label="Class A")
plt.scatter(X2[y==1,0], X2[y==1,1], s=12, alpha=0.7, label="Class B")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Exercise 2: PCA (5D → 2D) of Classes A and B")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("graficos_ex2/pca_ex2.png", dpi=150)
plt.show()
