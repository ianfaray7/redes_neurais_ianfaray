import numpy as np
import matplotlib.pyplot as plt

# ===== 1) Parâmetros e geração dos dados =====
rng = np.random.default_rng(42)

params = {
    0: {"mean": np.array([2.0, 3.0]),  "std": np.array([0.8, 2.5])},
    1: {"mean": np.array([5.0, 6.0]),  "std": np.array([1.2, 1.9])},
    2: {"mean": np.array([8.0, 1.0]),  "std": np.array([0.9, 0.9])},
    3: {"mean": np.array([15.0, 4.0]), "std": np.array([0.5, 2.0])},
}

def gen_class(mean, std, n=100, rng=None):
    """Gera n amostras 2D de uma Gaussiana com covariância diagonal (std^2)."""
    if rng is None:
        rng = np.random.default_rng()
    cov = np.diag(std**2)
    return rng.multivariate_normal(mean, cov, size=n)

# 100 pontos por classe (total = 400)
data = {c: gen_class(p["mean"], p["std"], n=100, rng=rng) for c, p in params.items()}

# ===== 2) Gráfico de dispersão (todas as classes) =====
plt.figure(figsize=(7,6))
for c in sorted(data.keys()):
    X = data[c]
    plt.scatter(X[:,0], X[:,1], s=18, alpha=0.75, label=f"Class {c}")
plt.title("Synthetic 2D Dataset (4 Gaussian Classes)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("graficos_ex1/scatter_ex1.png", dpi=150)
plt.show()

# ===== 3) Segundo gráfico com “fronteiras” esboçadas =====
plt.figure(figsize=(7,6))
for c in sorted(data.keys()):
    X = data[c]
    plt.scatter(X[:,0], X[:,1], s=18, alpha=0.75, label=f"Class {c}")

# (a) Linha vertical para isolar Classe 3 (à direita)
plt.axvline(11, linestyle="--", linewidth=2, label="Boundary: isolate Class 3")

# (b) Linha diagonal para separar Classe 2 das Classes 0/1
xline = np.linspace(3, 11, 200)
yline = 0.5 * xline + 0.25
plt.plot(xline, yline, linestyle="--", linewidth=2, color='green', label="Boundary: split Class 2")

# (c) Linha vertical para separar Classe 1 da Classe 0
xline3 = np.full(200, 3.5)  # x constante = linha vertical
yline3 = np.linspace(-3, 12, 200)  # y varia
plt.plot(xline3, yline3, linestyle="--", linewidth=2, color="purple", label="Boundary: split Class 1/0")


plt.title("Dataset with Sketched Candidate Decision Boundaries")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("graficos_ex1/boundaries_ex1.png", dpi=150)
plt.show()
