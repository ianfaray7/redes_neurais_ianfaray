import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import os

# ---------------------------
# 1) Geração dos dados (2D)
# ---------------------------
def generate_data(n_per_class=1000, seed=42):
    rng = np.random.default_rng(seed)
    mean0 = np.array([1.5, 1.5])
    cov0  = np.array([[0.5, 0.0], [0.0, 0.5]])
    mean1 = np.array([5.0, 5.0])
    cov1  = np.array([[0.5, 0.0], [0.0, 0.5]])
    X0 = rng.multivariate_normal(mean0, cov0, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov1, size=n_per_class)
    X  = np.vstack([X0, X1])
    y  = np.hstack([np.zeros(n_per_class, dtype=int),
                    np.ones(n_per_class,  dtype=int)])
    return X, y

# -----------------------------------------
# 2) Perceptron simples (treinamento)
# -----------------------------------------
def perceptron_train(X, y, lr=0.01, max_epochs=10, seed=42):
    """
    Perceptron para labels {0,1} (mapeados para {-1,+1}).
    Treina até convergir (nenhuma atualização na época) ou até max_epochs.
    """
    rng = np.random.default_rng(seed)
    y_signed = np.where(y == 1, 1, -1)

    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    acc_history = []

    for _ in range(max_epochs):
        idx = rng.permutation(n)
        errors = 0
        for i in idx:
            xi = X[i]; yi = y_signed[i]
            # Atualiza apenas se misclassificado (ou no limiar)
            if (np.dot(w, xi) + b) * yi <= 0.0:
                w += lr * yi * xi
                b += lr * yi
                errors += 1

        # Acurácia ao final da época
        preds = np.where(X @ w + b >= 0, 1, -1)
        acc = (preds == y_signed).mean()
        acc_history.append(acc)

        # Convergência: nenhuma atualização em um passe completo
        if errors == 0:
            break

    return w, b, np.array(acc_history), len(acc_history)

# -----------------------------------------
# 3) Avaliação
# -----------------------------------------
def evaluate(X, y, w, b):
    y_signed = np.where(y == 1, 1, -1)
    preds = np.where(X @ w + b >= 0, 1, -1)
    acc = (preds == y_signed).mean()
    mis_idx = np.where(preds != y_signed)[0]
    return acc, mis_idx

# -----------------------------------------
# 4) Visualizações
# -----------------------------------------
def plot_data_and_boundary(X, y, w, b, mis_idx, save_path=None):
    class0 = X[y == 0]; class1 = X[y == 1]
    plt.figure()
    plt.scatter(class0[:,0], class0[:,1], label="Class 0", alpha=0.6, marker="o")
    plt.scatter(class1[:,0], class1[:,1], label="Class 1", alpha=0.6, marker="^")

    # w1*x + w2*y + b = 0  ->  y = -(w1/w2)*x - b/w2
    if abs(w[1]) > 1e-12:
        xs = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200)
        ys = -(w[0]/w[1]) * xs - b / w[1]
        plt.plot(xs, ys, label="Decision boundary")
    elif abs(w[0]) > 1e-12:
        x_line = -b / w[0]
        plt.plot([x_line, x_line], [X[:,1].min()-0.5, X[:,1].max()+0.5], label="Decision boundary")

    if mis_idx.size > 0:
        plt.scatter(X[mis_idx,0], X[mis_idx,1], label="Misclassified", marker="x", s=60)

    plt.title("Dataset with Perceptron Decision Boundary")
    plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(loc="best")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_accuracy(acc_history, save_path=None, y_floor=0.99, top_headroom=0.002):
    """
    Plota a acurácia com epochs inteiros exatamente 1..N (N = len(acc_history)).
    Remove ticks fracionários no eixo X e deixa folga acima de 1.0.
    (Sem anotações de valores nos pontos.)
    """
    epochs = np.arange(1, len(acc_history) + 1)

    plt.figure()
    plt.plot(epochs, acc_history, marker="o", linewidth=2)
    plt.title("Training Accuracy per Epoch", pad=12)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")

    ax = plt.gca()
    ax.set_xticks(epochs)                  # apenas inteiros (ex.: 1 e 2)
    ax.set_xlim(epochs[0] - 0.2, epochs[-1] + 0.2)
    ax.xaxis.set_minor_locator(NullLocator())  # sem marcas fracionárias
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.grid(False, axis="x")

    # Base do Y perto de 1.0, topo com folga acima de 1.0
    y_min = float(np.min(acc_history))
    lower = min(y_floor, y_min - 0.001) if y_min >= 0.95 else max(0.0, y_min - 0.02)
    upper = 1.0 + top_headroom
    plt.ylim(lower, upper)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

# -----------------------------------------
# 5) Execução
# -----------------------------------------
if __name__ == "__main__":
    X, y = generate_data(n_per_class=1000, seed=42)

    # Treino (para por convergência; no seu caso, 2 épocas)
    w, b, acc_history, epochs_run = perceptron_train(X, y, lr=0.01, max_epochs=10, seed=42)

    # Avaliação
    final_acc, mis_idx = evaluate(X, y, w, b)

    # Logs (sem arredondar artificialmente)
    print("Final weights w:", w)
    print("Final bias b:", b)
    print("Epochs actually run:", epochs_run)
    print("Final training accuracy:", float(final_acc))

    # Salvar gráficos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "graficos_ex1")
    os.makedirs(out_dir, exist_ok=True)

    plot_data_and_boundary(X, y, w, b, mis_idx,
                           save_path=os.path.join(out_dir, "decision_boundary.png"))
    # Gráfico só para as épocas executadas (ex.: 1 e 2), sem decimais entre elas
    plot_accuracy(acc_history,
                  save_path=os.path.join(out_dir, "training_accuracy.png"),
                  y_floor=0.99, top_headroom=0.002)
