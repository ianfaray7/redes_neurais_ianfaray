import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------------------
# 1) Geração dos dados (2D)
# ---------------------------
def generate_data(n_per_class=1000, seed=42):
    rng = np.random.default_rng(seed)

    # Class 0
    mean0 = np.array([1.5, 1.5])
    cov0 = np.array([[0.5, 0.0], [0.0, 0.5]])  # variância 0.5, sem covariância

    # Class 1
    mean1 = np.array([5.0, 5.0])
    cov1 = np.array([[0.5, 0.0], [0.0, 0.5]])  # variância 0.5, sem covariância

    X0 = rng.multivariate_normal(mean0, cov0, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov1, size=n_per_class)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class, dtype=int),
                   np.ones(n_per_class, dtype=int)])
    return X, y


# -----------------------------------------
# 2) Perceptron simples (treinamento)
# -----------------------------------------
def perceptron_train(X, y, lr=0.01, max_epochs=100, seed=42):
    """
    Implementação do Perceptron para labels {0,1}.
    Internamente mapeamos para {-1, +1}.
    """
    rng = np.random.default_rng(seed)
    y_signed = np.where(y == 1, 1, -1)

    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    acc_history = []

    for epoch in range(max_epochs):
        idx = rng.permutation(n)
        errors = 0

        for i in idx:
            xi = X[i]
            yi = y_signed[i]

            activation = np.dot(w, xi) + b
            pred = 1 if activation >= 0 else -1

            if pred != yi:
                # Regra do perceptron:
                # w <- w + lr * yi * xi
                # b <- b + lr * yi
                w = w + lr * yi * xi
                b = b + lr * yi
                errors += 1

        # Acurácia após a época
        preds = np.where(X @ w + b >= 0, 1, -1)
        acc = (preds == y_signed).mean()
        acc_history.append(acc)

        # Critério de convergência: nenhuma atualização na época
        if errors == 0:
            break

    return w, b, np.array(acc_history), (epoch + 1)


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
    """
    Plota os pontos por classe e a fronteira de decisão w·x + b = 0.
    Amostras mal classificadas são destacadas com marcador 'x'.
    """
    class0 = X[y == 0]
    class1 = X[y == 1]

    plt.figure()
    plt.scatter(class0[:, 0], class0[:, 1], label="Class 0", alpha=0.6, marker="o")
    plt.scatter(class1[:, 0], class1[:, 1], label="Class 1", alpha=0.6, marker="^")

    # Fronteira de decisão:
    # w1*x + w2*y + b = 0  ->  y = -(w1/w2) * x - b/w2  (se w2 != 0)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    if abs(w[1]) > 1e-12:
        xs = np.linspace(x_min, x_max, 200)
        ys = -(w[0] / w[1]) * xs - b / w[1]
        plt.plot(xs, ys, label="Decision boundary")
    elif abs(w[0]) > 1e-12:
        # Fronteira vertical: x = -b / w0
        x_line = -b / w[0]
        plt.plot([x_line, x_line], [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5], label="Decision boundary")

    # Destacar mal classificadas
    if mis_idx.size > 0:
        plt.scatter(X[mis_idx, 0], X[mis_idx, 1], label="Misclassified", marker="x", s=60)

    plt.title("Dataset with Perceptron Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="best")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()


def plot_accuracy(acc_history, save_path=None):
    plt.figure()
    plt.plot(np.arange(1, len(acc_history) + 1), acc_history, marker="o")
    plt.title("Training Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()


# -----------------------------------------
# 5) Execução
# -----------------------------------------
if __name__ == "__main__":
    # Dados
    X, y = generate_data(n_per_class=1000, seed=42)

    # Treino
    w, b, acc_history, epochs_run = perceptron_train(
        X, y, lr=0.01, max_epochs=100, seed=42
    )

    # Avaliação
    final_acc, mis_idx = evaluate(X, y, w, b)

    # Relatório
    print("Final weights w:", w)
    print("Final bias b:", b)
    print("Epochs run:", epochs_run)
    print("Final training accuracy:", round(float(final_acc), 4))

    # Plots
    # Determinar o diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    graficos_dir = os.path.join(script_dir, "graficos_ex1")
    
    plot_data_and_boundary(X, y, w, b, mis_idx, 
                          save_path=os.path.join(graficos_dir, "decision_boundary.png"))
    plot_accuracy(acc_history, 
                  save_path=os.path.join(graficos_dir, "training_accuracy.png"))
