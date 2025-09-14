import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------------------
# 1) Geração dos dados (2D)
# ---------------------------
def generate_data_ex2(n_per_class=1000, seed=123):
    rng = np.random.default_rng(seed)

    # Class 0
    mean0 = np.array([3.0, 3.0])
    cov0 = np.array([[1.5, 0.0], [0.0, 1.5]])  # variância 1.5, sem covariância

    # Class 1
    mean1 = np.array([4.0, 4.0])
    cov1 = np.array([[1.5, 0.0], [0.0, 1.5]])  # variância 1.5, sem covariância

    X0 = rng.multivariate_normal(mean0, cov0, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov1, size=n_per_class)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class, dtype=int),
                   np.ones(n_per_class, dtype=int)])
    return X, y


# -----------------------------------------
# 2) Perceptron simples (treinamento)
# -----------------------------------------
def perceptron_train(X, y, lr=0.01, max_epochs=100, seed=0, random_init=True):
    """
    Implementação do Perceptron para labels {0,1}.
    Internamente mapeamos para {-1, +1}.
    Se random_init=True, começa com pesos aleatórios pequenos.
    """
    rng = np.random.default_rng(seed)
    y_signed = np.where(y == 1, 1, -1)

    n, d = X.shape
    if random_init:
        w = rng.normal(0.0, 1e-3, size=d)
        b = float(rng.normal(0.0, 1e-3))
    else:
        w = np.zeros(d, dtype=float)
        b = 0.0

    acc_history = []

    for epoch in range(max_epochs):
        idx = rng.permutation(n)  # embaralha a ordem
        errors = 0

        for i in idx:
            xi = X[i]
            yi = y_signed[i]

            activation = float(np.dot(w, xi) + b)
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
        acc = float((preds == y_signed).mean())
        acc_history.append(acc)

        # Em dados não separáveis, raramente zera erros; se zerar, para cedo
        if errors == 0:
            break

    return w, b, np.array(acc_history), (epoch + 1)


# -----------------------------------------
# 3) Avaliação
# -----------------------------------------
def evaluate(X, y, w, b):
    y_signed = np.where(y == 1, 1, -1)
    preds = np.where(X @ w + b >= 0, 1, -1)
    acc = float((preds == y_signed).mean())
    mis_idx = np.where(preds != y_signed)[0]
    return acc, mis_idx


# -----------------------------------------
# 4) Visualizações
# -----------------------------------------
def plot_data_and_boundary(X, y, w, b, mis_idx, save_path=None):
    """
    Plota os pontos por classe e a fronteira de decisão w·x + b = 0.
    Amostras mal classificadas são destacadas com 'x'.
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

    plt.title("Exercise 2: Data & Perceptron Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="best")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()


def plot_accuracy(acc_history, title_suffix="(single run)", save_path=None):
    plt.figure()
    plt.plot(np.arange(1, len(acc_history) + 1), acc_history, marker="o")
    plt.title(f"Training Accuracy per Epoch {title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()


def plot_mean_accuracy(histories, save_path=None):
    """
    Recebe uma lista de históricos (arrays 1D de tamanhos possivelmente distintos)
    e plota a média por época com faixa de ±1 desvio-padrão.
    """
    max_len = max(len(h) for h in histories)
    padded = np.full((len(histories), max_len), np.nan, dtype=float)
    for i, h in enumerate(histories):
        padded[i, :len(h)] = h

    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)
    epochs = np.arange(1, max_len + 1)

    plt.figure()
    plt.plot(epochs, mean, marker="o", label="Mean accuracy")
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.2, label="±1 std")
    plt.title("Mean Training Accuracy over Runs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()


# -----------------------------------------
# 5) Execução do experimento
# -----------------------------------------
if __name__ == "__main__":
    # Dados com sobreposição (enunciado)
    X, y = generate_data_ex2(n_per_class=1000, seed=2025)

    # Múltiplas inicializações
    runs = 5
    histories = []
    results = []
    best_run_idx = None
    best_acc = -np.inf

    for r in range(runs):
        w, b, acc_hist, epochs_run = perceptron_train(
            X, y, lr=0.01, max_epochs=100, seed=100 + r, random_init=True
        )
        final_acc, mis_idx = evaluate(X, y, w, b)
        histories.append(acc_hist)
        results.append((w, b, acc_hist, epochs_run, final_acc, mis_idx))
        if final_acc > best_acc:
            best_acc = final_acc
            best_run_idx = r

    # Relatório de runs
    mean_final = float(np.mean([res[4] for res in results]))
    std_final = float(np.std([res[4] for res in results]))

    print(f"Runs: {runs}")
    for i, (w, b, acc_hist, epochs_run, final_acc, _mis) in enumerate(results, 1):
        print(f"Run {i}: epochs={epochs_run}, final_acc={final_acc:.4f}, "
              f"best_epoch_acc={np.max(acc_hist):.4f}")
    print(f"\nBest run: {best_run_idx + 1} with final_acc={best_acc:.4f}")
    print(f"Final accuracy across runs: mean={mean_final:.4f}, std={std_final:.4f}")

    # Plots do melhor run + média entre runs
    # Determinar o diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    graficos_dir = os.path.join(script_dir, "graficos_ex2")
    
    w_best, b_best, acc_hist_best, epochs_best, final_acc_best, mis_best = results[best_run_idx]
    plot_data_and_boundary(X, y, w_best, b_best, mis_best, 
                          save_path=os.path.join(graficos_dir, "data_boundary_best_run.png"))
    plot_accuracy(acc_hist_best, title_suffix="(best run)", 
                  save_path=os.path.join(graficos_dir, "accuracy_best_run.png"))
    plot_mean_accuracy(histories, 
                      save_path=os.path.join(graficos_dir, "mean_accuracy_all_runs.png"))
