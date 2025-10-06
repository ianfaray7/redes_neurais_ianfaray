import math
import numpy as np

# ---------- Funções auxiliares (implementadas manualmente) ----------
def tanh_scalar(u: float) -> float:
    """tanh implementado manualmente (sem np.tanh)."""
    e_pos = math.exp(u)
    e_neg = math.exp(-u)
    return (e_pos - e_neg) / (e_pos + e_neg)

def dtanh_scalar(u: float) -> float:
    """Derivada da tanh: 1 - tanh(u)^2, usando a tanh acima."""
    t = tanh_scalar(u)
    return 1.0 - t * t

# ---------- Dados do exercício ----------
x  = np.array([0.5, -0.2])              # input (2,)
y  = 1.0                                 # target escalar

W1 = np.array([[0.3, -0.1],             # pesos hidden (2x2)  (linhas = neurônios)
               [0.2,  0.4]])
b1 = np.array([0.1, -0.2])              # bias hidden (2,)

W2 = np.array([0.5, -0.3])              # pesos saída (1x2) como vetor
b2 = 0.2                                 # bias saída (escalar)

# ---------- Forward pass ----------
# z1 = W1 @ x + b1  (uso do @ permitido: operação matricial)
z1 = W1 @ x + b1                         # (2,)
# h1 = tanh(z1)  elemento a elemento usando nossa tanh
h1 = np.array([tanh_scalar(float(z1[0])), tanh_scalar(float(z1[1]))])  # (2,)
# u2 = W2 @ h1 + b2
u2 = float(W2 @ h1 + b2)                 # escalar
# y_hat = tanh(u2)
y_hat = tanh_scalar(u2)                  # escalar

# Loss MSE (N=1)
L = (y - y_hat) ** 2

# ---------- Backprop ----------
# dL/dy_hat = 2*(y_hat - y)
dL_dyhat = 2.0 * (y_hat - y)
# dL/du2 = dL/dy_hat * dtanh(u2)
dL_du2   = dL_dyhat * dtanh_scalar(u2)

# Gradientes da camada de saída
# dL/dW2 = dL/du2 * h1
dL_dW2 = dL_du2 * h1                     # (2,)
# dL/db2 = dL/du2
dL_db2 = dL_du2                          # escalar

# Propagação p/ a hidden
# dL/dh1 = dL/du2 * W2  (elemento a elemento no vetor W2)
dL_dh1 = dL_du2 * W2                     # (2,)
# dL/dz1_i = dL/dh1_i * dtanh(z1_i)
dL_dz1 = np.array([
    dL_dh1[0] * dtanh_scalar(float(z1[0])),
    dL_dh1[1] * dtanh_scalar(float(z1[1]))
])                                       # (2,)

# Gradientes da camada oculta
# dL/dW1 = outer(dL/dz1, x)  (uso de outer permitido: operação matricial)
dL_dW1 = np.outer(dL_dz1, x)             # (2x2)
# dL/db1 = dL/dz1
dL_db1 = dL_dz1                          # (2,)

# ---------- Atualização de parâmetros ----------
def apply_updates(eta: float):
    W2_new = W2 - eta * dL_dW2
    b2_new = b2 - eta * dL_db2
    W1_new = W1 - eta * dL_dW1
    b1_new = b1 - eta * dL_db1
    return W1_new, b1_new, W2_new, b2_new

# Exemplos: η = 0.3 (enunciado) e η = 0.1 (passo 4)
W1_03, b1_03, W2_03, b2_03 = apply_updates(0.3)
W1_01, b1_01, W2_01, b2_01 = apply_updates(0.1)

# ---------- Impressão dos resultados ----------
np.set_printoptions(precision=6, floatmode="maxprec", suppress=False)
print("=== Forward ===")
print("x       =", x)
print("W1      =\n", W1)
print("b1      =", b1)
print("z1      =", z1)
print("h1=tanh =", h1)
print("W2      =", W2)
print("b2      =", b2)
print(f"u2      = {u2:.10f}")
print(f"y_hat   = {y_hat:.10f}")
print(f"L (MSE) = {L:.10f}")

print("\n=== Gradientes ===")
print(f"dL/dy_hat = {dL_dyhat:.10f}")
print(f"dL/du2    = {dL_du2:.10f}")
print("dL/dW2    =", dL_dW2)
print(f"dL/db2    = {dL_db2:.10f}")
print("dL/dW1    =\n", dL_dW1)
print("dL/db1    =", dL_db1)

print("\n=== Atualizações ===")
print("[eta = 0.3]")
print("W1_new =\n", W1_03)
print("b1_new =", b1_03)
print("W2_new =", W2_03)
print(f"b2_new = {b2_03:.10f}")

print("\n[eta = 0.1]")
print("W1_new =\n", W1_01)
print("b1_new =", b1_01)
print("W2_new =", W2_01)
print(f"b2_new = {b2_01:.10f}")
