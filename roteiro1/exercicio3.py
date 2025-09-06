# Exercício 3 — Spaceship Titanic (CSV local) • Pipeline completo p/ NN com tanh
# Requisitos: pandas, numpy, scikit-learn, matplotlib, scipy

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import sparse

# =========================
# 1) Ler CSVs locais
# =========================
TRAIN_CSV = "train.csv"   # ajuste o caminho se necessário
assert os.path.exists(TRAIN_CSV), f"Arquivo não encontrado: {TRAIN_CSV}"
df = pd.read_csv(TRAIN_CSV)
print(f"[OK] Carregado: {TRAIN_CSV} | shape: {df.shape}")

# =========================
# 2) Descrição do dataset
# =========================
print("\nObjetivo: prever 'Transported' (1=transportado; 0=não).")
assert "Transported" in df.columns, "Coluna 'Transported' não encontrada no train.csv"

# Separar alvo e remover IDs/Name
y = df["Transported"].astype(int)
X = df.drop(columns=["Transported", "PassengerI'd", "Name"], errors="ignore")

# Engenharia mínima de 'Cabin' -> Deck / CabNum / Side
if "Cabin" in df.columns:
    cab = df["Cabin"].astype(str).fillna("nan/nan/nan").str.split("/", expand=True)
    cab.columns = ["Deck", "CabNum", "Side"]
    cab["CabNum"] = pd.to_numeric(cab["CabNum"], errors="coerce")
    X = pd.concat([X.drop(columns=["Cabin"], errors="ignore"), cab], axis=1)

# Definir listas de features
num_cols = [c for c in ["Age","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck","CabNum"] if c in X.columns]
cat_cols = [c for c in ["HomePlanet","CryoSleep","Destination","VIP","Deck","Side"] if c in X.columns]

print("\nFeatures numéricas:", num_cols)
print("Features categóricas:", cat_cols)

# Relatório de faltantes (para o relatório)
missing = df.isna().sum().sort_values(ascending=False).to_frame("missing_count")
missing["missing_pct"] = (missing["missing_count"] / len(df)).round(4)
print("\nFaltantes (top 10):")
print(missing.head(10))

# =========================
# 3) Pré-processamento
# =========================
# Justificativa:
# - Numéricas: mediana (robusta a outliers) + MinMax [-1,1] (casa com tanh).
# - Categóricas: moda + OneHot (evita ordem arbitrária).
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler(feature_range=(-1, 1)))
])

# Compatibilidade OneHotEncoder (sklearn novo usa 'sparse_output', antigo usa 'sparse')
ohe_kwargs = {"handle_unknown": "ignore"}
if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
    ohe_kwargs["sparse_output"] = True
else:
    ohe_kwargs["sparse"] = True

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(**ohe_kwargs))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=1.0  # mantemos esparso (muitas dummies)
)

X_prepared = preprocessor.fit_transform(X)
print("\nShape X original:", X.shape, "| Shape X preparado:", X_prepared.shape)

# =========================
# 4) Visualizações: Antes vs Depois (Age, FoodCourt)
# =========================
def _scaled_column_from_pipeline(prepared, colname):
    used_num = [c for c in num_cols if c in X.columns]
    idx = used_num.index(colname)
    col = prepared[:, idx]
    return col.toarray().ravel() if sparse.issparse(col) else np.asarray(col).ravel()

def plot_before_after(col):
    if col not in num_cols:
        print(f"[skip] {col} não está em num_cols.")
        return
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.hist(X[col].dropna(), bins=30, alpha=0.85)
    plt.title(f"{col} — Antes")
    plt.grid(True, ls="--", alpha=0.4)

    plt.subplot(1,2,2)
    scaled = _scaled_column_from_pipeline(X_prepared, col)
    plt.hist(scaled, bins=30, alpha=0.85)
    plt.title(f"{col} — Após MinMax [-1, 1]")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"graficos_ex3/{col}_before_after.png", dpi=150)
    plt.show()

plot_before_after("Age")
plot_before_after("FoodCourt")

# =========================
# 5) Exportar dados prontos
# =========================
os.makedirs("preprocessed_out", exist_ok=True)

X_path = os.path.join("preprocessed_out", "X_minmax_n11.npz")
if sparse.issparse(X_prepared):
    sparse.save_npz(X_path, X_prepared)
else:
    np.savez_compressed(X_path, X_prepared=X_prepared)

y_path = os.path.join("preprocessed_out", "y.csv")
pd.Series(y, name="Transported").to_csv(y_path, index=False)

# Nomes de features (numéricas + dummies)
def get_feature_names(preprocessor, num_cols, cat_cols):
    num_feats = list(num_cols)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    # método disponível em versões recentes
    if hasattr(ohe, "get_feature_names_out"):
        cat_feats = ohe.get_feature_names_out(cat_cols).tolist()
    else:
        # fallback para versões bem antigas
        cat_feats = ohe.get_feature_names(cat_cols).tolist()
    return num_feats + cat_feats

try:
    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
    with open(os.path.join("preprocessed_out", "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))
except Exception as e:
    print("Aviso: não foi possível salvar nomes de features:", e)

print("\nArquivos salvos em ./preprocessed_out:")
print(" •", X_path)
print(" •", y_path)
print(" • feature_names.txt (se disponível)")
