import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Definir semente para reprodutibilidade
np.random.seed(42)

print("=" * 60)
print("VARIATIONAL AUTOENCODER (VAE) - Implementação NumPy")
print("Dataset: MNIST")
print("Sem dependências PyTorch/CUDA")
print("=" * 60)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def sigmoid(x):
    """Função sigmoid"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    """Função ReLU"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivada da ReLU"""
    return (x > 0).astype(float)

def binary_cross_entropy(y_true, y_pred, epsilon=1e-8):
    """Binary Cross-Entropy Loss"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ============================================================================
# 1. PREPARAÇÃO DOS DADOS
# ============================================================================

class PreparadorDados:
    """Classe para carregar e preparar o dataset MNIST"""
    
    def __init__(self, validacao_split=0.1):
        self.validacao_split = validacao_split
        
    def carregar_dados(self):
        """Carregar e preparar dataset MNIST"""
        print("\n" + "=" * 60)
        print("1. PREPARAÇÃO DOS DADOS")
        print("=" * 60)
        
        print("Carregando dataset MNIST...")
        # Carregar MNIST usando sklearn
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.to_numpy() / 255.0  # Normalizar para [0, 1]
        y = mnist.target.to_numpy().astype(int)
        
        # Dividir em treino e teste
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=10000, random_state=42, stratify=y
        )
        
        # Dividir treino em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=self.validacao_split, 
            random_state=42, 
            stratify=y_train_full
        )
        
        self.X_train = X_train.astype(np.float32)
        self.X_val = X_val.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"Dataset MNIST carregado com sucesso!")
        print(f"- Conjunto de treinamento: {len(X_train)} imagens")
        print(f"- Conjunto de validação: {len(X_val)} imagens")
        print(f"- Conjunto de teste: {len(X_test)} imagens")
        print(f"- Formato das imagens: 28x28 pixels (784 features)")
        print(f"- Range dos valores: [0, 1] (normalizado)")
        
        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)
    
    def visualizar_amostras(self, num_amostras=10):
        """Visualizar amostras do dataset"""
        print("\nVisualizando amostras do dataset...")
        
        # Selecionar amostras aleatórias
        indices = np.random.choice(len(self.X_train), num_amostras, replace=False)
        
        # Plotar amostras
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            if i < num_amostras:
                img = self.X_train[indices[i]].reshape(28, 28)
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Label: {self.y_train[indices[i]]}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('/home/ian/RedesNeurais/redes_neurais_ianfaray/docs/roteiro4/amostras_dataset.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Amostras salvas em: amostras_dataset.png")

# ============================================================================
# 2. IMPLEMENTAÇÃO DO VAE
# ============================================================================

class VAE:
    """
    Variational Autoencoder (VAE) implementado em NumPy puro
    
    Arquitetura:
    - Encoder: 784 -> 400 -> 20 (μ, σ)
    - Reparametrização: z = μ + σ * ε
    - Decoder: 20 -> 400 -> 784
    """
    
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        
        # Inicializar pesos do Encoder
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_mu = np.zeros(latent_dim)
        
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_logvar = np.zeros(latent_dim)
        
        # Inicializar pesos do Decoder
        self.W2 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.b2 = np.zeros(hidden_dim)
        
        self.W3 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(input_dim)
        
        # Gradientes acumulados (para Adam optimizer)
        self.m = {}  # Primeiro momento
        self.v = {}  # Segundo momento
        self.t = 0   # Timestep
        self._init_adam()
        
    def _init_adam(self):
        """Inicializar momentos do Adam"""
        params = ['W1', 'b1', 'W_mu', 'b_mu', 'W_logvar', 'b_logvar', 
                  'W2', 'b2', 'W3', 'b3']
        for param in params:
            self.m[param] = np.zeros_like(getattr(self, param))
            self.v[param] = np.zeros_like(getattr(self, param))
    
    def encoder(self, x):
        """Encoder: mapeia input para μ e log(σ²)"""
        self.h1 = relu(np.dot(x, self.W1) + self.b1)
        mu = np.dot(self.h1, self.W_mu) + self.b_mu
        logvar = np.dot(self.h1, self.W_logvar) + self.b_logvar
        return mu, logvar
    
    def reparametrizacao(self, mu, logvar):
        """Reparameterization Trick: z = μ + σ * ε"""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps
        return z, eps
    
    def decoder(self, z):
        """Decoder: reconstrói input a partir de z"""
        self.h2 = relu(np.dot(z, self.W2) + self.b2)
        x_recon = sigmoid(np.dot(self.h2, self.W3) + self.b3)
        return x_recon
    
    def forward(self, x):
        """Forward pass completo"""
        self.mu, self.logvar = self.encoder(x)
        self.z, self.eps = self.reparametrizacao(self.mu, self.logvar)
        self.x_recon = self.decoder(self.z)
        return self.x_recon, self.mu, self.logvar
    
    def compute_loss(self, x, x_recon, mu, logvar):
        """Calcular perda do VAE (ELBO)"""
        # Reconstruction loss (BCE)
        bce = binary_cross_entropy(x, x_recon)
        
        # KL divergence
        kld = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
        
        return bce + kld, bce, kld
    
    def backward(self, x):
        """Backward pass e atualização de pesos"""
        batch_size = x.shape[0]
        
        # Gradiente da loss de reconstrução
        dx_recon = (self.x_recon - x) / batch_size
        
        # Gradientes do decoder
        dW3 = np.dot(self.h2.T, dx_recon)
        db3 = np.sum(dx_recon, axis=0)
        
        dh2 = np.dot(dx_recon, self.W3.T) * relu_derivative(self.h2)
        dW2 = np.dot(self.z.T, dh2)
        db2 = np.sum(dh2, axis=0)
        
        # Gradiente do reparameterization trick
        dz = np.dot(dh2, self.W2.T)
        
        # Gradientes da KL divergence
        dmu_kl = self.mu / batch_size
        dlogvar_kl = (np.exp(self.logvar) - 1) / (2 * batch_size)
        
        # Gradientes combinados
        std = np.exp(0.5 * self.logvar)
        dmu = dz + dmu_kl
        dlogvar = dz * self.eps * std * 0.5 + dlogvar_kl
        
        # Gradientes do encoder
        dW_mu = np.dot(self.h1.T, dmu)
        db_mu = np.sum(dmu, axis=0)
        
        dW_logvar = np.dot(self.h1.T, dlogvar)
        db_logvar = np.sum(dlogvar, axis=0)
        
        dh1 = (np.dot(dmu, self.W_mu.T) + np.dot(dlogvar, self.W_logvar.T)) * relu_derivative(self.h1)
        dW1 = np.dot(x.T, dh1)
        db1 = np.sum(dh1, axis=0)
        
        # Atualizar pesos usando Adam
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        grads = {
            'W1': dW1, 'b1': db1,
            'W_mu': dW_mu, 'b_mu': db_mu,
            'W_logvar': dW_logvar, 'b_logvar': db_logvar,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3
        }
        
        for param_name, grad in grads.items():
            # Atualizar momentos
            self.m[param_name] = beta1 * self.m[param_name] + (1 - beta1) * grad
            self.v[param_name] = beta2 * self.v[param_name] + (1 - beta2) * (grad ** 2)
            
            # Correção de bias
            m_hat = self.m[param_name] / (1 - beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - beta2 ** self.t)
            
            # Atualizar pesos
            param = getattr(self, param_name)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, param_name, param)
    
    def gerar_amostras(self, num_amostras):
        """Gerar novas amostras do espaço latente"""
        z = np.random.randn(num_amostras, self.latent_dim)
        amostras = self.decoder(z)
        return amostras

# ============================================================================
# 3. TREINAMENTO DO VAE
# ============================================================================

class TreinadorVAE:
    """Classe para treinar o VAE"""
    
    def __init__(self, modelo, X_train, X_val, batch_size=128):
        self.modelo = modelo
        self.X_train = X_train
        self.X_val = X_val
        self.batch_size = batch_size
        
        # Histórico
        self.historico_treino = {'loss_total': [], 'loss_recon': [], 'loss_kl': []}
        self.historico_val = {'loss_total': [], 'loss_recon': [], 'loss_kl': []}
        
    def treinar_epoca(self):
        """Treinar uma época"""
        indices = np.arange(len(self.X_train))
        np.random.shuffle(indices)
        
        loss_total = 0
        loss_recon = 0
        loss_kl = 0
        num_batches = 0
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            X_batch = self.X_train[batch_indices]
            
            # Forward pass
            x_recon, mu, logvar = self.modelo.forward(X_batch)
            
            # Calcular loss
            total, bce, kld = self.modelo.compute_loss(X_batch, x_recon, mu, logvar)
            
            # Backward pass
            self.modelo.backward(X_batch)
            
            loss_total += total
            loss_recon += bce
            loss_kl += kld
            num_batches += 1
        
        return (loss_total / len(self.X_train), 
                loss_recon / len(self.X_train), 
                loss_kl / len(self.X_train))
    
    def validar(self):
        """Validar modelo"""
        loss_total = 0
        loss_recon = 0
        loss_kl = 0
        
        for i in range(0, len(self.X_val), self.batch_size):
            X_batch = self.X_val[i:i + self.batch_size]
            
            x_recon, mu, logvar = self.modelo.forward(X_batch)
            total, bce, kld = self.modelo.compute_loss(X_batch, x_recon, mu, logvar)
            
            loss_total += total
            loss_recon += bce
            loss_kl += kld
        
        return (loss_total / len(self.X_val), 
                loss_recon / len(self.X_val), 
                loss_kl / len(self.X_val))
    
    def treinar(self, num_epocas=30):
        """Treinar o VAE"""
        print("\n" + "=" * 60)
        print("2. TREINAMENTO DO VAE")
        print("=" * 60)
        
        print(f"Configuração de treinamento:")
        print(f"- Número de épocas: {num_epocas}")
        print(f"- Learning rate: {self.modelo.learning_rate}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Otimizador: Adam")
        print(f"- Arquitetura: {self.modelo.input_dim} → {self.modelo.hidden_dim} → {self.modelo.latent_dim}")
        
        for epoca in range(1, num_epocas + 1):
            # Treinar
            loss_treino, recon_treino, kl_treino = self.treinar_epoca()
            
            # Validar
            loss_val, recon_val, kl_val = self.validar()
            
            # Armazenar histórico
            self.historico_treino['loss_total'].append(loss_treino)
            self.historico_treino['loss_recon'].append(recon_treino)
            self.historico_treino['loss_kl'].append(kl_treino)
            
            self.historico_val['loss_total'].append(loss_val)
            self.historico_val['loss_recon'].append(recon_val)
            self.historico_val['loss_kl'].append(kl_val)
            
            # Imprimir progresso
            if epoca % 5 == 0 or epoca == 1:
                print(f'Época {epoca:3d}/{num_epocas}: '
                      f'Treino Loss={loss_treino:.2f} (Recon={recon_treino:.2f}, KL={kl_treino:.2f}) | '
                      f'Val Loss={loss_val:.2f} (Recon={recon_val:.2f}, KL={kl_val:.2f})')
        
        print("\n✓ Treinamento concluído!")
    
    def plotar_curvas_treinamento(self):
        """Plotar curvas de treinamento"""
        print("\nGerando gráficos de treinamento...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epocas = range(1, len(self.historico_treino['loss_total']) + 1)
        
        # Loss total
        axes[0].plot(epocas, self.historico_treino['loss_total'], 'b-', label='Treino', linewidth=2)
        axes[0].plot(epocas, self.historico_val['loss_total'], 'r-', label='Validação', linewidth=2)
        axes[0].set_title('Perda Total (ELBO)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Perda')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[1].plot(epocas, self.historico_treino['loss_recon'], 'b-', label='Treino', linewidth=2)
        axes[1].plot(epocas, self.historico_val['loss_recon'], 'r-', label='Validação', linewidth=2)
        axes[1].set_title('Perda de Reconstrução (BCE)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Perda')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # KL Divergence
        axes[2].plot(epocas, self.historico_treino['loss_kl'], 'b-', label='Treino', linewidth=2)
        axes[2].plot(epocas, self.historico_val['loss_kl'], 'r-', label='Validação', linewidth=2)
        axes[2].set_title('Divergência KL', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Época')
        axes[2].set_ylabel('Perda')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ian/RedesNeurais/redes_neurais_ianfaray/docs/roteiro4/curvas_treinamento.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Curvas salvas em: curvas_treinamento.png")

# ============================================================================
# 4. AVALIAÇÃO E VISUALIZAÇÃO
# ============================================================================

class AvaliadorVAE:
    """Classe para avaliar e visualizar o VAE"""
    
    def __init__(self, modelo, X_test, y_test):
        self.modelo = modelo
        self.X_test = X_test
        self.y_test = y_test
        
    def avaliar(self):
        """Avaliar performance no conjunto de teste"""
        print("\n" + "=" * 60)
        print("3. AVALIAÇÃO DO VAE")
        print("=" * 60)
        
        loss_total = 0
        loss_recon = 0
        loss_kl = 0
        batch_size = 128
        
        for i in range(0, len(self.X_test), batch_size):
            X_batch = self.X_test[i:i + batch_size]
            x_recon, mu, logvar = self.modelo.forward(X_batch)
            total, bce, kld = self.modelo.compute_loss(X_batch, x_recon, mu, logvar)
            
            loss_total += total
            loss_recon += bce
            loss_kl += kld
        
        loss_total /= len(self.X_test)
        loss_recon /= len(self.X_test)
        loss_kl /= len(self.X_test)
        
        print(f"Performance no conjunto de teste:")
        print(f"- Perda Total (ELBO): {loss_total:.2f}")
        print(f"- Perda de Reconstrução: {loss_recon:.2f}")
        print(f"- Divergência KL: {loss_kl:.2f}")
        
        return loss_total, loss_recon, loss_kl
    
    def visualizar_reconstrucoes(self, num_amostras=10):
        """Visualizar imagens originais vs reconstruídas"""
        print("\n" + "=" * 60)
        print("4. VISUALIZAÇÕES")
        print("=" * 60)
        print("Gerando reconstruções...")
        
        indices = np.random.choice(len(self.X_test), num_amostras, replace=False)
        X_amostras = self.X_test[indices]
        
        x_recon, _, _ = self.modelo.forward(X_amostras)
        
        fig, axes = plt.subplots(2, num_amostras, figsize=(15, 3))
        
        for i in range(num_amostras):
            # Original
            axes[0, i].imshow(X_amostras[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontweight='bold')
            
            # Reconstruída
            axes[1, i].imshow(x_recon[i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstruída', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/ian/RedesNeurais/redes_neurais_ianfaray/docs/roteiro4/reconstrucoes.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Reconstruções salvas em: reconstrucoes.png")
    
    def gerar_novas_amostras(self, num_amostras=20):
        """Gerar novas amostras do espaço latente"""
        print("Gerando novas amostras...")
        
        amostras = self.modelo.gerar_amostras(num_amostras)
        
        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
        
        for i, ax in enumerate(axes.flat):
            if i < num_amostras:
                ax.imshow(amostras[i].reshape(28, 28), cmap='gray')
                ax.axis('off')
        
        plt.suptitle('Amostras Geradas do Espaço Latente', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/home/ian/RedesNeurais/redes_neurais_ianfaray/docs/roteiro4/amostras_geradas.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Amostras geradas salvas em: amostras_geradas.png")
    
    def visualizar_espaco_latente(self, metodo='tsne', max_amostras=5000):
        """Visualizar espaço latente"""
        print(f"Visualizando espaço latente usando {metodo.upper()}...")
        
        # Limitar número de amostras para velocidade
        indices = np.random.choice(len(self.X_test), 
                                   min(max_amostras, len(self.X_test)), 
                                   replace=False)
        X_amostras = self.X_test[indices]
        y_amostras = self.y_test[indices]
        
        # Obter representações latentes
        mu, _ = self.modelo.encoder(X_amostras)
        
        # Reduzir dimensionalidade
        if metodo == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            latentes_2d = reducer.fit_transform(mu)
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
            latentes_2d = reducer.fit_transform(mu)
        
        # Plotar
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latentes_2d[:, 0], latentes_2d[:, 1], 
                            c=y_amostras, cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Dígito')
        plt.title(f'Visualização do Espaço Latente ({metodo.upper()})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Dimensão 1')
        plt.ylabel('Dimensão 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'/home/ian/RedesNeurais/redes_neurais_ianfaray/docs/roteiro4/espaco_latente_{metodo}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Espaço latente ({metodo}) salvo em: espaco_latente_{metodo}.png")
    
    def interpolacao_latente(self, num_steps=10):
        """Interpolar entre dois pontos no espaço latente"""
        print("Gerando interpolação no espaço latente...")
        
        # Selecionar duas amostras
        indices = np.random.choice(len(self.X_test), 2, replace=False)
        X_amostras = self.X_test[indices]
        
        # Codificar
        mu, _ = self.modelo.encoder(X_amostras)
        z1, z2 = mu[0], mu[1]
        
        # Interpolar
        alphas = np.linspace(0, 1, num_steps)
        interpolacoes = []
        
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            x_recon = self.modelo.decoder(z.reshape(1, -1))
            interpolacoes.append(x_recon[0].reshape(28, 28))
        
        # Plotar
        fig, axes = plt.subplots(1, num_steps, figsize=(15, 2))
        for i, ax in enumerate(axes):
            ax.imshow(interpolacoes[i], cmap='gray')
            ax.axis('off')
        
        plt.suptitle('Interpolação Linear no Espaço Latente', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/home/ian/RedesNeurais/redes_neurais_ianfaray/docs/roteiro4/interpolacao_latente.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Interpolação salva em: interpolacao_latente.png")

# ============================================================================
# 5. EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal"""
    print("\n" + "=" * 60)
    print("INICIANDO IMPLEMENTAÇÃO DO VAE")
    print("=" * 60)
    
    # 1. Preparar dados
    preparador = PreparadorDados(validacao_split=0.1)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preparador.carregar_dados()
    preparador.visualizar_amostras()
    
    # 2. Criar modelo
    vae = VAE(input_dim=784, hidden_dim=400, latent_dim=20, learning_rate=0.001)
    
    print(f"\nArquitetura do VAE:")
    print(f"- Dimensão de entrada: 784 (28x28)")
    print(f"- Dimensão oculta: 400")
    print(f"- Dimensão latente: 20")
    print(f"- Implementação: NumPy puro (sem PyTorch/CUDA)")
    
    # 3. Treinar modelo
    treinador = TreinadorVAE(vae, X_train, X_val, batch_size=128)
    treinador.treinar(num_epocas=30)
    treinador.plotar_curvas_treinamento()
    
    # 4. Avaliar modelo
    avaliador = AvaliadorVAE(vae, X_test, y_test)
    avaliador.avaliar()
    
    # 5. Visualizações
    avaliador.visualizar_reconstrucoes(num_amostras=10)
    avaliador.gerar_novas_amostras(num_amostras=20)
    avaliador.visualizar_espaco_latente(metodo='pca')
    avaliador.visualizar_espaco_latente(metodo='tsne')
    avaliador.interpolacao_latente(num_steps=10)
    
    print("\n" + "=" * 60)
    print("EXERCÍCIO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print("\nArquivos gerados:")
    print("- amostras_dataset.png")
    print("- curvas_treinamento.png")
    print("- reconstrucoes.png")
    print("- amostras_geradas.png")
    print("- espaco_latente_pca.png")
    print("- espaco_latente_tsne.png")
    print("- interpolacao_latente.png")

if __name__ == "__main__":
    main()
