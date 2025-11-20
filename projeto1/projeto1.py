import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Definir semente aleatória para reprodutibilidade
np.random.seed(42)

# ============================================================================
# 1. CARREGAMENTO E EXPLORAÇÃO INICIAL DO DATASET
# ============================================================================

class CarregadorDataset:
    """Classe para carregamento e exploração inicial do dataset EA FC 26"""
    
    def __init__(self, caminho_arquivo):
        self.caminho_arquivo = caminho_arquivo
        self.dados = None
        
    def carregar_dados(self):
        """Carregar o dataset e realizar exploração inicial"""
        print("=" * 60)
        print("1. SELEÇÃO E CARREGAMENTO DO DATASET")
        print("=" * 60)
        
        # Carregar dataset
        self.dados = pd.read_csv(self.caminho_arquivo)
        
        print(f"Fonte do Dataset: EA FC 26 Players Dataset")
        print(f"Tamanho do Dataset: {self.dados.shape[0]} jogadores, {self.dados.shape[1]} características")
        print(f"Uso de Memória: {self.dados.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Informações básicas
        print(f"\nVisão Geral do Dataset:")
        print(f"- Total de jogadores: {len(self.dados):,}")
        print(f"- Total de características: {len(self.dados.columns)}")
        print(f"- Tipos de dados: {self.dados.dtypes.value_counts().to_dict()}")
        
        return self.dados
    
    def explorar_posicoes(self):
        """Explorar posições dos jogadores para tarefa de classificação"""
        print("\n" + "=" * 60)
        print("2. EXPLICAÇÃO DO DATASET - ANÁLISE DE POSIÇÕES")
        print("=" * 60)
        
        # Analisar posições dos jogadores
        posicoes = self.dados['player_positions'].value_counts()
        print(f"Combinações únicas de posições: {len(posicoes)}")
        print(f"Top 10 combinações de posições:")
        print(posicoes.head(10))
        
        # Extrair posições primárias (primeira posição na string)
        self.dados['primary_position'] = self.dados['player_positions'].str.split(',').str[0].str.strip()
        posicoes_primarias = self.dados['primary_position'].value_counts()
        
        print(f"\nDistribuição das posições primárias:")
        print(posicoes_primarias)
        
        # Filtrar para manter apenas jogadores com posições comuns (>= 500 jogadores)
        min_jogadores = 500
        posicoes_comuns = posicoes_primarias[posicoes_primarias >= min_jogadores].index
        dados_filtrados = self.dados[self.dados['primary_position'].isin(posicoes_comuns)].copy()
        
        print(f"\nApós filtragem (posições com >= {min_jogadores} jogadores):")
        print(f"Posições mantidas: {list(posicoes_comuns)}")
        print(f"Jogadores restantes: {len(dados_filtrados):,}")
        
        return dados_filtrados

# ============================================================================
# 2. PRÉ-PROCESSAMENTO DE DADOS E ENGENHARIA DE CARACTERÍSTICAS
# ============================================================================

class PreprocessadorDados:
    """Classe para limpeza de dados, pré-processamento e engenharia de características"""
    
    def __init__(self, dados):
        self.dados = dados
        self.colunas_caracteristicas = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def limpar_dados(self):
        """Limpar o dataset tratando valores ausentes e outliers"""
        print("\n" + "=" * 60)
        print("3. LIMPEZA E PRÉ-PROCESSAMENTO DE DADOS")
        print("=" * 60)
        
        # Verificar valores ausentes
        valores_ausentes = self.dados.isnull().sum()
        porcentagem_ausentes = (valores_ausentes / len(self.dados)) * 100
        
        print("Análise de valores ausentes:")
        df_ausentes = pd.DataFrame({
            'Contagem Ausentes': valores_ausentes,
            'Porcentagem': porcentagem_ausentes
        }).sort_values('Porcentagem', ascending=False)
        
        print(df_ausentes[df_ausentes['Contagem Ausentes'] > 0].head(10))
        
        # Selecionar características numéricas para o modelo
        caracteristicas_numericas = [
            'overall', 'potential', 'age', 'height_cm', 'weight_kg',
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
            'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
            'attacking_short_passing', 'attacking_volleys', 'skill_dribbling',
            'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
            'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
            'movement_reactions', 'movement_balance', 'power_shot_power',
            'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
            'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
            'mentality_vision', 'mentality_penalties', 'mentality_composure',
            'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle'
        ]
        
        # Manter apenas linhas com dados completos para características selecionadas
        self.colunas_caracteristicas = caracteristicas_numericas
        dados_completos = self.dados[self.colunas_caracteristicas + ['primary_position']].dropna()
        
        print(f"\nCaracterísticas selecionadas: {len(self.colunas_caracteristicas)}")
        print(f"Amostras após remover valores ausentes: {len(dados_completos):,}")
        
        # Remover outliers usando método IQR
        Q1 = dados_completos[self.colunas_caracteristicas].quantile(0.25)
        Q3 = dados_completos[self.colunas_caracteristicas].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir limites de outliers
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Remover outliers
        mascara_outliers = ((dados_completos[self.colunas_caracteristicas] < limite_inferior) | 
                           (dados_completos[self.colunas_caracteristicas] > limite_superior)).any(axis=1)
        
        dados_limpos = dados_completos[~mascara_outliers]
        
        print(f"Outliers removidos: {sum(mascara_outliers):,}")
        print(f"Dataset final limpo: {len(dados_limpos):,} jogadores")
        
        self.dados = dados_limpos
        return dados_limpos
    
    def preparar_caracteristicas(self):
        """Preparar características e variáveis alvo"""
        # Características (X) e alvo (y)
        X = self.dados[self.colunas_caracteristicas].values
        y = self.dados['primary_position'].values
        
        # Codificar labels
        y_codificado = self.label_encoder.fit_transform(y)
        
        # Escalar características
        X_escalado = self.scaler.fit_transform(X)
        
        print(f"\nEscalonamento de características concluído")
        print(f"Forma da matriz de características: {X_escalado.shape}")
        print(f"Classes alvo: {len(self.label_encoder.classes_)}")
        print(f"Rótulos das classes: {list(self.label_encoder.classes_)}")
        
        return X_escalado, y_codificado
    
    def visualizar_dados(self):
        """Criar visualizações para exploração de dados"""
        print("\nCriando visualizações de dados...")
        
        # 1. Distribuição de posições
        plt.figure(figsize=(12, 6))
        contagem_posicoes = self.dados['primary_position'].value_counts()
        plt.subplot(1, 2, 1)
        contagem_posicoes.plot(kind='bar')
        plt.title('Distribuição das Posições dos Jogadores')
        plt.xlabel('Posição')
        plt.ylabel('Número de Jogadores')
        plt.xticks(rotation=45)
        
        # 2. Mapa de calor de correlação das características (amostra das principais características)
        caracteristicas_chave = ['overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
        plt.subplot(1, 2, 2)
        matriz_correlacao = self.dados[caracteristicas_chave].corr()
        sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação das Características')
        
        plt.tight_layout()
        plt.savefig('/home/ian/RedesNeurais/redes_neurais_ianfaray/docs/projeto/data_exploration.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Distribuições das características por posição
        plt.figure(figsize=(15, 10))
        caracteristicas_chave = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
        for i, caracteristica in enumerate(caracteristicas_chave, 1):
            plt.subplot(2, 3, i)
            for posicao in self.dados['primary_position'].unique():
                dados_posicao = self.dados[self.dados['primary_position'] == posicao][caracteristica]
                plt.hist(dados_posicao, alpha=0.6, label=posicao, bins=20)
            plt.title(f'Distribuição de {caracteristica} por Posição')
            plt.xlabel(caracteristica)
            plt.ylabel('Frequência')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('/home/ian/RedesNeurais/redes_neurais_ianfaray/docs/projeto/feature_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# 3. IMPLEMENTAÇÃO DO PERCEPTRON MULTI-CAMADAS
# ============================================================================

class MLP:
    """Implementação do Perceptron Multi-Camadas do zero"""
    
    def __init__(self, tamanho_entrada, tamanhos_camadas_ocultas, tamanho_saida, taxa_aprendizado=0.01):
        """
        Inicializar arquitetura do MLP
        
        Args:
            tamanho_entrada: Número de características de entrada
            tamanhos_camadas_ocultas: Lista dos tamanhos das camadas ocultas
            tamanho_saida: Número de classes de saída
            taxa_aprendizado: Taxa de aprendizado para descida do gradiente
        """
        self.tamanho_entrada = tamanho_entrada
        self.tamanhos_camadas_ocultas = tamanhos_camadas_ocultas
        self.tamanho_saida = tamanho_saida
        self.taxa_aprendizado = taxa_aprendizado
        
        # Inicializar arquitetura da rede
        self.camadas = [tamanho_entrada] + tamanhos_camadas_ocultas + [tamanho_saida]
        self.num_camadas = len(self.camadas)
        
        # Inicializar pesos e vieses
        self.pesos = []
        self.vieses = []
        
        for i in range(self.num_camadas - 1):
            # Inicialização Xavier
            peso = np.random.randn(self.camadas[i], self.camadas[i + 1]) * np.sqrt(2.0 / self.camadas[i])
            vies = np.zeros((1, self.camadas[i + 1]))
            
            self.pesos.append(peso)
            self.vieses.append(vies)
    
    def relu(self, x):
        """Função de ativação ReLU"""
        return np.maximum(0, x)
    
    def derivada_relu(self, x):
        """Derivada da ReLU"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Ativação softmax para camada de saída"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def propagacao_direta(self, X):
        """Propagação direta através da rede"""
        self.ativacoes = [X]
        self.valores_z = []
        
        entrada_atual = X
        
        # Propagação através das camadas ocultas
        for i in range(len(self.pesos) - 1):
            z = np.dot(entrada_atual, self.pesos[i]) + self.vieses[i]
            a = self.relu(z)
            
            self.valores_z.append(z)
            self.ativacoes.append(a)
            entrada_atual = a
        
        # Camada de saída (softmax)
        z_saida = np.dot(entrada_atual, self.pesos[-1]) + self.vieses[-1]
        saida = self.softmax(z_saida)
        
        self.valores_z.append(z_saida)
        self.ativacoes.append(saida)
        
        return saida
    
    def propagacao_reversa(self, X, y_verdadeiro, y_predito):
        """Propagação reversa para calcular gradientes"""
        m = X.shape[0]
        
        # Converter y_verdadeiro para codificação one-hot
        y_onehot = np.eye(self.tamanho_saida)[y_verdadeiro]
        
        # Inicializar gradientes
        gradientes_pesos = []
        gradientes_vieses = []
        
        # Gradiente da camada de saída
        delta = y_predito - y_onehot
        
        # Calcular gradientes para todas as camadas (reverso)
        for i in range(len(self.pesos) - 1, -1, -1):
            # Gradientes de pesos e vieses
            gradiente_peso = np.dot(self.ativacoes[i].T, delta) / m
            gradiente_vies = np.mean(delta, axis=0, keepdims=True)
            
            gradientes_pesos.insert(0, gradiente_peso)
            gradientes_vieses.insert(0, gradiente_vies)
            
            # Propagar erro para camada anterior (se não for camada de entrada)
            if i > 0:
                delta = np.dot(delta, self.pesos[i].T) * self.derivada_relu(self.valores_z[i-1])
        
        return gradientes_pesos, gradientes_vieses
    
    def atualizar_parametros(self, gradientes_pesos, gradientes_vieses):
        """Atualizar pesos e vieses usando gradientes"""
        for i in range(len(self.pesos)):
            self.pesos[i] -= self.taxa_aprendizado * gradientes_pesos[i]
            self.vieses[i] -= self.taxa_aprendizado * gradientes_vieses[i]
    
    def calcular_perda(self, y_verdadeiro, y_predito):
        """Calcular perda de entropia cruzada"""
        m = y_predito.shape[0]
        # Adicionar pequeno epsilon para prevenir log(0)
        epsilon = 1e-15
        y_predito = np.clip(y_predito, epsilon, 1 - epsilon)
        
        # Converter para one-hot se necessário
        if len(y_verdadeiro.shape) == 1:
            y_onehot = np.eye(self.tamanho_saida)[y_verdadeiro]
        else:
            y_onehot = y_verdadeiro
        
        perda = -np.sum(y_onehot * np.log(y_predito)) / m
        return perda
    
    def predizer(self, X):
        """Fazer predições em novos dados"""
        saida = self.propagacao_direta(X)
        return np.argmax(saida, axis=1)
    
    def predizer_proba(self, X):
        """Obter probabilidades de predição"""
        return self.propagacao_direta(X)

# ============================================================================
# 4. ESTRATÉGIA DE TREINAMENTO E VALIDAÇÃO
# ============================================================================

class TreinadorMLP:
    """Classe para treinamento e validação do MLP"""
    
    def __init__(self, mlp, X_treino, y_treino, X_val, y_val):
        self.mlp = mlp
        self.X_treino = X_treino
        self.y_treino = y_treino
        self.X_val = X_val
        self.y_val = y_val
        
        # Histórico de treinamento
        self.perdas_treino = []
        self.perdas_val = []
        self.acuracias_treino = []
        self.acuracias_val = []
    
    def treinar_epoca(self, tamanho_lote=32):
        """Treinar por uma época usando descida do gradiente mini-lote"""
        m = self.X_treino.shape[0]
        
        # Embaralhar dados
        indices = np.random.permutation(m)
        X_embaralhado = self.X_treino[indices]
        y_embaralhado = self.y_treino[indices]
        
        perda_epoca = 0
        num_lotes = 0
        
        # Treinamento mini-lote
        for i in range(0, m, tamanho_lote):
            lote_X = X_embaralhado[i:i+tamanho_lote]
            lote_y = y_embaralhado[i:i+tamanho_lote]
            
            # Propagação direta
            y_pred = self.mlp.propagacao_direta(lote_X)
            
            # Calcular perda
            perda_lote = self.mlp.calcular_perda(lote_y, y_pred)
            perda_epoca += perda_lote
            
            # Propagação reversa
            grad_pesos, grad_vieses = self.mlp.propagacao_reversa(lote_X, lote_y, y_pred)
            
            # Atualizar parâmetros
            self.mlp.atualizar_parametros(grad_pesos, grad_vieses)
            
            num_lotes += 1
        
        return perda_epoca / num_lotes
    
    def avaliar(self, X, y):
        """Avaliar modelo nos dados fornecidos"""
        y_pred = self.mlp.predizer(X)
        y_pred_proba = self.mlp.predizer_proba(X)
        
        acuracia = accuracy_score(y, y_pred)
        perda = self.mlp.calcular_perda(y, y_pred_proba)
        
        return perda, acuracia
    
    def treinar(self, epocas=100, tamanho_lote=32, parada_precoce=True, paciencia=10):
        """Treinar o MLP com parada precoce opcional"""
        print("\n" + "=" * 60)
        print("5. TREINAMENTO DO MODELO")
        print("=" * 60)
        
        print(f"Configuração de treinamento:")
        print(f"- Épochs: {epocas}")
        print(f"- Tamanho do lote: {tamanho_lote}")
        print(f"- Taxa de aprendizado: {self.mlp.taxa_aprendizado}")
        print(f"- Arquitetura: {self.mlp.camadas}")
        print(f"- Parada precoce: {parada_precoce} (paciência: {paciencia})")
        
        melhor_perda_val = float('inf')
        contador_paciencia = 0
        
        for epoca in range(epocas):
            # Treinamento
            perda_treino = self.treinar_epoca(tamanho_lote)
            
            # Validação
            perda_val, acuracia_val = self.avaliar(self.X_val, self.y_val)
            perda_treino_eval, acuracia_treino = self.avaliar(self.X_treino, self.y_treino)
            
            # Armazenar histórico
            self.perdas_treino.append(perda_treino_eval)
            self.perdas_val.append(perda_val)
            self.acuracias_treino.append(acuracia_treino)
            self.acuracias_val.append(acuracia_val)
            
            # Imprimir progresso
            if epoca % 10 == 0 or epoca == epocas - 1:
                print(f"Época {epoca+1:3d}/{epocas}: "
                      f"Perda Treino: {perda_treino_eval:.4f}, Acur Treino: {acuracia_treino:.4f}, "
                      f"Perda Val: {perda_val:.4f}, Acur Val: {acuracia_val:.4f}")
            
            # Parada precoce
            if parada_precoce:
                if perda_val < melhor_perda_val:
                    melhor_perda_val = perda_val
                    contador_paciencia = 0
                    # Salvar melhores pesos do modelo
                    self.melhores_pesos = [w.copy() for w in self.mlp.pesos]
                    self.melhores_vieses = [b.copy() for b in self.mlp.vieses]
                else:
                    contador_paciencia += 1
                    if contador_paciencia >= paciencia:
                        print(f"\nParada precoce na época {epoca+1}")
                        # Restaurar melhores pesos
                        self.mlp.pesos = self.melhores_pesos
                        self.mlp.vieses = self.melhores_vieses
                        break
        
        print(f"\nTreinamento concluído!")
        print(f"Melhor perda de validação: {melhor_perda_val:.4f}")
        
    def plotar_curvas_treinamento(self):
        """Plotar curvas de treinamento e validação"""
        print("\n" + "=" * 60)
        print("7. CURVAS DE ERRO E VISUALIZAÇÃO")
        print("=" * 60)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epocas = range(1, len(self.perdas_treino) + 1)
        
        # Curvas de perda
        ax1.plot(epocas, self.perdas_treino, 'b-', label='Perda de Treinamento', linewidth=2)
        ax1.plot(epocas, self.perdas_val, 'r-', label='Perda de Validação', linewidth=2)
        ax1.set_title('Perda do Modelo ao Longo das Épocas')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Perda')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Curvas de acurácia
        ax2.plot(epocas, self.acuracias_treino, 'b-', label='Acurácia de Treinamento', linewidth=2)
        ax2.plot(epocas, self.acuracias_val, 'r-', label='Acurácia de Validação', linewidth=2)
        ax2.set_title('Acurácia do Modelo ao Longo das Épocas')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ian/RedesNeurais/redes_neurais_ianfaray/docs/projeto/training_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Análise
        print("Análise do Treinamento:")
        print(f"- Acurácia final de treinamento: {self.acuracias_treino[-1]:.4f}")
        print(f"- Acurácia final de validação: {self.acuracias_val[-1]:.4f}")
        print(f"- Melhor acurácia de validação: {max(self.acuracias_val):.4f}")
        
        # Verificar overfitting
        if self.acuracias_treino[-1] - self.acuracias_val[-1] > 0.1:
            print("- Modelo mostra sinais de overfitting")
        else:
            print("- Modelo mostra boa generalização")

# ============================================================================
# 5. AVALIAÇÃO E MÉTRICAS
# ============================================================================

class AvaliadorModelo:
    """Classe para avaliação abrangente do modelo"""
    
    def __init__(self, mlp, X_teste, y_teste, label_encoder):
        self.mlp = mlp
        self.X_teste = X_teste
        self.y_teste = y_teste
        self.label_encoder = label_encoder
        
    def avaliar_modelo(self):
        """Avaliação abrangente do modelo"""
        print("\n" + "=" * 60)
        print("8. MÉTRICAS DE AVALIAÇÃO")
        print("=" * 60)
        
        # Fazer predições
        y_pred = self.mlp.predizer(self.X_teste)
        y_pred_proba = self.mlp.predizer_proba(self.X_teste)
        
        # Métricas básicas
        acuracia = accuracy_score(self.y_teste, y_pred)
        precisao = precision_score(self.y_teste, y_pred, average='weighted')
        recall = recall_score(self.y_teste, y_pred, average='weighted')
        f1 = f1_score(self.y_teste, y_pred, average='weighted')
        
        print("Métricas de Performance Geral:")
        print(f"- Acurácia:  {acuracia:.4f}")
        print(f"- Precisão:  {precisao:.4f}")
        print(f"- Recall:    {recall:.4f}")
        print(f"- F1-Score:  {f1:.4f}")
        
        # Relatório de classificação detalhado
        print("\nRelatório de Classificação Detalhado:")
        print(classification_report(self.y_teste, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        # Matriz de confusão
        cm = confusion_matrix(self.y_teste, y_pred)
        self.plotar_matriz_confusao(cm)
        
        # Métricas por classe
        self.analisar_performance_por_classe(y_pred)
        
        return {
            'acuracia': acuracia,
            'precisao': precisao,
            'recall': recall,
            'f1_score': f1,
            'matriz_confusao': cm
        }
    
    def plotar_matriz_confusao(self, cm):
        """Plotar mapa de calor da matriz de confusão"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Matriz de Confusão')
        plt.xlabel('Posição Predita')
        plt.ylabel('Posição Real')
        plt.tight_layout()
        plt.savefig('/home/ian/RedesNeurais/redes_neurais_ianfaray/docs/projeto/confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analisar_performance_por_classe(self, y_pred):
        """Analisar performance para cada classe"""
        print("\nAnálise por Classe:")
        
        for i, nome_classe in enumerate(self.label_encoder.classes_):
            # Verdadeiros positivos, falsos positivos, falsos negativos
            tp = np.sum((self.y_teste == i) & (y_pred == i))
            fp = np.sum((self.y_teste != i) & (y_pred == i))
            fn = np.sum((self.y_teste == i) & (y_pred != i))
            
            precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
            
            print(f"{nome_classe:3s}: Precisão={precisao:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# ============================================================================
# 6. EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função de execução principal"""
    print("Classificação de Posições de Jogadores EA FC 26 usando MLP")
    print("=" * 60)
    
    # 1. Carregar e explorar dataset
    carregador = CarregadorDataset('/home/ian/RedesNeurais/redes_neurais_ianfaray/docs/projeto/FC26_20250921.csv')
    dados = carregador.carregar_dados()
    dados_filtrados = carregador.explorar_posicoes()
    
    # 2. Pré-processamento de dados
    preprocessador = PreprocessadorDados(dados_filtrados)
    dados_limpos = preprocessador.limpar_dados()
    X, y = preprocessador.preparar_caracteristicas()
    preprocessador.visualizar_dados()
    
    # 3. Divisão treino/validação/teste
    print("\n" + "=" * 60)
    print("6. ESTRATÉGIA DE TREINAMENTO E TESTE")
    print("=" * 60)
    
    # Primeira divisão: 80% treino+val, 20% teste
    X_temp, X_teste, y_temp, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Segunda divisão: 75% treino, 25% validação (dos 80%)
    X_treino, X_val, y_treino, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print(f"Divisões do dataset:")
    print(f"- Conjunto de treinamento:   {X_treino.shape[0]:,} amostras ({X_treino.shape[0]/len(X)*100:.1f}%)")
    print(f"- Conjunto de validação: {X_val.shape[0]:,} amostras ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"- Conjunto de teste:       {X_teste.shape[0]:,} amostras ({X_teste.shape[0]/len(X)*100:.1f}%)")
    
    # 4. Inicializar e treinar MLP
    print("\n" + "=" * 60)
    print("4. IMPLEMENTAÇÃO DO MLP")
    print("=" * 60)
    
    # Configuração do MLP
    tamanho_entrada = X_treino.shape[1]
    tamanhos_camadas_ocultas = [128, 64, 32]  # Três camadas ocultas
    tamanho_saida = len(np.unique(y))
    taxa_aprendizado = 0.01
    
    print(f"Arquitetura do MLP:")
    print(f"- Camada de entrada:    {tamanho_entrada} neurônios")
    print(f"- Camadas ocultas:  {tamanhos_camadas_ocultas}")
    print(f"- Camada de saída:   {tamanho_saida} neurônios")
    print(f"- Ativação:     ReLU (ocultas), Softmax (saída)")
    print(f"- Função de perda:  Entropia cruzada")
    print(f"- Otimizador:      SGD mini-lote")
    
    # Inicializar MLP
    mlp = MLP(tamanho_entrada, tamanhos_camadas_ocultas, tamanho_saida, taxa_aprendizado)
    
    # Treinar MLP
    treinador = TreinadorMLP(mlp, X_treino, y_treino, X_val, y_val)
    treinador.treinar(epocas=200, tamanho_lote=64, parada_precoce=True, paciencia=15)
    
    # Plotar curvas de treinamento
    treinador.plotar_curvas_treinamento()
    
    # 5. Avaliar modelo
    avaliador = AvaliadorModelo(mlp, X_teste, y_teste, preprocessador.label_encoder)
    resultados = avaliador.avaliar_modelo()
    
    # 6. Resumo final
    print("\n" + "=" * 60)
    print("RESUMO DO PROJETO")
    print("=" * 60)
    
    print(f"Dataset: Jogadores EA FC 26 ({len(dados_limpos):,} amostras, {len(preprocessador.colunas_caracteristicas)} características)")
    print(f"Tarefa: Classificação multi-classe ({tamanho_saida} posições)")
    print(f"Modelo: MLP com arquitetura {mlp.camadas}")
    print(f"Acurácia Final no Teste: {resultados['acuracia']:.4f}")
    print(f"F1-Score Final no Teste: {resultados['f1_score']:.4f}")
    
    # Salvar resumo do modelo
    with open('/home/ian/RedesNeurais/redes_neurais_ianfaray/docs/projeto/model_summary.txt', 'w') as f:
        f.write("Classificação de Posições de Jogadores EA FC 26 - Resumo do Modelo\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {len(dados_limpos):,} jogadores, {len(preprocessador.colunas_caracteristicas)} características\n")
        f.write(f"Classes: {list(preprocessador.label_encoder.classes_)}\n")
        f.write(f"Arquitetura: {mlp.camadas}\n")
        f.write(f"Acurácia no Teste: {resultados['acuracia']:.4f}\n")
        f.write(f"F1-Score no Teste: {resultados['f1_score']:.4f}\n")
    
    print("\nProjeto concluído com sucesso!")
    print("Arquivos gerados:")
    print("- data_exploration.png")
    print("- feature_distributions.png") 
    print("- training_curves.png")
    print("- confusion_matrix.png")
    print("- model_summary.txt")

if __name__ == "__main__":
    main()
