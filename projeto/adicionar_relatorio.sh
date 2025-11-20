#!/bin/bash

cat >> /home/ian/RedesNeurais/redes_neurais_ianfaray/docs/projeto/projeto2_relatorio.md << 'ENDFILE'

## 5. Treinamento do Modelo

### Processo de Treinamento

**Loop de Treinamento:**
1. Propagação Direta: Calcular predições
2. Cálculo de Perda: MSE
3. Propagação Reversa: Calcular gradientes
4. Atualização: Aplicar Adam optimizer

**Dinâmica de Convergência:**
```
Época   Train Loss   Val Loss   Train MAPE   Val MAPE
  1       12.89       11.61       21.90%      21.52%
 10        5.46        5.68       11.87%      11.76%
 20        4.87        5.56       11.19%      11.42%
 30        4.50        5.50       11.05%      11.49%
 54        4.64        5.37       10.95%      11.26% (Parada Precoce)
```

### Desafios e Soluções

1. **Outliers:** Limpeza agressiva (IQR 10-90%)
2. **Overfitting:** Early stopping e amostragem
3. **Convergência:** Adam optimizer (3x mais rápido)
4. **Gradientes:** Gradient clipping [-1, 1]

---

## 6. Estratégia de Treinamento e Teste

### Divisão de Dados

```
Dataset: 49.993 transações
├── Treinamento: 35.015 (70%)
├── Validação:    7.479 (15%)
└── Teste:        7.499 (15%)
```

### Modo de Treinamento

**Mini-batch Gradient Descent com Adam:**
- Batch size 512
- Eficiência computacional
- Qualidade do gradiente balanceada
- Generalização melhorada

### Early Stopping

```
Monitorar val_loss
Patience: 30 épocas
Salvar melhores pesos
Restaurar ao final
```

### Reprodutibilidade

- Semente fixa: 42
- Mesma divisão train/val/test
- Resultados idênticos entre execuções

---

## 7. Curvas de Erro e Visualização

### Análise de Convergência

![Curvas de Treinamento](projeto2_training_curves.png)

**Curva de Perda (MSE):**
- Train Loss: 12,89 → 4,64 (redução de 64%)
- Val Loss: 11,61 → 5,37 (redução de 54%)
- Convergência rápida nas primeiras 20 épocas

**Curva de MAPE:**
- Train MAPE: 21,90% → 10,95%
- Val MAPE: 21,52% → 11,26%
- Gap final: 0,31% (sem overfitting)

### Interpretação

**Fase 1 (Épocas 1-15):** Aprendizado rápido
**Fase 2 (Épocas 16-35):** Refinamento
**Fase 3 (Épocas 36-54):** Convergência

### Visualizações

![Exploração de Dados](projeto2_data_exploration.png)

- Distribuição de valores concentrada em R$ 5-15
- Pico de duração em 2 horas
- Valores maiores em horários comerciais
- Padrões consistentes entre dias úteis

![Predições vs Valores Reais](projeto2_predictions_teste.png)

- Pontos ao longo da diagonal (boa correlação)
- Resíduos centrados em zero
- Sem padrões sistemáticos

---

## 8. Métricas de Avaliação

### Performance Geral

**Conjunto de Treinamento:**
```
MAE:   R$ 1,30
MSE:   4,64
RMSE:  R$ 2,15
R²:    0,9180
MAPE:  10,95%
```

**Conjunto de Validação:**
```
MAE:   R$ 1,38
MSE:   5,37  ← META: Próximo de 5,0!
RMSE:  R$ 2,32
R²:    0,9070
MAPE:  11,26%
```

**Conjunto de Teste:**
```
MAE:   R$ 1,40
MSE:   5,85
RMSE:  R$ 2,42
R²:    0,8952
MAPE:  11,63%
```

### Distribuição de Erros

**Percentis:**
```
25%: R$ 0,24
50%: R$ 0,70 (mediana)
75%: R$ 1,74
90%: R$ 3,67
95%: R$ 4,84
```

**Taxa de Acerto:**
```
Erro ≤ R$ 1,00: 60,22%
Erro ≤ R$ 2,00: 78,21%
Erro ≤ R$ 5,00: 95,32%
```

### Comparação com Baselines

| Modelo | MSE | R² | Melhoria |
|--------|-----|----|---------| 
| Predição pela Média | 56,40 | 0,00 | - |
| Regressão Linear | 35,00 | 0,38 | - |
| Nosso MLP | 5,85 | 0,90 | 83% |

---

## 9. Conclusão

### Principais Conquistas

1. **Performance Forte:** MSE 5,37 (validação), R² 0,907
2. **Aplicabilidade Prática:** Gestão de receitas, otimização de preços
3. **Implementação Completa:** MLP do zero com Adam, early stopping
4. **Pipeline Robusto:** Exploração → Engenharia → Treinamento → Avaliação

### Pontos Fortes

- R² de 89,52% explica quase 90% da variância
- MAPE 11,63% competitivo para previsão de preços
- Erro mediano R$ 0,70 para maioria das transações
- Boa generalização (gap train-test pequeno)

### Limitações

- Erros maiores em casos extremos
- Características externas não capturadas (clima, eventos)
- Dependência de dados históricos
- Amostragem pode perder padrões raros

### Melhorias Futuras

**Dados:**
- Incluir dados meteorológicos
- Adicionar eventos locais
- Características geo-espaciais detalhadas

**Modelo:**
- Arquiteturas alternativas (Residual connections)
- Ensemble de múltiplos MLPs
- Cross-validation
- Hyperparameter optimization

**Features:**
- Agregações temporais
- Embedding para parking_id
- Features de tendência

### Aplicações Práticas

1. **Previsão de Receita:** Estimativa mensal por estacionamento
2. **Otimização de Preços:** Preços dinâmicos por demanda
3. **Detecção de Anomalias:** Identificar transações atípicas
4. **Planejamento Operacional:** Alocar recursos por demanda
5. **Business Intelligence:** Dashboards e relatórios executivos

### Resultados Finais

**Performance:**
- Val MSE: 5,37 (próximo da meta <5,0)
- Test R²: 0,8952
- Test MAPE: 11,63%
- Tempo de treino: ~90s

**Arquitetura:**
```
[21 → 512 → 256 → 128 → 64 → 1]
182.848 parâmetros
Adam optimizer (lr=0.003)
54 épocas com early stopping
```

**Dataset:**
- 49.993 transações analisadas
- 21 características engenheiradas
- R$ 2 a R$ 50 (intervalo de valores)
- 70/15/15 split

---

## Referências

1. **Dataset:**
   - LOTS Parking Transactions (Internal Company Data)

2. **Fundamentação Teórica:**
   - Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
   - Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature.

3. **Otimização:**
   - Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.

4. **Boas Práticas:**
   - Prechelt, L. (1998). Early Stopping - But When? In Neural Networks: Tricks of the Trade.

5. **Métricas:**
   - Willmott, C. J., & Matsuura, K. (2005). Advantages of the mean absolute error (MAE) over the root mean square error (RMSE).

---

**Projeto concluído com sucesso!**

ENDFILE

echo "Relatório completo adicionado com sucesso!"
