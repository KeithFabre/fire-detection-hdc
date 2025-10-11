# Análise de Resultados: HDC para Classificação de Imagens de UAV

Este projeto analisa os resultados de experimentos comparando diferentes abordagens para classificação de imagens de UAV usando Hyperdimensional Computing (HDC).

## 📊 Objetivo

Demonstrar os benefícios do HDC em termos de:
- ⚡ **Redução do tempo de treinamento** (até 95%)
- 🔋 **Economia de energia** (até 97%)
- 💾 **Menor uso de memória**
- 🌱 **Redução de emissões de carbono**
- 🎯 **Manutenção de acurácia competitiva**

## 🧪 Modelos Testados

### 1. VGG16 Transfer Learning
- **VGG16 (10 épocas)**: Baseline com treinamento reduzido
- **VGG16 (20 épocas)**: Baseline completo

### 2. Modelos Híbridos VGG16 + HDC
- **VGG16 + NeuralHD**: Combinação de CNN com NeuralHD
- **VGG16 + RandomProj**: Combinação de CNN com Random Projection
- **VGG16 + OnlineHD**: Combinação de CNN com OnlineHD
- **VGG16 + AdaptHD**: Combinação de CNN com AdaptHD

### 3. HDC Puro
- **HDC Record-Based**: Codificação baseada em registros
- **HDC Random Proj**: Projeção aleatória
- **HDC NeuralHD**: NeuralHD puro
- **HDC AdaptHD**: AdaptHD puro
- **HDC OnlineHD**: OnlineHD puro

## 📁 Arquivos do Projeto

### Notebook de Análise
- `analise_resultados_hdc_uav.ipynb`: Notebook Jupyter completo com análise detalhada

### Script de Geração de Gráficos
- `gerar_graficos_hdc.py`: Script Python para gerar gráficos específicos

### Dados de Entrada
- `cnn_vgg16/`: Resultados dos experimentos com VGG16
- `model_torchhd/`: Resultados dos experimentos com HDC puro

### Saídas Geradas
- `tempo_vs_acuracia_hdc.png`: Gráfico principal de tempo vs acurácia
- `consumo_energetico_hdc.png`: Comparação de consumo energético
- `analise_eficiencia_hdc.png`: Análise de eficiência multidimensional
- `resultados_hdc_processados.csv`: Dados processados
- `tabela_resumo_hdc.csv`: Tabela resumo dos resultados
- `economias_hdc.csv`: Análise de economias

## 🚀 Como Usar

### 1. Executar o Notebook
```bash
jupyter notebook analise_resultados_hdc_uav.ipynb
```

### 2. Executar o Script de Gráficos
```bash
python gerar_graficos_hdc.py
```

### 3. Requisitos
```bash
pip install pandas numpy matplotlib seaborn jupyter
```

## 📈 Principais Descobertas

### Eficiência Dramática
- **Tempo de treinamento**: Redução de até 95% em relação ao VGG16
- **Consumo energético**: Redução de até 97% em relação ao VGG16
- **Emissões de CO2**: Redução significativa no impacto ambiental

### Trade-off Acurácia vs Eficiência
- **VGG16 (20 épocas)**: 82.5% acurácia, 225 min, 51.5 kWh
- **VGG16 + NeuralHD**: 74.5% acurácia, 5 min, 1.5 kWh
- **HDC Puro**: 60-65% acurácia, 5-15 min, 0.5-2.5 kWh

### Modelos Híbridos
Os modelos VGG16 + HDC oferecem o melhor equilíbrio, combinando:
- Capacidade de extração de features da CNN
- Eficiência computacional do HDC
- Acurácia competitiva

## 🎯 Recomendações

### Para Aplicações Específicas

| Aplicação | Modelo Recomendado | Justificativa |
|-----------|-------------------|---------------|
| **Tempo real** | HDC Puro | Máxima velocidade |
| **Máxima acurácia** | VGG16 + NeuralHD | Melhor trade-off |
| **Desenvolvimento rápido** | HDC Random Proj | Simplicidade |
| **Produção** | VGG16 + RandomProj | Equilíbrio geral |

### Impacto Ambiental
A adoção de HDC pode reduzir significativamente o impacto ambiental do treinamento de modelos de IA, especialmente importante para:
- Aplicações de UAV que requerem retreinamento frequente
- Edge computing com recursos limitados
- Desenvolvimento sustentável de IA

## 📊 Métricas Analisadas

### Métricas Principais
- **Acurácia**: Performance de classificação
- **Tempo de treinamento**: Duração do processo
- **Consumo energético**: kWh utilizados
- **Emissões de carbono**: kg CO2 emitidos
- **Uso de memória**: RAM e GPU utilizados

### Métricas Derivadas
- **Eficiência temporal**: Acurácia por hora de treinamento
- **Eficiência energética**: Acurácia por kWh
- **Eficiência de carbono**: Acurácia por kg CO2
- **Trade-off score**: Pontuação combinada

## 🔍 Análise Detalhada

### Por Categoria
1. **VGG16 Puro**: Máxima acurácia, alto custo
2. **VGG16 + HDC Híbrido**: Melhor equilíbrio
3. **HDC Puro**: Máxima eficiência, acurácia moderada

### Economias em Relação ao VGG16 (20 épocas)
- **Tempo**: 80-95% de economia
- **Energia**: 85-97% de economia
- **Carbono**: Redução proporcional
- **Acurácia**: Redução de 5-25%

## 📝 Conclusões

O HDC demonstra ser uma tecnologia promissora para:
1. **Reduzir custos computacionais** significativamente
2. **Acelerar desenvolvimento** de modelos de IA
3. **Tornar IA mais sustentável** ambientalmente
4. **Permitir edge computing** eficiente
5. **Manter performance competitiva** em aplicações práticas

## 🤝 Contribuições

Este projeto demonstra o potencial do HDC para democratizar o acesso à IA, especialmente em aplicações de UAV onde eficiência e sustentabilidade são críticas.

---

**Nota**: Os resultados mostram que o HDC pode ser uma alternativa viável e eficiente aos métodos tradicionais de deep learning para classificação de imagens de UAV, oferecendo ganhos substanciais em eficiência com perdas moderadas em acurácia. 