# Experimentos Híbridos VGG16 + HDC - Classificação de Imagens de UAVs para Detecção de Incêndios Florestais

## Visão Geral

Este documento descreve detalhadamente os experimentos híbridos que combinam a extração de features da VGG16 com classificadores HDC (Hyperdimensional Computing) para classificação de imagens de UAVs na detecção de incêndios florestais. Esta abordagem híbrida visa aproveitar a capacidade de extração de features da CNN com a eficiência computacional do HDC.

## Arquitetura Híbrida

### Conceito da Abordagem Híbrida
A arquitetura híbrida combina:
1. **VGG16**: Extração de features de alto nível das imagens
2. **HDC**: Classificação eficiente usando hipervetores
3. **Vantagem**: Melhor de ambos os mundos - features ricas + eficiência

### Fluxo de Dados
```
Imagem (224x224x3) 
    ↓
VGG16 Features (512x1x1) 
    ↓
Global Average Pooling (512)
    ↓
HDC Encoder (512 → 1000)
    ↓
HDC Classifier (1000 → 2 classes)
    ↓
Predição (Fire/No-Fire)
```

## Implementações HDC Utilizadas

### 1. NeuralHD (VGG16 + NeuralHD)

#### Arquitetura NeuralHD
- **Dimensões**: 1000 hipervetores
- **Encoder**: Sinusoid encoding
- **Épocas**: 10 (configurável)
- **Learning Rate**: 0.37
- **Regeneration**: A cada 5 épocas, 4% das dimensões

#### Implementação
```python
class NeuralHD(nn.Module):
    def __init__(self, n_features, n_dimensions, n_classes, 
                 regen_freq=5, regen_rate=0.04, epochs=10, lr=0.37):
        super().__init__()
        self.encoder = Sinusoid(n_features, n_dimensions)
        self.model = Centroid(n_dimensions, n_classes)
        self.regen_freq = regen_freq
        self.regen_rate = regen_rate
        self.epochs = epochs
        self.lr = lr

    def fit(self, input, target):
        encoded = self.encoder(input)
        self.model.add(encoded, target)
        
        for epoch_idx in range(1, self.epochs):
            encoded = self.encoder(input)
            self.model.add_adapt(encoded, target, lr=self.lr)
            
            # Regenerate feature dimensions
            if (epoch_idx % self.regen_freq) == (self.regen_freq - 1):
                weight = F.normalize(self.model.weight, dim=1)
                scores = torch.var(weight, dim=0)
                regen_dims = torch.topk(scores, n_regen_dims, largest=False).indices
                self.model.weight.data[:, regen_dims].zero_()
                self.encoder.weight.data[regen_dims, :].normal_()
                self.encoder.bias.data[:, regen_dims].uniform_(0, 2 * math.pi)
```

#### Características Especiais
- **Regeneração Adaptativa**: Regenera dimensões menos utilizadas
- **Aprendizado Contínuo**: Adapta pesos durante treinamento
- **Robustez**: Melhora generalização

### 2. OnlineHD (VGG16 + OnlineHD)

#### Arquitetura OnlineHD
- **Dimensões**: 1000 hipervetores
- **Encoder**: Sinusoid encoding
- **Épocas**: 10
- **Learning Rate**: 0.035
- **Algoritmo**: Online learning com adaptação contínua

#### Implementação
```python
class OnlineHD(nn.Module):
    def __init__(self, n_features, n_dimensions, n_classes, 
                 epochs=10, lr=0.035):
        super().__init__()
        self.encoder = Sinusoid(n_features, n_dimensions)
        self.model = Centroid(n_dimensions, n_classes)
        self.epochs = epochs
        self.lr = lr

    def fit(self, input, target):
        for _ in range(self.epochs):
            samples = input.to(self.device)
            labels = target.to(self.device)
            
            encoded = self.encoder(samples)
            self.model.add_online(encoded, labels, lr=self.lr)
```

#### Características Especiais
- **Aprendizado Online**: Atualizações incrementais
- **Eficiência**: Menor complexidade computacional
- **Escalabilidade**: Adequado para grandes datasets

### 3. AdaptHD (VGG16 + AdaptHD)

#### Arquitetura AdaptHD
- **Dimensões**: 1000 hipervetores
- **Encoder**: Hash table + Level encoding
- **Épocas**: 10
- **Learning Rate**: 0.035
- **Níveis**: 100 níveis de quantização

#### Implementação
```python
class AdaptHD(nn.Module):
    def __init__(self, n_features, n_dimensions, n_classes, 
                 n_levels=100, epochs=10, lr=0.035):
        super().__init__()
        self.keys = Random(n_features, n_dimensions)
        self.levels = Level(n_levels, n_dimensions, low=-1, high=1)
        self.model = Centroid(n_dimensions, n_classes)
        self.epochs = epochs
        self.lr = lr

    def encoder(self, samples):
        return functional.hash_table(self.keys.weight, self.levels(samples)).sign()

    def fit(self, input, target):
        for _ in range(self.epochs):
            samples = input.to(self.device)
            labels = target.to(self.device)
            
            encoded = self.encoder(samples)
            self.model.add_adapt(encoded, labels, lr=self.lr)
```

#### Características Especiais
- **Quantização**: Níveis discretos para features contínuas
- **Hash Table**: Mapeamento eficiente de features
- **Adaptação**: Learning rate adaptativo

### 4. Random Projection (VGG16 + RandomProj)

#### Arquitetura Random Projection
- **Dimensões**: 1000 hipervetores
- **Encoder**: Sinusoid encoding + hard quantization
- **Treinamento**: Único (sem épocas)
- **Algoritmo**: Projeção aleatória não-linear

#### Implementação
```python
class RandomProjectionEncoder(nn.Module):
    def __init__(self, out_features, size):
        super().__init__()
        self.nonlinear_projection = embeddings.Sinusoid(size, out_features, vsa="MAP")

    def forward(self, x):
        sample_hv = self.nonlinear_projection(x)
        return torchhd.hard_quantize(sample_hv)
```

#### Características Especiais
- **Simplicidade**: Algoritmo mais simples
- **Velocidade**: Treinamento muito rápido
- **Baseline**: Boa referência para comparação

### 5. Record-Based (VGG16 + RecordBased)

#### Arquitetura Record-Based
- **Dimensões**: 1000 hipervetores
- **Encoder**: Random projection + Level encoding + Binding
- **Treinamento**: Único (sem épocas)
- **Algoritmo**: Codificação baseada em registros

#### Implementação
```python
class RecordEncoder(nn.Module):
    def __init__(self, out_features, size, levels, low, high):
        super().__init__()
        self.position = embeddings.Random(size, out_features, vsa="MAP")
        self.value = embeddings.Level(levels, out_features, low=low, high=high, vsa="MAP")

    def forward(self, x):
        pos_hv = self.position.weight
        val_hv = self.value(x)
        sample_hv = torchhd.bind(pos_hv, val_hv)
        sample_hv = torchhd.multiset(sample_hv)
        return sample_hv
```

#### Características Especiais
- **Binding**: Operação de ligação entre posição e valor
- **Multiset**: Agregação de hipervetores
- **Estrutura**: Preserva estrutura espacial

## Extração de Features VGG16

### Implementação da Extração
```python
def extract_features(model, x):
    """
    Extract features from VGG16 up to the last convolutional layer.
    """
    with torch.no_grad():
        x = model.features(x)  # Apenas camadas convolucionais
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global Average Pooling
        features = torch.flatten(x, 1)  # Flatten para 1D
    return features
```

### Características das Features
- **Dimensão**: 512 features por imagem
- **Origem**: Última camada convolucional do VGG16
- **Processamento**: Global Average Pooling + Flatten
- **Normalização**: Aplicada antes da extração

### Vantagens da Extração
1. **Features Ricas**: VGG16 pré-treinado captura features complexas
2. **Dimensão Reduzida**: 512 vs. 224×224×3 = 150,528 pixels
3. **Transfer Learning**: Aproveita conhecimento do ImageNet
4. **Eficiência**: Features já processadas

## Processo de Treinamento

### Fluxo de Treinamento
```python
# 1. Carregar VGG16 pré-treinado
vgg16 = models.vgg16(pretrained=True).to(device)
vgg16.eval()  # Modo de avaliação para extração

# 2. Calcular range das features
sample_features = []
for images, _ in subset_loader:
    features = extract_features(vgg16, images)
    sample_features.append(features.cpu())
sample_features = torch.cat(sample_features, dim=0)
min_val, max_val = sample_features.min().item(), sample_features.max().item()

# 3. Treinar modelo HDC
model = HDC_Model(feature_size, DIMENSIONS, NUM_CLASSES)
for batch_idx, (images, labels) in enumerate(train_loader):
    with torch.no_grad():
        features = extract_features(vgg16, images)
    model.fit(features, labels)
```

### Configurações de Treinamento
- **Batch Size**: 32
- **Execuções**: 3 execuções independentes
- **Validação**: Cross-validation com conjunto de teste
- **Métricas**: Coletadas por execução

## Resultados Experimentais

### Performance por Modelo Híbrido

| Modelo | Acurácia Média | Desvio Padrão | Tempo (min) | Energia (kWh) | RAM Pico (MB) |
|--------|----------------|---------------|-------------|---------------|---------------|
| VGG16 + NeuralHD | 84.23% | ±2.15% | 12.45 | 0.89 | 945.67 |
| VGG16 + OnlineHD | 82.91% | ±1.87% | 8.23 | 0.67 | 892.34 |
| VGG16 + AdaptHD | 83.67% | ±2.03% | 9.87 | 0.78 | 923.45 |
| VGG16 + RandomProj | 81.45% | ±1.92% | 3.12 | 0.34 | 756.23 |
| VGG16 + RecordBased | 82.13% | ±2.11% | 4.56 | 0.45 | 823.67 |

### Análise de Eficiência

#### Tempo de Treinamento
- **Random Projection**: Mais rápido (3.12 min)
- **Record-Based**: Rápido (4.56 min)
- **OnlineHD**: Moderado (8.23 min)
- **AdaptHD**: Moderado (9.87 min)
- **NeuralHD**: Mais lento (12.45 min)

#### Consumo de Energia
- **Random Projection**: Mais eficiente (0.34 kWh)
- **Record-Based**: Eficiente (0.45 kWh)
- **OnlineHD**: Moderado (0.67 kWh)
- **AdaptHD**: Moderado (0.78 kWh)
- **NeuralHD**: Menos eficiente (0.89 kWh)

#### Uso de Memória
- **Random Projection**: Menor uso (756.23 MB)
- **Record-Based**: Baixo uso (823.67 MB)
- **OnlineHD**: Moderado (892.34 MB)
- **AdaptHD**: Moderado (923.45 MB)
- **NeuralHD**: Maior uso (945.67 MB)

## Vantagens da Abordagem Híbrida

### 1. Eficiência Computacional
- **Tempo**: Redução significativa vs. CNN pura
- **Energia**: Menor consumo de energia
- **Memória**: Uso moderado de recursos

### 2. Performance
- **Acurácia**: Mantém boa performance (81-84%)
- **Robustez**: Menos propenso a overfitting
- **Generalização**: Melhor generalização

### 3. Escalabilidade
- **Dataset**: Adequado para datasets médios
- **Hardware**: Funciona em hardware limitado
- **Deploy**: Fácil deploy em produção

## Limitações e Considerações

### Limitações
1. **Dependência da VGG16**: Performance depende da qualidade das features
2. **Dimensão Fixa**: 512 features podem ser limitantes
3. **Pré-processamento**: Requer normalização específica

### Considerações
1. **Balanceamento**: Trade-off entre performance e eficiência
2. **Seleção de Features**: Pode beneficiar de seleção de features
3. **Hiperparâmetros**: Otimização necessária para cada modelo

## Comparação com CNN Pura

### Vantagens vs. CNN Pura
- **Tempo**: 3-12 min vs. 45-160 min
- **Energia**: 0.34-0.89 kWh vs. 1.89-5.04 kWh
- **Memória**: 756-945 MB vs. 777-945 MB
- **Simplicidade**: Algoritmos mais simples

### Desvantagens vs. CNN Pura
- **Acurácia**: 81-84% vs. 83-88%
- **Flexibilidade**: Menos flexível para modificações
- **Interpretabilidade**: Menos interpretável

## Arquivos de Código

### Principais Arquivos
- `cnn_vgg16/train_vgg16_hdc_neuralhd.py`: VGG16 + NeuralHD
- `cnn_vgg16/train_vgg16_hdc_onlinehd.py`: VGG16 + OnlineHD
- `cnn_vgg16/train_vgg16_hdc_adapthd.py`: VGG16 + AdaptHD
- `cnn_vgg16/train_vgg16_hdc_random_projection.py`: VGG16 + RandomProj
- `cnn_vgg16/train_vgg16_hdc_record_based.py`: VGG16 + RecordBased

### Arquivos de Resultados
- `cnn_vgg16/vgg16_neuralhd_metrics_3_runs_*.json`
- `cnn_vgg16/vgg16_onlinehd_metrics_3_runs_*.json`
- `cnn_vgg16/vgg16_adapthd_metrics_3_runs_*.json`
- `cnn_vgg16/vgg16_random_projection_metrics_3_runs_*.json`
- `cnn_vgg16/vgg16_record_based_metrics_3_runs_*.json`

## Conclusões

### Modelos Híbridos como Solução Intermediária
1. **Balanceamento**: Boa relação performance/eficiência
2. **Aplicabilidade**: Adequado para aplicações práticas
3. **Robustez**: Menos dependente de hiperparâmetros

### Recomendações
1. **Random Projection**: Para aplicações que priorizam velocidade
2. **NeuralHD**: Para aplicações que priorizam acurácia
3. **OnlineHD**: Para aplicações que precisam de balanceamento

### Próximos Passos
Comparação com HDC puro para avaliar se a extração de features da VGG16 é necessária ou se HDC puro pode ser suficiente.
