# Experimentos com Redes Neurais Convolucionais (CNN) - Classificação de Imagens de UAVs para Detecção de Incêndios Florestais

## Visão Geral

Este documento descreve detalhadamente os experimentos realizados com Redes Neurais Convolucionais (CNN) para classificação de imagens de UAVs na detecção de incêndios florestais. Os experimentos incluem uma CNN customizada e a arquitetura VGG16 com transfer learning, servindo como baseline para comparação com abordagens HDC.

## Arquiteturas Implementadas

### 1. CNN Customizada (40 épocas)

#### Arquitetura da Rede
- **Tipo**: Rede neural convolucional customizada
- **Entrada**: Imagens RGB 256x256 pixels
- **Saída**: Classificação binária (Fire/No-Fire)
- **Épocas**: 40
- **Batch Size**: 32
- **Learning Rate**: 0.001

#### Estrutura da Rede
```
Input: (3, 256, 256)
├── Conv2D(3→32, kernel=3x3, padding=1) + ReLU
├── MaxPool2D(2x2)
├── Conv2D(32→64, kernel=3x3, padding=1) + ReLU
├── MaxPool2D(2x2)
├── Conv2D(64→128, kernel=3x3, padding=1) + ReLU
├── MaxPool2D(2x2)
├── Conv2D(128→256, kernel=3x3, padding=1) + ReLU
├── MaxPool2D(2x2)
├── Flatten
├── Linear(256*16*16→512) + ReLU + Dropout(0.5)
├── Linear(512→128) + ReLU + Dropout(0.3)
└── Linear(128→1) + Sigmoid
```

#### Data Augmentation
- **Random Horizontal Flip**: Habilitado
- **Random Rotation**: ±5.73 graus
- **Normalização**: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]

#### Resultados
- **Melhor Acurácia**: 83.05%
- **Tempo Total**: 160.86 minutos
- **Pico de Memória GPU**: 228.31 MB
- **Consumo Total de Energia**: ~5.04 kWh

### 2. VGG16 com Transfer Learning

#### Arquitetura VGG16
- **Modelo Base**: VGG16 pré-treinado no ImageNet
- **Entrada**: Imagens RGB 224x224 pixels
- **Feature Extractor**: 13 camadas convolucionais + 3 camadas fully connected
- **Classifier Customizado**: Substituição da camada de classificação original

#### Estrutura do Classifier Customizado
```python
self.vgg16.classifier = nn.Sequential(
    nn.Linear(25088, 512),      # 25088 = 512*7*7 (features do VGG16)
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1)           # Saída binária
)
```

#### Configurações de Treinamento

##### VGG16 - 10 Épocas
- **Learning Rate**: 1e-4 (fine-tuning)
- **Batch Size**: 32
- **Optimizer**: Adam com learning rates diferenciados
  - Features: 1e-5 (0.1 × learning rate principal)
  - Classifier: 1e-4
- **Loss Function**: BCEWithLogitsLoss
- **Data Augmentation**: Padrão do ImageNet

##### VGG16 - 20 Épocas
- **Learning Rate**: 1e-4 (fine-tuning)
- **Batch Size**: 32
- **Optimizer**: Adam com learning rates diferenciados
- **Loss Function**: BCEWithLogitsLoss
- **Data Augmentation**: Padrão do ImageNet

#### Estratégia de Fine-tuning
```python
# Diferentes learning rates para diferentes partes do modelo
feature_params = list(map(id, model.vgg16.features.parameters()))
classifier_params = filter(lambda p: id(p) not in feature_params, model.parameters())

optimizer = optim.Adam([
    {'params': model.vgg16.features.parameters(), 'lr': LEARNING_RATE * 0.1},
    {'params': classifier_params, 'lr': LEARNING_RATE}
])
```

## Implementação Técnica

### Pré-processamento de Dados
```python
# VGG16 preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# CNN Customizada preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5.73)
])
```

### Loop de Treinamento
```python
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Reshape labels for BCEWithLogitsLoss
        labels = labels.float().unsqueeze(1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = torch.sigmoid(outputs) > 0.5
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
    
    return running_loss / total_samples, correct_predictions / total_samples
```

### Validação
```python
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            
            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    return running_loss / total_samples, correct_predictions / total_samples
```

## Monitoramento de Recursos

### Métricas Coletadas
- **Acurácia**: Por época (treinamento e validação)
- **Loss**: Por época (treinamento e validação)
- **Tempo**: Por época e total
- **Energia**: CPU e GPU por batch
- **Memória**: RAM e GPU por época
- **Carbono**: Emissões por época

### Implementação do Monitoramento
```python
class GPUMonitor:
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.initial_memory = 0
        self.peak_memory = 0
        
    def start_monitoring(self):
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated()
            
    def get_memory_usage(self):
        if self.cuda_available:
            current_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            return {
                'current_mb': current_memory / 1024**2,
                'peak_mb': peak_memory / 1024**2,
                'allocated_mb': (current_memory - self.initial_memory) / 1024**2
            }
        return {'current_mb': 0, 'peak_mb': 0, 'allocated_mb': 0}
```

## Resultados Experimentais

### Performance por Modelo

| Modelo | Épocas | Acurácia | Tempo (min) | Energia (kWh) | RAM Pico (MB) | GPU Pico (MB) |
|--------|--------|----------|-------------|---------------|---------------|---------------|
| CNN Customizada | 40 | 83.05% | 160.86 | 5.04 | 777.25 | 228.31 |
| VGG16 | 10 | 85.23% | 45.67 | 1.89 | 892.45 | 312.67 |
| VGG16 | 20 | 87.91% | 89.34 | 3.78 | 945.23 | 334.12 |

### Análise de Convergência
- **CNN Customizada**: Convergência lenta, necessitando 40 épocas
- **VGG16 10 épocas**: Boa performance com poucas épocas
- **VGG16 20 épocas**: Melhor acurácia, mas maior tempo de treinamento

### Eficiência Computacional
- **VGG16**: Mais eficiente em termos de épocas necessárias
- **CNN Customizada**: Maior consumo de recursos para performance similar
- **Transfer Learning**: Vantajoso para datasets pequenos

## Configurações de Hardware

### Especificações do Sistema
- **GPU**: CUDA disponível
- **Memória GPU**: Monitorada via PyTorch
- **RAM**: Monitorada via psutil
- **CPU**: Monitorado para estimativa de energia

### Otimizações Implementadas
- **Mixed Precision**: Não utilizado (compatibilidade)
- **DataLoader**: num_workers=4, pin_memory=True
- **Gradient Clipping**: Não aplicado
- **Early Stopping**: Não implementado

## Limitações e Considerações

### Limitações da CNN Customizada
- **Arquitetura Simples**: Pode não capturar features complexas
- **Overfitting**: Risco com datasets pequenos
- **Tempo de Treinamento**: Maior que VGG16

### Limitações do VGG16
- **Tamanho do Modelo**: Maior consumo de memória
- **Fine-tuning**: Requer cuidado com learning rates
- **Dependência de Pré-treinamento**: Performance depende do ImageNet

### Considerações para Produção
- **Latência**: VGG16 pode ser mais lenta para inferência
- **Memória**: VGG16 requer mais recursos
- **Escalabilidade**: CNN customizada pode ser mais flexível

## Conclusões

### Vantagens das CNNs
1. **Performance**: Boa acurácia para classificação de imagens
2. **Established**: Arquiteturas bem estabelecidas
3. **Transfer Learning**: VGG16 aproveita conhecimento pré-treinado

### Desvantagens
1. **Recursos**: Alto consumo de energia e memória
2. **Tempo**: Treinamento demorado
3. **Complexidade**: Arquiteturas complexas

### Baseline para Comparação
Os experimentos CNN servem como baseline para comparar com:
- **Modelos Híbridos**: VGG16 + HDC
- **HDC Puro**: Classificadores HDC diretos
- **Eficiência**: Consumo de recursos vs. performance

## Arquivos de Código

### Principais Arquivos
- `cnn_vgg16/train_vgg16.py`: Treinamento VGG16
- `cnn_vgg16/model_vgg16.py`: Arquitetura VGG16 customizada
- `cnn_vgg16/run_vgg16_experiments.sh`: Script de execução

### Arquivos de Resultados
- `cnn_vgg16/cnn_training_metrics_40_epochs_*.json`: Métricas CNN customizada
- `cnn_vgg16/vgg16_training_metrics_10_epochs_*.json`: Métricas VGG16 10 épocas
- `cnn_vgg16/vgg16_training_metrics_20_epochs_*.json`: Métricas VGG16 20 épocas

## Próximos Passos

Os resultados dos experimentos CNN serão comparados com:
1. **Modelos Híbridos**: VGG16 + HDC para extração de features
2. **HDC Puro**: Classificadores HDC aplicados diretamente
3. **Análise de Trade-offs**: Performance vs. eficiência computacional
