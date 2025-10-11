# Metodologia de Coleta de Métricas - Classificação de Imagens de UAVs para Detecção de Incêndios Florestais

## Visão Geral

Este documento descreve detalhadamente como as métricas de desempenho foram coletadas e calculadas nos experimentos de classificação de imagens de UAVs para detecção de incêndios florestais. O objetivo é fornecer transparência total sobre a metodologia utilizada para justificar os resultados obtidos.

## Arquitetura dos Experimentos

Os experimentos foram divididos em três categorias principais:

1. **CNN Pura**: Modelos de rede neural convolucional tradicionais (VGG16) como baseline
2. **Modelo Híbrido**: VGG16 para extração de features + HDC para classificação
3. **HDC Puro**: Classificadores HDC aplicados diretamente nas imagens

## Métricas Coletadas

### 1. Acurácia de Classificação
- **Definição**: Porcentagem de predições corretas em relação ao total de amostras
- **Cálculo**: `(predições_corretas / total_amostras) × 100`
- **Coleta**: Medida no conjunto de teste após o treinamento
- **Unidade**: Porcentagem (%)

### 2. Tempo de Treinamento
- **Definição**: Tempo total necessário para treinar o modelo
- **Cálculo**: Soma do tempo de todas as épocas de treinamento
- **Coleta**: Medido usando `time.time()` antes e depois do loop de treinamento
- **Unidade**: Minutos (min)

### 3. Tempo de Inferência
- **Definição**: Tempo necessário para classificar todas as amostras do conjunto de teste
- **Cálculo**: Tempo total do loop de predição no conjunto de teste
- **Coleta**: Medido usando `time.time()` durante a fase de teste
- **Unidade**: Segundos (s)

### 4. Consumo de Energia

#### 4.1 Energia do CPU
- **Método**: Estimativa baseada na utilização do CPU
- **Fórmula**: `(cpu_percent / 100) × 65 Watts × tempo_duracao / 3600`
- **Parâmetros**:
  - `cpu_percent`: Utilização do CPU obtida via `psutil.cpu_percent(interval=0.1)`
  - `65 Watts`: Potência máxima estimada do CPU
  - `tempo_duracao`: Duração do batch em segundos
- **Coleta**: A cada batch durante treinamento e teste
- **Unidade**: kWh

#### 4.2 Energia da GPU
- **Método**: Estimativa baseada na utilização da GPU
- **Fórmula**: `(gpu_utilization / 100) × 250 Watts × tempo_duracao / 3600`
- **Parâmetros**:
  - `gpu_utilization`: Utilização da GPU calculada como `(memória_alocada / memória_total) × 100`
  - `250 Watts`: Potência máxima estimada da GPU
  - `tempo_duracao`: Duração do batch em segundos
- **Coleta**: A cada batch durante treinamento e teste
- **Unidade**: kWh

#### 4.3 Energia Total
- **Cálculo**: Soma da energia do CPU e GPU
- **Fórmula**: `energia_cpu + energia_gpu`
- **Unidade**: kWh

### 5. Emissões de Carbono
- **Método**: Utilização da biblioteca CodeCarbon
- **Implementação**: 
  ```python
  tracker = EmissionsTracker()
  tracker.start()
  # ... execução do código ...
  emissions = tracker.stop()
  ```
- **Coleta**: Medida separadamente para treinamento e teste
- **Unidade**: kg CO2

### 6. Consumo de Memória RAM

#### 6.1 Monitoramento de RAM
- **Método**: Utilização da biblioteca `psutil`
- **Métricas coletadas**:
  - **RSS (Resident Set Size)**: Memória física realmente utilizada
  - **VMS (Virtual Memory Size)**: Memória virtual total
  - **Percentual**: Porcentagem de utilização da RAM do sistema
- **Fórmula**: `process.memory_info().rss / 1024**2` (conversão para MB)
- **Coleta**: 
  - Início e fim de cada fase (treinamento/teste)
  - Pico de utilização durante execução
- **Unidade**: MB

#### 6.2 Memória do Sistema
- **Métricas**: Memória total, disponível, utilizada e percentual
- **Coleta**: Via `psutil.virtual_memory()`
- **Unidade**: MB

### 7. Consumo de Memória GPU

#### 7.1 Monitoramento de GPU
- **Método**: Utilização das APIs do PyTorch para CUDA
- **Métricas coletadas**:
  - **Memória atual**: `torch.cuda.memory_allocated()`
  - **Pico de memória**: `torch.cuda.max_memory_allocated()`
  - **Memória alocada**: Diferença entre memória atual e inicial
- **Fórmula**: `memória_bytes / 1024**2` (conversão para MB)
- **Coleta**: 
  - Reset do pico antes de cada fase
  - Medição do pico ao final de cada fase
- **Unidade**: MB

## Metodologia de Coleta por Tipo de Experimento

### Experimentos CNN Pura (VGG16)
- **Arquivos**: `train_vgg16.py`
- **Estrutura**: Métricas coletadas por época
- **Processamento**: Agregação de métricas de todas as épocas
- **Execuções**: 1 execução com múltiplas épocas (10 ou 20)

### Experimentos Híbridos (VGG16 + HDC)
- **Arquivos**: `train_vgg16_hdc_*.py`
- **Estrutura**: Métricas coletadas por execução
- **Processamento**: Média e desvio padrão de 3 execuções
- **Execuções**: 3 execuções independentes

### Experimentos HDC Puro
- **Arquivos**: `model_torchhd/classifier_*.py`
- **Estrutura**: Métricas coletadas por execução
- **Processamento**: Média e desvio padrão de 3 execuções
- **Execuções**: 3 execuções independentes

## Configurações dos Experimentos

### Parâmetros Comuns
- **Batch Size**: 32 (VGG16), 8 (HDC puro)
- **Tamanho da Imagem**: 224x224 (VGG16), 64x64 (HDC puro)
- **Dimensões HDC**: 1000-1024
- **Número de Classes**: 2 (Fire/No-Fire)

### Parâmetros Específicos por Modelo
- **VGG16**: 10-20 épocas, learning rate 1e-4
- **NeuralHD**: 3-10 épocas, regen_freq=5, lr=0.37
- **OnlineHD**: 10 épocas, lr=0.035
- **AdaptHD**: 10 épocas, lr=0.035
- **Random Projection**: Sem épocas (treinamento único)
- **Record-Based**: Sem épocas (treinamento único)

## Validação e Reproduzibilidade

### Controle de Variáveis
- **Hardware**: Mesmo sistema para todos os experimentos
- **Software**: Versões fixas das bibliotecas
- **Dados**: Mesmo dataset de treinamento e teste
- **Sementes**: Não fixadas (variação natural)

### Tratamento de Erros
- **CodeCarbon**: Fallback para 0.0 se não disponível
- **GPU**: Fallback para 0 se CUDA não disponível
- **Memória**: Tratamento de exceções em monitoramento

## Processamento dos Dados

### Agregação de Métricas
- **CNN Pura**: Soma de métricas de todas as épocas
- **HDC**: Média aritmética de 3 execuções
- **Desvio Padrão**: Calculado para experimentos HDC

### Normalização
- **Tempo**: Conversão para minutos para treinamento
- **Energia**: Conversão para kWh
- **Memória**: Conversão para MB
- **Acurácia**: Conversão para porcentagem

## Limitações e Considerações

### Estimativas de Energia
- **CPU**: Baseada em utilização percentual (pode subestimar)
- **GPU**: Baseada em memória alocada (não reflete computação real)
- **Valores**: Estimativas conservadoras (65W CPU, 250W GPU)

### Monitoramento de Memória
- **RAM**: Medição do processo Python (não inclui overhead do sistema)
- **GPU**: Medição da memória alocada (não inclui cache)
- **Picos**: Capturados durante execução, podem não refletir uso sustentado

### Emissões de Carbono
- **Precisão**: Dependente da localização geográfica e mix energético
- **CodeCarbon**: Estimativas baseadas em médias regionais
- **Granularidade**: Medição por fase (treinamento/teste)

## Justificativa da Metodologia

### Escolha das Métricas
1. **Acurácia**: Métrica principal de desempenho
2. **Tempo**: Importante para aplicações em tempo real
3. **Energia**: Relevante para dispositivos com limitações energéticas
4. **Carbono**: Importante para sustentabilidade
5. **Memória**: Crítico para dispositivos com recursos limitados

### Metodologia de Coleta
- **Granularidade**: Coleta por batch para capturar variações
- **Repetibilidade**: Múltiplas execuções para HDC (variação natural)
- **Consistência**: Mesma metodologia para todos os experimentos
- **Transparência**: Código fonte disponível para verificação

### Validação dos Resultados
- **Comparabilidade**: Mesmas condições para todos os experimentos
- **Reproduzibilidade**: Metodologia documentada e código disponível
- **Robustez**: Tratamento de erros e fallbacks implementados

## Conclusão

A metodologia de coleta de métricas foi projetada para fornecer uma comparação justa e transparente entre diferentes abordagens de classificação de imagens. Todas as métricas são coletadas de forma consistente e os resultados podem ser reproduzidos seguindo a metodologia documentada.

Para mais detalhes sobre a implementação, consulte os arquivos de código fonte correspondentes a cada tipo de experimento.
