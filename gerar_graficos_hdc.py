#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gerar gráficos específicos dos resultados de HDC para classificação de UAVs
Autor: Análise de Resultados HDC
Data: 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração para gráficos em português
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# Cores para os gráficos
colors = {
    'CNN_40_epochs': '#e74c3c',
    'VGG16_10_epochs': '#c0392b',
    'VGG16_20_epochs': '#8e44ad',
    'VGG16_NeuralHD': '#3498db',
    'VGG16_RandomProj': '#2ecc71',
    'VGG16_OnlineHD': '#1abc9c',
    'VGG16_AdaptHD': '#f1c40f',
    'VGG16_RecordBased': '#9b59b6',
    'HDC_RandomProj': '#16a085',
    'HDC_NeuralHD': '#f39c12',
    'HDC_AdaptHD': '#8e44ad',
    'HDC_OnlineHD': '#2980b9',
    'HDC_RecordBased': '#d7bde2'
}

# Mapeamento de nomes para labels mais amigáveis
name_mapping = {
    'CNN_40_epochs': 'CNN (40 épocas)',
    'VGG16_10_epochs': 'VGG16 (10 épocas)',
    'VGG16_20_epochs': 'VGG16 (20 épocas)',
    'VGG16_NeuralHD': 'VGG16 + NeuralHD',
    'VGG16_RandomProj': 'VGG16 + RandomProj',
    'VGG16_OnlineHD': 'VGG16 + OnlineHD',
    'VGG16_AdaptHD': 'VGG16 + AdaptHD',
    'VGG16_RecordBased': 'VGG16 RecordBased',
    'HDC_RandomProj': 'HDC RandomProj',
    'HDC_NeuralHD': 'HDC NeuralHD',
    'HDC_AdaptHD': 'HDC AdaptHD',
    'HDC_OnlineHD': 'HDC OnlineHD',
    'HDC_RecordBased': 'HDC RecordBased'
}


def load_and_process_data():
    """Carrega e processa todos os dados dos experimentos"""
    
    def load_json_data(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Erro ao carregar {file_path}: {e}")
            return None

    def extract_vgg16_metrics(data, model_name):
        if not data:
            return None
        
        exp_info = data.get('experiment_info', {})
        total_energy = 0
        total_carbon = 0
        
        # Track peak RAM and GPU usage across all epochs
        peak_ram_mb = 0
        peak_gpu_mb = 0
        
        for epoch in data.get('individual_epochs', []):
            energy = epoch.get('energy_consumption', {})
            total_energy += energy.get('training', {}).get('total_kwh', 0)
            total_energy += energy.get('testing', {}).get('total_kwh', 0)
            
            carbon = epoch.get('carbon_emissions', {})
            total_carbon += carbon.get('total', 0)
            
            # Update peak RAM usage
            ram_usage = epoch.get('ram_usage', {})
            training_ram = ram_usage.get('training', {}).get('peak_rss_mb', 0)
            testing_ram = ram_usage.get('testing', {}).get('peak_rss_mb', 0)
            peak_ram_mb = max(peak_ram_mb, training_ram, testing_ram)
            
            # Update peak GPU usage
            gpu_memory = epoch.get('gpu_memory', {})
            training_gpu = gpu_memory.get('training_peak_mb', 0)
            testing_gpu = gpu_memory.get('testing_peak_mb', 0)
            peak_gpu_mb = max(peak_gpu_mb, training_gpu, testing_gpu)
        
        # Para VGG16/CNN, usar tempo de teste da última época como prediction time
        prediction_time = 0
        if data.get('individual_epochs'):
            last_epoch = data['individual_epochs'][-1]
            prediction_time = last_epoch.get('testing_time', 0)
        
        return {
            'model_name': model_name,
            'accuracy': exp_info.get('best_accuracy', 0) * 100,
            'total_energy_kwh': total_energy,
            'total_carbon_kg': total_carbon,
            'training_time_minutes': exp_info.get('total_time_minutes', 0),
            'epochs': exp_info.get('total_epochs', 0),
            'peak_ram_mb': peak_ram_mb,
            'peak_gpu_mb': peak_gpu_mb,
            'prediction_time_seconds': prediction_time
        }

    def extract_hdc_metrics(data, model_name):
        if not data:
            return None
        
        accuracies = []
        energies = []
        carbons = []
        training_times = []
        prediction_times = []
        peak_ram_values = []
        peak_gpu_values = []
        
        for run in data.get('individual_runs', []):
            accuracies.append(run.get('accuracy', 0))
            
            energy = run.get('energy_consumption', {})
            total_energy = (energy.get('training', {}).get('total_kwh', 0) + 
                           energy.get('testing', {}).get('total_kwh', 0))
            energies.append(total_energy)
            
            carbon = run.get('carbon_emissions', {})
            carbons.append(carbon.get('total', 0))
            
            training_times.append(run.get('training_time', 0) / 60)  # Convert to minutes
            prediction_times.append(run.get('prediction_time', 0))   # Keep in seconds
            
            # Extract peak RAM usage (maximum of training and testing peaks)
            ram_usage = run.get('ram_usage', {})
            training_ram = ram_usage.get('training', {}).get('peak_rss_mb', 0)
            testing_ram = ram_usage.get('testing', {}).get('peak_rss_mb', 0)
            peak_ram_values.append(max(training_ram, testing_ram))
            
            # Extract peak GPU usage (maximum of training and testing peaks)
            gpu_memory = run.get('gpu_memory', {})
            training_gpu = gpu_memory.get('training_peak_mb', 0)
            testing_gpu = gpu_memory.get('testing_peak_mb', 0)
            peak_gpu_values.append(max(training_gpu, testing_gpu))
        
        return {
            'model_name': model_name,
            'accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'total_energy_kwh': np.mean(energies),
            'total_carbon_kg': np.mean(carbons),
            'training_time_minutes': np.mean(training_times),
            'prediction_time_seconds': np.mean(prediction_times),
            'peak_ram_mb': np.mean(peak_ram_values),
            'peak_gpu_mb': np.mean(peak_gpu_values)
        }

    # Carregar todos os dados
    data_files = {
        'CNN_40_epochs': 'cnn_vgg16/cnn_training_metrics_40_epochs_20250907_193418.json',
        'VGG16_10_epochs': 'cnn_vgg16/vgg16_training_metrics_10_epochs_20250903_182427.json',
        'VGG16_20_epochs': 'cnn_vgg16/vgg16_training_metrics_20_epochs_20250902_172216.json',
        'VGG16_NeuralHD': 'cnn_vgg16/vgg16_neuralhd_metrics_3_runs_20250902_105430.json',
        'VGG16_RandomProj': 'cnn_vgg16/vgg16_random_projection_metrics_3_runs_20250902_091831.json',
        'VGG16_RecordBased': 'cnn_vgg16/vgg16_record_based_metrics_3_runs_20250902_094904.json',
        'VGG16_OnlineHD': 'cnn_vgg16/vgg16_onlinehd_metrics_3_runs_20250924_153603.json',
        'VGG16_AdaptHD': 'cnn_vgg16/vgg16_adapthd_metrics_3_runs_20250924_160643.json',
        'HDC_RandomProj': 'model_torchhd/hdc_random_projection_metrics_3_runs_20250902_065657.json',
        'HDC_NeuralHD': 'model_torchhd/hdc_neuralhd_classifier_metrics_20250903_132034.json',
        'HDC_AdaptHD': 'model_torchhd/hdc_adapthd_classifier_metrics_20250903_115212.json',
        'HDC_OnlineHD': 'model_torchhd/hdc_onlinehd_classifier_metrics_20250903_144931.json',
        'HDC_RecordBased': 'model_torchhd/hdc_record_based_metrics_3_runs_20250902_075259.json'
    }

    results = []
    for model_name, file_path in data_files.items():
        data = load_json_data(file_path)
        
        if any(x in model_name for x in ['VGG16_10_epochs', 'VGG16_20_epochs', 'CNN_']):
            metrics = extract_vgg16_metrics(data, model_name)
        else:
            metrics = extract_hdc_metrics(data, model_name)
        
        if metrics:
            results.append(metrics)
            print(f"✓ {model_name}: Acurácia {metrics['accuracy']:.1f}%, Tempo {metrics['training_time_minutes']:.1f}min, Predição {metrics['prediction_time_seconds']:.3f}s")

    df = pd.DataFrame(results)
    
    # Categorizar modelos
    df['category'] = df['model_name'].apply(lambda x: 
        'CNN' if 'CNN_' in x else
        'VGG16' if 'VGG16_' in x and 'epochs' in x else
        'VGG16_HDC_Hibrido' if 'VGG16_' in x else
        'HDC'
    )
    
    # Adicionar nome amigável
    df['friendly_name'] = df['model_name'].map(name_mapping)
    
    return df


def create_comparison_charts(df):
    """Cria gráficos de comparação principais (PNG)"""
    
    # Configurar fontes maiores para PDF
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 16
    
    # 1. Gráfico principal: Tempo vs Acurácia
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for idx, row in df.iterrows():
        color = colors.get(row['model_name'], '#000000')  # preto como fallback
        size = 500 + (row['total_energy_kwh'] * 100)  # pontos ainda maiores
        ax.scatter(row['training_time_minutes'], row['accuracy'], 
                  c=color, s=size, alpha=0.9, edgecolors='black', linewidth=2.5)
        ax.annotate(row['friendly_name'].replace(' ', '\n'),
                    (row['training_time_minutes'], row['accuracy']),
                    xytext=(8, 8), textcoords='offset points', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='black'))
    ax.set_xlabel('Tempo de Treinamento (min)', fontsize=18, fontweight='bold', color='black')
    ax.set_ylabel('Acurácia (%)', fontsize=18, fontweight='bold', color='black')
    ax.set_title('Tempo de Treinamento vs Acurácia', fontsize=22, fontweight='bold', color='black', pad=20)
    ax.tick_params(axis='both', labelsize=16, colors='black')
    ax.grid(True, alpha=0.4, color='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.tight_layout()
    plt.savefig('tempo_vs_acuracia_hdc.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # 2. Gráfico de barras: Consumo Energético
    fig, ax = plt.subplots(figsize=(16, 10))
    bars = ax.bar(range(len(df)), df['total_energy_kwh'], 
                  color=[colors.get(name, '#000000') for name in df['model_name']],
                  alpha=0.9, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([name_mapping.get(name, name).replace(' ', '\n') for name in df['model_name']], 
                       rotation=45, ha='right', fontsize=14, fontweight='bold', color='black')
    ax.set_ylabel('Consumo Energético (kWh)', fontsize=18, fontweight='bold', color='black')
    ax.set_title('Comparação de Consumo Energético (kWh)', fontsize=22, fontweight='bold', color='black', pad=20)
    ax.tick_params(axis='y', labelsize=16, colors='black')
    ax.grid(True, alpha=0.4, axis='y', color='black')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2., h+0.5, f'{h:.1f}', 
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.tight_layout()
    plt.savefig('consumo_energetico_hdc.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # 3. NOVO: Gráfico de barras para Tempo de Predição
    fig, ax = plt.subplots(figsize=(12, 8))  # Reduzir tamanho da figura
    bars = ax.bar(range(len(df)), df['prediction_time_seconds'], 
                  color=[colors.get(name, '#000000') for name in df['model_name']],
                  alpha=0.9, edgecolor='black', linewidth=1.5)  # Reduzir linewidth
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([name_mapping.get(name, name).replace(' ', '\n') for name in df['model_name']], 
                       rotation=45, ha='right', fontsize=12, fontweight='bold', color='black')  # Reduzir fonte
    ax.set_ylabel('Tempo de Predição (segundos)', fontsize=16, fontweight='bold', color='black')  # Reduzir fonte
    ax.set_title('Comparação de Tempo de Predição', fontsize=18, fontweight='bold', color='black', pad=15)  # Reduzir fonte
    ax.tick_params(axis='y', labelsize=14, colors='black')  # Reduzir fonte
    ax.grid(True, alpha=0.4, axis='y', color='black')
    
    # Usar escala logarítmica se houver diferenças grandes
    try:
        max_time = df['prediction_time_seconds'].max()
        min_time = df['prediction_time_seconds'].min()
        if max_time / min_time > 100:  # Se a diferença for maior que 100x
            ax.set_yscale('log')
            ax.set_ylabel('Tempo de Predição (segundos) - escala log', fontsize=16, fontweight='bold', color='black')
    except:
        pass  # Se houver erro, continuar sem escala log
    
    # Adicionar valores apenas se não forem muitos pontos
    if len(df) <= 15:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2., h*1.05, f'{h:.3f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.tight_layout()
    
    # Tentar salvar com DPI menor se houver problema de memória
    try:
        plt.savefig('tempo_predicao_hdc.png', dpi=300, bbox_inches='tight', facecolor='white')
    except MemoryError:
        print("Aviso: Usando DPI menor devido a limitações de memória")
        plt.savefig('tempo_predicao_hdc.png', dpi=150, bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    plt.clf()  # Limpar completamente


def create_efficiency_analysis(df):
    """Cria análise de eficiência incluindo tempo de predição (PNG)"""
    df = df.copy()
    df['efficiency_score'] = (df['accuracy'] / 100) / (df['training_time_minutes'] / 60)
    df['energy_efficiency'] = (df['accuracy'] / 100) / df['total_energy_kwh']
    df['carbon_efficiency'] = (df['accuracy'] / 100) / df['total_carbon_kg']
    df['prediction_efficiency'] = (df['accuracy'] / 100) / df['prediction_time_seconds']  # Nova métrica
    
    # Configurar fontes maiores para PDF
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 16
    
    # Criar um subplot 2x3 para incluir o novo gráfico
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Análise de Eficiência dos Modelos HDC', fontsize=24, fontweight='bold', color='black', y=0.98)
    
    # Eficiência Temporal
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(df)), df['efficiency_score'], 
                    color=[colors.get(name, '#000000') for name in df['model_name']],
                    alpha=0.9, edgecolor='black', linewidth=2)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels([name_mapping.get(name, name).replace(' ', '\n') for name in df['model_name']], 
                        rotation=45, ha='right', fontsize=11, fontweight='bold', color='black')
    ax1.set_ylabel('Acurácia/Hora', fontsize=18, fontweight='bold', color='black')
    ax1.set_title('Eficiência Temporal', fontsize=20, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelsize=14, colors='black')
    ax1.grid(True, alpha=0.4, axis='y', color='black')
    for spine in ax1.spines.values():
        spine.set_color('black')
    
    # Eficiência Energética
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(df)), df['energy_efficiency'], 
                    color=[colors.get(name, '#000000') for name in df['model_name']],
                    alpha=0.9, edgecolor='black', linewidth=2)
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([name_mapping.get(name, name).replace(' ', '\n') for name in df['model_name']], 
                        rotation=45, ha='right', fontsize=11, fontweight='bold', color='black')
    ax2.set_ylabel('Acurácia/kWh', fontsize=18, fontweight='bold', color='black')
    ax2.set_title('Eficiência Energética', fontsize=20, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelsize=14, colors='black')
    ax2.grid(True, alpha=0.4, axis='y', color='black')
    for spine in ax2.spines.values():
        spine.set_color('black')
    
    # NOVO: Eficiência de Predição
    ax3 = axes[0, 2]
    bars3 = ax3.bar(range(len(df)), df['prediction_efficiency'], 
                    color=[colors.get(name, '#000000') for name in df['model_name']],
                    alpha=0.9, edgecolor='black', linewidth=2)
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels([name_mapping.get(name, name).replace(' ', '\n') for name in df['model_name']], 
                        rotation=45, ha='right', fontsize=11, fontweight='bold', color='black')
    ax3.set_ylabel('Acurácia/segundo', fontsize=18, fontweight='bold', color='black')
    ax3.set_title('Eficiência de Predição', fontsize=20, fontweight='bold', color='black')
    ax3.tick_params(axis='y', labelsize=14, colors='black')
    ax3.grid(True, alpha=0.4, axis='y', color='black')
    # Usar escala logarítmica se necessário
    max_pred_eff = df['prediction_efficiency'].max()
    min_pred_eff = df['prediction_efficiency'].min()
    if max_pred_eff / min_pred_eff > 100:
        ax3.set_yscale('log')
        ax3.set_ylabel('Acurácia/segundo (log)', fontsize=18, fontweight='bold', color='black')
    for spine in ax3.spines.values():
        spine.set_color('black')
    
    # Eficiência de Carbono
    ax4 = axes[1, 0]
    bars4 = ax4.bar(range(len(df)), df['carbon_efficiency'], 
                    color=[colors.get(name, '#000000') for name in df['model_name']],
                    alpha=0.9, edgecolor='black', linewidth=2)
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels([name_mapping.get(name, name).replace(' ', '\n') for name in df['model_name']], 
                        rotation=45, ha='right', fontsize=11, fontweight='bold', color='black')
    ax4.set_ylabel('Acurácia/kg CO2', fontsize=18, fontweight='bold', color='black')
    ax4.set_title('Eficiência de Carbono', fontsize=20, fontweight='bold', color='black')
    ax4.tick_params(axis='y', labelsize=14, colors='black')
    ax4.grid(True, alpha=0.4, axis='y', color='black')
    for spine in ax4.spines.values():
        spine.set_color('black')
    
    # Eficiência por Categoria
    ax5 = axes[1, 1]
    category_means = df.groupby('category')['efficiency_score'].mean()
    bars5 = ax5.bar(range(len(category_means)), category_means.values,
                    color=['#e74c3c', '#c0392b', '#3498db', '#9b59b6'], alpha=0.9, edgecolor='black', linewidth=2)
    ax5.set_xticks(range(len(category_means)))
    ax5.set_xticklabels(['CNN Puro', 'VGG16 Puro', 'VGG16+HDC\nHíbrido', 'HDC Puro'], 
                        rotation=0, fontsize=14, fontweight='bold', color='black')
    ax5.set_ylabel('Eficiência Temporal Média', fontsize=18, fontweight='bold', color='black')
    ax5.set_title('Eficiência por Categoria', fontsize=20, fontweight='bold', color='black')
    ax5.tick_params(axis='y', labelsize=14, colors='black')
    ax5.grid(True, alpha=0.4, axis='y', color='black')
    for spine in ax5.spines.values():
        spine.set_color('black')
    
    # NOVO: Scatter Tempo de Predição vs Acurácia
    ax6 = axes[1, 2]
    for idx, row in df.iterrows():
        color = colors.get(row['model_name'], '#000000')
        size = 200
        ax6.scatter(row['prediction_time_seconds'], row['accuracy'], 
                   c=color, s=size, alpha=0.9, edgecolors='black', linewidth=2)
        ax6.annotate(row['friendly_name'].split(' ')[0],  # Nome mais curto
                    (row['prediction_time_seconds'], row['accuracy']),
                    xytext=(3, 3), textcoords='offset points', fontsize=10, fontweight='bold')
    ax6.set_xlabel('Tempo de Predição (s)', fontsize=18, fontweight='bold', color='black')
    ax6.set_ylabel('Acurácia (%)', fontsize=18, fontweight='bold', color='black')
    ax6.set_title('Tempo de Predição vs Acurácia', fontsize=20, fontweight='bold', color='black')
    ax6.tick_params(axis='both', labelsize=14, colors='black')
    ax6.grid(True, alpha=0.4, color='black')
    # Usar escala log no eixo X se necessário
    max_time = df['prediction_time_seconds'].max()
    min_time = df['prediction_time_seconds'].min()
    if max_time / min_time > 100:
        ax6.set_xscale('log')
        ax6.set_xlabel('Tempo de Predição (s) - escala log', fontsize=18, fontweight='bold', color='black')
    for spine in ax6.spines.values():
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('analise_eficiencia_hdc.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_prediction_time_analysis(df):
    """Cria análises específicas do tempo de predição"""
    
    # Configurar fontes menores para economizar memória
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 12
    
    # Usar figura menor
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análise Detalhada do Tempo de Predição', fontsize=18, fontweight='bold', color='black', y=0.96)
    
    # 1. Comparação por categoria - tempo de predição
    ax1 = axes[0, 0]
    try:
        category_pred_times = df.groupby('category')['prediction_time_seconds'].mean()
        bars1 = ax1.bar(range(len(category_pred_times)), category_pred_times.values,
                        color=['#e74c3c', '#c0392b', '#3498db', '#9b59b6'], alpha=0.9, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(category_pred_times)))
        ax1.set_xticklabels(['CNN Puro', 'VGG16 Puro', 'VGG16+HDC\nHíbrido', 'HDC Puro'], 
                            fontsize=12, fontweight='bold', color='black')
        ax1.set_ylabel('Tempo Médio de Predição (s)', fontsize=14, fontweight='bold', color='black')
        ax1.set_title('Tempo de Predição por Categoria', fontsize=16, fontweight='bold', color='black')
        ax1.tick_params(axis='y', labelsize=11, colors='black')
        ax1.grid(True, alpha=0.4, axis='y', color='black')
        
        # Adicionar valores nas barras apenas se não forem muitos
        if len(category_pred_times) <= 6:
            for bar in bars1:
                h = bar.get_height()
                ax1.text(bar.get_x()+bar.get_width()/2., h*1.05, f'{h:.3f}s', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    except Exception as e:
        print(f"Erro no gráfico 1: {e}")
    
    for spine in ax1.spines.values():
        spine.set_color('black')
    
    # 2. Relação Treinamento vs Predição (escala log-log) - versão simplificada
    ax2 = axes[0, 1]
    try:
        # Limitar número de pontos para economizar memória
        df_sample = df.sample(min(10, len(df))) if len(df) > 10 else df
        
        for idx, row in df_sample.iterrows():
            color = colors.get(row['model_name'], '#000000')
            size = 150  # Tamanho menor
            ax2.scatter(row['training_time_minutes']*60, row['prediction_time_seconds'], 
                       c=color, s=size, alpha=0.9, edgecolors='black', linewidth=1.5)
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Tempo de Treinamento (s)', fontsize=14, fontweight='bold', color='black')
        ax2.set_ylabel('Tempo de Predição (s)', fontsize=14, fontweight='bold', color='black')
        ax2.set_title('Treinamento vs Predição', fontsize=16, fontweight='bold', color='black')
        ax2.tick_params(axis='both', labelsize=11, colors='black')
        ax2.grid(True, alpha=0.4, color='black')
    except Exception as e:
        print(f"Erro no gráfico 2: {e}")
        ax2.text(0.5, 0.5, 'Gráfico indisponível', transform=ax2.transAxes, ha='center')
    
    for spine in ax2.spines.values():
        spine.set_color('black')
    
    # 3. Eficiência de Predição detalhada - versão horizontal simples
    ax3 = axes[1, 0]
    try:
        df['prediction_efficiency'] = (df['accuracy'] / 100) / df['prediction_time_seconds']
        df_sorted = df.sort_values('prediction_efficiency', ascending=True)
        
        # Limitar a 10 modelos se houver muitos
        if len(df_sorted) > 10:
            df_sorted = df_sorted.tail(10)
        
        bars3 = ax3.barh(range(len(df_sorted)), df_sorted['prediction_efficiency'], 
                         color=[colors.get(name, '#000000') for name in df_sorted['model_name']],
                         alpha=0.9, edgecolor='black', linewidth=1.5)
        ax3.set_yticks(range(len(df_sorted)))
        ax3.set_yticklabels([name_mapping.get(name, name)[:15] + '...' if len(name_mapping.get(name, name)) > 15 
                            else name_mapping.get(name, name) for name in df_sorted['model_name']], 
                            fontsize=10, fontweight='bold', color='black')
        ax3.set_xlabel('Eficiência (Acurácia/segundo)', fontsize=14, fontweight='bold', color='black')
        ax3.set_title('Ranking de Eficiência', fontsize=16, fontweight='bold', color='black')
        ax3.tick_params(axis='x', labelsize=11, colors='black')
        ax3.grid(True, alpha=0.4, axis='x', color='black')
    except Exception as e:
        print(f"Erro no gráfico 3: {e}")
        ax3.text(0.5, 0.5, 'Gráfico indisponível', transform=ax3.transAxes, ha='center')
    
    for spine in ax3.spines.values():
        spine.set_color('black')
    
    # 4. Speedup simples
    ax4 = axes[1, 1]
    try:
        cnn_models = df[df['model_name'].str.contains('CNN', case=False)]
        if not cnn_models.empty:
            cnn_pred_time = cnn_models['prediction_time_seconds'].iloc[0]
            df_speedup = df.copy()
            df_speedup['prediction_speedup'] = cnn_pred_time / df_speedup['prediction_time_seconds']
            
            # Limitar número de modelos
            if len(df_speedup) > 10:
                df_speedup = df_speedup.nlargest(10, 'prediction_speedup')
            
            bars4 = ax4.bar(range(len(df_speedup)), df_speedup['prediction_speedup'], 
                            color=[colors.get(name, '#000000') for name in df_speedup['model_name']],
                            alpha=0.9, edgecolor='black', linewidth=1.5)
            ax4.set_xticks(range(len(df_speedup)))
            ax4.set_xticklabels([name_mapping.get(name, name)[:10] + '...' if len(name_mapping.get(name, name)) > 10 
                                else name_mapping.get(name, name) for name in df_speedup['model_name']], 
                                rotation=45, ha='right', fontsize=10, fontweight='bold', color='black')
            ax4.set_ylabel('Speedup vs CNN', fontsize=14, fontweight='bold', color='black')
            ax4.set_title('Speedup de Predição', fontsize=16, fontweight='bold', color='black')
            ax4.tick_params(axis='y', labelsize=11, colors='black')
            ax4.grid(True, alpha=0.4, axis='y', color='black')
            ax4.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        else:
            ax4.text(0.5, 0.5, 'CNN não encontrado\npara comparação', transform=ax4.transAxes, ha='center')
    except Exception as e:
        print(f"Erro no gráfico 4: {e}")
        ax4.text(0.5, 0.5, 'Gráfico indisponível', transform=ax4.transAxes, ha='center')
    
    for spine in ax4.spines.values():
        spine.set_color('black')
    
    plt.tight_layout()
    
    # Tentar salvar com fallback de DPI
    try:
        plt.savefig('analise_tempo_predicao_detalhada.png', dpi=200, bbox_inches='tight', facecolor='white')
    except MemoryError:
        print("Aviso: Usando DPI menor para análise detalhada devido a limitações de memória")
        plt.savefig('analise_tempo_predicao_detalhada.png', dpi=100, bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    plt.clf()  # Limpar memória
    for spine in ax1.spines.values():
        spine.set_color('black')
    
    # 2. Relação Treinamento vs Predição (escala log-log)
    ax2 = axes[0, 1]
    for idx, row in df.iterrows():
        color = colors.get(row['model_name'], '#000000')
        size = 300
        ax2.scatter(row['training_time_minutes']*60, row['prediction_time_seconds'], 
                   c=color, s=size, alpha=0.9, edgecolors='black', linewidth=2)
        # Adicionar labels apenas para alguns pontos para evitar sobreposição
        if row['model_name'] in ['HDC_RandomProj', 'HDC_NeuralHD', 'VGG16_20_epochs', 'CNN_40_epochs']:
            ax2.annotate(row['friendly_name'].split(' ')[0],
                        (row['training_time_minutes']*60, row['prediction_time_seconds']),
                        xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Tempo de Treinamento (s) - escala log', fontsize=18, fontweight='bold', color='black')
    ax2.set_ylabel('Tempo de Predição (s) - escala log', fontsize=18, fontweight='bold', color='black')
    ax2.set_title('Treinamento vs Predição', fontsize=20, fontweight='bold', color='black')
    ax2.tick_params(axis='both', labelsize=14, colors='black')
    ax2.grid(True, alpha=0.4, color='black')
    for spine in ax2.spines.values():
        spine.set_color('black')
    
    # 3. Eficiência de Predição detalhada
    ax3 = axes[1, 0]
    df_sorted = df.sort_values('prediction_efficiency', ascending=True)
    bars3 = ax3.barh(range(len(df_sorted)), df_sorted['prediction_efficiency'], 
                     color=[colors.get(name, '#000000') for name in df_sorted['model_name']],
                     alpha=0.9, edgecolor='black', linewidth=2)
    ax3.set_yticks(range(len(df_sorted)))
    ax3.set_yticklabels([name_mapping.get(name, name) for name in df_sorted['model_name']], 
                        fontsize=12, fontweight='bold', color='black')
    ax3.set_xlabel('Eficiência de Predição (Acurácia/segundo)', fontsize=18, fontweight='bold', color='black')
    ax3.set_title('Ranking de Eficiência de Predição', fontsize=20, fontweight='bold', color='black')
    ax3.tick_params(axis='x', labelsize=14, colors='black')
    ax3.grid(True, alpha=0.4, axis='x', color='black')
    
    # Usar escala log se necessário
    max_eff = df_sorted['prediction_efficiency'].max()
    min_eff = df_sorted['prediction_efficiency'].min()
    if max_eff / min_eff > 100:
        ax3.set_xscale('log')
        ax3.set_xlabel('Eficiência de Predição (log)', fontsize=18, fontweight='bold', color='black')
    
    for spine in ax3.spines.values():
        spine.set_color('black')
    
    # 4. Speedup de Predição (comparado com CNN)
    ax4 = axes[1, 1]
    cnn_pred_time = df[df['model_name'] == 'CNN_40_epochs']['prediction_time_seconds'].iloc[0]
    df_speedup = df.copy()
    df_speedup['prediction_speedup'] = cnn_pred_time / df_speedup['prediction_time_seconds']
    
    bars4 = ax4.bar(range(len(df_speedup)), df_speedup['prediction_speedup'], 
                    color=[colors.get(name, '#000000') for name in df_speedup['model_name']],
                    alpha=0.9, edgecolor='black', linewidth=2)
    ax4.set_xticks(range(len(df_speedup)))
    ax4.set_xticklabels([name_mapping.get(name, name).replace(' ', '\n') for name in df_speedup['model_name']], 
                        rotation=45, ha='right', fontsize=11, fontweight='bold', color='black')
    ax4.set_ylabel('Speedup vs CNN (vezes mais rápido)', fontsize=18, fontweight='bold', color='black')
    ax4.set_title('Speedup de Predição (base: CNN)', fontsize=20, fontweight='bold', color='black')
    ax4.tick_params(axis='y', labelsize=14, colors='black')
    ax4.grid(True, alpha=0.4, axis='y', color='black')
    ax4.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.text(len(df_speedup)/2, 1.1, 'Baseline CNN', ha='center', fontsize=12, 
            color='red', fontweight='bold')
    
    # Adicionar valores de speedup
    for bar in bars4:
        h = bar.get_height()
        ax4.text(bar.get_x()+bar.get_width()/2., h*1.05, f'{h:.1f}x', 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
    
    for spine in ax4.spines.values():
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('analise_tempo_predicao_detalhada.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_html_report(df):
    """Gera um relatório HTML com Chart.js no mesmo estilo do results.html"""
    
    # Assegurar ordenação estável: por categoria e depois por nome
    df_ord = df.copy()
    category_order = {"CNN": 0, "VGG16": 1, "VGG16_HDC_Hibrido": 2, "HDC": 3}
    df_ord["category_idx"] = df_ord["category"].map(category_order)
    df_ord = df_ord.sort_values(["category_idx", "model_name"]).reset_index(drop=True)

    # Paleta por categoria
    category_colors = {
        "CNN": "rgba(231, 76, 60, 0.95)",        # vermelho
        "VGG16": "rgba(192, 57, 43, 0.95)",      # vermelho escuro
        "VGG16_HDC_Hibrido": "rgba(39, 174, 96, 0.95)", # verde
        "HDC": "rgba(241, 196, 15, 0.95)",       # amarelo
    }

    # Dados para os bar charts (todos)
    labels = [name_mapping.get(name, name).replace(" ", "\n") for name in df_ord["model_name"].tolist()]
    energy_vals = df_ord["total_energy_kwh"].round(2).tolist()
    carbon_vals = df_ord["total_carbon_kg"].round(6).tolist()
    accuracy_vals = df_ord["accuracy"].round(2).tolist()
    prediction_vals = df_ord["prediction_time_seconds"].round(6).tolist()
    
    # Cores por categoria para os gráficos de barras
    bar_colors = []
    for _, row in df_ord.iterrows():
        cat = row["category"]
        bar_colors.append(category_colors.get(cat, "rgba(127, 127, 127, 0.95)"))

    # Construir datasets do scatter por categoria
    scatter_datasets_js = []
    for cat in ["CNN", "VGG16", "VGG16_HDC_Hibrido", "HDC"]:
        sub = df_ord[df_ord["category"] == cat]
        if sub.empty:
            continue
        points_js = ", ".join([
            "{x: %s, y: %s}" % (round(float(r["training_time_minutes"]), 2), round(float(r["accuracy"]), 2))
            for _, r in sub.iterrows()
        ])
        ds_js = "{" + \
            f"label: '{cat.replace('_', ' ')}', " + \
            f"data: [{points_js}], " + \
            f"backgroundColor: '{category_colors.get(cat, 'rgba(127, 127, 127, 0.95)')}', " + \
            "pointRadius: 14, pointBorderColor: '#ffffff', pointBorderWidth: 2}"
        
        scatter_datasets_js.append(ds_js)

    # Versão com pontos maiores
    scatter_datasets_js_larger = [ds.replace("pointRadius: 14", "pointRadius: 18") for ds in scatter_datasets_js]

    # Scatter para Predição vs Acurácia
    scatter_prediction_datasets_js = []
    for cat in ["CNN", "VGG16", "VGG16_HDC_Hibrido", "HDC"]:
        sub = df_ord[df_ord["category"] == cat]
        if sub.empty:
            continue
        points_js = ", ".join([
            "{x: %s, y: %s}" % (round(float(r["prediction_time_seconds"]), 6), round(float(r["accuracy"]), 2))
            for _, r in sub.iterrows()
        ])
        ds_js = "{" + \
            f"label: '{cat.replace('_', ' ')}', " + \
            f"data: [{points_js}], " + \
            f"backgroundColor: '{category_colors.get(cat, 'rgba(127, 127, 127, 0.95)')}', " + \
            "pointRadius: 18, pointBorderColor: '#ffffff', pointBorderWidth: 2}"
        
        scatter_prediction_datasets_js.append(ds_js)

    # Montar HTML
    html = f"""<!DOCTYPE html>
<html lang=\"pt-BR\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Relatório HDC vs CNN - Com Análise de Predição</title>
  <script src=\"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js\"></script>
  <style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin:0; padding:20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height:100vh; }}
    .container {{ max-width: 1400px; margin: 0 auto; background: rgba(255,255,255,0.95); border-radius: 20px; padding: 30px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }}
    h1 {{ text-align: center; color: #2c3e50; font-size: 2.3rem; margin-bottom: 10px; background: linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .chart-container {{ background: #f8f4f0; border-radius: 15px; padding: 25px; margin: 25px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1); }}
    .chart-title {{ font-size: 1.2rem; font-weight: bold; color: #2c3e50; margin-bottom: 12px; text-align: center; }}
    .actions {{ text-align: right; margin-top: 10px; }}
    .btn {{ background: #2c3e50; color: #fff; border: none; padding: 8px 12px; border-radius: 8px; cursor: pointer; }}
    .btn:hover {{ background: #1f2a35; }}
    .highlight {{ background: linear-gradient(135deg, #ff9a56 0%, #ff6b6b 100%); }}
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>Relatório Comparativo HDC vs CNN</h1>

    <div class=\"chart-container\">
      <div class=\"chart-title\">Comparação de Consumo Energético (kWh)</div>
      <canvas id=\"energyChart\"></canvas>
      <div class=\"actions\"><button class=\"btn\" onclick=\"downloadCanvas('energyChart','grafico_energia.png')\">Baixar PNG</button></div>
    </div>

    <div class=\"chart-container\">
      <div class=\"chart-title\">Emissão de Carbono (kg CO₂)</div>
      <canvas id=\"carbonChart\"></canvas>
      <div class=\"actions\"><button class=\"btn\" onclick=\"downloadCanvas('carbonChart','grafico_carbono.png')\">Baixar PNG</button></div>
    </div>

    <div class=\"chart-container\">
      <div class=\"chart-title\">Acurácia por Modelo (%)</div>
      <canvas id=\"accuracyChart\"></canvas>
      <div class=\"actions\"><button class=\"btn\" onclick=\"downloadCanvas('accuracyChart','grafico_acuracia.png')\">Baixar PNG</button></div>
    </div>

    <div class=\"chart-container highlight\">
      <div class=\"chart-title\">⚡ Tempo de Predição por Modelo (segundos) - NOVA ANÁLISE</div>
      <canvas id=\"predictionChart\"></canvas>
      <div class=\"actions\"><button class=\"btn\" onclick=\"downloadCanvas('predictionChart','grafico_predicao.png')\">Baixar PNG</button></div>
    </div>

    <div class=\"chart-container\">
      <div class=\"chart-title\">Tempo de Treinamento vs Acurácia (escala log no tempo)</div>
      <canvas id=\"scatterChart\"></canvas>
      <div class=\"actions\"><button class=\"btn\" onclick=\"downloadCanvas('scatterChart','tempo_vs_acuracia_log.png')\">Baixar PNG</button></div>
    </div>

    <div class=\"chart-container\">
      <div class=\"chart-title\">Tempo de Treinamento vs Acurácia (zoom 0–30 min)</div>
      <canvas id=\"scatterChartZoom\"></canvas>
      <div class=\"actions\"><button class=\"btn\" onclick=\"downloadCanvas('scatterChartZoom','tempo_vs_acuracia_zoom.png')\">Baixar PNG</button></div>
    </div>

    <div class=\"chart-container highlight\">
      <div class=\"chart-title\">⚡ Tempo de Predição vs Acurácia (escala log na predição) - NOVA ANÁLISE</div>
      <canvas id=\"scatterPredictionChart\"></canvas>
      <div class=\"actions\"><button class=\"btn\" onclick=\"downloadCanvas('scatterPredictionChart','predicao_vs_acuracia.png')\">Baixar PNG</button></div>
    </div>
  </div>

  <script>
    function downloadCanvas(canvasId, filename) {{
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      canvas.toBlob(function(blob) {{
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        link.click();
        URL.revokeObjectURL(link.href);
      }}, 'image/png');
    }}

    // Energia (barras) - cores por categoria
    const energyCtx = document.getElementById('energyChart').getContext('2d');
    new Chart(energyCtx, {{
      type: 'bar',
      data: {{
        labels: {labels},
        datasets: [{{
          label: 'Energia (kWh)',
          data: {energy_vals},
          backgroundColor: {bar_colors},
          borderColor: 'rgba(255, 255, 255, 1)',
          borderWidth: 2
        }}]
      }},
      options: {{
        responsive: true,
        backgroundColor: '#f8f4f0',
        scales: {{ 
          y: {{ 
            beginAtZero: true, 
            title: {{ display: true, text: 'Energia (kWh)', font: {{ size: 20, weight: 'bold', color: '#000000' }} }}, 
            ticks: {{ font: {{ size: 16, color: '#000000' }} }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }}, 
          x: {{ 
            ticks: {{ font: {{ size: 16, color: '#000000' }} }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }} 
        }},
        plugins: {{ legend: {{ display: false, labels: {{ font: {{ size: 18, color: '#000000' }} }} }} }}
      }}
    }});

    // Carbono (barras) - cores por categoria
    const carbonCtx = document.getElementById('carbonChart').getContext('2d');
    new Chart(carbonCtx, {{
      type: 'bar',
      data: {{
        labels: {labels},
        datasets: [{{
          label: 'CO₂ (kg)',
          data: {carbon_vals},
          backgroundColor: {bar_colors},
          borderColor: 'rgba(255, 255, 255, 1)',
          borderWidth: 2
        }}]
      }},
      options: {{
        responsive: true,
        backgroundColor: '#f8f4f0',
        scales: {{ 
          y: {{ 
            beginAtZero: true, 
            title: {{ display: true, text: 'CO₂ (kg)', font: {{ size: 20, weight: 'bold', color: '#000000' }} }}, 
            ticks: {{ font: {{ size: 16, color: '#000000' }} }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }}, 
          x: {{ 
            ticks: {{ font: {{ size: 16, color: '#000000' }} }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }} 
        }},
        plugins: {{ legend: {{ display: false, labels: {{ font: {{ size: 18, color: '#000000' }} }} }} }}
      }}
    }});

    // Acurácia (barras) - cores por categoria
    const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
    new Chart(accuracyCtx, {{
      type: 'bar',
      data: {{
        labels: {labels},
        datasets: [{{
          label: 'Acurácia (%)',
          data: {accuracy_vals},
          backgroundColor: {bar_colors},
          borderColor: 'rgba(255, 255, 255, 1)',
          borderWidth: 2
        }}]
      }},
      options: {{
        responsive: true,
        backgroundColor: '#f8f4f0',
        scales: {{ 
          y: {{ 
            beginAtZero: true, 
            max: 100,
            title: {{ display: true, text: 'Acurácia (%)', font: {{ size: 20, weight: 'bold', color: '#000000' }} }}, 
            ticks: {{ font: {{ size: 16, color: '#000000' }} }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }}, 
          x: {{ 
            ticks: {{ font: {{ size: 16, color: '#000000' }} }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }} 
        }},
        plugins: {{ legend: {{ display: false, labels: {{ font: {{ size: 18, color: '#000000' }} }} }} }}
      }}
    }});

    // NOVO: Tempo de Predição (barras) - cores por categoria
    const predictionCtx = document.getElementById('predictionChart').getContext('2d');
    new Chart(predictionCtx, {{
      type: 'bar',
      data: {{
        labels: {labels},
        datasets: [{{
          label: 'Tempo de Predição (s)',
          data: {prediction_vals},
          backgroundColor: {bar_colors},
          borderColor: 'rgba(255, 255, 255, 1)',
          borderWidth: 2
        }}]
      }},
      options: {{
        responsive: true,
        backgroundColor: '#f8f4f0',
        scales: {{ 
          y: {{ 
            type: 'logarithmic',
            title: {{ display: true, text: 'Tempo de Predição (s) [log]', font: {{ size: 20, weight: 'bold', color: '#000000' }} }}, 
            ticks: {{ 
              callback: function(value) {{ return value.toFixed(4); }},
              font: {{ size: 16, color: '#000000' }} 
            }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }}, 
          x: {{ 
            ticks: {{ font: {{ size: 16, color: '#000000' }} }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }} 
        }},
        plugins: {{ legend: {{ display: false, labels: {{ font: {{ size: 18, color: '#000000' }} }} }} }}
      }}
    }});

    // Scatter Tempo vs Acurácia por categoria (escala logarítmica no tempo)
    const scatterCtx = document.getElementById('scatterChart').getContext('2d');
    new Chart(scatterCtx, {{
      type: 'scatter',
      data: {{ datasets: [ {', '.join(scatter_datasets_js_larger)} ] }},
      options: {{
        responsive: true,
        backgroundColor: '#f8f4f0',
        scales: {{
          x: {{
            type: 'logarithmic',
            title: {{ display: true, text: 'Tempo de Treinamento (min) [log]', font: {{ size: 20, weight: 'bold', color: '#000000' }} }},
            ticks: {{ callback: (v) => v, min: 1, font: {{ size: 18, color: '#000000', weight: 'bold' }} }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }},
          y: {{ 
            title: {{ display: true, text: 'Acurácia (%)', font: {{ size: 20, weight: 'bold', color: '#000000' }} }}, 
            ticks: {{ font: {{ size: 18, color: '#000000', weight: 'bold' }} }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }}
        }},
        plugins: {{ legend: {{ display: true, position: 'top', labels: {{ font: {{ size: 18, weight: 'bold', color: '#000000' }} }} }} }}
      }}
    }});

    // Scatter com zoom linear (0–30 min)
    const scatterZoomCtx = document.getElementById('scatterChartZoom').getContext('2d');
    new Chart(scatterZoomCtx, {{
      type: 'scatter',
      data: {{ datasets: [ {', '.join(scatter_datasets_js_larger)} ] }},
      options: {{
        responsive: true,
        backgroundColor: '#f8f4f0',
        scales: {{
          x: {{
            type: 'linear',
            min: 0,
            max: 30,
            title: {{ display: true, text: 'Tempo de Treinamento (min)', font: {{ size: 20, weight: 'bold', color: '#000000' }} }},
            ticks: {{ font: {{ size: 18, color: '#000000', weight: 'bold' }} }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }},
          y: {{ 
            title: {{ display: true, text: 'Acurácia (%)', font: {{ size: 20, weight: 'bold', color: '#000000' }} }}, 
            ticks: {{ font: {{ size: 18, color: '#000000', weight: 'bold' }} }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }}
        }},
        plugins: {{ legend: {{ display: true, position: 'top', labels: {{ font: {{ size: 18, weight: 'bold', color: '#000000' }} }} }} }}
      }}
    }});

    // NOVO: Scatter Predição vs Acurácia por categoria (escala log na predição)
    const scatterPredictionCtx = document.getElementById('scatterPredictionChart').getContext('2d');
    new Chart(scatterPredictionCtx, {{
      type: 'scatter',
      data: {{ datasets: [ {', '.join(scatter_prediction_datasets_js)} ] }},
      options: {{
        responsive: true,
        backgroundColor: '#f8f4f0',
        scales: {{
          x: {{
            type: 'logarithmic',
            title: {{ display: true, text: 'Tempo de Predição (s) [log]', font: {{ size: 20, weight: 'bold', color: '#000000' }} }},
            ticks: {{ 
              callback: function(value) {{ return value.toFixed(4); }},
              font: {{ size: 18, color: '#000000', weight: 'bold' }} 
            }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }},
          y: {{ 
            title: {{ display: true, text: 'Acurácia (%)', font: {{ size: 20, weight: 'bold', color: '#000000' }} }}, 
            ticks: {{ font: {{ size: 18, color: '#000000', weight: 'bold' }} }},
            grid: {{ color: 'rgba(0, 0, 0, 0.3)' }}
          }}
        }},
        plugins: {{ legend: {{ display: true, position: 'top', labels: {{ font: {{ size: 18, weight: 'bold', color: '#000000' }} }} }} }}
      }}
    }});
  </script>
</body>
</html>"""

    Path('report_results_hdc.html').write_text(html, encoding='utf-8')
    print("Relatório HTML gerado: report_results_hdc.html")


def main():
    print("=== ANÁLISE DE RESULTADOS HDC PARA CLASSIFICAÇÃO DE UAVs ===\n")
    df = load_and_process_data()
    print(f"\nModelos carregados: {len(df)}")

    print("\nGerando gráficos PNG...")
    create_comparison_charts(df)
    create_efficiency_analysis(df)
    create_prediction_time_analysis(df)

    print("\nGerando relatório HTML estilo results.html...")
    create_html_report(df)

    # Salvar CSVs com formatação para evitar perda de precisão
    # Configurar opções de exibição para mostrar mais casas decimais
    pd.set_option('display.float_format', '{:.10f}'.format)
    
    # Criar uma cópia do DataFrame para formatação específica
    df_csv = df.copy()
    
    # Formatar colunas numéricas para garantir precisão adequada
    float_cols = df_csv.select_dtypes(include=['float64']).columns
    for col in float_cols:
        # Usar formatação científica para valores muito pequenos
        df_csv[col] = df_csv[col].apply(lambda x: f"{x:.10f}" if abs(x) < 0.001 else f"{x:.6f}")
    
    # Salvar CSV preservando a formatação
    df_csv.to_csv('resultados_hdc_processados.csv', index=False, float_format='%.10f')
    
    # Também salvar uma versão com notação científica para valores muito pequenos
    df_scientific = df.copy()
    df_scientific.to_csv('resultados_hdc_processados_cientifico.csv', index=False, float_format='%.6e')
    
    print("\nArquivos gerados:")
    print("- PNG: tempo_vs_acuracia_hdc.png")
    print("- PNG: consumo_energetico_hdc.png") 
    print("- PNG: tempo_predicao_hdc.png [NOVO]")
    print("- PNG: analise_eficiencia_hdc.png (atualizado com predição)")
    print("- PNG: analise_tempo_predicao_detalhada.png [NOVO]")
    print("- HTML: report_results_hdc.html (com gráficos de predição)")
    print("- CSV: resultados_hdc_processados.csv (formato decimal)")
    print("- CSV: resultados_hdc_processados_cientifico.csv (notação científica)")
    
    # Estatísticas de predição
    print(f"\n=== ESTATÍSTICAS DE TEMPO DE PREDIÇÃO ===")
    print(f"Menor tempo: {df['prediction_time_seconds'].min():.6f}s ({df.loc[df['prediction_time_seconds'].idxmin(), 'friendly_name']})")
    print(f"Maior tempo: {df['prediction_time_seconds'].max():.6f}s ({df.loc[df['prediction_time_seconds'].idxmax(), 'friendly_name']})")
    
    # Comparar categorias
    hdc_times = df[df['category'] == 'HDC']['prediction_time_seconds']
    cnn_times = df[df['category'] == 'CNN']['prediction_time_seconds']
    vgg16_times = df[df['category'] == 'VGG16']['prediction_time_seconds']
    hybrid_times = df[df['category'] == 'VGG16_HDC_Hibrido']['prediction_time_seconds']
    
    print(f"\nTempo médio por categoria:")
    print(f"HDC Puro: {hdc_times.mean():.6f}s ± {hdc_times.std():.6f}s")
    if not cnn_times.empty:
        print(f"CNN: {cnn_times.mean():.6f}s")
    if not vgg16_times.empty:
        print(f"VGG16: {vgg16_times.mean():.6f}s ± {vgg16_times.std():.6f}s")
    if not hybrid_times.empty:
        print(f"VGG16+HDC: {hybrid_times.mean():.6f}s ± {hybrid_times.std():.6f}s")
    
    # Speedup comparado com CNN
    if not cnn_times.empty:
        cnn_baseline = cnn_times.iloc[0]
        print(f"\nSpeedup comparado com CNN:")
        for _, row in df.iterrows():
            speedup = cnn_baseline / row['prediction_time_seconds']
            print(f"{row['friendly_name']}: {speedup:.1f}x mais rápido")


if __name__ == "__main__":
    main()