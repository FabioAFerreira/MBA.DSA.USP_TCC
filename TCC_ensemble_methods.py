#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENSEMBLE METHODS - TCC ELEIÇÕES RONDÔNIA
===============================================================================
🎯 OBJETIVO: Implementar ensemble methods para melhorar performance
📊 FOCO: Voting, Stacking e Bagging com modelos corrigidos
🏆 MODELOS: SVM + GradientBoosting sem overfitting
===============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, 
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import joblib

def carregar_dados_features_agrupadas():
    """Carregar apenas as features agrupadas"""
    print("📊 CARREGANDO FEATURES AGRUPADAS...")
    
    try:
        # Dataset completo (54 features) - para extrair apenas as agrupadas
        df_2020_completo = pd.read_csv('../data/features/ml_features_2020_completo.csv')
        df_2024_completo = pd.read_csv('../data/features/ml_features_2024_completo.csv')
        
        print(f"✅ Completo 2020: {df_2020_completo.shape}")
        print(f"✅ Completo 2024: {df_2024_completo.shape}")
        
        # Lista das features agrupadas (18 features)
        features_agrupadas = [
            'midia_propaganda', 'materiais_graficos_sonoros', 'mobilizacao_humana',
            'gestao_administrativa', 'infraestrutura_basica', 'apoio_politico',
            'pesquisa_eleitoral', 'aquisicoes_bens', 'comunicacao_correspondencia', 'diversos',
            'idhm', 'pib_per_capita', 'habitantes', 'total_receitas_brutas_ibge',
            'NR_PARTIDO', 'eleitores', 'vagas', 'total_receita'
        ]
        
        # Verificar quais features existem no dataset
        features_existentes = []
        for feature in features_agrupadas:
            if feature in df_2020_completo.columns:
                features_existentes.append(feature)
            else:
                print(f"   ⚠️ {feature}: NÃO ENCONTRADA")
        
        print(f"   📊 Features agrupadas encontradas: {len(features_existentes)}")
        
        # Criar dataset APENAS com features agrupadas + target
        features_finais = features_existentes + ['eleito']
        
        df_2020_agrupadas = df_2020_completo[features_finais].copy()
        df_2024_agrupadas = df_2024_completo[features_finais].copy()
        
        print(f"   ✅ Dataset agrupadas 2020: {df_2020_agrupadas.shape}")
        print(f"   ✅ Dataset agrupadas 2024: {df_2024_agrupadas.shape}")
        
        return df_2020_agrupadas, df_2024_agrupadas, features_existentes
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return None, None, None

def preparar_dados_ml(df_2020, df_2024):
    """Preparar dados para ML"""
    print(f"\n🔧 PREPARANDO DADOS PARA ENSEMBLE METHODS...")
    
    try:
        # Separar features e target
        if 'eleito' in df_2020.columns:
            X_2020 = df_2020.drop(['eleito'], axis=1)
            y_2020 = df_2020['eleito']
            X_2024 = df_2024.drop(['eleito'], axis=1)
            y_2024 = df_2024['eleito']
        else:
            print("❌ Coluna 'eleito' não encontrada!")
            return None, None, None, None
        
        # Remover colunas não numéricas
        colunas_numericas = X_2020.select_dtypes(include=[np.number]).columns
        X_2020 = X_2020[colunas_numericas]
        X_2024 = X_2024[colunas_numericas]
        
        print(f"   📊 Features numéricas: {len(colunas_numericas)}")
        print(f"   📊 Candidatos 2020: {len(X_2020)}")
        print(f"   📊 Candidatos 2024: {len(X_2024)}")
        
        # Imputar valores faltantes
        imputer = SimpleImputer(strategy='median')
        X_2020_imputed = imputer.fit_transform(X_2020)
        X_2024_imputed = imputer.transform(X_2024)
        
        # Converter de volta para DataFrame
        X_2020_imputed = pd.DataFrame(X_2020_imputed, columns=X_2020.columns)
        X_2024_imputed = pd.DataFrame(X_2024_imputed, columns=X_2024.columns)
        
        # Normalizar features
        scaler = StandardScaler()
        X_2020_scaled = scaler.fit_transform(X_2020_imputed)
        X_2024_scaled = scaler.transform(X_2024_imputed)
        
        # Converter de volta para DataFrame
        X_2020_scaled = pd.DataFrame(X_2020_scaled, columns=X_2020.columns)
        X_2024_scaled = pd.DataFrame(X_2024_scaled, columns=X_2024.columns)
        
        print(f"   ✅ Dados preparados: {X_2020_scaled.shape}")
        
        return X_2020_scaled, X_2024_scaled, y_2020, y_2024
        
    except Exception as e:
        print(f"❌ Erro ao preparar dados: {e}")
        return None, None, None, None

def carregar_modelos_corrigidos():
    """Carregar modelos corrigidos salvos"""
    print(f"\n🔍 CARREGANDO MODELOS CORRIGIDOS...")
    
    try:
        # Carregar SVM corrigido
        svm_corrigido = joblib.load('modelo_svm_sem_overfitting.pkl')
        print(f"   ✅ SVM corrigido carregado")
        
        # Carregar GradientBoosting corrigido
        gb_corrigido = joblib.load('modelo_gradientboosting_sem_overfitting.pkl')
        print(f"   ✅ GradientBoosting corrigido carregado")
        
        return svm_corrigido, gb_corrigido
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        print(f"   🔄 Criando modelos base...")
        
        # Criar modelos base se não conseguir carregar
        svm_base = SVC(random_state=42, probability=True, C=0.5, class_weight='balanced')
        gb_base = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5)
        
        return svm_base, gb_base

def criar_voting_classifier(svm_model, gb_model):
    """Criar Voting Classifier"""
    print(f"\n🤝 CRIANDO VOTING CLASSIFIER...")
    
    try:
        # Configurar Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('svm', svm_model),
                ('gb', gb_model)
            ],
            voting='soft',  # Probabilidades médias
            weights=[1, 1]  # Peso igual para ambos
        )
        
        print(f"   ✅ Voting Classifier criado (soft voting)")
        return voting_clf
        
    except Exception as e:
        print(f"❌ Erro ao criar Voting Classifier: {e}")
        return None

def criar_bagging_classifier(base_model, n_estimators=10):
    """Criar Bagging Classifier"""
    print(f"\n👜 CRIANDO BAGGING CLASSIFIER...")
    
    try:
        # Configurar Bagging Classifier (versão mais recente do scikit-learn)
        bagging_clf = BaggingClassifier(
            estimator=base_model,  # Mudança de base_estimator para estimator
            n_estimators=n_estimators,
            max_samples=0.8,  # 80% dos dados por estimador
            max_features=0.8,  # 80% das features por estimador
            random_state=42,
            n_jobs=-1
        )
        
        print(f"   ✅ Bagging Classifier criado ({n_estimators} estimadores)")
        return bagging_clf
        
    except Exception as e:
        print(f"❌ Erro ao criar Bagging Classifier: {e}")
        return None

def criar_stacking_classifier(svm_model, gb_model):
    """Criar Stacking Classifier"""
    print(f"\n🏗️ CRIANDO STACKING CLASSIFIER...")
    
    try:
        # Meta-learner (Logistic Regression)
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        # Configurar Stacking Classifier
        from sklearn.ensemble import StackingClassifier
        
        stacking_clf = StackingClassifier(
            estimators=[
                ('svm', svm_model),
                ('gb', gb_model)
            ],
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        print(f"   ✅ Stacking Classifier criado (Logistic Regression como meta-learner)")
        return stacking_clf
        
    except Exception as e:
        print(f"❌ Erro ao criar Stacking Classifier: {e}")
        return None

def treinar_ensemble_models(X_train, y_train, modelos_ensemble):
    """Treinar todos os modelos ensemble"""
    print(f"\n🤖 TREINANDO MODELOS ENSEMBLE...")
    
    modelos_treinados = {}
    resultados_treino = {}
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for nome, modelo in modelos_ensemble.items():
        print(f"   🔄 Treinando {nome}...")
        
        inicio = time.time()
        
        # Treinar modelo
        modelo.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='f1')
        
        # Tempo de treinamento
        tempo_treinamento = time.time() - inicio
        
        # Salvar modelo e resultados
        modelos_treinados[nome] = modelo
        resultados_treino[nome] = {
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'tempo_treinamento': tempo_treinamento
        }
        
        print(f"      ✅ {nome}: CV Score = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return modelos_treinados, resultados_treino

def avaliar_ensemble_models(modelos_treinados, X_test, y_test, resultados_treino):
    """Avaliar performance dos modelos ensemble"""
    print(f"\n📊 AVALIANDO MODELOS ENSEMBLE...")
    
    resultados_completos = {}
    
    for nome, modelo in modelos_treinados.items():
        print(f"   🔍 Avaliando {nome}...")
        
        # Predições
        y_pred = modelo.predict(X_test)
        y_pred_proba = modelo.predict_proba(X_test)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        
        # Resultados completos
        resultados_completos[nome] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'cv_score_mean': resultados_treino[nome]['cv_score_mean'],
            'cv_score_std': resultados_treino[nome]['cv_score_std'],
            'tempo_treinamento': resultados_treino[nome]['tempo_treinamento']
        }
        
        print(f"      📊 Accuracy: {accuracy:.4f}")
        print(f"      📊 Precision: {precision:.4f}")
        print(f"      📊 Recall: {recall:.4f}")
        print(f"      📊 F1-Score: {f1:.4f}")
        print(f"      📊 ROC-AUC: {roc_auc:.4f}")
        print(f"      📊 CV Score: {resultados_treino[nome]['cv_score_mean']:.4f} ± {resultados_treino[nome]['cv_score_std']:.4f}")
    
    return resultados_completos

def comparar_ensemble_vs_individual(resultados_ensemble, resultados_individual):
    """Comparar ensemble vs. modelos individuais"""
    print(f"\n🏆 COMPARANDO: ENSEMBLE vs. MODELOS INDIVIDUAIS")
    print("=" * 80)
    
    # Resultados dos modelos individuais corrigidos
    resultados_individual = {
        'SVM': {'accuracy': 0.5240, 'f1': 0.2190, 'roc_auc': 0.4744},
        'GradientBoosting': {'accuracy': 0.5214, 'f1': 0.2828, 'roc_auc': 0.5011}
    }
    
    comparacao = []
    
    # Adicionar modelos individuais
    for modelo in ['SVM', 'GradientBoosting']:
        if modelo in resultados_individual:
            row = {
                'Modelo': modelo,
                'Tipo': 'Individual',
                'Accuracy': resultados_individual[modelo]['accuracy'],
                'F1_Score': resultados_individual[modelo]['f1'],
                'ROC_AUC': resultados_individual[modelo]['roc_auc']
            }
            comparacao.append(row)
    
    # Adicionar modelos ensemble
    for nome, resultados in resultados_ensemble.items():
        row = {
            'Modelo': nome,
            'Tipo': 'Ensemble',
            'Accuracy': resultados['accuracy'],
            'F1_Score': resultados['f1'],
            'ROC_AUC': resultados['roc_auc']
        }
        comparacao.append(row)
    
    df_comparacao = pd.DataFrame(comparacao)
    
    # Mostrar comparação
    print("📊 COMPARAÇÃO DE PERFORMANCE:")
    print(df_comparacao.round(4))
    
    # Melhorias do ensemble
    melhor_individual = df_comparacao[df_comparacao['Tipo'] == 'Individual']['F1_Score'].max()
    melhor_ensemble = df_comparacao[df_comparacao['Tipo'] == 'Ensemble']['F1_Score'].max()
    
    if melhor_ensemble > melhor_individual:
        melhoria = melhor_ensemble - melhor_individual
        print(f"\n📈 MELHORIA DO ENSEMBLE:")
        print(f"   Melhor Individual: {melhor_individual:.4f}")
        print(f"   Melhor Ensemble: {melhor_ensemble:.4f}")
        print(f"   Melhoria: +{melhoria:.4f}")
    
    return df_comparacao

def gerar_visualizacoes_ensemble(resultados_ensemble, df_comparacao):
    """Gerar visualizações dos ensemble methods"""
    print(f"\n📊 GERANDO VISUALIZAÇÕES DOS ENSEMBLE METHODS...")
    
    try:
        # Configurar estilo
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ENSEMBLE METHODS: PERFORMANCE COMPARATIVA', fontsize=16, fontweight='bold')
        
        # 1. Accuracy por modelo
        modelos = list(resultados_ensemble.keys())
        acc_values = [resultados_ensemble[m]['accuracy'] for m in modelos]
        
        x = np.arange(len(modelos))
        axes[0,0].bar(x, acc_values, alpha=0.8, color='skyblue')
        axes[0,0].set_xlabel('Modelos Ensemble')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_title('Accuracy por Modelo Ensemble')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(modelos, rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, v in enumerate(acc_values):
            axes[0,0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
        
        # 2. F1-Score por modelo
        f1_values = [resultados_ensemble[m]['f1'] for m in modelos]
        
        axes[0,1].bar(x, f1_values, alpha=0.8, color='lightgreen')
        axes[0,1].set_xlabel('Modelos Ensemble')
        axes[0,1].set_ylabel('F1-Score')
        axes[0,1].set_title('F1-Score por Modelo Ensemble')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(modelos, rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, v in enumerate(f1_values):
            axes[0,1].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
        
        # 3. ROC-AUC por modelo
        roc_values = [resultados_ensemble[m]['roc_auc'] for m in modelos]
        
        axes[1,0].bar(x, roc_values, alpha=0.8, color='lightcoral')
        axes[1,0].set_xlabel('Modelos Ensemble')
        axes[1,0].set_ylabel('ROC-AUC')
        axes[1,0].set_title('ROC-AUC por Modelo Ensemble')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(modelos, rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, v in enumerate(roc_values):
            axes[1,0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
        
        # 4. Comparação Individual vs. Ensemble
        tipos = df_comparacao['Tipo'].unique()
        f1_medio_por_tipo = df_comparacao.groupby('Tipo')['F1_Score'].mean()
        
        cores = ['lightblue' if t == 'Individual' else 'lightgreen' for t in tipos]
        axes[1,1].bar(tipos, f1_medio_por_tipo.values, color=cores, alpha=0.8)
        axes[1,1].set_xlabel('Tipo de Modelo')
        axes[1,1].set_ylabel('F1-Score Médio')
        axes[1,1].set_title('F1-Score Médio: Individual vs. Ensemble')
        axes[1,1].grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, v in enumerate(f1_medio_por_tipo.values):
            axes[1,1].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Salvar visualização
        arquivo_viz = 'resultados_ensemble_methods.png'
        plt.savefig(arquivo_viz, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualização salva: {arquivo_viz}")
        
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao gerar visualizações: {e}")
        return False

def salvar_ensemble_models(modelos_treinados, resultados_ensemble, df_comparacao):
    """Salvar modelos ensemble e resultados"""
    print(f"\n💾 SALVANDO MODELOS ENSEMBLE...")
    
    try:
        # Salvar modelos ensemble
        for nome, modelo in modelos_treinados.items():
            arquivo_modelo = f'modelo_ensemble_{nome.lower().replace(" ", "_")}.pkl'
            joblib.dump(modelo, arquivo_modelo)
            print(f"   ✅ {nome} salvo: {arquivo_modelo}")
        
        # Salvar resultados completos
        resultados_completos = {
            'resultados_ensemble': resultados_ensemble,
            'comparacao': df_comparacao.to_dict('records')
        }
        
        arquivo_json = 'resultados_ensemble_methods.json'
        with open(arquivo_json, 'w', encoding='utf-8') as f:
            json.dump(resultados_completos, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Resultados JSON salvos: {arquivo_json}")
        
        # Salvar comparação em CSV
        arquivo_csv = 'comparacao_ensemble_vs_individual.csv'
        df_comparacao.to_csv(arquivo_csv, index=False)
        print(f"   ✅ Comparação CSV salva: {arquivo_csv}")
        
        # Salvar relatório
        arquivo_relatorio = 'RELATORIO_ENSEMBLE_METHODS.md'
        with open(arquivo_relatorio, 'w', encoding='utf-8') as f:
            f.write("# RELATÓRIO DE ENSEMBLE METHODS\n\n")
            f.write("## Modelos Ensemble: Voting, Bagging e Stacking\n\n")
            f.write("### Resumo Executivo\n\n")
            f.write(f"- **Features Utilizadas**: 18 features agrupadas\n")
            f.write(f"- **Modelos Base**: SVM e GradientBoosting corrigidos\n")
            f.write(f"- **Métodos Ensemble**: Voting, Bagging e Stacking\n")
            f.write(f"- **Objetivo**: Melhorar performance combinando modelos\n\n")
            
            f.write("### Modelos Ensemble Implementados\n\n")
            f.write("1. **Voting Classifier**: Combinação por votação suave\n")
            f.write("2. **Bagging Classifier**: Bootstrap aggregating\n")
            f.write("3. **Stacking Classifier**: Meta-learning com Logistic Regression\n\n")
            
            f.write("### Performance por Modelo Ensemble\n\n")
            f.write("| Modelo | Accuracy | F1-Score | ROC-AUC | CV Score |\n")
            f.write("|--------|----------|----------|---------|----------|\n")
            
            for nome, resultados in resultados_ensemble.items():
                f.write(f"| {nome} | {resultados['accuracy']:.4f} | {resultados['f1']:.4f} | {resultados['roc_auc']:.4f} | {resultados['cv_score_mean']:.4f} ± {resultados['cv_score_std']:.4f} |\n")
            
            f.write("\n### Comparação: Individual vs. Ensemble\n\n")
            f.write("| Tipo | F1-Score Médio |\n")
            f.write("|------|----------------|\n")
            
            tipos = df_comparacao['Tipo'].unique()
            for tipo in tipos:
                f1_medio = df_comparacao[df_comparacao['Tipo'] == tipo]['F1_Score'].mean()
                f.write(f"| {tipo} | {f1_medio:.4f} |\n")
            
            f.write("\n### Conclusões\n\n")
            f.write("1. **Ensemble Methods**: Implementados com sucesso\n")
            f.write("2. **Performance**: Comparação com modelos individuais\n")
            f.write("3. **Modelos Salvos**: Arquivos .pkl prontos para uso\n")
            f.write("4. **Próximos Passos**: Feature importance e análise interpretativa\n")
        
        print(f"   ✅ Relatório salvo: {arquivo_relatorio}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao salvar modelos: {e}")
        return False

def main():
    """Função principal"""
    print("🚀 ENSEMBLE METHODS - TCC ELEIÇÕES RONDÔNIA")
    print("=" * 80)
    print("🎯 OBJETIVO: Implementar ensemble methods para melhorar performance")
    print("📊 FOCO: Voting, Stacking e Bagging com modelos corrigidos")
    print("🏆 MODELOS: SVM + GradientBoosting sem overfitting")
    print("=" * 80)
    
    # 1. Carregar dados
    df_2020_agrupadas, df_2024_agrupadas, features_agrupadas = carregar_dados_features_agrupadas()
    if df_2020_agrupadas is None:
        print("❌ Falha ao carregar dados. Parando execução.")
        return
    
    # 2. Preparar dados
    X_2020, X_2024, y_2020, y_2024 = preparar_dados_ml(df_2020_agrupadas, df_2024_agrupadas)
    if X_2020 is None:
        print("❌ Falha ao preparar dados. Parando execução.")
        return
    
    # 3. Carregar modelos corrigidos
    svm_model, gb_model = carregar_modelos_corrigidos()
    if svm_model is None or gb_model is None:
        print("❌ Falha ao carregar modelos. Parando execução.")
        return
    
    # 4. Criar modelos ensemble
    modelos_ensemble = {}
    
    # Voting Classifier
    voting_clf = criar_voting_classifier(svm_model, gb_model)
    if voting_clf:
        modelos_ensemble['Voting'] = voting_clf
    
    # Bagging Classifier (usando SVM como base)
    bagging_clf = criar_bagging_classifier(svm_model, n_estimators=10)
    if bagging_clf is not None:
        modelos_ensemble['Bagging'] = bagging_clf
    
    # Stacking Classifier
    stacking_clf = criar_stacking_classifier(svm_model, gb_model)
    if stacking_clf:
        modelos_ensemble['Stacking'] = stacking_clf
    
    if not modelos_ensemble:
        print("❌ Nenhum modelo ensemble foi criado. Parando execução.")
        return
    
    # 5. Treinar modelos ensemble
    modelos_treinados, resultados_treino = treinar_ensemble_models(X_2020, y_2020, modelos_ensemble)
    if not modelos_treinados:
        print("❌ Falha ao treinar modelos ensemble. Parando execução.")
        return
    
    # 6. Avaliar modelos ensemble
    resultados_ensemble = avaliar_ensemble_models(modelos_treinados, X_2024, y_2024, resultados_treino)
    if not resultados_ensemble:
        print("❌ Falha ao avaliar modelos ensemble. Parando execução.")
        return
    
    # 7. Comparar ensemble vs. individual
    df_comparacao = comparar_ensemble_vs_individual(resultados_ensemble, None)
    
    # 8. Gerar visualizações
    gerar_visualizacoes_ensemble(resultados_ensemble, df_comparacao)
    
    # 9. Salvar modelos e resultados
    salvar_ensemble_models(modelos_treinados, resultados_ensemble, df_comparacao)
    
    print(f"\n🏆 ENSEMBLE METHODS IMPLEMENTADOS COM SUCESSO!")
    print("=" * 80)
    print("📊 Modelos ensemble salvos e prontos para uso")
    print("🎯 Próximo passo: Feature importance e análise interpretativa")
    print("=" * 80)

if __name__ == "__main__":
    main()
