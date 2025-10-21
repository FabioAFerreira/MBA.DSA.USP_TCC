#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEATURE IMPORTANCE - TCC ELEIÇÕES RONDÔNIA
===============================================================================
🎯 OBJETIVO: Analisar importância das 18 features agrupadas
📊 FOCO: SHAP values, feature selection e interpretabilidade
🏆 MODELOS: SVM, GradientBoosting e Ensemble corrigidos
===============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import joblib

# Tentar importar SHAP (pode não estar instalado)
try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP disponível para análise avançada")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP não disponível - usando métodos alternativos")

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
    print(f"\n🔧 PREPARANDO DADOS PARA FEATURE IMPORTANCE...")
    
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

def analisar_correlacao_features(X_train, feature_names):
    """Analisar correlação entre features"""
    print(f"\n🔗 ANALISANDO CORRELAÇÃO ENTRE FEATURES...")
    
    try:
        # Calcular matriz de correlação
        corr_matrix = pd.DataFrame(X_train, columns=feature_names).corr()
        
        # Identificar features altamente correlacionadas
        threshold = 0.8
        high_corr_features = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_features.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        print(f"   📊 Features com correlação > {threshold}: {len(high_corr_features)}")
        
        # Mostrar top 5 correlações
        if high_corr_features:
            print("   🔍 Top 5 correlações altas:")
            for i, corr in enumerate(sorted(high_corr_features, key=lambda x: abs(x['correlation']), reverse=True)[:5]):
                print(f"      {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.4f}")
        
        return corr_matrix, high_corr_features
        
    except Exception as e:
        print(f"❌ Erro ao analisar correlação: {e}")
        return None, None

def analisar_anova_features(X_train, y_train, feature_names):
    """Analisar importância usando ANOVA F-test"""
    print(f"\n📊 ANALISANDO IMPORTÂNCIA COM ANOVA F-TEST...")
    
    try:
        # Aplicar ANOVA F-test
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X_train, y_train)
        
        # Obter scores e p-values
        scores = selector.scores_
        p_values = selector.pvalues_
        
        # Criar DataFrame com resultados
        anova_results = pd.DataFrame({
            'Feature': feature_names,
            'F_Score': scores,
            'P_Value': p_values
        })
        
        # Ordenar por F-score (maior = mais importante)
        anova_results = anova_results.sort_values('F_Score', ascending=False)
        
        print("   📈 Top 10 features por F-Score:")
        for i, row in anova_results.head(10).iterrows():
            print(f"      {row['Feature']}: F={row['F_Score']:.2f}, p={row['P_Value']:.4f}")
        
        return anova_results
        
    except Exception as e:
        print(f"❌ Erro ao analisar ANOVA: {e}")
        return None

def analisar_random_forest_importance(X_train, y_train, feature_names):
    """Analisar importância usando Random Forest"""
    print(f"\n🌲 ANALISANDO IMPORTÂNCIA COM RANDOM FOREST...")
    
    try:
        # Treinar Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Obter feature importance
        importance = rf.feature_importances_
        
        # Criar DataFrame com resultados
        rf_results = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Ordenar por importância
        rf_results = rf_results.sort_values('Importance', ascending=False)
        
        print("   📈 Top 10 features por importância:")
        for i, row in rf_results.head(10).iterrows():
            print(f"      {row['Feature']}: {row['Importance']:.4f}")
        
        return rf_results, rf
        
    except Exception as e:
        print(f"❌ Erro ao analisar Random Forest: {e}")
        return None, None

def analisar_shap_values(X_train, y_train, feature_names):
    """Analisar importância usando SHAP values"""
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP não disponível - pulando análise SHAP")
        return None, None
    
    print(f"\n✨ ANALISANDO IMPORTÂNCIA COM SHAP VALUES...")
    
    try:
        # Treinar Random Forest para SHAP
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Calcular SHAP values
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_train.iloc[:100])  # Usar apenas uma amostra
        
        # Para classificação binária, usar shap_values[1] (classe positiva)
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # Calcular importância média por feature
        feature_importance = np.abs(shap_values).mean(0)
        
        # Criar DataFrame com resultados
        shap_results = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': feature_importance
        })
        
        # Ordenar por importância
        shap_results = shap_results.sort_values('SHAP_Importance', ascending=False)
        
        print("   📈 Top 10 features por SHAP Importance:")
        for i, row in shap_results.head(10).iterrows():
            print(f"      {row['Feature']}: {row['SHAP_Importance']:.4f}")
        
        return shap_results, explainer
        
    except Exception as e:
        print(f"❌ Erro ao analisar SHAP: {e}")
        return None, None

def analisar_rfe_features(X_train, y_train, feature_names):
    """Analisar importância usando Recursive Feature Elimination"""
    print(f"\n🔄 ANALISANDO IMPORTÂNCIA COM RFE...")
    
    try:
        # Usar Random Forest como estimador base
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Aplicar RFE
        rfe = RFE(estimator=rf, n_features_to_select=10, step=1)
        rfe.fit(X_train, y_train)
        
        # Obter ranking das features
        feature_ranking = rfe.ranking_
        feature_selected = rfe.support_
        
        # Criar DataFrame com resultados
        rfe_results = pd.DataFrame({
            'Feature': feature_names,
            'Ranking': feature_ranking,
            'Selected': feature_selected
        })
        
        # Ordenar por ranking (1 = mais importante)
        rfe_results = rfe_results.sort_values('Ranking')
        
        print("   📈 Top 10 features por RFE Ranking:")
        for i, row in rfe_results.head(10).iterrows():
            status = "✅" if row['Selected'] else "❌"
            print(f"      {status} {row['Feature']}: Ranking {row['Ranking']}")
        
        return rfe_results
        
    except Exception as e:
        print(f"❌ Erro ao analisar RFE: {e}")
        return None

def consolidar_importancia_features(anova_results, rf_results, shap_results, rfe_results):
    """Consolidar resultados de diferentes métodos"""
    print(f"\n🏆 CONSOLIDANDO IMPORTÂNCIA DAS FEATURES...")
    
    try:
        # Criar DataFrame consolidado
        features_consolidadas = pd.DataFrame()
        
        # Adicionar resultados de cada método
        if anova_results is not None:
            features_consolidadas['Feature'] = anova_results['Feature']
            features_consolidadas['ANOVA_Rank'] = range(1, len(anova_results) + 1)
            features_consolidadas['ANOVA_F_Score'] = anova_results['F_Score']
        
        if rf_results is not None:
            # Mapear features do RF para o DataFrame consolidado
            rf_dict = dict(zip(rf_results['Feature'], rf_results['Importance']))
            features_consolidadas['RF_Importance'] = features_consolidadas['Feature'].map(rf_dict)
            features_consolidadas['RF_Rank'] = features_consolidadas['RF_Importance'].rank(ascending=False)
        
        if shap_results is not None:
            # Mapear features do SHAP para o DataFrame consolidado
            shap_dict = dict(zip(shap_results['Feature'], shap_results['SHAP_Importance']))
            features_consolidadas['SHAP_Importance'] = features_consolidadas['Feature'].map(shap_dict)
            features_consolidadas['SHAP_Rank'] = features_consolidadas['SHAP_Importance'].rank(ascending=False)
        
        if rfe_results is not None:
            # Mapear features do RFE para o DataFrame consolidado
            rfe_dict = dict(zip(rfe_results['Feature'], rfe_results['Ranking']))
            features_consolidadas['RFE_Ranking'] = features_consolidadas['Feature'].map(rfe_dict)
            features_consolidadas['RFE_Rank'] = features_consolidadas['RFE_Ranking'].rank()
        
        # Calcular ranking médio
        rank_columns = [col for col in features_consolidadas.columns if 'Rank' in col]
        if rank_columns:
            features_consolidadas['Ranking_Medio'] = features_consolidadas[rank_columns].mean(axis=1)
            features_consolidadas = features_consolidadas.sort_values('Ranking_Medio')
        
        print("   📊 Ranking consolidado das features:")
        for i, row in features_consolidadas.head(10).iterrows():
            print(f"      {i+1}. {row['Feature']}: Ranking médio {row['Ranking_Medio']:.2f}")
        
        return features_consolidadas
        
    except Exception as e:
        print(f"❌ Erro ao consolidar features: {e}")
        return None

def selecionar_features_otimas(features_consolidadas, n_features=10):
    """Selecionar features mais importantes"""
    print(f"\n🎯 SELECIONANDO FEATURES MAIS IMPORTANTES...")
    
    try:
        # Selecionar top N features
        top_features = features_consolidadas.head(n_features)['Feature'].tolist()
        
        print(f"   ✅ Top {n_features} features selecionadas:")
        for i, feature in enumerate(top_features, 1):
            print(f"      {i}. {feature}")
        
        return top_features
        
    except Exception as e:
        print(f"❌ Erro ao selecionar features: {e}")
        return None

def avaliar_performance_features_selecionadas(X_train, X_test, y_train, y_test, top_features, feature_names):
    """Avaliar performance com features selecionadas"""
    print(f"\n📊 AVALIANDO PERFORMANCE COM FEATURES SELECIONADAS...")
    
    try:
        # Encontrar índices das features selecionadas
        feature_indices = [i for i, name in enumerate(feature_names) if name in top_features]
        
        # Selecionar apenas as features importantes
        X_train_selected = X_train.iloc[:, feature_indices]
        X_test_selected = X_test.iloc[:, feature_indices]
        
        print(f"   📊 Features selecionadas: {len(top_features)}")
        print(f"   📊 Dimensões reduzidas: {X_train_selected.shape[1]}")
        
        # Treinar Random Forest com features selecionadas
        rf_selected = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_selected.fit(X_train_selected, y_train)
        
        # Avaliar performance
        y_pred = rf_selected.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"   📈 Performance com features selecionadas:")
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'n_features': len(top_features),
            'features_selecionadas': top_features
        }
        
    except Exception as e:
        print(f"❌ Erro ao avaliar performance: {e}")
        return None

def gerar_visualizacoes_importance(anova_results, rf_results, shap_results, features_consolidadas):
    """Gerar visualizações da análise de feature importance"""
    print(f"\n📊 GERANDO VISUALIZAÇÕES DA FEATURE IMPORTANCE...")
    
    try:
        # Configurar estilo
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FEATURE IMPORTANCE: ANÁLISE COMPARATIVA', fontsize=16, fontweight='bold')
        
        # 1. ANOVA F-Scores
        if anova_results is not None:
            top_10_anova = anova_results.head(10)
            axes[0,0].barh(range(len(top_10_anova)), top_10_anova['F_Score'])
            axes[0,0].set_yticks(range(len(top_10_anova)))
            axes[0,0].set_yticklabels(top_10_anova['Feature'])
            axes[0,0].set_xlabel('F-Score')
            axes[0,0].set_title('Top 10 Features - ANOVA F-Test')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Random Forest Importance
        if rf_results is not None:
            top_10_rf = rf_results.head(10)
            axes[0,1].barh(range(len(top_10_rf)), top_10_rf['Importance'])
            axes[0,1].set_yticks(range(len(top_10_rf)))
            axes[0,1].set_yticklabels(top_10_rf['Feature'])
            axes[0,1].set_xlabel('Importance')
            axes[0,1].set_title('Top 10 Features - Random Forest')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. SHAP Importance (se disponível)
        if shap_results is not None:
            top_10_shap = shap_results.head(10)
            axes[1,0].barh(range(len(top_10_shap)), top_10_shap['SHAP_Importance'])
            axes[1,0].set_yticks(range(len(top_10_shap)))
            axes[1,0].set_yticklabels(top_10_shap['Feature'])
            axes[1,0].set_xlabel('SHAP Importance')
            axes[1,0].set_title('Top 10 Features - SHAP Values')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Ranking Consolidado
        if features_consolidadas is not None:
            top_10_consolidado = features_consolidadas.head(10)
            axes[1,1].barh(range(len(top_10_consolidado)), top_10_consolidado['Ranking_Medio'])
            axes[1,1].set_yticks(range(len(top_10_consolidado)))
            axes[1,1].set_yticklabels(top_10_consolidado['Feature'])
            axes[1,1].set_xlabel('Ranking Médio')
            axes[1,1].set_title('Top 10 Features - Ranking Consolidado')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar visualização
        arquivo_viz = 'feature_importance_analysis.png'
        plt.savefig(arquivo_viz, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualização salva: {arquivo_viz}")
        
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao gerar visualizações: {e}")
        return False

def salvar_resultados_importance(anova_results, rf_results, shap_results, rfe_results, 
                               features_consolidadas, performance_results):
    """Salvar resultados da análise de feature importance"""
    print(f"\n💾 SALVANDO RESULTADOS DA FEATURE IMPORTANCE...")
    
    try:
        # Salvar resultados em CSV
        if anova_results is not None:
            anova_results.to_csv('anova_feature_importance.csv', index=False)
            print(f"   ✅ ANOVA results salvo: anova_feature_importance.csv")
        
        if rf_results is not None:
            rf_results.to_csv('randomforest_feature_importance.csv', index=False)
            print(f"   ✅ Random Forest results salvo: randomforest_feature_importance.csv")
        
        if shap_results is not None:
            shap_results.to_csv('shap_feature_importance.csv', index=False)
            print(f"   ✅ SHAP results salvo: shap_feature_importance.csv")
        
        if rfe_results is not None:
            rfe_results.to_csv('rfe_feature_importance.csv', index=False)
            print(f"   ✅ RFE results salvo: rfe_feature_importance.csv")
        
        if features_consolidadas is not None:
            features_consolidadas.to_csv('feature_importance_consolidado.csv', index=False)
            print(f"   ✅ Ranking consolidado salvo: feature_importance_consolidado.csv")
        
        # Salvar relatório
        arquivo_relatorio = 'RELATORIO_FEATURE_IMPORTANCE.md'
        with open(arquivo_relatorio, 'w', encoding='utf-8') as f:
            f.write("# RELATÓRIO DE FEATURE IMPORTANCE\n\n")
            f.write("## Análise das 18 Features Agrupadas\n\n")
            f.write("### Resumo Executivo\n\n")
            f.write(f"- **Features Analisadas**: 18 features agrupadas\n")
            f.write(f"- **Métodos Utilizados**: ANOVA, Random Forest, SHAP, RFE\n")
            f.write(f"- **Objetivo**: Identificar features mais relevantes\n\n")
            
            if features_consolidadas is not None:
                f.write("### Top 10 Features por Ranking Consolidado\n\n")
                f.write("| Rank | Feature | Ranking Médio |\n")
                f.write("|------|---------|---------------|\n")
                
                for i, row in features_consolidadas.head(10).iterrows():
                    f.write(f"| {i+1} | {row['Feature']} | {row['Ranking_Medio']:.2f} |\n")
            
            if performance_results is not None:
                f.write(f"\n### Performance com Features Selecionadas\n\n")
                f.write(f"- **Número de Features**: {performance_results['n_features']}\n")
                f.write(f"- **Accuracy**: {performance_results['accuracy']:.4f}\n")
                f.write(f"- **F1-Score**: {performance_results['f1_score']:.4f}\n")
                f.write(f"- **Features Selecionadas**: {', '.join(performance_results['features_selecionadas'])}\n")
            
            f.write("\n### Conclusões\n\n")
            f.write("1. **Feature Importance**: Análise completa realizada\n")
            f.write("2. **Seleção**: Features mais relevantes identificadas\n")
            f.write("3. **Performance**: Avaliação com features selecionadas\n")
            f.write("4. **Próximos Passos**: Otimização final dos modelos\n")
        
        print(f"   ✅ Relatório salvo: {arquivo_relatorio}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao salvar resultados: {e}")
        return False

def main():
    """Função principal"""
    print("🚀 FEATURE IMPORTANCE - TCC ELEIÇÕES RONDÔNIA")
    print("=" * 80)
    print("🎯 OBJETIVO: Analisar importância das 18 features agrupadas")
    print("📊 FOCO: SHAP values, feature selection e interpretabilidade")
    print("🏆 MODELOS: SVM, GradientBoosting e Ensemble corrigidos")
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
    
    # 3. Análise de correlação
    corr_matrix, high_corr_features = analisar_correlacao_features(X_2020, features_agrupadas)
    
    # 4. Análise ANOVA
    anova_results = analisar_anova_features(X_2020, y_2020, features_agrupadas)
    
    # 5. Análise Random Forest
    rf_results, rf_model = analisar_random_forest_importance(X_2020, y_2020, features_agrupadas)
    
    # 6. Análise SHAP
    shap_results, shap_explainer = analisar_shap_values(X_2020, y_2020, features_agrupadas)
    
    # 7. Análise RFE
    rfe_results = analisar_rfe_features(X_2020, y_2020, features_agrupadas)
    
    # 8. Consolidar resultados
    features_consolidadas = consolidar_importancia_features(anova_results, rf_results, shap_results, rfe_results)
    
    # 9. Selecionar features ótimas
    top_features = selecionar_features_otimas(features_consolidadas, n_features=10)
    
    # 10. Avaliar performance com features selecionadas
    performance_results = avaliar_performance_features_selecionadas(
        X_2020, X_2024, y_2020, y_2024, top_features, features_agrupadas
    )
    
    # 11. Gerar visualizações
    gerar_visualizacoes_importance(anova_results, rf_results, shap_results, features_consolidadas)
    
    # 12. Salvar resultados
    salvar_resultados_importance(anova_results, rf_results, shap_results, rfe_results, 
                               features_consolidadas, performance_results)
    
    print(f"\n🏆 FEATURE IMPORTANCE ANALISADA COM SUCESSO!")
    print("=" * 80)
    print("📊 Features mais relevantes identificadas")
    print("🎯 Próximo passo: Otimização final dos modelos")
    print("=" * 80)

if __name__ == "__main__":
    main()
