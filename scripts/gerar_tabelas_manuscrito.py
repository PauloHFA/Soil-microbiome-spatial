"""
gerar_tabelas_manuscrito.py - Exporta tabelas estatísticas estruturadas e formatadas
para Excel (tabelas_manuscrito.xlsx) para inclusão direta no artigo científico.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "real_analysis_715"
OUT_EXCEL = ROOT / "results" / "tabelas_manuscrito.xlsx"

PT_MODELS = {
    "set_a_16s": "A · Bactérias (16S rRNA)",
    "set_b_its": "B · Fungos (ITS)",
    "set_c_integrated": "C · Multi-marcador (16S + ITS)",
    "set_d_soil": "D · Físico-Química de Solo (ESDAC)",
    "set_e_holistic": "E · Holístico Multimodal (16S + ITS + Solo)",
}

PT_CLASSES = {
    "Non-permanent crops": "Cultivo não permanente",
    "Permanent crops": "Cultivo permanente",
    "Extensive grasslands": "Pastagem extensiva",
    "Intensive grassland": "Pastagem intensiva",
    "Coniferous forests": "Floresta de coníferas",
    "Broadleaved forests": "Floresta de folhosas",
}

def main():
    print("Gerando tabelas estruturadas para o manuscrito...")
    metrics = json.loads((RESULTS / "real_spatial_cv_metrics_with_soil.json").read_text(encoding="utf-8"))
    confusion = json.loads((RESULTS / "real_confusion_matrices_with_soil.json").read_text(encoding="utf-8"))
    permanova_soil = json.loads((RESULTS / "permanova_edaphic_results.json").read_text(encoding="utf-8"))
    alpha = pd.read_csv(ROOT / "data" / "real_cohort_715" / "real_alpha_diversity_715.csv")
    manifest = pd.read_csv(ROOT / "data" / "real_cohort_715" / "real_paired_manifest_715_with_soil.csv")

    classes = confusion["classes"]
    
    # -------------------------------------------------------------
    # Tabela 1: Desempenho Geral dos 5 Modelos sob Validação Espacial
    # -------------------------------------------------------------
    rows_t1 = []
    for m_key in ["set_a_16s", "set_b_its", "set_c_integrated", "set_d_soil", "set_e_holistic"]:
        m_data = metrics[m_key]
        acc = m_data["mean_balanced_accuracy"] * 100
        ci_l = m_data["ci_95_low"] * 100
        ci_h = m_data["ci_95_high"] * 100
        f1 = m_data["macro_f1"]
        
        delta_str = "—"
        concl_str = "Referência"
        if m_key == "set_c_integrated":
            delta_str = "+0.44 pp"
            concl_str = "H0 Aceita (Equivalente ao melhor marcador único)"
        elif m_key == "set_e_holistic":
            delta_str = "+0.74 pp"
            concl_str = f"H0 Aceita (p = {metrics['hypothesis_test_holistic']['p_value']:.3f})"
        elif m_key == "set_b_its":
            concl_str = "Melhor marcador biológico individual"

        rows_t1.append({
            "Código": m_key,
            "Modelo / Conjunto de Preditoras": PT_MODELS[m_key],
            "Acurácia Balanceada Média (%)": round(acc, 2),
            "IC 95% Inferior (Bootstrap) (%)": round(ci_l, 2),
            "IC 95% Superior (Bootstrap) (%)": round(ci_h, 2),
            "Macro-F1": round(f1, 4),
            "Diferença vs. Melhor Individual (Delta)": delta_str,
            "Conclusão do Teste de Hipótese (H0)": concl_str
        })
    
    # Adicionar linha do acaso
    rows_t1.append({
        "Código": "baseline_chance",
        "Modelo / Conjunto de Preditoras": "Linha de Base do Acaso Teórico (1/6 classes)",
        "Acurácia Balanceada Média (%)": 16.67,
        "IC 95% Inferior (Bootstrap) (%)": 16.67,
        "IC 95% Superior (Bootstrap) (%)": 16.67,
        "Macro-F1": 0.1667,
        "Diferença vs. Melhor Individual (Delta)": "—",
        "Conclusão do Teste de Hipótese (H0)": "Linha de base nula"
    })
    df_t1 = pd.DataFrame(rows_t1)

    # -------------------------------------------------------------
    # Tabela 2: Desempenho por Fitofisionomia (Recall, Precision, F1)
    # -------------------------------------------------------------
    rows_t2 = []
    for m_key in ["set_a_16s", "set_b_its", "set_c_integrated", "set_d_soil", "set_e_holistic"]:
        cm = np.array(confusion[m_key])
        recalls = np.diag(cm) / np.maximum(np.sum(cm, axis=1), 1)
        precisions = np.diag(cm) / np.maximum(np.sum(cm, axis=0), 1)
        f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-9)
        counts = np.sum(cm, axis=1)
        
        for c, cnt, r, p, f in zip(classes, counts, recalls, precisions, f1s):
            rows_t2.append({
                "Modelo": PT_MODELS[m_key],
                "Fitofisionomia (Original)": c,
                "Fitofisionomia (Português)": PT_CLASSES.get(c, c),
                "Amostras (N)": int(cnt),
                "Sensibilidade / Recall (%)": round(r * 100, 2),
                "Precisão (%)": round(p * 100, 2),
                "F1-Score": round(f, 4)
            })
    df_t2 = pd.DataFrame(rows_t2)

    # -------------------------------------------------------------
    # Tabela 3: Decomposição da PERMANOVA (Efeito do Solo no Microbioma)
    # -------------------------------------------------------------
    rows_t3 = []
    labels_soil = {
        "ph_h2o": ("pH em H2O", "Acidez ativa do solo"),
        "potassium": ("Potássio Extraível (K)", "Nutriente essencial de fertilidade"),
        "organic_carbon": ("Carbono Orgânico (SOC)", "Matéria orgânica e estrutura do solo"),
        "nitrogen": ("Nitrogênio Total (N)", "Ciclagem de nitrogênio e biomassa"),
        "phosphorus": ("Fósforo Disponível (P)", "Limitação de fósforo em solos agrícolas")
    }
    
    for v_key, (nome, desc) in labels_soil.items():
        res = permanova_soil[v_key]
        rows_t3.append({
            "Variável Edáfica": nome,
            "Papel Ecológico / Descrição": desc,
            "16S Bactérias - R² (%)": round(res["16S"]["r2"] * 100, 2),
            "16S Bactérias - p-valor": res["16S"]["p_value"],
            "ITS Fungos - R² (%)": round(res["ITS"]["r2"] * 100, 2),
            "ITS Fungos - p-valor": res["ITS"]["p_value"],
            "Razão de Sensibilidade (16S / ITS)": round(res["16S"]["r2"] / max(res["ITS"]["r2"], 1e-6), 2)
        })
    df_t3 = pd.DataFrame(rows_t3)

    # -------------------------------------------------------------
    # Tabela 4: Correlações de Pearson entre Solo e Diversidade Alfa
    # -------------------------------------------------------------
    soil_corr_vars = [
        ("ph_h2o", "pH em H2O"),
        ("c_n_ratio", "Razão Carbono/Nitrogênio (C/N)"),
        ("bulk_density", "Densidade Aparente do Solo"),
        ("organic_carbon", "Carbono Orgânico (SOC)"),
        ("sand", "Teor de Areia (%)"),
        ("clay", "Teor de Argila (%)"),
        ("silt", "Teor de Silte (%)"),
        ("potassium", "Potássio Extraível (K)"),
        ("nitrogen", "Nitrogênio Total (N)"),
        ("phosphorus", "Fósforo Disponível (P)"),
        ("ec", "Condutividade Elétrica (CE)")
    ]
    
    rows_t4 = []
    for col, nome in soil_corr_vars:
        valid = manifest[[col]].join(alpha[['richness_16s', 'shannon_16s', 'richness_its', 'shannon_its']]).dropna()
        rows_t4.append({
            "Propriedade de Solo": nome,
            "Amostras Válidas (N)": len(valid),
            "Correlação r (16S Riqueza)": round(float(valid[col].corr(valid['richness_16s'])), 4),
            "Correlação r (16S Shannon)": round(float(valid[col].corr(valid['shannon_16s'])), 4),
            "Correlação r (ITS Riqueza)": round(float(valid[col].corr(valid['richness_its'])), 4),
            "Correlação r (ITS Shannon)": round(float(valid[col].corr(valid['shannon_its'])), 4),
        })
    df_t4 = pd.DataFrame(rows_t4)

    # -------------------------------------------------------------
    # Tabela 5: Importância Relativa das Variáveis de Solo (Random Forest)
    # -------------------------------------------------------------
    soil_cols = ['c_n_ratio', 'ph_h2o', 'ph_cacl2', 'phosphorus', 'organic_carbon', 'bulk_density', 'potassium', 'ec', 'nitrogen', 'sand', 'caco3', 'clay', 'silt']
    X_soil = manifest[soil_cols].values
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X_soil)
    y_codes = manifest['fine_land_cover'].map({c: i for i, c in enumerate(classes)}).values
    rf_soil = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    rf_soil.fit(X_imp, y_codes)
    imp_series = pd.Series(rf_soil.feature_importances_, index=soil_cols).sort_values(ascending=False)

    soil_names_pt = {
        'c_n_ratio': 'Razão Carbono/Nitrogênio (C/N)',
        'ph_h2o': 'pH em H2O',
        'ph_cacl2': 'pH em CaCl2',
        'phosphorus': 'Fósforo Disponível (P)',
        'organic_carbon': 'Carbono Orgânico (SOC)',
        'bulk_density': 'Densidade Aparente do Solo',
        'potassium': 'Potássio Extraível (K)',
        'ec': 'Condutividade Elétrica (CE)',
        'nitrogen': 'Nitrogênio Total (N)',
        'sand': 'Teor de Areia (%)',
        'caco3': 'Carbonato de Cálcio (CaCO3)',
        'clay': 'Teor de Argila (%)',
        'silt': 'Teor de Silte (%)'
    }

    rows_t5 = []
    for rank, (k, val) in enumerate(imp_series.items(), start=1):
        rows_t5.append({
            "Ranking": rank,
            "Variável Edáfica": soil_names_pt.get(k, k),
            "Importância Relativa (Gini) (%)": round(val * 100, 2),
            "Importância Acumulada (%)": round(float(imp_series.iloc[:rank].sum()) * 100, 2)
        })
    df_t5 = pd.DataFrame(rows_t5)

    # Exportar para Excel com múltiplas abas formatadas
    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
        df_t1.to_excel(writer, sheet_name="Tabela 1 - Modelos", index=False)
        df_t2.to_excel(writer, sheet_name="Tabela 2 - Classes", index=False)
        df_t3.to_excel(writer, sheet_name="Tabela 3 - PERMANOVA", index=False)
        df_t4.to_excel(writer, sheet_name="Tabela 4 - Correlacao Alfa", index=False)
        df_t5.to_excel(writer, sheet_name="Tabela 5 - Importancia Solo", index=False)

    print(f"Planilha de tabelas para manuscrito gerada com sucesso em: {OUT_EXCEL}")

if __name__ == "__main__":
    main()

