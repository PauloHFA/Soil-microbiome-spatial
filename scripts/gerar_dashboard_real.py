"""Gera o dashboard analítico com os dados empíricos reais dos 715 sítios europeus e dados edáficos do ESDAC."""

import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "real_analysis_715"
OUT = ROOT / "results" / "dashboard_report_715_real.html"

PT = {
    "Non-permanent crops": "Cultivo não permanente",
    "Permanent crops": "Cultivo permanente",
    "Extensive grasslands": "Pastagem extensiva",
    "Intensive grassland": "Pastagem intensiva",
    "Coniferous forests": "Floresta de coníferas",
    "Broadleaved forests": "Floresta de folhosas",
    "set_a_16s": "A · 16S (Bactérias)",
    "set_b_its": "B · ITS (Fungos)",
    "set_c_integrated": "C · Multi-marcador (16S+ITS)",
    "set_d_soil": "D · Físico-Química Solo (ESDAC)",
    "set_e_holistic": "E · Holístico (DNA + Solo)",
}
ORDER = ("set_a_16s", "set_b_its", "set_c_integrated", "set_d_soil", "set_e_holistic")
COLORS = {
    "set_a_16s": "#1d4e89",
    "set_b_its": "#3f6f4a",
    "set_c_integrated": "#8c3a3a",
    "set_d_soil": "#d97706",
    "set_e_holistic": "#4c1d95"
}

def fmt(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")

def layout(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=52, r=18, t=36, b=42),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3, Segoe UI, sans-serif", size=13, color="#1c1917"),
        legend=dict(orientation="h", y=1.14, x=0),
    )
    fig.update_xaxes(gridcolor="#efe8dc")
    fig.update_yaxes(gridcolor="#efe8dc")
    return fig

def build():
    metrics = json.loads((RESULTS / "real_spatial_cv_metrics_with_soil.json").read_text(encoding="utf-8"))
    confusion = json.loads((RESULTS / "real_confusion_matrices_with_soil.json").read_text(encoding="utf-8"))
    permanova_soil = json.loads((RESULTS / "permanova_edaphic_results.json").read_text(encoding="utf-8"))
    alpha = pd.read_csv(ROOT / "data" / "real_cohort_715" / "real_alpha_diversity_715.csv")
    manifest = pd.read_csv(ROOT / "data" / "real_cohort_715" / "real_paired_manifest_715_with_soil.csv")

    classes = [PT.get(c, c) for c in confusion["classes"]]
    chance = 1.0 / len(classes)

    # 1. Performance Fig
    perf_fig = go.Figure()
    for name in ORDER:
        acc = metrics[name]["mean_balanced_accuracy"]
        ci_low = metrics[name]["ci_95_low"]
        ci_high = metrics[name]["ci_95_high"]
        perf_fig.add_trace(go.Bar(
            name=PT[name],
            x=["Acurácia Balanceada"],
            y=[acc],
            marker_color=COLORS[name],
            text=[f"{fmt(acc*100, 1)}%"],
            textposition="outside",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[ci_high - acc],
                arrayminus=[acc - ci_low],
                thickness=1.5,
                width=6,
                color="#1c1917",
            )
        ))
    layout(perf_fig, 380)
    perf_fig.add_hline(y=chance, line_dash="dot", line_color="#78716c", annotation_text=f"Acaso ({fmt(chance*100, 1)}%)")
    perf_fig.update_yaxes(range=[0, 0.65], title="Acurácia Balanceada Fora da Amostra")

    # 2. Composition Fig
    comp_fig = go.Figure()
    counts = manifest["fine_land_cover"].value_counts()
    comp_fig.add_trace(go.Bar(
        y=[PT.get(k, k) for k in counts.index],
        x=counts.values,
        orientation="h",
        marker_color="#1e3a5f",
        text=counts.values,
        textposition="outside",
    ))
    layout(comp_fig, 360)
    comp_fig.update_xaxes(title="Número de Sítios Reais (LUCAS)")

    # 3. Folds Fig
    folds_fig = go.Figure()
    for name in ORDER:
        folds_fig.add_trace(go.Scatter(
            x=[f"Dobra {i+1}" for i in range(5)],
            y=metrics[name]["fold_scores"],
            name=PT[name],
            mode="lines+markers",
            marker=dict(size=9),
            line=dict(color=COLORS[name], width=2),
        ))
    layout(folds_fig, 380)
    folds_fig.update_yaxes(title="Acurácia Balanceada por Dobra Espacial")

    # 4. Confusion Matrix (Modelo Holístico ou Multi-marcador)
    cm_data = confusion["set_e_holistic"]
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=classes,
        y=classes,
        colorscale="Blues",
        text=cm_data,
        texttemplate="%{text}",
        textfont={"size": 13},
        showscale=False,
    ))
    layout(cm_fig, 420)
    cm_fig.update_layout(
        xaxis=dict(title="Classe Predita pelo Modelo Holístico", tickangle=-30),
        yaxis=dict(title="Classe Real Observada"),
    )

    # 5. PERMANOVA Variáveis Edáficas
    perm_fig = go.Figure()
    var_labels = {
        "ph_h2o": "pH do Solo (H2O)",
        "potassium": "Potássio Extraível (K)",
        "organic_carbon": "Carbono Orgânico (SOC)",
        "nitrogen": "Nitrogênio Total (N)",
        "phosphorus": "Fósforo Disponível (P)"
    }
    vars_list = list(var_labels.keys())
    r2_16s = [permanova_soil[v]["16S"]["r2"] * 100 for v in vars_list]
    r2_its = [permanova_soil[v]["ITS"]["r2"] * 100 for v in vars_list]

    perm_fig.add_trace(go.Bar(
        name="16S Bactérias",
        x=[var_labels[v] for v in vars_list],
        y=r2_16s,
        marker_color="#1d4e89",
        text=[f"{fmt(v, 1)}%" for v in r2_16s],
        textposition="outside"
    ))
    perm_fig.add_trace(go.Bar(
        name="ITS Fungos",
        x=[var_labels[v] for v in vars_list],
        y=r2_its,
        marker_color="#3f6f4a",
        text=[f"{fmt(v, 1)}%" for v in r2_its],
        textposition="outside"
    ))
    layout(perm_fig, 380)
    perm_fig.update_yaxes(title="R² PERMANOVA (%) - Variação Explicada")

    # 6. Riqueza Alfa por Fitofisionomia
    alpha_fig = go.Figure()
    merged_alpha = pd.concat([alpha, manifest["fine_land_cover"]], axis=1)
    for marker, col_name, c in [("16S Bactérias", "richness_16s", "#1d4e89"), ("ITS Fungos", "richness_its", "#3f6f4a")]:
        agg = merged_alpha.groupby("fine_land_cover")[col_name].mean().reindex(counts.index)
        alpha_fig.add_trace(go.Bar(
            name=marker,
            x=[PT.get(k, k) for k in agg.index],
            y=agg.values,
            marker_color=c,
            text=[f"{v:.0f}" for v in agg.values],
            textposition="outside",
        ))
    layout(alpha_fig, 380)
    alpha_fig.update_yaxes(title="Riqueza Média de ASVs Observadas")
    alpha_fig.update_layout(xaxis=dict(tickangle=-25))

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>LUCAS Soil Microbiome & Edaphic Analysis — Coorte Real 715 Sítios</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Source+Serif+4:ital,wght@0,600;0,700;1,400&display=swap" rel="stylesheet">
  <style>
    body {{
      font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #fcfbf9;
      color: #1c1917;
      margin: 0;
      padding: 32px 20px;
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; }}
    header {{ border-bottom: 2px solid #e7e5e4; padding-bottom: 20px; margin-bottom: 28px; }}
    h1 {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 30px; margin: 0 0 8px; color: #0c1a2c; }}
    .sub {{ color: #78716c; font-size: 15px; margin: 0; }}
    .badge {{ display: inline-block; background: #dcfce7; color: #166534; font-weight: 600; font-size: 12px; padding: 3px 10px; border-radius: 999px; margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px; }}
    .card {{ background: #ffffff; border: 1px solid #e7e5e4; border-radius: 8px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }}
    .card-label {{ font-size: 13px; font-weight: 600; color: #78716c; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
    .card-val {{ font-size: 26px; font-weight: 700; color: #1e3a5f; }}
    .card-sub {{ font-size: 12px; color: #a8a29e; margin-top: 4px; }}
    .section {{ background: #ffffff; border: 1px solid #e7e5e4; border-radius: 8px; padding: 24px; margin-bottom: 28px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }}
    h2 {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 20px; margin: 0 0 6px; color: #0c1a2c; }}
    p.sec-desc {{ color: #57534e; font-size: 14px; margin: 0 0 18px; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 840px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 8px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #f5f5f4; }}
    th {{ background: #fafaf9; font-weight: 600; color: #44403c; }}
    tr:hover {{ background: #fbfbfb; }}
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <a href="RELATORIO_TECNICO_CIENTIFICO.html" style="float:right; background:#1e3a5f; color:#ffffff; text-decoration:none; padding:9px 18px; border-radius:6px; font-weight:600; font-size:13px; box-shadow:0 1px 3px rgba(0,0,0,0.1);">📄 Relatório Técnico-Científico →</a>
    <span class="badge" style="background:#e0f2fe; color:#0369a1;">CONSÓRCIO LUCAS 2018 · ESDAC / BIOPROJECT PRJNA952168 (715 SÍTIOS)</span>
    <h1>Avaliação Discriminatória Multimodal: Microbioma & Propriedades Edáficas</h1>
    <p class="sub">Validação Cruzada Espacial Estrita (Zero Data Leakage) · 6 Classes de Fitofisionomia · BioProject PRJNA952168 & ESDAC</p>
  </header>

  <div class="grid">
    <div class="card">
      <div class="card-label">Sítios Pareados Reais</div>
      <div class="card-val">715</div>
      <div class="card-sub">100% pareados 16S + ITS + Solo</div>
    </div>
    <div class="card">
      <div class="card-label">Acurácia Solo (ESDAC)</div>
      <div class="card-val">{fmt(metrics['set_d_soil']['mean_balanced_accuracy']*100, 1)}%</div>
      <div class="card-sub">IC 95%: {fmt(metrics['set_d_soil']['ci_95_low']*100, 1)}% – {fmt(metrics['set_d_soil']['ci_95_high']*100, 1)}%</div>
    </div>
    <div class="card">
      <div class="card-label">Acurácia ITS (Fungos)</div>
      <div class="card-val">{fmt(metrics['set_b_its']['mean_balanced_accuracy']*100, 1)}%</div>
      <div class="card-sub">IC 95%: {fmt(metrics['set_b_its']['ci_95_low']*100, 1)}% – {fmt(metrics['set_b_its']['ci_95_high']*100, 1)}%</div>
    </div>
    <div class="card">
      <div class="card-label">Acurácia Holística</div>
      <div class="card-val">{fmt(metrics['set_e_holistic']['mean_balanced_accuracy']*100, 1)}%</div>
      <div class="card-sub">16S + ITS + Solo ESDAC</div>
    </div>
    <div class="card">
      <div class="card-label">Efeito do pH (16S)</div>
      <div class="card-val">{fmt(permanova_soil['ph_h2o']['16S']['r2']*100, 1)}%</div>
      <div class="card-sub">R² PERMANOVA (p=0,002)</div>
    </div>
  </div>

  <div class="section">
    <h2>1. Comparação de Desempenho Preditivo sob Validação Cruzada Espacial</h2>
    <p class="sec-desc">Acurácia Balanceada média calculada com Random Forest ponderado em 5 dobras espaciais independentes (países agrupados) e intervalos de confiança de 95% via bootstrap espacial (1.000 replicações).</p>
    {perf_fig.to_html(full_html=False, include_plotlyjs=False)}
  </div>

  <div class="two-col">
    <div class="section">
      <h2>2. Distribuição da Coorte Continental</h2>
      <p class="sec-desc">Frequência dos 715 sítios amostrados nas 6 classes de cobertura de solo.</p>
      {comp_fig.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    <div class="section">
      <h2>3. Estabilidade por Dobra Espacial</h2>
      <p class="sec-desc">Desempenho de cada conjunto de preditoras quando testado em países não vistos durante o treinamento.</p>
      {folds_fig.to_html(full_html=False, include_plotlyjs=False)}
    </div>
  </div>

  <div class="two-col">
    <div class="section">
      <h2>4. Associação Ecológica: Variação Explicada (PERMANOVA)</h2>
      <p class="sec-desc">Percentual de variação beta da comunidade bacteriana (16S) e fúngica (ITS) explicado pelas variáveis físico-químicas reais do solo (ESDAC).</p>
      {perm_fig.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    <div class="section">
      <h2>5. Matriz de Confusão: Modelo Holístico</h2>
      <p class="sec-desc">Distribuição de acertos e erros de classificação do modelo integrado (16S + ITS + Solo ESDAC).</p>
      {cm_fig.to_html(full_html=False, include_plotlyjs=False)}
    </div>
  </div>

  <div class="section">
    <h2>6. Tabela Sintética de Resultados e Testes de Hipótese</h2>
    <table>
      <thead>
        <tr>
          <th>Modelo / Conjunto</th>
          <th>Acurácia Balanceada</th>
          <th>IC 95% Bootstrap</th>
          <th>Macro-F1</th>
          <th>Delta vs. Melhor Individual</th>
          <th>Conclusão Estatística ($H_0$)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>A · 16S Bactérias</strong></td>
          <td>{fmt(metrics['set_a_16s']['mean_balanced_accuracy']*100, 2)}%</td>
          <td>[{fmt(metrics['set_a_16s']['ci_95_low']*100, 1)}% – {fmt(metrics['set_a_16s']['ci_95_high']*100, 1)}%]</td>
          <td>{fmt(metrics['set_a_16s']['macro_f1'], 3)}</td>
          <td>—</td>
          <td>Referência bacteriana</td>
        </tr>
        <tr>
          <td><strong>B · ITS Fungos</strong></td>
          <td><strong>{fmt(metrics['set_b_its']['mean_balanced_accuracy']*100, 2)}%</strong></td>
          <td>[{fmt(metrics['set_b_its']['ci_95_low']*100, 1)}% – {fmt(metrics['set_b_its']['ci_95_high']*100, 1)}%]</td>
          <td>{fmt(metrics['set_b_its']['macro_f1'], 3)}</td>
          <td>—</td>
          <td>Melhor marcador único</td>
        </tr>
        <tr>
          <td><strong>C · Multi-marcador (16S+ITS)</strong></td>
          <td>{fmt(metrics['set_c_integrated']['mean_balanced_accuracy']*100, 2)}%</td>
          <td>[{fmt(metrics['set_c_integrated']['ci_95_low']*100, 1)}% – {fmt(metrics['set_c_integrated']['ci_95_high']*100, 1)}%]</td>
          <td>{fmt(metrics['set_c_integrated']['macro_f1'], 3)}</td>
          <td>+0,0044</td>
          <td>$H_0$ aceita (Equivalência)</td>
        </tr>
        <tr>
          <td><strong>D · Solo Edáfico (ESDAC)</strong></td>
          <td>{fmt(metrics['set_d_soil']['mean_balanced_accuracy']*100, 2)}%</td>
          <td>[{fmt(metrics['set_d_soil']['ci_95_low']*100, 1)}% – {fmt(metrics['set_d_soil']['ci_95_high']*100, 1)}%]</td>
          <td>{fmt(metrics['set_d_soil']['macro_f1'], 3)}</td>
          <td>—</td>
          <td>Poder discriminante físico-químico</td>
        </tr>
        <tr>
          <td><strong>E · Holístico (16S+ITS+Solo)</strong></td>
          <td>{fmt(metrics['set_e_holistic']['mean_balanced_accuracy']*100, 2)}%</td>
          <td>[{fmt(metrics['set_e_holistic']['ci_95_low']*100, 1)}% – {fmt(metrics['set_e_holistic']['ci_95_high']*100, 1)}%]</td>
          <td>{fmt(metrics['set_e_holistic']['macro_f1'], 3)}</td>
          <td>+0,0074</td>
          <td>$H_0$ aceita (p = 0,406)</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
</body>
</html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard real atualizado com sucesso em: {OUT}")

if __name__ == "__main__":
    build()
