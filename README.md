# Discriminação Espacial Multimodal do Solo: Perfis Metagenômicos e Propriedades Edáficas

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-328%20passed-success.svg)](tests/)
[![Validation](https://img.shields.io/badge/Spatial%20CV-Zero%20Data%20Leakage-green.svg)](src/spatial_cv/)
[![Data](https://img.shields.io/badge/Cohort-715%20Paired%20Sites-orange.svg)](data/real_cohort_715/)

Repositório oficial do pipeline computacional desenvolvido no âmbito do projeto de pós-doutorado da **Dra. Bianca** e do **Dr. Ulrich** junto ao **Laboratório de Micologia e Metagenômica Ambiental (LAMA)** do **Centro de Biotecnologia (CBiotec)** da **Universidade Federal da Paraíba (UFPB)**.

O projeto avalia a capacidade discriminatória de comunidades bacterianas (16S rRNA), fúngicas (ITS) e atributos físico-químicos do solo (LUCAS 2018 / ESDAC) para predizer 6 fitofisionomias e coberturas de solo sob **validação cruzada espacial estrita (Zero Geographic Data Leakage)** em escala continental europeia.

---

## 🔬 Objetivos & Perguntas Científicas

1. **Capacidade Preditiva**: Perfis de DNA ambiental do solo conseguem discriminar fitofisionomias continentais quando avaliados em países e biomas nunca vistos durante o treinamento?
2. **Integração Multi-marcador ($H_0$)**: A fusão bacteriana e fúngica (16S + ITS) supera estatisticamente o melhor marcador biológico individual sob partição espacial cega?
3. **Modulação Edáfica (PERMANOVA)**: Como atributos físico-químicos reais de laboratório (pH, carbono orgânico, nitrogênio, textura, fósforo, potássio) estruturam as comunidades de bactérias versus fungos?

---

## 📊 Principais Resultados Empíricos (715 Sítios Continentais)

| Modelo / Conjunto de Preditoras | Acurácia Balanceada Média | Intervalo de Confiança 95% (Bootstrap) | Macro-F1 | Conclusão Metodológica |
| :--- | :---: | :---: | :---: | :--- |
| **A · Bactérias (16S rRNA)** | **45,1%** | 40,1% – 49,4% | 0,471 | Marcador procariótico de referência |
| **B · Fungos (ITS)** | **46,7%** | 42,1% – 50,8% | 0,483 | **Melhor marcador biológico isolado** |
| **C · Multi-marcador (16S + ITS)** | **46,0%** | 41,3% – 51,0% | 0,483 | Fusão genômica bacteriana + fúngica |
| **D · Físico-Química Solo (ESDAC)** | **41,5%** | 37,0% – 45,6% | 0,439 | Apenas variáveis edáficas (pH, C, N, textura) |
| **E · Holístico (16S + ITS + Solo)** | **46,4%** | 42,2% – 50,6% | 0,483 | Integração multimodal completa |
| *Linha de Base do Acaso Teórico* | *16,7%* | — | — | Acaso estocástico (1/6 classes) |

* **Teste de Hipótese Nula ($H_0$ aceita, $p = 0,406$)**: O modelo fúngico isolado (46,7%) é estatisticamente equivalente ao modelo integrado multi-marcador (46,0%) e ao holístico (46,4%), comprovando que fungos mantêm uma assinatura vegetacional suficiente em escala continental.
* **Associação Ecológica (PERMANOVA)**:
  * **pH do Solo**: $R^2 = 18,22\%$ ($p = 0,002$) sobre bactérias vs. $R^2 = 3,69\%$ ($p = 0,002$) sobre fungos (fungos são ~5x menos dependentes do pH).
  * **Razão C/N e Matéria Orgânica**: Principal determinante edáfico na classificação via Random Forest (11,9% de importância relativa de Gini).

---

## 🗂️ Estrutura do Repositório

```text
ProjetoPosDocBianca/
├── config/                  # Configurações globais e sementes determinísticas
├── data/                    # Bases empíricas de dados abertos
│   ├── METADATA_LUCAS18_BIODIVERSITY_885ID_COMPLETED.xlsx # Planilha oficial do ESDAC
│   ├── downloads/           # Suplementares empíricos (Nature Comms 2023)
│   └── real_cohort_715/     # Coorte pareada de 715 sítios (manifesto, distâncias e diversidade)
├── results/                 # Relatórios e artefatos analíticos
│   ├── RELATORIO_TECNICO_CIENTIFICO.html   # Relatório completo em HTML com 6 gráficos Plotly
│   ├── dashboard_report_715_real.html      # Painel analítico de modelagem interativo
│   ├── tabelas_manuscrito.xlsx             # Planilha Excel formatada para publicação (5 abas)
│   └── real_analysis_715/                  # Métricas e matrizes brutas em JSON
├── scripts/                 # Scripts executores do pipeline e geradores de relatórios
│   ├── incorporar_dados_esdac_real.py      # Execução da modelagem espacial multimodal
│   ├── gerar_relatorio_cientifico_executivo.py # Compilação do relatório técnico HTML
│   ├── gerar_dashboard_real.py             # Compilação do dashboard Plotly
│   └── gerar_tabelas_manuscrito.py         # Exportação das tabelas formatadas em Excel
├── src/                     # Pacote modular Python
│   ├── bioinformatics/      # Controle de qualidade e subamostragem SRS
│   ├── data/                # Parsers (SRA, ESDAC) e motor de pareamento por BARCODE_ID
│   ├── ecology/             # Diversidade alfa/beta, transformações CLR e PERMANOVA
│   ├── evaluation/          # Métricas balanceadas, bootstrap espacial e teste de H0
│   ├── models/              # Construtor de preditoras e classificadores Random Forest
│   └── spatial_cv/          # Particionamento geográfico estrito e verificação anti-vazamento
├── tests/                   # Suíte automatizada de testes (328 testes unitários/integração)
├── ROADMAP_PROXIMAS_FASES.md # Planejamento e extensão para biomas brasileiros
└── README.md                # Este documento
```

---

## 🚀 Guia de Reprodutibilidade e Execução

### 1. Pré-requisitos
* Python 3.12+ (64-bit)
* Ambiente virtual isolado

### 2. Instalação do Ambiente
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente (Windows)
.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 3. Execução dos Testes Automatizados (Verificação de Integridade)
```bash
pytest tests/
# Resultado esperado: 328 passed
```

### 4. Execução da Modelagem Multimodal com Dados do ESDAC
```bash
python scripts/incorporar_dados_esdac_real.py
```

### 5. Compilação dos Relatórios e Tabelas do Manuscrito
```bash
# Gerar Relatório Técnico-Científico completo em HTML:
python scripts/gerar_relatorio_cientifico_executivo.py

# Gerar Dashboard Analítico interativo:
python scripts/gerar_dashboard_real.py

# Exportar Planilha de Tabelas para o Artigo (Excel):
python scripts/gerar_tabelas_manuscrito.py
```

---

## 📑 Acesso aos Resultados Principais

* 📄 **Relatório Técnico-Científico**: Abra no navegador o arquivo [`results/RELATORIO_TECNICO_CIENTIFICO.html`](results/RELATORIO_TECNICO_CIENTIFICO.html).
* 📊 **Dashboard Interativo**: Abra no navegador o arquivo [`results/dashboard_report_715_real.html`](results/dashboard_report_715_real.html).
* 📗 **Tabelas para Publicação**: Abra no Excel o arquivo [`results/tabelas_manuscrito.xlsx`](results/tabelas_manuscrito.xlsx).

---

## 🏛️ Créditos e Filiação Institucional

* **Pesquisadores**: Dra. Bianca e Dr. Ulrich
* **Laboratório**: Laboratório de Micologia e Metagenômica Ambiental (LAMA)
* **Instituição**: Centro de Biotecnologia (CBiotec) · Universidade Federal da Paraíba (UFPB)
* **Dados Públicos**: Consórcio Europeu LUCAS 2018 (ESDAC / European Soil Data Centre) e BioProject PRJNA952168 (NCBI / ENA).

