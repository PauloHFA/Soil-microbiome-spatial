# Discriminação Espacial Multimodal do Solo: Perfis Metagenômicos e Propriedades Edáficas

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-328%20passed-success.svg)](tests/)
[![Validation](https://img.shields.io/badge/Spatial%20CV-Zero%20Data%20Leakage-green.svg)](src/spatial_cv/)
[![Data](https://img.shields.io/badge/Cohort-715%20Paired%20Sites-orange.svg)](data/real_cohort_715/)

Repositório com o pipeline computacional e estatístico desenvolvido para avaliação do potencial discriminatório de comunidades bacterianas (16S rRNA), fúngicas (ITS) e atributos físico-químicos de solo na classificação de fitofisionomias continentais sob regime estrito de **Validação Cruzada Espacial (Spatial Group K-Fold / Zero Geographic Data Leakage)**.

A investigação utiliza dados empíricos observados a partir de bases públicas consolidadas: o **Consórcio Europeu LUCAS 2018 Soil Biodiversity**, metadados edáficos do **ESDAC (European Soil Data Centre)** e o BioProject **PRJNA952168** (ENA/NCBI).

---

## 🔬 Perguntas de Pesquisa

1. **Capacidade Preditiva**: Perfis de DNA ambiental do solo conseguem discriminar categorias de cobertura vegetal em áreas e países geograficamente inéditos (não vistos no treinamento)?
2. **Integração Multi-marcador ($H_0$)**: A fusão simultânea de bactérias e fungos (16S + ITS) é estatisticamente superior ao melhor marcador biológico individual sob validação espacial cega?
3. **Modulação Físico-Química (PERMANOVA)**: Como atributos edáficos mensurados em bancada (pH, carbono orgânico, nitrogênio, textura, fósforo, potássio) estruturam diferencialmente as comunidades de bactérias versus fungos?

---

## 📊 Síntese dos Resultados Observados (715 Sítios Continentais)

Sob validação cruzada espacial em 5 dobras independentes (agrupadas por país/bloco geográfico, sem qualquer sobreposição espacial entre treino e teste), foram avaliados 5 paradigmas preditivos:

| Modelo / Conjunto de Preditoras | Acurácia Balanceada Média | Intervalo de Confiança 95% (Bootstrap) | Macro-F1 | Diferença ($\Delta$) vs. Melhor Único | Conclusão Estatística ($H_0$) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **A · Bactérias (16S rRNA)** | **45,13%** | 40,09% – 49,44% | 0,4713 | — | Marcador bacteriano de referência |
| **B · Fungos (ITS)** | **46,67%** | 42,08% – 50,75% | 0,4827 | — | **Melhor marcador biológico individual** |
| **C · Multi-marcador (16S + ITS)** | **46,04%** | 41,28% – 50,97% | 0,4830 | +0,44 pp | $H_0$ aceita (Equivalência estatística) |
| **D · Físico-Química Solo (ESDAC)** | **41,45%** | 37,03% – 45,61% | 0,4385 | — | Poder discriminante de atributos de bancada |
| **E · Holístico (16S + ITS + Solo)** | **46,35%** | 42,15% – 50,55% | 0,4826 | +0,74 pp | $H_0$ aceita ($p = 0,406$) |
| *Linha de Base do Acaso Teórico* | *16,67%* | — | 0,1667 | — | Acaso estocástico (1/6 classes) |

### Destaques Ecológicos e Estatísticos:
* **Princípio de Parsimônia ($H_0$ aceita)**: O teste pareado de Wilcoxon signed-rank demonstrou que a integração de marcadores não supera estatisticamente o modelo fúngico isolado ($p = 0,406$). O marcador ITS isolado captura a quase totalidade da assinatura vegetal necessária.
* **Associação Multivariada (PERMANOVA de Anderson 2001, $p < 0,005$)**:
  * **pH do Solo**: Explica **18,22%** da variação beta da comunidade bacteriana (16S), mas apenas **3,69%** da fúngica (ITS). As bactérias são aproximadamente 5 vezes mais sensíveis ao gradiente de acidez do que os fungos.
  * **Potássio Extraível (K)**: Explica 6,64% em 16S vs. 1,46% em ITS.
  * **Carbono Orgânico (SOC)**: Explica 5,48% em 16S vs. 1,93% em ITS.
* **Importância Relativa no Random Forest**: A razão Carbono/Nitrogênio (C/N, 11,86%) e o pH (10,13%) foram os atributos edáficos mais importantes na classificação de fitofisionomias.

> 📖 **Para conferir a discussão aprofundada, as matrizes de confusão e os dados por fitofisionomia, consulte o [Relatório Técnico em Markdown (`REPORT.md`)](REPORT.md).**

---

## 🗂️ Estrutura do Repositório

```text
Soil-microbiome-spatial/
├── config/                  # Configurações globais e esquemas de variáveis
├── data/                    # Bases observadas processadas
│   └── real_cohort_715/     # Coorte pareada de 715 sítios (manifesto, distâncias e diversidade)
├── results/                 # Resultados estruturados
│   ├── tabelas_manuscrito.xlsx # Planilha consolidada de tabelas formatadas (5 abas)
│   └── real_analysis_715/   # Métricas e matrizes brutas em JSON
├── scripts/                 # Scripts executores do pipeline
│   ├── incorporar_dados_esdac_real.py # Execução da modelagem espacial multimodal
│   └── gerar_tabelas_manuscrito.py    # Exportação das tabelas estatísticas em Excel
├── src/                     # Pacote modular em Python
│   ├── bioinformatics/      # Subamostragem SRS e controle de qualidade
│   ├── data/                # Parsers (SRA, ESDAC) e pareamento por BARCODE_ID
│   ├── ecology/             # Diversidade alfa/beta, transformações CLR e PERMANOVA
│   ├── evaluation/          # Métricas balanceadas, bootstrap espacial e teste de H0
│   ├── models/              # Construtores de preditoras e classificadores Random Forest
│   └── spatial_cv/          # Particionamento geográfico estrito e guardas anti-vazamento
├── tests/                   # Suíte automatizada de testes (328 testes unitários e de integração)
├── REPORT.md                # Relatório técnico-científico completo em Markdown
├── ROADMAP_PROXIMAS_FASES.md # Planejamento e extensão metodológica
└── README.md                # Este documento
```

---

## 🚀 Guia de Execução e Reprodutibilidade

### 1. Instalação do Ambiente
```bash
# Criar e ativar ambiente virtual
python -m venv .venv
.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Verificação de Integridade (Testes Automatizados)
```bash
pytest tests/
# Resultado esperado: 328 passed
```

### 3. Execução da Modelagem Multimodal com Dados Edáficos
```bash
python scripts/incorporar_dados_esdac_real.py
```

### 4. Exportação das Tabelas Formatadas
```bash
python scripts/gerar_tabelas_manuscrito.py
# Gera o arquivo: results/tabelas_manuscrito.xlsx
```

---

## 📑 Acesso aos Resultados Principais

* 📄 **Relatório Técnico Completo**: Consulte diretamente no repositório o arquivo [`REPORT.md`](REPORT.md).
* 📗 **Tabelas para Artigo (Excel)**: Planilha formatada pronta para publicação em [`results/tabelas_manuscrito.xlsx`](results/tabelas_manuscrito.xlsx).

---

## 🏛️ Dados Públicos & Citações de Referência

* **Consórcio LUCAS 2018 Soil Biodiversity**: European Soil Data Centre (ESDAC) e European Nucleotide Archive (ENA BioProject PRJNA952168).
* **Estudo Empírico de Referência**: Labouyrie, M. et al. (2023). *Patterns in soil microbial diversity across Europe*. Nature Communications, 14, 3311.
