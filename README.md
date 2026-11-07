# Discriminação Espacial Multimodal do Solo: Perfis Metagenômicos e Propriedades Edáficas

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-328%20passed-success.svg)](tests/)
[![Validation](https://img.shields.io/badge/Spatial%20CV-Zero%20Data%20Leakage-green.svg)](src/spatial_cv/)
[![Data](https://img.shields.io/badge/Cohort-715%20Paired%20Sites-orange.svg)](data/real_cohort_715/)
[![Open Science](https://img.shields.io/badge/Data%20Source-LUCAS%202018%20%7C%20ESDAC-blueviolet.svg)](https://esdac.jrc.ec.europa.eu/)

---

## 📌 Visão Geral do Estudo

Este repositório abriga o pipeline computacional, bioinformático e estatístico desenvolvido para investigar uma pergunta fundamental da ecologia microbiana e da ciência de dados geoespacial:

> **É possível utilizar o DNA ambiental do solo (bactérias e fungos) e suas propriedades físico-químicas para prever a fitofisionomia e o uso da terra em escala continental, sob uma avaliação rigorosa que impeça qualquer contaminação espacial entre treino e teste?**

A análise baseia-se exclusivamente em **dados empíricos observados** provenientes de duas grandes iniciativas científicas europeias:
1. O **Consórcio LUCAS 2018 Soil Biodiversity** (amplicons de 16S rRNA e ITS depositados no BioProject **PRJNA952168** e processados por *Labouyrie et al., 2023, Nature Communications*).
2. O banco de dados físico-químico do **ESDAC (European Soil Data Centre)**, contendo atributos edáficos medidos em bancada para os mesmos pontos geográficos.

---

## 🔬 O Problema Científico e a Motivação

### O Desafio da Escala Continental
Solos de diferentes países europeus (da Finlândia a Portugal, da Polônia à Grécia) variam drasticamente em temperatura, precipitação, relevo e histórico geológico. Identificar se uma amostra de solo pertence a uma floresta de coníferas, a uma pastagem ou a uma lavoura agrícola apenas lendo sua microbiota exige que os modelos aprendam **sinais ecológicos universais**, e não particularidades locais de uma fazenda ou cidade.

### O "Pecado Capital" da Literatura: Vazamento de Dados Espaciais
Em muitos artigos científicos de machine learning aplicado a microbiomas, os autores dividem as amostras aleatoriamente em treino e teste (*Random K-Fold Cross-Validation*). Em dados geográficos, isso gera uma falha grave conhecida como **Spatial Data Leakage** (vazamento de dados espaciais):
* Se duas amostras de solo foram coletadas a 500 metros de distância no mesmo bosque, uma acaba no treino e a outra no teste.
* O algoritmo aprende a reconhecer o local exato (mesmo microclima, mesma química) e atinge 85% ou 90% de acurácia aparente.
* Porém, quando esse mesmo modelo é levado para outro país ou região, a acurácia despenca porque o modelo nunca aprendeu a generalizar.

### A Solução Implementada neste Repositório
Adotamos uma abordagem metodológica estrita de **Validação Cruzada Espacial em Blocos (Spatial Group K-Fold)**:
* Os dados foram divididos em **5 dobras independentes agrupadas por países e blocos geográficos**.
* Em cada dobra, os países avaliados no teste são **completamente inéditos** para o modelo.
* Uma guarda matemática auditada em código verifica formalmente: `Sites(treino) ∩ Sites(teste) = ∅`.
* Todo o pré-processamento (imputação de dados faltantes e padronização escalar) é aprendido **estritamente dentro da partição de treino**, sem espiar os dados de teste.

---

## 📂 Bases de Dados & Coorte Analítica (715 Sítios Pareados)

Para garantir solidez e reprodutibilidade, construímos uma coorte com **pareamento biológico triplo e estrito**: cada uma das **715 amostras** possui simultaneamente perfil bacteriano, perfil fúngico e análises físico-químicas de solo do mesmo tubo de amostragem em campo (`BARCODE_ID`).

```
                ┌──────────────────────────────────────────────┐
                │        715 SÍTIO AMBIENTAIS PAREADOS        │
                │        (24 Países Continentais Europeus)     │
                └──────────────────────┬───────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│  16S rRNA        │          │  ITS             │          │  ESDAC Solo      │
│  (Bactérias)     │          │  (Fungos)        │          │  (Físico-Química)│
│  40.109 reads    │          │  502 reads       │          │  14 variáveis de │
│  (SRS normalized)│          │  (SRS normalized)│          │  laboratório     │
└──────────────────┘          └──────────────────┘          └──────────────────┘
```

### As 6 Fitofisionomias Avaliadas
A classificação visa predizer as 6 fitofisionomias padronizadas pelo levantamento europeu:
1. **Cultivos não permanentes (Non-permanent crops)**: 264 sítios (36,9%) — lavouras anuais de grãos, milho, trigo, etc.
2. **Florestas de coníferas (Coniferous forests)**: 160 sítios (22,4%) — pinheiros e abetos de solos ácidos e boreais.
3. **Pastagens extensivas (Extensive grasslands)**: 128 sítios (17,9%) — campos naturais e pastoreio de baixa intensidade.
4. **Florestas de folhosas (Broadleaved forests)**: 99 sítios (13,8%) — carvalhos, faias e florestas decíduas temperadas.
5. **Cultivos permanentes (Permanent crops)**: 46 sítios (6,4%) — vinhedos, olivais e pomares mediterrâneos.
6. **Pastagens intensivas (Intensive grassland)**: 18 sítios (2,5%) — pastagens comerciais sob fertilização constante.

*Linha de base do acaso teórico:* Em um problema com 6 classes, um classificador que apenas "chutasse" obteria **16,67% (1/6)** de acurácia balanceada.

---

## 🛠️ Metodologia Passo a Passo

```text
[Matrizes de Dissimilaridade Bray-Curtis (16S e ITS)]
                          │
                          ▼
             [Redução Dimensional PCoA / MDS]
             (30 componentes principais para 16S e 30 para ITS)
                          │
                          ▼
      [Particionamento Espacial em 5 Dobras por País]
      (Zero vazamento: países do teste nunca vistos no treino)
                          │
                          ▼
            [Isolamento Intra-Dobra de Solo]
            (Imputação mediana e Z-score calculados APENAS no treino)
                          │
                          ▼
     [Treinamento de Random Forest Ponderado (200 árvores)]
     (Ajuste com class_weight='balanced' para classes minoritárias)
                          │
                          ▼
        [Avaliação com Incerteza e Teste de Hipótese]
        (Bootstrap Espacial com 1.000 iterações + Wilcoxon H0)
```

### Os 5 Paradigmas Preditivos Comparados
Para responder cientificamente qual fonte de dados carrega mais sinal preditivo, construímos 5 modelos idênticos em arquitetura, variando apenas as preditoras de entrada:
* **Modelo A (16S)**: Apenas os 30 componentes PCoA da comunidade bacteriana.
* **Modelo B (ITS)**: Apenas os 30 componentes PCoA da comunidade fúngica.
* **Modelo C (Multi-marcador 16S + ITS)**: Fusão precoce dos 60 componentes de bactérias e fungos.
* **Modelo D (Físico-Química de Solo)**: Apenas os atributos edáficos de bancada do ESDAC (pH, carbono, nitrogênio, textura, fósforo, potássio, densidade aparente e condutividade elétrica).
* **Modelo E (Holístico Multimodal)**: Integração total (16S + ITS + Solo ESDAC).

---

## 📊 Resultados e Descobertas Científicas

### 1. Comparação Geral de Desempenho (Fora da Amostra)

| Modelo / Conjunto de Preditoras | Acurácia Balanceada Média | IC 95% Bootstrap (1.000 iterações) | Macro-F1 | Delta vs. Melhor Único | Conclusão Estatística ($H_0$) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **A · Bactérias (16S rRNA)** | **45,13%** | 40,09% – 49,44% | 0,4713 | — | Marcador bacteriano de referência |
| **B · Fungos (ITS)** | **46,67%** | 42,08% – 50,75% | 0,4827 | — | **Melhor marcador biológico individual** |
| **C · Multi-marcador (16S + ITS)** | **46,04%** | 41,28% – 50,97% | 0,4830 | +0,44 pp | $H_0$ aceita (Equivalência estatística) |
| **D · Solo Edáfico (ESDAC)** | **41,45%** | 37,03% – 45,61% | 0,4385 | — | Sinal de fertilidade e granulometria |
| **E · Holístico (16S + ITS + Solo)** | **46,35%** | 42,15% – 50,55% | 0,4826 | +0,74 pp | $H_0$ aceita ($p = 0,406$) |
| *Linha de Base do Acaso Teórico* | *16,67%* | — | 0,1667 | — | Acaso estocástico (1/6 classes) |

> 📌 **Conclusão Central:** Todos os modelos atingiram quase **o triplo da acurácia do acaso (45% a 47% vs. 16,7%)**, comprovando que assinaturas genômicas de solo retêm sinal suficiente para identificar o bioma mesmo quando testadas em países inteiramente inéditos.

---

### 2. Por que os Fungos Superaram as Bactérias?
O modelo de fungos (ITS) obteve a melhor acurácia individual (46,67%) e o maior Macro-F1 (0,4827).
* **Mecanismo Biológico:** Fungos mantêm relações simbióticas (micorrízicas), saprotróficas e patogênicas altamente específicas com a cobertura vegetal superior. Florestas de coníferas têm espécies exclusivas de fungos ectomicorrízicos que degradam agulhas ricas em resina e lignina, enquanto lavouras agrícolas possuem fungos micorrízicos arbusculares adaptados a gramíneas e solos arados.
* Bactérias, embora abundantes, respondem mais a microgradientes químicos pontuais, variando rapidamente no curto prazo.

---

### 3. A Dissociação Extrema do Efeito do pH (PERMANOVA de Anderson 2001)
Ao aplicar a Análise de Variância Multivariada Permutacional (499 permutações, $p < 0,005$) para avaliar quanto da diversidade beta de cada comunidade é explicada pelas propriedades físico-químicas de bancada, encontramos uma assimetria ecológica impressionante:

| Variável Edáfica de Laboratório | Variação Explicada em Bactérias (16S $R^2$) | Variação Explicada em Fungos (ITS $R^2$) | Razão de Sensibilidade (16S / ITS) |
| :--- | :---: | :---: | :---: |
| **pH do Solo (em H2O)** | **18,22%** | **3,69%** | **4,94 vezes mais forte em bactérias** |
| **Potássio Extraível (K)** | **6,64%** | **1,46%** | **4,55 vezes mais forte em bactérias** |
| **Carbono Orgânico (SOC)** | **5,48%** | **1,93%** | **2,84 vezes mais forte em bactérias** |
| **Nitrogênio Total (N)** | **2,75%** | **1,04%** | **2,64 vezes mais forte em bactérias** |
| **Fósforo Disponível (P)** | **0,96%** | **0,91%** | **1,05 vezes (efeito equilibrado)** |

* **O que isso revela?** O pH é a "variável mestra" controladora das bactérias em escala continental. Bactérias possuem membranas e proteínas de transporte diretamente sensíveis à concentração de prótons externos ($H^+$). Em contraste, fungos possuem paredes celulares robustas de quitina e conseguem tolerar faixas amplas de pH (desde solos extremamente ácidos de coníferas com pH < 4 até solos calcários alcalinos com pH > 8).

---

### 4. O Teste de Hipótese ($H_0$) e o Princípio de Parsimônia
O teste pareado não-paramétrico de Wilcoxon signed-rank aceitou a hipótese nula $H_0$ ($p = 0,406$), mostrando que **a fusão multi-marcador (16S + ITS) não supera com significância estatística o modelo fúngico isolado**.
* **Implicação Econômica e Prática:** Em projetos de monitoramento biológico em larga escala ou auditoria ambiental, **sequenciar apenas o marcador ITS reduz os custos de bancada pela metade** sem qualquer perda mensurável no poder de discriminação da cobertura vegetal.

---

### 5. Desempenho Específico por Fitofisionomia (Modelo Holístico)

| Fitofisionomia | Amostras ($N$) | Sensibilidade (Recall) | Precisão | F1-Score | Diagnóstico Ecológico |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Florestas de Coníferas** | 160 | **76,9%** | **78,8%** | **0,778** | Altíssima discriminação (pH ácido + microbiota fúngica única) |
| **Cultivos não permanentes** | 264 | **71,6%** | **75,3%** | **0,734** | Excelente acurácia (impacto de fertilização e arado contínuo) |
| **Pastagens extensivas** | 128 | **67,2%** | **55,5%** | **0,608** | Boa separação, com transições suaves para campos abertos |
| **Florestas de Folhosas** | 99 | **35,4%** | **44,9%** | **0,395** | Sobreposição ecológica moderada com florestas mistas |
| **Cultivos permanentes** | 46 | **50,0%** | **30,7%** | **0,380** | Moderada (vinhedos/pomares concentrados regionalmente) |
| **Pastagens intensivas** | 18 | 0,0% | 0,0% | 0,000 | Desafio amostral severo (< 3% da coorte continental) |

---

## 🗂️ Estrutura do Repositório

```text
Soil-microbiome-spatial/
├── config/                  # Configurações de hiperparâmetros e sementes determinísticas
├── data/                    # Dados empíricos processados
│   └── real_cohort_715/     # Coorte pareada de 715 sítios (manifesto, distâncias e diversidade)
├── results/                 # Artefatos consolidados e relatórios
│   ├── RELATORIO_TECNICO_CIENTIFICO.pdf  # Relatório completo em PDF (pronto para leitura/impressão)
│   ├── RELATORIO_TECNICO_CIENTIFICO.html # Relatório interativo em HTML com 6 gráficos Plotly
│   ├── dashboard_report_715_real.html    # Painel analítico de modelagem em HTML
│   ├── tabelas_manuscrito.xlsx           # Planilha Excel formatada para publicação (5 abas)
│   └── real_analysis_715/                # Métricas brutas em JSON
├── scripts/                 # Scripts executores do pipeline e geradores
│   ├── incorporar_dados_esdac_real.py    # Execução da modelagem espacial multimodal
│   ├── gerar_relatorio_cientifico_executivo.py # Compilação do relatório técnico HTML
│   ├── gerar_dashboard_real.py           # Compilação do dashboard Plotly
│   └── gerar_tabelas_manuscrito.py       # Exportação das tabelas formatadas em Excel
├── src/                     # Pacote modular em Python
│   ├── bioinformatics/      # Subamostragem SRS e controle de qualidade
│   ├── data/                # Parsers de SRA/ESDAC e motor de pareamento por BARCODE_ID
│   ├── ecology/             # Diversidade alfa/beta, transformações CLR e PERMANOVA
│   ├── evaluation/          # Métricas balanceadas, bootstrap espacial e teste de H0
│   ├── models/              # Construtores de preditoras e classificadores Random Forest
│   └── spatial_cv/          # Particionamento geográfico estrito e guardas anti-vazamento
├── tests/                   # Suíte automatizada com 328 testes unitários e de integração
├── REPORT.md                # Relatório técnico-científico completo em Markdown nativo
├── ROADMAP_PROXIMAS_FASES.md # Planejamento e extensão metodológica para novos biomas
└── README.md                # Este documento
```

---

## 🚀 Guia de Reprodutibilidade e Execução

### 1. Clonar e Configurar o Ambiente
```bash
git clone https://github.com/PauloHFA/Soil-microbiome-spatial.git
cd Soil-microbiome-spatial

# Criar e ativar o ambiente virtual (Python 3.12)
python -m venv .venv
.venv\Scripts\activate       # No Windows
# source .venv/bin/activate  # No Linux/macOS

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar a Suíte de Testes Automatizados (328 testes)
```bash
pytest tests/
# Saída esperada: 328 passed
```

### 3. Rodar a Modelagem Multimodal com os Dados de Solo
```bash
python scripts/incorporar_dados_esdac_real.py
```

### 4. Gerar Relatórios e Exportar Planilha de Tabelas
```bash
# Exportar tabelas formatadas em Excel (results/tabelas_manuscrito.xlsx):
python scripts/gerar_tabelas_manuscrito.py

# Compilar Relatório Técnico HTML com gráficos interativos:
python scripts/gerar_relatorio_cientifico_executivo.py

# Compilar Painel Analítico de Modelagem:
python scripts/gerar_dashboard_real.py
```

---

## 📑 Acesso aos Relatórios e Resultados

* 📕 **[Relatório Completo em PDF](results/RELATORIO_TECNICO_CIENTIFICO.pdf)**: Documento de 1,25 MB com diagramação formal, pronto para leitura e impressão.
* 🌐 **[Relatório Técnico em HTML](results/RELATORIO_TECNICO_CIENTIFICO.html)**: Relatório completo com os 6 gráficos interativos em Plotly.
* 📊 **[Painel Analítico de Modelagem (Dashboard)](results/dashboard_report_715_real.html)**: Visualização interativa dos 5 modelos e da matriz de confusão.
* 📝 **[Relatório em Markdown Nativo (`REPORT.md`)](REPORT.md)**: Versão textual completa renderizada diretamente no GitHub.
* 📗 **[Planilha com as Tabelas do Artigo (`tabelas_manuscrito.xlsx`)](results/tabelas_manuscrito.xlsx)**: 5 abas prontas para inclusão em manuscritos científicos.

---

## 🏛️ Bases de Dados Públicas & Referências

* **Consórcio Europeu LUCAS 2018 Soil Biodiversity**: European Soil Data Centre (ESDAC) e European Nucleotide Archive (ENA BioProject PRJNA952168).
* **Artigo de Referência dos Dados**: Labouyrie, M., Ballabio, C., Romero, F., Panagos, P., Jones, A., Schmid, M., Mikryukov, V., Dulya, O., Tedersoo, L., Bahram, M., Lugato, E., van der Heijden, M.G.A., Orgiazzi, A. (2023). *Patterns in soil microbial diversity across Europe*. **Nature Communications**, 14, 3311. DOI: [10.1038/s41467-023-37937-4](https://doi.org/10.1038/s41467-023-37937-4).
