# Roadmap Metodológico: Conclusão da Fase 1 e Planejamento das Fases Posteriores

Documento técnico de planejamento metodológico para a esteira de bioinformática e modelagem espacial de microbiomas e atributos edáficos de solo a partir de dados abertos observados.

---

## 1. Conquistas e Entregas Consolidadas na Fase 1 (Europa / LUCAS / ESDAC)

A presente fase conclui com êxito o estabelecimento da esteira analítica e a validação empírica continental:

- [x] **Ingestão e Auditoria de Dados Públicos Reais:**
  - Pareamento biológico estrito triplo de **715 sítios continentais** em 24 países europeus via `BARCODE_ID`.
  - Ingestão de matrizes de dissimilaridade de Bray-Curtis padronizadas por SRS (40.109 leituras para 16S e 502 leituras para ITS) a partir do estudo de referência (*Labouyrie et al., 2023, Nature Communications*).
  - Ingestão da planilha oficial do ESDAC (`METADATA_LUCAS18_BIODIVERSITY_885ID_COMPLETED.xlsx`) cobrindo 18 variáveis físico-químicas de bancada (pH, textura, matéria orgânica, C/N, P, K, condutividade e densidade).
- [x] **Validação Cruzada Espacial Estrita (Zero Geographic Data Leakage):**
  - Implementação de particionamento geoespacial em 5 dobras agrupadas por país/região (`SpatialGroupKFold`).
  - Verificação formal de que nenhum sítio ou país do teste jamais esteve no treino (`Sites(treino) ∩ Sites(teste) = ∅`).
  - Pré-processamento intra-dobra rigoroso (imputação mediana e escala Z-score aprendidas estritamente na partição de treino).
- [x] **Modelagem Multimodal Preditiva (5 Paradigmas Comparados):**
  - **16S Bactérias**: 45,1% de acurácia balanceada (Macro-F1 = 0,471).
  - **ITS Fungos**: 46,7% de acurácia balanceada (Macro-F1 = 0,483) — **Melhor modelo individual**.
  - **Multi-marcador (16S + ITS)**: 46,0% de acurácia balanceada (Macro-F1 = 0,483).
  - **Solo Edáfico Puro**: 41,5% de acurácia balanceada (Macro-F1 = 0,439).
  - **Holístico Multimodal (16S + ITS + Solo)**: 46,4% de acurácia balanceada (Macro-F1 = 0,483).
  - Todos os modelos superaram o nível do acaso teórico (16,7%) em quase 3 vezes.
- [x] **Inferência Estatística e Ecológica Formal:**
  - Intervalos de confiança de 95% via bootstrap espacial (1.000 replicações independentes).
  - Teste da hipótese nula $H_0$ via Wilcoxon pareado ($p = 0,406$, aceitando a equivalência estatística entre o marcador fúngico isolado e as integrações multi-marcador).
  - PERMANOVA multivariada (Anderson 2001, 499 permutações): o pH explica 18,22% da variação beta bacteriana ($p = 0,002$), mas apenas 3,69% da fúngica.
- [x] **Qualidade de Software & Reprodutibilidade:**
  - Suíte com **328 testes automatizados** passando com 100% de sucesso (`pytest`).
  - Relatório técnico em Markdown ([`REPORT.md`](REPORT.md)).
  - Planilha de tabelas formatadas para publicação ([`results/tabelas_manuscrito.xlsx`](results/tabelas_manuscrito.xlsx)).

---

## 2. Planejamento para as Fases Posteriores

Com a arquitetura analítica pronta e homologada, as próximas etapas seguirão o seguinte cronograma:

### Etapa 2.1: Redação e Submissão do Manuscrito Continental
* **Objetivo:** Submeter artigo focado no potencial preditivo continental do microbioma e a dissociação da sensibilidade edáfica entre bactérias e fungos.
* **Periódicos Alvo:** *Soil Biology and Biochemistry*, *Microbiome* ou *Geoderma*.
* **Recursos Disponíveis:** As 5 tabelas prontas em `results/tabelas_manuscrito.xlsx` e as métricas consolidadas em `REPORT.md`.

### Etapa 2.2: Integração do Benchmark Tropical (Manguezais / PRJNA283936)
* **Objetivo:** Avaliar a transferibilidade da esteira metodológica em biomas costeiros e tropicais.
* **Metodologia:**
  * Ingestão das matrizes de ASVs e metadados de salinidade/pH do BioProject `PRJNA283936`.
  * Execução do mesmo protocolo de validação espacial e testes de hipótese ($H_0$).
  * Comparação entre a estabilidade das assinaturas fúngicas em solos temperados agrícolas vs. solos anaeróbios de manguezal.

### Etapa 2.3: Interpretabilidade Aprofundada (SHAP & Associações Taxonômicas Específicas)
* **Objetivo:** Identificar os táxons (famílias e gêneros de fungos e bactérias) mais determinantes para cada bioma.
* **Técnica:** Aplicação de valores de SHAP (*SHapley Additive exPlanations*) sobre os modelos Random Forest para extrair biomarcadores taxonômicos específicos de solos florestais vs. agrícolas.

---

## 3. Estado de Prontidão da Infraestrutura Computacional

Toda a árvore do código foi consolidada em módulos independentes e reutilizáveis dentro de `src/`:
- Qualquer nova matriz de contagens ou distâncias (seja de novos biomas tropicais ou novas coortes públicas) pode ser plugada diretamente através de `src/models/feature_builder.py` e `src/spatial_cv/spatial_partitioner.py`.
- O repositório está limpo, estável e pronto para arquivamento ou versionamento contínuo em Git.
