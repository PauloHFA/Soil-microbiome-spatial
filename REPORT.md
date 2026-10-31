# Relatório Técnico-Científico: Discriminação Espacial Multimodal do Solo

Este relatório documenta a análise computacional e ecológica conduzida sobre dados observados de sequenciamento de amplicons bacterianos (16S rRNA), fúngicos (ITS) e propriedades físico-químicas de solo do Consórcio Europeu LUCAS 2018 (ESDAC / BioProject PRJNA952168).

A investigação avalia o potencial discriminatório dessas três fontes de dados sobre 6 fitofisionomias continentais sob regime estrito de **Validação Cruzada Espacial (Spatial Group K-Fold)** garantindo ausência total de vazamento geográfico (`Zero Data Leakage`).

---

## 1. Bases de Dados e Pareamento de Amostras

A coorte analítica reúne **715 sítios com pareamento biológico triplo e estrito** (16S + ITS + Solo ESDAC) em 24 países europeus:

* **BioProject PRJNA952168 / ENA**: Perfis brutos de amplicons para bactérias e fungos.
* **Matrizes Ecológicas de Referência (Nature Communications, Labouyrie et al. 2023)**: Matrizes pareadas de dissimilaridade de Bray-Curtis padronizadas por subamostragem repetida (SRS) em **40.109 leituras para 16S** e **502 leituras para ITS**.
* **ESDAC LUCAS 2018 (`Soil_properties_2018`)**: Propriedades físico-químicas de laboratório: pH (em H2O e CaCl2), carbono orgânico (SOC), nitrogênio total, razão C/N, fósforo, potássio, condutividade elétrica, densidade aparente, teores de argila, silte e areia.
* **Pareamento**: Amostras pareadas estritamente pelo identificador físico de campo (`BARCODE_ID`).

---

## 2. Metodologia de Validação Espacial & Modelagem

Para evitar a inflação artificial de métricas causada por autocorrelação espacial (mesma região em treino e teste):
1. **Spatial Group K-Fold**: Particionamento em 5 dobras independentes agrupadas por país/bloco geográfico. Verificação formal: `Sites(treino) ∩ Sites(teste) = ∅`.
2. **Isolamento Intra-Dobra**: Imputação de valores faltantes (mediana) e padronização escalar (Z-score) calculadas estritamente na partição de treino de cada dobra.
3. **Classificadores**: Random Forest Classifier (200 árvores) ponderado (`class_weight='balanced'`) para tratar o desbalanceamento amostral natural entre biomas.
4. **Quantificação de Incerteza**: Intervalos de confiança de 95% obtidos por 1.000 replicações de bootstrap espacial.
5. **Teste de Hipótese Nula ($H_0$)**: Teste pareado de Wilcoxon signed-rank avaliando se a integração multi-marcador supera o melhor marcador individual.

---

## 3. Desempenho Preditivo Comparativo (5 Paradigmas)

| Modelo / Conjunto de Preditoras | Acurácia Balanceada Média | IC 95% Bootstrap | Macro-F1 | Delta vs. Melhor Individual | Conclusão Estatística ($H_0$) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **A · Bactérias (16S rRNA)** | **45,13%** | 40,09% – 49,44% | 0,4713 | — | Referência bacteriana |
| **B · Fungos (ITS)** | **46,67%** | 42,08% – 50,75% | 0,4827 | — | **Melhor marcador biológico único** |
| **C · Multi-marcador (16S + ITS)** | **46,04%** | 41,28% – 50,97% | 0,4830 | +0,44 pp | $H_0$ aceita (Equivalência) |
| **D · Físico-Química Solo (ESDAC)** | **41,45%** | 37,03% – 45,61% | 0,4385 | — | Poder preditivo edáfico isolado |
| **E · Holístico (16S + ITS + Solo)** | **46,35%** | 42,15% – 50,55% | 0,4826 | +0,74 pp | $H_0$ aceita ($p = 0,406$) |
| *Linha de Base do Acaso Teórico* | *16,67%* | — | 0,1667 | — | Acaso estocástico (1/6 classes) |

---

## 4. Decomposição Ecológica: Associação com Propriedades do Solo (PERMANOVA)

Avaliação da porcentagem de variação beta das comunidades microbianas explicada por cada variável físico-química de bancada (PERMANOVA de Anderson 2001, 499 permutações, $p < 0,005$):

| Variável Edáfica | Papel / Descrição | 16S Bactérias ($R^2$) | ITS Fungos ($R^2$) | Razão de Sensibilidade (16S / ITS) |
| :--- | :--- | :---: | :---: | :---: |
| **pH em H2O** | Acidez ativa do solo | **18,22%** | **3,69%** | **4,94x** mais forte em bactérias |
| **Potássio (K)** | Nutriente e fertilidade | **6,64%** | **1,46%** | **4,55x** mais forte em bactérias |
| **Carbono Orgânico (SOC)** | Matéria orgânica e estrutura | **5,48%** | **1,93%** | **2,84x** mais forte em bactérias |
| **Nitrogênio Total (N)** | Ciclagem de nitrogênio e biomassa | **2,75%** | **1,04%** | **2,64x** mais forte em bactérias |
| **Fósforo Disponível (P)** | Limitação de fósforo disponível | **0,96%** | **0,91%** | **1,05x** (efeito similar em ambos) |

---

## 5. Desempenho Detalhado por Fitofisionomia (Modelo Holístico)

| Fitofisionomia | Amostras ($N$) | Sensibilidade (Recall) | Precisão | F1-Score | Comportamento Observado |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Floresta de Coníferas** | 160 (22,4%) | **76,9%** | **78,8%** | **0,778** | Altíssima separação (solos ácidos e micorrizas específicas) |
| **Cultivo não permanente** | 264 (36,9%) | **71,6%** | **75,3%** | **0,734** | Excelente acurácia (impacto de manejo e fertilizantes) |
| **Pastagem extensiva** | 128 (17,9%) | **67,2%** | **55,5%** | **0,608** | Boa discriminação com transições suaves |
| **Floresta de Folhosas** | 99 (13,8%) | **35,4%** | **44,9%** | **0,395** | Sobreposição ecológica parcial com florestas mistas |
| **Cultivo permanente** | 46 (6,4%) | **50,0%** | **30,7%** | **0,380** | Moderada (vinhedos/pomares restritos regionalmente) |
| **Pastagem intensiva** | 18 (2,5%) | 0,0% | 0,0% | 0,000 | Desafio amostral (< 3% do total de observações) |

---

## 6. Importância Relativa das Propriedades de Solo (Random Forest)

1. **Razão Carbono/Nitrogênio (C/N)**: 11,86%
2. **pH em H2O**: 10,13%
3. **pH em CaCl2**: 9,99%
4. **Fósforo Disponível (P)**: 7,63%
5. **Carbono Orgânico (SOC)**: 7,39%
6. **Densidade Aparente do Solo**: 7,05%
7. **Potássio Extraível (K)**: 6,63%
8. **Condutividade Elétrica (CE)**: 6,53%
9. **Nitrogênio Total (N)**: 6,14%
10. **Teor de Areia (%)**: 5,76%

---

## 7. Principais Conclusões Científicas

1. **Eficácia Preditiva Superior ao Acaso**: Todos os modelos atingiram acurácias balanceadas entre 41% e 47% sob particionamento espacial cego, demonstrando que tanto as comunidades microbianas quanto as variáveis edáficas capturam assinaturas ecológicas autênticas de cobertura de solo.
2. **Papel Singular da Comunidade Fúngica**: Os fungos (ITS) apresentaram a maior acurácia balanceada individual (46,7%), refletindo uma associação simbiótica e saprotrófica mais estreita e restrita com as espécies vegetais dominantes de cada fitofisionomia.
3. **Dissociação Bacteriana vs. Fúngica em Relação ao pH**: A PERMANOVA demonstrou que o pH é a variável mestra determinante para as bactérias ($R^2 = 18,2\%$), ao passo que os fungos toleram variações de acidez com plasticidade muito maior ($R^2 = 3,7\%$).
4. **Princípio de Parsimônia ($H_0$ aceita)**: Como o marcador fúngico isolado alcança desempenho estatisticamente indistinguível da integração multi-marcador ($p = 0,406$), o perfil de ITS isolado se apresenta como a escolha mais custo-efetiva para programas de monitoramento de vegetação em escala continental.
