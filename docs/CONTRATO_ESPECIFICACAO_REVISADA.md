# Contrato da especificação revisada

Fonte normativa única: `PROJETO_POS_DOC_BIANCA_REVISADO.pdf` (18 páginas).

Não normativo: `PROJETO PÓS-DOC_BIANCA_ULRICH_2026.pdf`. Números, prazos e desenhos que só aparecem nesse arquivo não obrigam o código.

O texto abaixo não acrescenta regra. Só registra as cláusulas do PDF revisado que o fluxo analítico tem de obedecer.

## O que o estudo é, e o que não é

1. O alvo é a categoria ambiental compartilhada entre locais, não o identificador de um local e não a origem de uma amostra desconhecida.
2. Discriminação ambiental não vira critério de procedência nem conclusão de laudo.
3. Perfil de 16S ou ITS descreve composição. Não é atividade metabólica.
4. Resultado negativo, ou ausência de ganho da integração, é desfecho válido. A integração não é presumida superior.

## Coorte da comparação principal

5. A comparação principal usa amostras com 16S e ITS da mesma unidade biológica de solo, com metadados mínimos. A base candidata é LUCAS / PRJNA952168, com metadados ESDAC quando o acesso existir.
6. 885 amostras, 881 locais e o subconjunto de 715 solos não são tamanho final. Pareamento, duplicata técnica, qualidade e metadados são auditados antes da inclusão.
7. Não se combina 16S de uma amostra ou estudo com ITS de outra para simular microbioma integrado.
8. Estudos distintos só entram na mesma análise quantitativa se marcador, região, plataforma, processamento, unidade amostral e metadados forem compatíveis. Correção de lote não basta quando lote e categoria estão confundidos.
9. Duplicata técnica não conta como observação independente. Réplicas do mesmo local permanecem no mesmo lado da validação.
10. PRJNA283936 é trilha complementar só de 16S. Não é teste de integração bacteriana–fúngica e não é referência dos solos da Paraíba.
11. Solos paraibanos, coleta nova e sequenciamento local não são dependência da análise principal. Se só um marcador for obtido, não se afirma integração.
12. Se não houver pareamento ou a estrutura espacial não se sustentar, a base sai da comparação integrada. Não se alega teste de integração nem validação geográfica que não ocorreu.

## Categorias e ecologia

13. As categorias previstas na base principal são florestas de coníferas, florestas de folhosas, pastagem extensiva, pastagem intensiva, cultivo permanente e cultivo não permanente, conforme o metadado.
14. Comparação dentro da mesma categoria é exploratória e não valida procedência.
15. Diversidade alfa, beta composicional, PERMANOVA e teste de dispersão multivariada acompanham a descrição. Diferença de dispersão limita a leitura da PERMANOVA.
16. Variável ausente não derruba o projeto inteiro, mas não é tratada como medida que não existe. Variável medida e variável estimada ficam distintas.

## Validação

17. Treino e teste separam locais ou blocos espaciais. O mesmo local não aparece dos dois lados.
18. Toda categoria avaliada no teste tem de estar representada por outros locais no treino.
19. Falta de replicação espacial não se contorna com divisão aleatória.
20. Se os metadados não sustentarem avaliação espacial, a limitação é explícita e não se alega validação geográfica.
21. Quando o número de locais permitir, a seleção de variáveis, transformações com parâmetro estimado, modelo e hiperparâmetro ocorre só no treino. Caso contrário, usam-se modelos mais simples, poucos folds agrupados ou leave-one-site-out, e o resultado é exploratório.
22. Os conjuntos A (bactérias), B (fungos) e C (integração) usam as mesmas amostras pareadas, as mesmas categorias e as mesmas partições.
23. O melhor marcador isolado é escolhido no treino. O teste não escolhe o comparador.
24. Métricas: balanced accuracy, macro-F1, matrizes de confusão e intervalo de incerteza que respeite a unidade espacial. Diferença pequena não é ganho estabelecido. Há referencial de desempenho ao acaso compatível com o desenho.
25. Semente aleatória fica registrada. Versões, parâmetros, filtros e exclusões também.
