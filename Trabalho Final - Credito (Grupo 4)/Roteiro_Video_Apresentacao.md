# Roteiro do Vídeo de Apresentação — Trabalho Final de IA (Grupo 4)
## Sistema Inteligente de Análise de Risco de Crédito — ML + Lógica Fuzzy

**Duração-alvo:** ~10 minutos (teto de 15) · **Apresentadores:** 2 · **Tela:** notebook do Colab + relatório

> **Como usar este roteiro:** o texto em fonte normal é para **ler/narrar**. As linhas **[TELA: …]** indicam o que deixar na tela naquele momento. **[A]** e **[B]** indicam quem fala. Substituam *Apresentador A* e *Apresentador B* pelos nomes reais.

### 🎬 Dicas rápidas de gravação
- **Ensaiem 1 vez cronometrando** — o alvo é 10 min; se passar de 12, cortem exemplos, não conteúdo.
- Gravem a tela com **OBS Studio** (grátis) ou o gravador de tela do próprio sistema; falem de forma pausada.
- Deixem o notebook **já executado** (com as saídas visíveis) antes de gravar, para não esperar célula rodando.
- No YouTube, publiquem como **"Não listado" (por link)** ou **"Público"**, e coloquem o link no AVA.

---

## BLOCO 1 — Abertura e problema · `00:00–01:00` · [A]
**[TELA: capa do relatório ou primeira célula do notebook]**

[A] Olá, professor. Somos o **Grupo 4** da disciplina de Inteligência Artificial. Eu sou o *[Apresentador A]* e comigo está o *[Apresentador B]*. Neste vídeo vamos apresentar nosso trabalho final: um **sistema inteligente de análise de risco de crédito**, que combina **Machine Learning** e **Lógica Fuzzy**.

[A] O problema que escolhemos — o **Tema 4** — é o seguinte: quando alguém pede um empréstimo, o banco precisa decidir se aquele cliente é de **risco baixo, médio ou alto**. Errar para um lado gera calote; errar para o outro, perda de bons clientes. É uma decisão **incerta** e cheia de conceitos imprecisos, como "renda alta" ou "histórico ruim" — exatamente o tipo de problema em que IA e Lógica Fuzzy se complementam.

---

## BLOCO 2 — Objetivo e visão geral · `01:00–01:45` · [A]
**[TELA: o diagrama de fluxo no resumo do notebook — "Dados → Árvore → prob → Fuzzy → Risco"]**

[A] Nossa solução tem **dois cérebros que trabalham juntos**. O primeiro é um modelo de **Machine Learning**, que aprende com 1.000 casos reais e calcula a **probabilidade de inadimplência** de cada cliente. O segundo é um **sistema de Lógica Fuzzy**, que pega essa probabilidade, junta com o comprometimento da renda e o histórico de crédito, e aplica **regras do tipo SE–ENTÃO** para gerar um **risco final fácil de explicar**.

[A] A ligação entre os dois é o que o enunciado chama de **Integração**: a saída do Machine Learning vira a entrada do sistema fuzzy. Já explico por que escolhemos esse caminho.

---

## BLOCO 3 — Base de dados e pré-processamento · `01:45–03:00` · [A]
**[TELA: Colab, Fase 2 e Fase 3 — carregamento da base e EDA]**

[A] Usamos uma base **pública e real**: a *German Credit Data*, do repositório UCI, com **1.000 registros** de pedidos de crédito. Optamos por uma base pública, em vez de simular dados, porque dá **mais credibilidade** ao trabalho — o modelo aprende padrões reais de inadimplência.

[A] Na análise exploratória, percebemos que a base é **desbalanceada**: 700 bons pagadores e 300 inadimplentes. Esse detalhe é importante e vamos retomá-lo nos resultados.

[A] Um ponto de rigor: a coluna de comprometimento da renda, o *installment rate*, não tem a direção da escala documentada. Em vez de chutar, **descobrimos pelos próprios dados** — calculamos a correlação com a inadimplência, que deu **positiva, 0,072**, confirmando que valores maiores significam maior comprometimento. Também tratamos as variáveis categóricas, convertendo texto em número, e separamos a base em **80% para treino e 20% para teste**.

---

## BLOCO 4 — Modelo de Machine Learning e métricas · `03:00–04:45` · [A]
**[TELA: Colab, Fase 4 — código da árvore, depois as métricas e a matriz de confusão]**

[A] Para o Machine Learning, escolhemos a **Árvore de Decisão**. E essa escolha tem dois motivos: é o algoritmo de **preferência da disciplina** e, principalmente, é **interpretável** — dá para ver as "perguntas" que o modelo faz para decidir, o que conversa diretamente com as regras da Lógica Fuzzy.

**[TELA: rolar até a saída com acurácia, relatório e matriz de confusão]**

[A] Avaliando no conjunto de teste, o modelo alcançou **acurácia de 74,5%**. Mas, como a base é desbalanceada, a acurácia sozinha engana — por isso olhamos também **precisão, recall e F1**. Na matriz de confusão, de 200 clientes, ele acertou 120 bons pagadores e 29 maus pagadores.

[A] Aqui está uma observação honesta: o **recall dos inadimplentes é 0,48** — o modelo identifica menos da metade dos maus pagadores. Isso é consequência direta do desbalanceamento, e vamos tratar como limitação lá na frente. Por fim, em vez de só dizer "bom" ou "mau", pedimos ao modelo a **probabilidade de inadimplência** — e é esse número que entregamos ao sistema fuzzy. Passo a palavra para o *[Apresentador B]*.

---

## BLOCO 5 — Sistema de Inferência Fuzzy · `04:45–06:15` · [B]
**[TELA: Colab, Fase 5 — variáveis, funções de pertinência e o gráfico das pertinências]**

[B] Obrigado. Agora o segundo cérebro: o **sistema fuzzy**. Ele tem **três variáveis de entrada** — a probabilidade que veio do Machine Learning, o comprometimento da renda e o score do histórico — e **uma variável de saída**, o risco de crédito. Cada uma é dividida em **três termos linguísticos**, como baixo, médio e alto.

**[TELA: mostrar o gráfico das funções de pertinência triangulares]**

[B] Usamos **funções de pertinência triangulares**, conforme a preferência do enunciado. Repare que elas se **sobrepõem**: um valor pode pertencer parcialmente a "médio" e a "alto" ao mesmo tempo — é isso que dá ao sistema a capacidade de raciocinar com gradações, e não com fronteiras rígidas.

**[TELA: rolar até a célula das 9 regras]**

[B] O coração do sistema são **9 regras SE–ENTÃO**, criadas por nós. As três primeiras ancoram a decisão na probabilidade do modelo e garantem que **sempre haja uma regra ativa**. As outras seis **refinam** o risco usando o comprometimento e o histórico — como faria um analista de crédito. No fim, a **defuzzificação pelo método do centroide** converte tudo num número de 0 a 100.

---

## BLOCO 6 — Integração e demonstração · `06:15–07:45` · [B]
**[TELA: Colab, Fase 6 e Fase 7 — rodar a célula de cenários ao vivo]**

[B] Aqui está a parte central do trabalho: a **integração**. Para cada cliente, a probabilidade da árvore entra no sistema fuzzy junto com o comprometimento e o histórico, passa por **fuzzificação, inferência e defuzzificação**, e sai um risco final.

[B] Vamos ver funcionando. **[TELA: executar/mostrar a tabela de cenários]** Olhem os quatro perfis de teste: um **cliente ótimo** — probabilidade baixa, bom histórico — recebe risco **14, BAIXO**. Um **cliente ruim** — probabilidade alta, comprometimento alto — recebe risco **84, ALTO**. E os perfis intermediários caem em **MÉDIO**, com uma transição suave. Ou seja: o sistema **separa bem** os clientes e a decisão é totalmente **rastreável** pelas regras.

---

## BLOCO 7 — Resultados e defesa das escolhas · `07:45–09:00` · [B]
**[TELA: Colab — distribuição do risco e a célula "O que a Lógica Fuzzy acrescenta"]**

[B] Aplicando ao conjunto de teste inteiro, o sistema distribuiu os 200 clientes em **93 de risco baixo, 84 médio e 23 alto** — não concentrou tudo numa faixa só, o que mostra que ele realmente discrimina os perfis.

[B] E aqui respondemos à pergunta mais importante: **a integração agrega algo, ou só reembrulha o modelo?** Nós medimos: o sistema fuzzy **mudou a classificação de 28% dos clientes** em relação a usar apenas a probabilidade do Machine Learning. São casos em que o comprometimento alto ou o histórico ruim deslocaram o risco — exatamente o **ganho de interpretabilidade** que justifica a integração. Foi por isso que escolhemos **Integrar** em vez de só **Comparar**: é o próprio exemplo do enunciado e agrega valor real e mensurável.

---

## BLOCO 8 — Limitações e conclusão · `09:00–09:45` · [B] → [A]
**[TELA: seção de discussão crítica do relatório]**

[B] Sobre as **limitações**, somos transparentes: por causa do desbalanceamento, o recall dos inadimplentes ficou em 0,48. Como **melhoria futura**, aplicaríamos técnicas de balanceamento, como SMOTE, ou testaríamos outros modelos. As funções de pertinência também são, por natureza, subjetivas.

[A] **Concluindo:** entregamos uma solução que **integra Machine Learning e Lógica Fuzzy** — a árvore fornece a probabilidade aprendida dos dados, e o sistema fuzzy a transforma num risco interpretável. Exercitamos todos os conceitos da disciplina: fuzzificação, variáveis linguísticas, funções de pertinência, inferência e defuzzificação.

---

## BLOCO 9 — Encerramento · `09:45–10:00` · [A]
**[TELA: capa / tela final]**

[A] Esse foi o trabalho do **Grupo 4**. O relatório, o notebook comentado e o README estão nos entregáveis do AVA. Agradecemos a atenção, professor!

---

### ✅ Checklist de cobertura (o vídeo contempla todos os itens exigidos)
- [x] Problema escolhido e contexto — Bloco 1
- [x] Base de dados e características — Bloco 3
- [x] Técnica de ML aplicada e justificativa — Bloco 4
- [x] Resultados do ML e métricas — Bloco 4
- [x] Sistema fuzzy: variáveis, pertinências e regras — Bloco 5
- [x] Demonstração das regras e da inferência — Bloco 6
- [x] Integração das duas abordagens — Blocos 6 e 7
- [x] Conclusão crítica com limitações e melhorias — Bloco 8
