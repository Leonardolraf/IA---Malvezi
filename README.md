# Trabalho Final — IA: Lógica Fuzzy + Machine Learning
## Tema 4 — Análise de Risco de Concessão de Crédito (Grupo 4)

Sistema inteligente que classifica o risco de crédito de um solicitante (**baixo / médio / alto**),
combinando **Machine Learning** (Árvore de Decisão) e **Lógica Fuzzy** por **integração**:
a probabilidade de inadimplência prevista pelo modelo vira entrada do sistema fuzzy.

---

### 📁 Arquivos
- `trabalho_credito_fuzzy_ml.ipynb` — notebook principal (código completo e comentado).
- `README.md` — este arquivo.

### ▶️ Como executar (Google Colab — recomendado)
1. Acesse https://colab.research.google.com
2. **Arquivo → Carregar notebook** e selecione `trabalho_credito_fuzzy_ml.ipynb`.
3. **Ambiente de execução → Executar tudo** (ou `Ctrl+F9`).
4. A primeira célula instala o `scikit-fuzzy`; o Colab pode pedir para reiniciar — basta executar tudo de novo.

> O notebook **baixa a base automaticamente** do UCI (precisa de internet). Não é necessário subir nenhum arquivo.

### ▶️ Como executar (Jupyter local)
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-fuzzy
jupyter notebook trabalho_credito_fuzzy_ml.ipynb
```

### 🗂️ Base de dados
**Statlog (German Credit Data)** — UCI Machine Learning Repository (1.000 registros, pública).
- Link: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- Por ser pública, não precisa ser entregue no AVA (apenas o link, conforme o enunciado).

### 🧱 Estrutura do notebook (7 fases)
1. Preparação do ambiente
2. Carregar e entender a base
3. Análise exploratória (EDA) + pré-processamento
4. Modelo de Machine Learning (Árvore de Decisão) + métricas
5. Sistema de inferência fuzzy (variáveis, pertinências, 9 regras)
6. Integração ML → Fuzzy
7. Cenários de teste e discussão crítica

### 👥 Grupo 4 — Inteligência Artificial
Prof. William Malvezzi · Universidade Católica de Brasília (UCB)
