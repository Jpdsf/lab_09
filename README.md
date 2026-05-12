# Laboratório 09 — Pipeline RAG avançado: HNSW + HyDE + Cross-Encoder

## Sobre o projeto

Este laboratório implementa um pipeline de **Retrieval-Augmented Generation (RAG)** voltado para busca em manuais médicos simulados. O sistema recebe uma **query coloquial** (como de um paciente) e aplica **quatro etapas** antes de exibir os trechos que seriam injetados em um modelo gerador:

1. **Indexação** — embeddings via API Gemini (`gemini-embedding-001`, 768 dimensões com MRL) + índice **HNSW** no FAISS (produto interno após normalização L2 ≈ similaridade de cosseno).
2. **HyDE** — o Gemini (`gemini-2.5-flash`) reescreve a pergunta em **um parágrafo técnico** estilo manual clínico; esse texto hipotético vira vetor de **consulta** (`RETRIEVAL_QUERY`).
3. **Recuperação larga** — busca dos **Top-10** fragmentos no HNSW (bi-encoder / mesmo espaço de embedding).
4. **Re-ranking** — **Cross-Encoder** local (`cross-encoder/ms-marco-MiniLM-L-6-v2`) sobre pares (query original, texto do candidato); seleção dos **Top-3** finais.

Os fragmentos estão em `dados_medicos.py` (**20** entradas), com vocabulário clínico em português, cobrindo temas como neurologia, cardiologia, pneumologia, gastroenterologia/hepatologia, endocrinologia, nefrologia, infectologia, reumatologia e hematologia.

---

## Como rodar no VS Code

1. Abra a pasta do projeto e, no terminal integrado (**Ctrl+`**), crie e ative um ambiente virtual (recomendado):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure a chave da API Gemini. Copie o modelo de variáveis (se ainda não existir o `.env`):

   ```bash
   cp .env.example .env
   ```

   Edite o arquivo **`.env`** na raiz do projeto:

   ```env
   GEMINI_API_KEY=sua-chave-aqui
   ```

   Chaves em: [Google AI Studio](https://aistudio.google.com/apikey).

4. Execute o pipeline:

   ```bash
   python rag_pipeline.py
   ```

O script usa `python-dotenv` e chama `load_dotenv()` em `rag_pipeline.py`, carregando automaticamente o `.env`. Também é possível exportar a variável no shell (`export GEMINI_API_KEY=...`) sem usar arquivo `.env`.

---

## Por que Gemini (Google AI Studio)?

Este lab usa o SDK **`google-genai`** com a **Gemini API** (mesma conta do Google AI Studio). Há **cota gratuita** para experimentação; políticas e limites mudam com o tempo — consulte a documentação oficial de preços e uso.

**Embeddings e geração** ficam no mesmo ecossistema: `embed_content` para vetores e `generate_content` para o texto HyDE, o que simplifica credenciais (uma chave) e evita combinar provedores diferentes para indexação e LLM.

### Como trocar modelos ou dimensão de embedding

No topo de `rag_pipeline.py`, ajuste as constantes:

| Constante | Função |
|-----------|--------|
| `MODELO_EMBEDDING` | Modelo de embedding (hoje `gemini-embedding-001`) |
| `EMBEDDING_DIMENSAO_SAIDA` | Dimensão MRL (hoje `768`) — **documentos e queries devem usar o mesmo valor** |
| `MODELO_GERACAO` | LLM do HyDE (hoje `gemini-2.5-flash`) |

Se mudar a dimensão ou o modelo de embedding, é preciso **reconstruir o índice** (o pipeline já refaz o índice a cada execução neste script).

---

## Passo a passo da implementação

### Passo 1 — Base de dados e indexação HNSW

Foram definidos **20 fragmentos** de manuais médicos fictícios em `fragmentos_medicos`, com terminologia clínica em português.

Cada texto é enviado à API Gemini com `task_type="RETRIEVAL_DOCUMENT"` e `output_dimensionality=768`. Os vetores são normalizados em L2; o índice é `faiss.IndexHNSWFlat` com **`faiss.METRIC_INNER_PRODUCT`**.

Hiperparâmetros atuais: **`HNSW_M = 16`**, **`HNSW_EF_CONSTRUCTION = 200`**, **`HNSW_EF_SEARCH = 50`**.

Exemplo de saída esperada (valores aproximados conforme ambiente):

```text
PASSO 1: Construindo índice HNSW com FAISS + Gemini Embeddings
[1/3] Modelo de embedding: gemini-embedding-001
[2/3] Gerando vetores para 20 fragmentos via Gemini API...
      Dimensão dos vetores: 768
[3/3] Criando índice HNSW (M=16, ef_construction=200)...
      Total de vetores indexados: 20
```

### Passo 2 — HyDE: transformação da query

A função `passo2_hyde` envia a pergunta coloquial ao Gemini com instruções para produzir **um único parágrafo técnico** alinhado a manual médico. Esse **documento hipotético** é o que será embedado como consulta no passo seguinte.

Exemplo ilustrativo (o texto exato varia a cada chamada):

```text
Documento Hipotético gerado:
--------------------------------------------------
A cefaleia pulsátil unilateral, associada a fotofobia e náuseas, sugere no
quadro clínico migrânea a investigação de critérios diagnósticos ICHD-3...
--------------------------------------------------
```

### Passo 3 — Busca no índice HNSW

O HyDE é vetorizado com `task_type="RETRIEVAL_QUERY"` e comparado ao índice; recuperam-se os **10** melhores (`TOP_K_RETRIEVAL`). Os scores impressos vêm do **produto interno** no espaço normalizado (interpretação análoga à similaridade de cosseno).

A saída lista `Rank`, `Score` e `Título` de cada fragmento.

### Passo 4 — Re-ranking com Cross-Encoder

Os 10 candidatos são avaliados em pares **(query original do usuário, texto do fragmento)** pelo modelo **`cross-encoder/ms-marco-MiniLM-L-6-v2`**, executado localmente (depende de PyTorch, instalado como dependência do `sentence-transformers`). O resultado é ordenado por `score_cross` e mantidos os **3** primeiros (`TOP_K_FINAL`).

### HyDE em relação à busca direta

Neste repositório o fluxo **sempre** usa HyDE para o vetor de busca no passo 3 (vetor do documento hipotético, não da query bruta). Em um experimento comparativo, espera-se que o HyDE aproxime a consulta da **linguagem dos manuais**, melhorando o alinhamento lexical e semântico frente à embedação literal de gírias ou descrições vagas.

---

## Hiperparâmetros HNSW: `M`, `ef_construction` e `efSearch`

O **KNN exato** compara a query a todos os vetores — custo típico **O(n)** por consulta, caro em bases grandes.

O **HNSW** (Hierarchical Navigable Small World) organiza os pontos em um grafo em camadas, buscando vizinhos sem visitar todos os documentos — escalabilidade muito melhor para **n** grande.

- **`M`**: grau aproximado de conexões por nó. Valores maiores tendem a **aumentar recall** e **memória/tempo de construção**. Aqui usa-se **`M = 16`** (mais enxuto que exemplos didáticos com `M = 32`).
- **`efConstruction`**: largura da busca greedy na **construção** do grafo; afeta qualidade do grafo, não o tamanho do vetor em si. Valor atual: **200**.
- **`efSearch`**: largura na **consulta**; trade-off velocidade vs. recall na busca. Valor atual: **50**.

---

## Problemas comuns durante o desenvolvimento

### Erro — `GEMINI_API_KEY` não encontrada

Mensagem do tipo: variável de ambiente `GEMINI_API_KEY` não encontrada.

**Causa:** o processo Python não enxerga a chave (terminal sem `export`, ou `.env` ausente / fora da raiz).

**Solução:** criar `.env` na raiz com `GEMINI_API_KEY=...` (ver `.env.example`) ou exportar no shell antes de rodar o script.

---

### Erro — `404 NOT_FOUND` no modelo de embedding

Exemplo: `models/text-embedding-004 is not found ... or is not supported for embedContent`.

**Causa:** o nome do modelo não existe ou não suporta `embedContent` na **Gemini Developer API** usada pelo `google-genai`.

**Solução:** usar um modelo de embedding suportado (no código atual, **`gemini-embedding-001`**) e, se necessário, definir **`output_dimensionality`** compatível com o modelo.

---

### Erro — dimensão inconsistente entre índice e query

**Causa:** trocar `EMBEDDING_DIMENSAO_SAIDA` ou o modelo de embedding só em parte do fluxo, ou misturar vetores antigos com novos.

**Solução:** garantir a **mesma** configuração de embedding para `RETRIEVAL_DOCUMENT` e `RETRIEVAL_QUERY` e reconstruir o índice.

---

## Sobre os scores do Cross-Encoder

Os valores são **logits / scores brutos** do modelo, **não** probabilidades normalizadas entre 0 e 1. É comum aparecerem **números negativos**. O re-ranking usa apenas a **ordem relativa** (maior score = par query–documento mais compatível segundo o cross-encoder).

---

## Declaração de integridade acadêmica

Partes deste laboratório foram **geradas ou complementadas com IA**, **revisadas e validadas** por **João Paulo**.

O uso de ferramentas de IA generativa limitou-se a apoio em estrutura do pipeline, redação de fragmentos fictícios de manuais e rascunhos de código, todos revistos antes da entrega. Este README foi reescrito com base em um modelo de documentação anterior, **adaptado** ao código real deste repositório (`rag_pipeline.py`, `dados_medicos.py`, Gemini e `requirements.txt`).
