import os

import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from google import genai
from google.genai import types

from dados_medicos import fragmentos_medicos

load_dotenv()

MODELO_EMBEDDING = "gemini-embedding-001"
EMBEDDING_DIMENSAO_SAIDA = 768
MODELO_GERACAO = "gemini-2.5-flash"

MODELO_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Hiperparâmetros do HNSW
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 50

TOP_K_RETRIEVAL = 10
TOP_K_FINAL = 3


def criar_cliente_gemini() -> genai.Client:
    chave = os.environ.get("GEMINI_API_KEY")
    if not chave:
        raise EnvironmentError(
            "Variável de ambiente GEMINI_API_KEY não encontrada.\n"
            "Crie um arquivo .env na raiz do projeto (veja .env.example) ou execute:\n"
            "  export GEMINI_API_KEY='sua-chave-aqui'"
        )
    return genai.Client(api_key=chave)


def gemini_embed(
    cliente: genai.Client,
    textos: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> np.ndarray:
    """
    task_type:
      - "RETRIEVAL_DOCUMENT" → indexação dos fragmentos da base
      - "RETRIEVAL_QUERY"    → vetorização da query/HyDE na busca
    """
    vetores = []
    for texto in textos:
        resposta = cliente.models.embed_content(
            model=MODELO_EMBEDDING,
            contents=texto,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSAO_SAIDA,
            ),
        )
        vetores.append(resposta.embeddings[0].values)

    arr = np.array(vetores, dtype=np.float32)
    normas = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.clip(normas, 1e-10, None)
    return arr


def separador(titulo: str) -> None:
    largura = 70
    print("\n" + "=" * largura)
    print(f"  {titulo}")
    print("=" * largura)


def passo1_construir_indice(cliente: genai.Client, textos: list[str]) -> faiss.Index:
    separador("PASSO 1: Construindo índice HNSW com FAISS + Gemini Embeddings")

    print(f"[1/3] Modelo de embedding: {MODELO_EMBEDDING}")
    print(
        f"[2/3] Gerando vetores para {len(textos)} fragmentos via Gemini API...")
    vetores = gemini_embed(cliente, textos, task_type="RETRIEVAL_DOCUMENT")

    dimensao = vetores.shape[1]
    print(f"      Dimensão dos vetores: {dimensao}")

    # IndexHNSWFlat com produto interno (≡ cosseno após normalização L2)
    print(
        f"[3/3] Criando índice HNSW (M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION})...")
    indice = faiss.IndexHNSWFlat(dimensao, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    indice.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    indice.hnsw.efSearch = HNSW_EF_SEARCH

    indice.add(vetores)
    print(f"      Total de vetores indexados: {indice.ntotal}")

    return indice


def passo2_hyde(cliente: genai.Client, query_usuario: str) -> str:
    separador("PASSO 2: HyDE — Geração do Documento Hipotético via Gemini")

    print(f"Query original do usuário:\n  \"{query_usuario}\"\n")

    prompt_hyde = (
        "Você é um especialista médico redator de manuais clínicos. "
        "A seguir, uma pergunta coloquial de um paciente. "
        "Escreva um ÚNICO parágrafo técnico, como se fosse extraído de um manual médico, "
        "que respondesse precisamente essa pergunta usando terminologia clínica formal "
        "(jargão médico, nomes de sinais, sintomas, diagnósticos e tratamentos em português). "
        "Não inclua saudações, introduções ou explicações. Apenas o parágrafo técnico.\n\n"
        f"Pergunta do paciente: {query_usuario}"
    )

    print(
        f"Chamando {MODELO_GERACAO} para gerar o Documento Hipotético (HyDE)...")
    resposta = cliente.models.generate_content(
        model=MODELO_GERACAO,
        contents=prompt_hyde,
    )

    documento_hipotetico = resposta.text.strip()
    print(
        f"\nDocumento Hipotético gerado:\n{'-'*50}\n{documento_hipotetico}\n{'-'*50}")

    return documento_hipotetico


def passo3_recuperar_top10(
    cliente: genai.Client,
    documento_hipotetico: str,
    indice: faiss.Index,
    fragmentos: list[dict],
) -> list[dict]:
    separador("PASSO 3: Busca rápida via Bi-Encoder (Gemini) no índice HNSW")

    print("Vetorizando o Documento Hipotético via Gemini...")
    vetor_hyde = gemini_embed(
        cliente, [documento_hipotetico], task_type="RETRIEVAL_QUERY")

    print(
        f"Buscando Top-{TOP_K_RETRIEVAL} documentos mais próximos no índice HNSW...")
    scores, indices = indice.search(vetor_hyde, TOP_K_RETRIEVAL)

    resultados = []
    print(f"\n{'Rank':<6} {'Score':<10} {'Título'}")
    print("-" * 65)
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        fragmento = fragmentos[idx]
        resultados.append(
            {"rank_bi": rank, "score_bi": float(score), **fragmento})
        print(f"  {rank:<4} {score:<10.4f} {fragmento['titulo']}")

    return resultados


def passo4_reranking(
    query_original: str,
    candidatos: list[dict],
) -> list[dict]:
    separador("PASSO 4: Re-ranking com Cross-Encoder (local)")

    print(f"Carregando Cross-Encoder: {MODELO_CROSS_ENCODER}")
    cross_encoder = CrossEncoder(MODELO_CROSS_ENCODER)

    pares = [(query_original, c["texto"]) for c in candidatos]

    print(f"Calculando scores de atenção profunda para {len(pares)} pares...")
    scores_cross = cross_encoder.predict(pares)

    for candidato, score in zip(candidatos, scores_cross):
        candidato["score_cross"] = float(score)

    rerankeados = sorted(
        candidatos, key=lambda x: x["score_cross"], reverse=True)

    print(f"\nTop-{TOP_K_FINAL} documentos após re-ranking:\n")
    print("=" * 70)
    for rank, doc in enumerate(rerankeados[:TOP_K_FINAL], start=1):
        print(f"\n[#{rank}] {doc['titulo']}")
        print(f"  Score Cross-Encoder : {doc['score_cross']:.4f}")
        print(
            f"  Score Bi-Encoder    : {doc['score_bi']:.4f}  (rank original: #{doc['rank_bi']})")
        print(f"  Texto:\n  {doc['texto'][:200]}...")
        print("-" * 70)

    return rerankeados[:TOP_K_FINAL]


def executar_pipeline(query_usuario: str) -> None:
    separador("INICIANDO PIPELINE RAG AVANÇADO — Lab 09")
    print(f"Total de fragmentos na base: {len(fragmentos_medicos)}")

    cliente = criar_cliente_gemini()
    textos = [f["texto"] for f in fragmentos_medicos]

    indice = passo1_construir_indice(cliente, textos)

    documento_hipotetico = passo2_hyde(cliente, query_usuario)

    candidatos = passo3_recuperar_top10(
        cliente, documento_hipotetico, indice, fragmentos_medicos)

    top3_final = passo4_reranking(query_usuario, candidatos)

    separador("RESULTADO FINAL — Documentos que seriam injetados no LLM")
    for rank, doc in enumerate(top3_final, start=1):
        print(f"\n[Documento {rank}] {doc['titulo']}")
        print(f"  {doc['texto']}")

    separador("Pipeline concluído")


if __name__ == "__main__":
    query = "tô com dor de cabeça latejante de um lado só e a luz me incomoda muito"
    executar_pipeline(query)
