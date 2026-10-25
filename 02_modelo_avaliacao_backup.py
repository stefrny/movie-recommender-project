"""Divisão 80/20, Pearson entre filmes, KNN e métricas de recomendação."""
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
SAIDA = BASE / "resultados"
SEMENTE = 42
K_VIZINHOS = 20
N_RECOMENDACOES = 10
LIMIAR_RELEVANTE = 4.0
MIN_COAVALIACOES = 3

def calcular_similaridades(ratings_treino):
    """Pearson por pares de filmes que tenham usuários avaliadores em comum."""
    por_usuario = ratings_treino.groupby("userId")
    pares = defaultdict(list)

    # Para cada usuário, guardar as notas dos pares de filmes que avaliou.
    for _, grupo in por_usuario:
        linhas = list(grupo[["movieId", "rating"]].itertuples(index=False, name=None))
        for (filme_a, nota_a), (filme_b, nota_b) in combinations(linhas, 2):
            if filme_a > filme_b:
                filme_a, filme_b = filme_b, filme_a
                nota_a, nota_b = nota_b, nota_a
            pares[(int(filme_a), int(filme_b))].append((float(nota_a), float(nota_b)))

    medias = ratings_treino.groupby("movieId")["rating"].mean().to_dict()
    similaridades = {}
    registros = []

    for (filme_a, filme_b), observacoes in pares.items():
        if len(observacoes) < MIN_COAVALIACOES:
            continue
        a = np.array([x[0] for x in observacoes], dtype=float)
        b = np.array([x[1] for x in observacoes], dtype=float)
        # Pearson pareado: centraliza as notas dos usuários que avaliaram ambos os filmes.
        # Isso mantém a correlação dentro do intervalo [-1, 1].
        da = a - float(a.mean())
        db = b - float(b.mean())
        denominador = np.sqrt(np.sum(da ** 2) * np.sum(db ** 2))
        if denominador == 0:
            continue
        corr = float(np.sum(da * db) / denominador)
        if not np.isfinite(corr):
            continue
        similaridades[(filme_a, filme_b)] = corr
        registros.append({
            "movieId1": filme_a,
            "movieId2": filme_b,
            "similarity": corr,
            "co_ratings": len(observacoes)
        })
    return similaridades, medias, pd.DataFrame(
        registros, columns=["movieId1", "movieId2", "similarity", "co_ratings"]
    )

def obter_similaridade(similaridades, filme_a, filme_b):
    if filme_a == filme_b:
        return 1.0
    chave = (min(int(filme_a), int(filme_b)), max(int(filme_a), int(filme_b)))
    return similaridades.get(chave)

def prever_nota(usuario, filme_alvo, notas_usuario, similaridades, media_filmes, media_global):
    """Média ponderada das notas do usuário nos K filmes mais similares."""
    candidatos = []
    for filme_visto, nota in notas_usuario.items():
        sim = obter_similaridade(similaridades, filme_alvo, filme_visto)
        if sim is not None and sim != 0:
            candidatos.append((sim, float(nota)))
    candidatos.sort(key=lambda item: abs(item[0]), reverse=True)
    vizinhos = candidatos[:K_VIZINHOS]
    if not vizinhos:
        return float(media_filmes.get(filme_alvo, media_global))
    numerador = sum(sim * nota for sim, nota in vizinhos)
    denominador = sum(abs(sim) for sim, _ in vizinhos)
    if denominador == 0:
        return float(media_filmes.get(filme_alvo, media_global))
    return float(np.clip(numerador / denominador, 0.5, 5.0))

def main():
    SAIDA.mkdir(exist_ok=True)
    ratings_path = DADOS / "ratings.csv"
    movies_path = DADOS / "movies.csv"
    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError("Execute primeiro: python 01_baixar_dados.py")

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    # Embaralhamento aleatório reprodutível; a divisão é feita por avaliação.
    embaralhado = ratings.sample(frac=1, random_state=SEMENTE).reset_index(drop=True)
    corte = int(len(embaralhado) * 0.8)
    treino = embaralhado.iloc[:corte].copy()
    teste = embaralhado.iloc[corte:].copy()

    print(f"Avaliações totais: {len(ratings)}")
    print(f"Treino: {len(treino)} ({len(treino)/len(ratings):.0%})")
    print(f"Teste: {len(teste)} ({len(teste)/len(ratings):.0%})")
    print("Calculando similaridades de Pearson. Isso pode levar alguns minutos...")
    similaridades, medias, tabela_sim = calcular_similaridades(treino)
    tabela_sim.to_csv(SAIDA / "similaridades.csv", index=False)

    notas_por_usuario = {
        int(usuario): dict(zip(grupo["movieId"].astype(int), grupo["rating"].astype(float)))
        for usuario, grupo in treino.groupby("userId")
    }
    media_global = float(treino["rating"].mean())

    # Métricas de previsão de nota calculadas apenas nos pares usuário-filme do teste.
    erros = []
    teste_com_previsoes = []
    for linha in teste.itertuples(index=False):
        notas_usuario = notas_por_usuario.get(int(linha.userId), {})
        previsao = prever_nota(
            int(linha.userId), int(linha.movieId), notas_usuario,
            similaridades, medias, media_global
        )
        erros.append(float(linha.rating) - previsao)
        teste_com_previsoes.append({
            "userId": int(linha.userId), "movieId": int(linha.movieId),
            "rating_real": float(linha.rating), "rating_previsto": previsao
        })
    pred_df = pd.DataFrame(teste_com_previsoes)
    mae = float(np.mean(np.abs(erros)))
    rmse = float(np.sqrt(np.mean(np.square(erros))))
    pred_df.to_csv(SAIDA / "previsoes_teste.csv", index=False)

    # Relevante = nota real >= 4. Recomendações são itens do teste ordenados
    # pela nota prevista; itens que não aparecem no teste não entram no denominador.
    # Essa avaliação mede ranking entre itens retidos no teste, não catálogo completo.
    catalogo_teste = teste.merge(movies[["movieId", "title"]], on="movieId", how="left")
    catalogo_teste = catalogo_teste.merge(
        pred_df[["userId", "movieId", "rating_previsto"]],
        on=["userId", "movieId"], how="left"
    )
    precisao_lista = []
    recall_lista = []
    recomendacoes = []
    for usuario, grupo in catalogo_teste.groupby("userId"):
        relevantes = set(grupo.loc[grupo["rating"] >= LIMIAR_RELEVANTE, "movieId"].astype(int))
        ranking = grupo.sort_values("rating_previsto", ascending=False).head(N_RECOMENDACOES)
        recomendados = set(ranking["movieId"].astype(int))
        acertos = len(relevantes & recomendados)
        precisao_lista.append(acertos / len(recomendados) if recomendados else 0.0)
        recall_lista.append(acertos / len(relevantes) if relevantes else 0.0)
        for posicao, item in enumerate(ranking.itertuples(index=False), start=1):
            recomendacoes.append({
                "userId": int(usuario), "posicao": posicao, "movieId": int(item.movieId),
                "title": item.title, "nota_real_teste": float(item.rating),
                "nota_prevista": float(item.rating_previsto)
            })

    pd.DataFrame(recomendacoes).to_csv(SAIDA / "recomendacoes_teste.csv", index=False)
    metricas = {
        "semente": SEMENTE,
        "percentual_treino": 0.8,
        "percentual_teste": 0.2,
        "avaliacoes_treino": int(len(treino)),
        "avaliacoes_teste": int(len(teste)),
        "pares_similares_calculados": int(len(tabela_sim)),
        "k_vizinhos": K_VIZINHOS,
        "n_recomendacoes": N_RECOMENDACOES,
        "limiar_relevante": LIMIAR_RELEVANTE,
        "precisao_media_por_usuario": float(np.mean(precisao_lista)) if precisao_lista else 0.0,
        "recall_medio_por_usuario": float(np.mean(recall_lista)) if recall_lista else 0.0,
        "mae_previsao_notas": mae,
        "rmse_previsao_notas": rmse,
        "observacao": "Precisão/recall são calculados entre itens do conjunto de teste de cada usuário."
    }
    (SAIDA / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    treino.to_csv(SAIDA / "ratings_treino.csv", index=False)
    print("\nMétricas:")
    print(json.dumps(metricas, ensure_ascii=False, indent=2))
    print(f"\nArquivos gravados em: {SAIDA}")

if __name__ == "__main__":
    main()
