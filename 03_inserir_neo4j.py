
"""Insere filmes, usuários, avaliações de treino e similaridades no Neo4j."""

from pathlib import Path
import os

import pandas as pd
from neo4j import GraphDatabase


# Caminhos do projeto
BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
SAIDA = BASE / "resultados"

# Configuração de conexão com o Neo4j
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USUARIO = os.getenv("NEO4J_USER", "neo4j")
SENHA = os.getenv("NEO4J_PASSWORD", "troque_esta_senha")


def main():
    # Carrega os arquivos de dados
    movies = pd.read_csv(DADOS / "movies.csv").fillna("")

    treino_path = SAIDA / "ratings_treino.csv"
    sim_path = SAIDA / "similaridades.csv"

    if not treino_path.exists() or not sim_path.exists():
        raise FileNotFoundError(
            "Execute primeiro: python 02_modelo_avaliacao.py"
        )

    ratings = pd.read_csv(treino_path)
    similarities = pd.read_csv(sim_path)

    # Calcula a nota média de cada filme usando somente os dados de treino.
    medias_filmes = (
        ratings.groupby("movieId")["rating"]
        .mean()
        .to_dict()
    )

    # Média global usada como alternativa para filmes sem avaliações de treino.
    media_global = float(ratings["rating"].mean())

    # Prepara os filmes e seus gêneros.
    filmes = []

    for row in movies.to_dict("records"):
        generos = [
            genero
            for genero in str(row["genres"]).split("|")
            if genero and genero != "(no genres listed)"
        ]

        movie_id = int(row["movieId"])

        filmes.append({
            "movieId": movie_id,
            "title": str(row["title"]),
            "genres": generos,
            "media_rating": float(
                medias_filmes.get(movie_id, media_global)
            )
        })

    # Prepara as avaliações dos usuários.
    avaliacoes = []

    for r in ratings.itertuples(index=False):
        avaliacoes.append({
            "userId": int(r.userId),
            "movieId": int(r.movieId),
            "rating": float(r.rating),
            "timestamp": (
                int(r.timestamp)
                if hasattr(r, "timestamp")
                else None
            )
        })

    # Prepara as relações de similaridade entre filmes.
    relacoes = []

    for r in similarities.itertuples(index=False):
        relacoes.append({
            "movieId1": int(r.movieId1),
            "movieId2": int(r.movieId2),
            "similarity": float(r.similarity),
            "co_ratings": int(r.co_ratings)
        })

    # Conecta ao Neo4j e importa os dados.
    driver = GraphDatabase.driver(
        URI,
        auth=(USUARIO, SENHA)
    )

    try:
        driver.verify_connectivity()

        with driver.session() as session:
            # Restrições e índice para melhorar as consultas.
            session.run("""
                CREATE CONSTRAINT usuario_id IF NOT EXISTS
                FOR (u:Usuario)
                REQUIRE u.userId IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT filme_id IF NOT EXISTS
                FOR (f:Filme)
                REQUIRE f.movieId IS UNIQUE
            """)

            session.run("""
                CREATE INDEX filme_titulo IF NOT EXISTS
                FOR (f:Filme)
                ON (f.title)
            """)

            # Importa os filmes, incluindo suas notas médias.
            for inicio in range(0, len(filmes), 500):
                lote = filmes[inicio:inicio + 500]

                session.run("""
                    UNWIND $lote AS item
                    MERGE (f:Filme {movieId: item.movieId})
                    SET f.title = item.title,
                        f.genres = item.genres,
                        f.media_rating = item.media_rating
                """, lote=lote)

            # Importa os usuários e suas avaliações.
            for inicio in range(0, len(avaliacoes), 1000):
                lote = avaliacoes[inicio:inicio + 1000]

                session.run("""
                    UNWIND $lote AS item
                    MERGE (u:Usuario {userId: item.userId})
                    WITH u, item
                    MATCH (f:Filme {movieId: item.movieId})
                    MERGE (u)-[a:AVALIOU]->(f)
                    SET a.rating = item.rating,
                        a.timestamp = item.timestamp
                """, lote=lote)

            # Importa as similaridades de Pearson entre os filmes.
            for inicio in range(0, len(relacoes), 1000):
                lote = relacoes[inicio:inicio + 1000]

                session.run("""
                    UNWIND $lote AS item
                    MATCH (a:Filme {movieId: item.movieId1})
                    MATCH (b:Filme {movieId: item.movieId2})
                    MERGE (a)-[s:SIMILAR_A]->(b)
                    SET s.pearson = item.similarity,
                        s.co_ratings = item.co_ratings
                """, lote=lote)

        print(
            "Importação concluída: "
            f"{len(filmes)} filmes, "
            f"{len(avaliacoes)} avaliações de treino, "
            f"{len(relacoes)} relações de similaridade."
        )

    finally:
        driver.close()


if __name__ == "__main__":
    main()
