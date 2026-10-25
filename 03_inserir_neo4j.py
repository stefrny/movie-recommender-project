

from pathlib import Path
import os

import pandas as pd
from neo4j import GraphDatabase


BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
SAIDA = BASE / "resultados"

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USUARIO = os.getenv("NEO4J_USER", "neo4j")
SENHA = os.getenv("NEO4J_PASSWORD", "troque_esta_senha")


def main():
    movies = pd.read_csv(DADOS / "movies.csv").fillna("")

    treino_path = SAIDA / "ratings_treino.csv"
    sim_path = SAIDA / "similaridades.csv"

    if not treino_path.exists() or not sim_path.exists():
        raise FileNotFoundError(
            "Execute primeiro: python 02_modelo_avaliacao.py"
        )

    ratings = pd.read_csv(treino_path)
    similarities = pd.read_csv(sim_path)

    medias_filmes = (
        ratings.groupby("movieId")["rating"]
        .mean()
        .to_dict()
    )

    media_global = float(ratings["rating"].mean())

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

    relacoes = []

    for r in similarities.itertuples(index=False):
        relacoes.append({
            "movieId1": int(r.movieId1),
            "movieId2": int(r.movieId2),
            "similarity": float(r.similarity),
            "co_ratings": int(r.co_ratings)
        })

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

            for inicio in range(0, len(filmes), 500):
                lote = filmes[inicio:inicio + 500]

                session.run("""
                    UNWIND $lote AS item
                    MERGE (f:Filme {movieId: item.movieId})
                    SET f.title = item.title,
                        f.genres = item.genres,
                        f.media_rating = item.media_rating
                """, lote=lote)

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
