"""Consultas de exemplo ao grafo Neo4j."""
import os
from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USUARIO = os.getenv("NEO4J_USER", "neo4j")
SENHA = os.getenv("NEO4J_PASSWORD", "Pass789123")

def executar(session, titulo, cypher, parametros):
    print(f"\n--- {titulo} ---")
    resultado = session.run(cypher, parametros)
    registros = [registro.data() for registro in resultado]
    if not registros:
        print("Nenhum resultado.")
    for registro in registros:
        print(registro)

def main():
    # Altere para um userId existente no MovieLens.
    usuario_x = int(os.getenv("USUARIO_X", "1"))
    limite = int(os.getenv("LIMITE_RECOMENDACOES", "10"))
    driver = GraphDatabase.driver(URI, auth=(USUARIO, SENHA))
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            executar(session, "Comédias avaliadas com nota >= 4 pelo usuário X", """
                MATCH (u:Usuario {userId: $userId})-[a:AVALIOU]->(f:Filme)
                WHERE a.rating >= 4 AND 'Comedy' IN f.genres
                RETURN f.title AS filme, a.rating AS nota
                ORDER BY nota DESC, filme
            """, {"userId": usuario_x})

            executar(session, "Quantidade de filmes de comédia avaliados positivamente", """
                MATCH (u:Usuario {userId: $userId})-[a:AVALIOU]->(f:Filme)
                WHERE a.rating >= 4 AND 'Comedy' IN f.genres
                RETURN count(f) AS quantidade
            """, {"userId": usuario_x})

            # Recomendações usando notas previstas já calculadas pelo modelo.
            # A consulta usa SIMILAR_A e ratings conhecidos pelo usuário.
            
            # Recomendações com média do filme-alvo e desvios das notas.
            executar(session, "Filmes similares a filmes bem avaliados pelo usuário", """
                MATCH (u:Usuario {userId: $userId})-[r:AVALIOU]->(visto:Filme)
            WHERE r.rating >= 4

            MATCH (visto)-[s:SIMILAR_A]-(candidato:Filme)
            WHERE NOT EXISTS {
                MATCH (u)-[:AVALIOU]->(candidato)
            }

            WITH candidato, visto, r, s
            ORDER BY abs(s.pearson) DESC

            WITH candidato,
                collect({
                    pearson: s.pearson,
                    nota: r.rating,
                    media_vizinho: visto.media_rating
                })[0..20] AS vizinhos

            WITH candidato,
                candidato.media_rating AS media_alvo,
                reduce(
                    num = 0.0,
                    v IN vizinhos |
                    num + v.pearson * (v.nota - v.media_vizinho)
                ) AS numerador,
                reduce(
                    den = 0.0,
                    v IN vizinhos |
                    den + abs(v.pearson)
                ) AS denominador,
                size(vizinhos) AS evidencias

            WHERE denominador > 0
            AND media_alvo IS NOT NULL

            WITH candidato,
                CASE
                    WHEN media_alvo + numerador / denominador < 0.5
                        THEN 0.5
                    WHEN media_alvo + numerador / denominador > 5.0
                        THEN 5.0
                    ELSE media_alvo + numerador / denominador
                END AS nota_estimada,
                evidencias

            RETURN candidato.movieId AS movieId,
                candidato.title AS filme,
                round(nota_estimada * 100) / 100.0 AS nota_estimada,
                evidencias

            ORDER BY nota_estimada DESC, evidencias DESC
            LIMIT $limite
            """, {"userId": usuario_x, "limite": limite})

    finally:
        driver.close()

if __name__ == "__main__":
    main()
