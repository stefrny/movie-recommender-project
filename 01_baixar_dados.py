
"""Baixa o MovieLens Small e extrai movies.csv e ratings.csv."""
import ssl
import certifi
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

BASE = Path(__file__).resolve().parent
PASTA_DADOS = BASE / "dados"
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def main():
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)

    movies = PASTA_DADOS / "movies.csv"
    ratings = PASTA_DADOS / "ratings.csv"

    if movies.exists() and ratings.exists():
        print("Os arquivos movies.csv e ratings.csv já existem.")
        print("Nenhum download é necessário.")
        return

    print("Baixando MovieLens Small...")
    contexto_ssl = ssl.create_default_context(cafile=certifi.where())

    with urlopen(URL, timeout=60, context=contexto_ssl) as resposta:
        conteudo = resposta.read()

    with ZipFile(BytesIO(conteudo)) as arquivo_zip:
        for nome in ("movies.csv", "ratings.csv"):
            caminho_interno = next(
                (n for n in arquivo_zip.namelist()
                 if n.endswith("/" + nome)),
                None
            )
            if caminho_interno is None:
                raise FileNotFoundError(f"{nome} não foi encontrado no ZIP.")

            destino = PASTA_DADOS / nome
            destino.write_bytes(arquivo_zip.read(caminho_interno))
            print(f"Arquivo salvo: {destino}")

    print("Download concluído.")


if __name__ == "__main__":
    main()

