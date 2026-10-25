# Sistema de Recomendação de Filmes — MovieLens + Neo4j + KNN

Projeto acadêmico de recomendação colaborativa de filmes utilizando correlação de Pearson entre filmes, previsão de notas baseada em vizinhos (KNN) e armazenamento de dados em um grafo no Neo4j.

## 1. Tecnologias

- Python 3.10+
- pandas e NumPy para manipulação de dados e cálculos.
- Neo4j para armazenar e consultar o grafo.
- Driver oficial `neo4j` para Python.

Não são utilizadas bibliotecas prontas de recomendação, como scikit-learn. O cálculo das similaridades, as previsões e as métricas de avaliação são implementados no próprio projeto.

## 2. Dados

O script `01_baixar_dados.py` baixa o conjunto MovieLens Latest Small, disponibilizado pelo GroupLens.

Fonte: [MovieLens Latest Small — GroupLens](https://grouplens.org/datasets/movielens/latest/)

O conjunto contém aproximadamente 100 mil avaliações e os arquivos `movies.csv` e `ratings.csv`.

O projeto utiliza esses arquivos para calcular as similaridades entre filmes, gerar previsões e avaliar o modelo. Os CSVs não são versionados no Git, pois podem ser baixados novamente pelo script de preparação dos dados.

## 3. Modelagem do grafo

O grafo representa usuários, filmes, avaliações e similaridades entre filmes.

### Nós

**`(:Usuario)`**

- `userId`: identificador numérico único do usuário.

**`(:Filme)`**

- `movieId`: identificador numérico único do filme.
- `title`: título do filme.
- `genres`: lista de gêneros do filme.
- `media_rating`: média das avaliações do filme no conjunto de treino. Para filmes sem avaliações de treino, utiliza-se a média global das avaliações de treino.

### Relacionamentos

**`(:Usuario)-[:AVALIOU]->(:Filme)`**

- `rating`: nota atribuída pelo usuário ao filme, na escala de 0,5 a 5,0.
- `timestamp`: data e hora da avaliação, representadas pelo timestamp original do conjunto de dados, quando disponível.

**`(:Filme)-[:SIMILAR_A]->(:Filme)`**

- `pearson`: correlação de Pearson entre os dois filmes, calculada com as avaliações do conjunto de treino.
- `co_ratings`: quantidade de usuários que avaliaram ambos os filmes.

Cada par de filmes é armazenado uma vez, e a relação pode ser consultada nos dois sentidos utilizando `-[:SIMILAR_A]-`. Não são criados nós separados para cada avaliação, pois a nota é armazenada como propriedade do relacionamento entre usuário e filme.

### Esquema visual

```text
(:Usuario {userId})
      |
      | [:AVALIOU {rating, timestamp}]
      v
(:Filme {
    movieId,
    title,
    genres,
    media_rating
})
      |
      | [:SIMILAR_A {pearson, co_ratings}]
      v
(:Filme)
```

## 4. Instalação e execução

### Passo 1 — Instalar dependências e baixar os dados

Na pasta do projeto, execute:

```bash
pip install -r requirements.txt
python 01_baixar_dados.py
```

O primeiro comando instala as dependências necessárias. O segundo baixa e prepara os arquivos utilizados pelo modelo.

### Passo 2 — Treinar e avaliar o modelo

Execute:

```bash
python 02_modelo_avaliacao.py
```

Esse script divide as avaliações aleatoriamente em 80% para treino e 20% para teste, utilizando uma semente fixa para permitir a reprodutibilidade da divisão.

O modelo calcula as similaridades de Pearson entre filmes com pelo menos três coavaliações, utiliza até 20 vizinhos para realizar as previsões e gera métricas de avaliação.

Os arquivos de saída são armazenados na pasta `resultados/`.

- `ratings_treino.csv`: avaliações utilizadas no treinamento e na importação para o grafo.
- `similaridades.csv`: pares de filmes, correlação de Pearson e quantidade de coavaliações.
- `previsoes_teste.csv`: notas reais e previstas para as avaliações do conjunto de teste.
- `recomendacoes_teste.csv`: recomendações geradas a partir das previsões, conforme a implementação do script.
- `metricas.json`: resultados das métricas de precisão, recall, MAE e RMSE.

### Passo 3 — Configurar o Neo4j

Inicie uma instância local do Neo4j e configure o usuário e a senha de acesso.

Por padrão, o script utiliza:

- URI: `bolt://localhost:7687`
- Usuário: `neo4j`

Configure as variáveis de ambiente antes de executar a importação.

**Linux/macOS:**

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="SUA_SENHA"
```

**Windows PowerShell:**

```powershell
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="SUA_SENHA"
```

Substitua `SUA_SENHA` pela senha configurada no seu banco.

Em seguida, execute:

```bash
python 03_inserir_neo4j.py
```

O script cria as restrições e o índice necessários e importa os filmes, as avaliações de treino e as relações de similaridade.

A importação utiliza `MERGE` e restrições de unicidade para evitar a duplicação dos nós e das relações correspondentes quando o script é executado novamente.

Somente as avaliações do conjunto de treino são importadas para os relacionamentos `AVALIOU`, evitando que as notas do teste sejam utilizadas diretamente no grafo durante a recomendação.

### Passo 4 — Executar as consultas

Execute:

```bash
python 04_consultas_neo4j.py
```

Por padrão, o script utiliza o usuário de identificador `1`.

Para consultar outro usuário no Windows PowerShell:

```powershell
$env:USUARIO_X="10"
python 04_consultas_neo4j.py
```

O exemplo utiliza o usuário de identificador `10`. O resultado depende das avaliações e das relações de similaridade disponíveis para esse usuário.

## 5. Consultas Cypher úteis

As consultas abaixo podem ser executadas no Neo4j Browser, desde que a importação tenha sido concluída.

### Quantas comédias o usuário 1 avaliou com nota igual ou superior a 4?

```cypher
MATCH (u:Usuario {userId: 1})-[a:AVALIOU]->(f:Filme)
WHERE a.rating >= 4
  AND 'Comedy' IN f.genres
RETURN count(f) AS quantidade;
```

### Quais são essas comédias?

```cypher
MATCH (u:Usuario {userId: 1})-[a:AVALIOU]->(f:Filme)
WHERE a.rating >= 4
  AND 'Comedy' IN f.genres
RETURN f.title AS filme,
       a.rating AS nota
ORDER BY nota DESC, filme ASC;
```

### Recomendações para um usuário

A consulta correspondente está implementada no script `04_consultas_neo4j.py`.

O processo parte dos filmes avaliados positivamente pelo usuário, percorre as relações de similaridade, exclui os filmes que o usuário já avaliou e calcula uma nota estimada para os candidatos com vizinhos disponíveis.

A consulta retorna os filmes recomendados, suas notas estimadas e a quantidade de evidências utilizadas na previsão.

## 6. Método de recomendação

O projeto utiliza filtragem colaborativa baseada em itens (*item-based collaborative filtering*).

A similaridade entre dois filmes é calculada pela correlação de Pearson das avaliações dos usuários que avaliaram ambos os filmes. O cálculo utiliza somente as avaliações do conjunto de treino, e são considerados pares com pelo menos três coavaliações.

Para prever a nota de um usuário `u` para um filme-alvo `i`, o modelo seleciona até 20 filmes `j` que o usuário já avaliou e cuja similaridade com o filme-alvo esteja disponível.

A previsão implementada no modelo Python é:

\[
\hat r_{ui} =
\bar r_i +
\frac{
\sum_{j \in N_k(i;u)}
c(i,j)(r_{uj}-\bar r_j)
}{
\sum_{j \in N_k(i;u)} |c(i,j)|
}
\]

Em que:

- \(\hat r_{ui}\): nota prevista para o usuário `u` no filme `i`.
- \(\bar r_i\): nota média do filme-alvo no conjunto de treino.
- \(r_{uj}\): nota atribuída pelo usuário ao filme vizinho `j`.
- \(\bar r_j\): nota média do filme vizinho no conjunto de treino.
- \(c(i,j)\): correlação de Pearson entre o filme-alvo e o vizinho.
- \(N_k(i;u)\): conjunto de até 20 filmes avaliados pelo usuário, selecionados pela magnitude absoluta da similaridade.

A fórmula considera a diferença entre a nota atribuída pelo usuário e a média do filme vizinho. Essa diferença é ponderada pela similaridade entre os filmes e somada à média do filme-alvo.

O denominador utiliza a soma dos valores absolutos das similaridades para normalizar o ajuste.

As previsões são limitadas ao intervalo de 0,5 a 5,0. Quando não existem vizinhos úteis, o modelo utiliza a média do filme no treino; se essa média não estiver disponível, utiliza a média global das avaliações de treino.

Na consulta do Neo4j, as recomendações são direcionadas a filmes similares aos que o usuário avaliou positivamente. Por isso, a consulta de recomendações pode utilizar um subconjunto diferente de vizinhos daquele utilizado pelo avaliador Python, que considera os filmes avaliados pelo usuário com similaridade disponível.

## 7. Avaliação do modelo

O projeto utiliza as seguintes métricas:

- **Precisão@10:** proporção dos filmes recomendados no top 10 que são relevantes.
- **Recall@10:** proporção dos filmes relevantes recuperados entre os itens recomendados.
- **MAE (*Mean Absolute Error*):** média dos erros absolutos entre as notas previstas e reais.
- **RMSE (*Root Mean Squared Error*):** raiz quadrada da média dos erros quadráticos entre as notas previstas e reais.

Um filme é considerado relevante quando sua nota real no conjunto de teste é igual ou superior a 4.

### Limitações metodológicas

Na implementação atual, precisão e recall são calculados com base nos itens retidos no conjunto de teste de cada usuário, ordenados pela nota prevista. Portanto, essas métricas avaliam o ranking dentro dos itens de teste, não uma recomendação contra todo o catálogo de filmes não vistos.

Para avaliar recomendações contra o catálogo completo, seria necessário calcular previsões para os filmes não avaliados por cada usuário e comparar o top 10 resultante com os itens relevantes do conjunto de teste.

A divisão dos dados é aleatória por avaliação, e não temporal. Uma divisão temporal baseada nos timestamps seria uma possível extensão do projeto.

Os valores das métricas devem ser obtidos a partir da execução do script `02_modelo_avaliacao.py` e do arquivo `metricas.json`, sem inserir resultados não calculados.

## 8. Estrutura do repositório

```text
movie_recommender_project/
├── dados/
│   └── .gitkeep
├── resultados/
│   └── .gitkeep
├── 01_baixar_dados.py
├── 02_modelo_avaliacao.py
├── 03_inserir_neo4j.py
├── 04_consultas_neo4j.py
├── requirements.txt
├── .gitignore
└── README.md
```

Os diretórios `dados/` e `resultados/` armazenam os arquivos utilizados e gerados durante a execução. Os arquivos CSV e JSON podem ser recriados executando os scripts correspondentes.

## 9. O que apresentar na entrega

1. Explicar os nós, os relacionamentos e seus atributos.
2. Demonstrar a execução do script de importação no Neo4j.
3. Executar as consultas de comédias avaliadas com nota igual ou superior a 4 e de recomendações.
4. Mostrar o conteúdo do arquivo `metricas.json` e explicar o significado de precisão, recall, MAE e RMSE.
5. Explicar como a divisão entre treino e teste é realizada.
6. Informar que as similaridades e as médias utilizadas pelo modelo são calculadas com as avaliações de treino.
7. Apresentar as limitações metodológicas da avaliação de ranking.

Os resultados numéricos devem ser obtidos pela execução dos scripts. O README documenta o funcionamento do projeto e os procedimentos necessários para reproduzir os resultados.