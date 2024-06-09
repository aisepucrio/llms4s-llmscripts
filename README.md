# Análise de Sentimentos e Avaliação de LLMs

Este repositório contém códigos para a análise de sentimentos em um dataset e para criação de matrizes de confusão para avaliar o desempenho de Large Language Models (LLMs).

## Estrutura do Repositório

- `docs/`: Contém os arquivos de documentação e requisitos.
- `src/`: Contém os códigos-fonte e scripts utilizados no projeto.
  - `prompts/`: Diretório onde os prompts devem ser adicionados.
  - `scripts/`: Scripts Python para processamento de dados, análise com LLMs e criação de matrizes de confusão.
- `tests/`: Possui os scripts de teste
  
## Requisitos

A primeira coisa que você precisa fazer é baixar os requirements na pasta de `docs` com o comando:

```sh
pip install -r docs/requirements.txt
```

## Utilização Dentro da pasta src, adicione os prompts na pasta prompts seguindo a numeração.

- Após adicionar os prompts, vá para a pasta scripts dentro de src.

- O código run_llm.py é responsável por analisar o dataset com as LLMs. Note que há uma lista chamada models no início do código onde você pode colocar os nomes dos modelos que deseja utilizar. No final do código, há a função main que recebe como parâmetro o prompt que você irá analisar, os modelos, e a quantidade de exemplos. Se você deixar o sample_size como None, ele fará a análise para o dataset completo.

- O código confusion_matrix.py irá gerar uma matriz de confusão para cada análise feita.
