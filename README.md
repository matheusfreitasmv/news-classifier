# News Classifier API

Classificador de categorias de notícias.
O projeto implementa um pipeline completo de Machine Learning para classificação automática de notícias, contemplando desde a análise exploratória dos dados até a disponibilização do modelo por meio de uma API REST desenvolvida com **FastAPI**.

# Objetivo

O objetivo deste projeto é desenvolver um modelo capaz de classificar automaticamente notícias em suas respectivas categorias utilizando técnicas de Processamento de Linguagem Natural (NLP).

O projeto contempla todas as etapas necessárias para uma solução completa:

- Análise Exploratória dos Dados (EDA);
- Pré-processamento dos textos;
- Engenharia de atributos;
- Treinamento do modelo;
- Avaliação utilizando métricas de classificação;
- Persistência do modelo treinado;
- Disponibilização do modelo através de uma API REST;
- Containerização da aplicação utilizando Docker.

#  Demonstração

A API foi disponibilizada temporariamente para fins de demonstração do projeto. Os links abaixo poderão ficar indisponíveis após o período de avaliação.

- **API:** https://news-classifier-ejgc.onrender.com
- **Documentação (Swagger):** https://news-classifier-ejgc.onrender.com/docs


#  Dataset

O conjunto de dados utilizado está disponível publicamente no [Kaggle](https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol/data
).


O dataset contém aproximadamente **167 mil notícias** publicadas pela **Folha de São Paulo**, contendo as seguintes informações:

- title
- text
- date
- category
- subcategory
- link

Após o pré-processamento, foram utilizadas apenas as informações consideradas relevantes para a tarefa de classificação.

# Tecnologias Utilizadas

- Python 3.11
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- Joblib
- Docker

#  Estrutura do Projeto

```text
.
├── app/
│   ├── __init__.py
│   ├── main.py              # API FastAPI
│   ├── predictor.py         # Carregamento do modelo e inferência
│   └── schemas.py           # Modelos Pydantic
│
├── data/
│   └── articles.csv
│
├── models/
│   ├── *.pkl                # Modelos treinados
│   └── *.json               # Métricas dos modelos
│
├── notebooks/
│   ├── eda_and_preprocess.ipynb
│   └── train.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

# Pipeline de Machine Learning

O modelo foi desenvolvido utilizando o **Pipeline** do Scikit-learn, garantindo que todas as etapas de pré-processamento aplicadas durante o treinamento também sejam executadas durante a inferência.

O pipeline é composto por:

1. Vetorização dos textos utilizando **TF-IDF**;
2. Classificação utilizando **LinearSVC**.

Essa abordagem permite encapsular todo o fluxo de processamento em um único objeto, simplificando tanto o treinamento quanto a utilização do modelo em produção.

# Resultados

O modelo final foi avaliado utilizando um conjunto de teste contendo **33.409 notícias**.

| Métrica | Valor |
|---------|------:|
| **Accuracy** | **0.87** |
| **Precision (Macro)** | **0.58** |
| **Recall (Macro)** | **0.46** |
| **F1-score (Macro)** | **0.49** |
| **Precision (Weighted)** | **0.86** |
| **Recall (Weighted)** | **0.87** |
| **F1-score (Weighted)** | **0.86** |

A diferença entre as métricas **Macro** e **Weighted** é explicada pelo forte desbalanceamento entre as categorias do conjunto de dados. Enquanto o modelo apresenta bom desempenho nas classes mais representativas, seu desempenho é inferior em categorias com poucas amostras.

<details>

<summary><strong>Classification Report</strong></summary>

```text
                             precision    recall  f1-score   support

ambiente                        0.49      0.47      0.48        98
asmais                          0.45      0.08      0.14       110
banco-de-dados                  1.00      0.08      0.14        13
bbc                             0.67      0.31      0.42       196
cenarios-2017                   0.00      0.00      0.00         9
ciencia                         0.68      0.64      0.66       267
colunas                         0.84      0.84      0.84      4324
comida                          0.68      0.63      0.65       166
cotidiano                       0.86      0.89      0.87      3393
dw                              0.00      0.00      0.00        10
educacao                        0.78      0.87      0.82       424
empreendedorsocial              0.81      0.73      0.76       168
equilibrioesaude                0.65      0.66      0.65       262
especial                        0.00      0.00      0.00         9
esporte                         0.95      0.98      0.97      3946
euronews                        0.00      0.00      0.00         1
folhinha                        0.78      0.63      0.70       175
guia-de-livros-discos-filmes    0.65      0.38      0.48        29
guia-de-livros-filmes-discos    0.00      0.00      0.00         6
ilustrada                       0.84      0.92      0.88      3269
ilustrissima                    0.78      0.46      0.58       282
infograficos                    0.00      0.00      0.00         9
mercado                         0.85      0.87      0.86      4194
mulher                          0.00      0.00      0.00         3
multimidia                      0.00      0.00      0.00         5
mundo                           0.87      0.93      0.90      3426
o-melhor-de-sao-paulo           0.42      0.13      0.20        38
opiniao                         0.97      0.91      0.94       905
paineldoleitor                  0.99      0.96      0.98       802
poder                           0.89      0.90      0.90      4404
rfi                             0.00      0.00      0.00         6
saopaulo                        0.84      0.79      0.81       791
seminariosfolha                 0.87      0.51      0.64        76
serafina                        0.25      0.01      0.03        67
sobretudo                       0.78      0.64      0.70       211
tec                             0.74      0.69      0.71       452
topofmind                       0.92      0.65      0.76        17
treinamento                     1.00      0.25      0.40         4
treinamentocienciaesaude        0.00      0.00      0.00         4
turismo                         0.79      0.73      0.76       381
tv                              0.81      0.58      0.68       428
vice                            0.43      0.10      0.17        29

accuracy                                            0.87     33409
macro avg                       0.58      0.46      0.49     33409
weighted avg                    0.86      0.87      0.86     33409
```

</details> <br>

Analisando o _Classification Report_ acima, percebe-se que o modelo obteve **87% de acurácia** e **F1-score ponderado de 0,86**, indicando bom desempenho geral. Entretanto, o **F1-score Macro de 0,49** evidencia dificuldades nas categorias com poucas amostras, refletindo o forte desbalanceamento do conjunto de dados. Em contrapartida, categorias com maior representatividade, como **esporte**, **poder**, **mercado**, **mundo**, **cotidiano** e **ilustrada**, apresentaram F1-scores superiores a 0,85.

# Como Executar o Projeto

## 1. Clonar o repositório

```bash
git clone https://github.com/matheusfreitasmv/news-classifier.git

cd news-classifier
```

## 2. Criar um ambiente virtual

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

# Treinamento

Antes de realizar o treinamento, certifique-se de que o dataset `articles.csv` encontra-se no diretório `data/`. 

Para treinar um novo modelo execute:

```bash
python -m src.train
```

Ao final do treinamento serão gerados automaticamente:

- modelo treinado (`.pkl`);
- arquivo contendo as métricas (`.json`).

Os arquivos serão armazenados no diretório:

```text
models/
```

# Predição

Para realizar inferências utilizando um modelo treinado execute:

```bash
python -m src.predict
```

O script carregará automaticamente o modelo com maior F1-score presente no diretório `models/` (caso nenhum modelo seja informado) e solicitará que um texto seja digitado no terminal.

Exemplo:

```text
Digite um texto: O Brasil conquista mais uma medalha olímpica
Categoria prevista: esporte

Digite um texto: O dólar fecha em alta nesta sexta-feira
Categoria prevista: mercado
```

Caso deseje utilizar um modelo específico, informe seu caminho por meio do parâmetro `--model_path`:

```bash
python -m src.predict --model_path ./models/news_classifier_f1_0.87_20260712_103015.pkl
```

# Executando a API

Inicie a API utilizando:

```bash
uvicorn app.main:app --reload
```

A API ficará disponível em:

```
http://127.0.0.1:8000
```

# Documentação da API

A documentação interativa é gerada automaticamente pelo FastAPI.

Swagger UI

```
http://127.0.0.1:8000/docs
```

ReDoc

```
http://127.0.0.1:8000/redoc
```


# Endpoints

## GET /

Verifica se a API está em execução.

### Resposta

```json
{
    "status": "running"
}
```

---

## POST /predict

Realiza a classificação de uma notícia.

### Corpo da requisição

```json
{
    "text": "O Brasil vence a Copa do Mundo e conquista o hexa."
}
```

### Resposta

```json
{
    "category": "esporte"
}
```

---

# Executando com Docker

## Construindo a imagem

```bash
docker build -t news-classifier .
```

## Executando o container

```bash
docker run -p 8000:8000 news-classifier
```

A API ficará disponível em:

```
http://localhost:8000
```

Documentação:

```
http://localhost:8000/docs
```


# Decisões de Projeto

Durante o desenvolvimento optou-se por utilizar um pipeline baseado em **TF-IDF** e **LinearSVC**.

Essa combinação apresenta bom desempenho para tarefas de classificação de textos, além de oferecer:

- Baixo custo computacional;
- Treinamento rápido;
- Inferência eficiente;
- Simplicidade de implantação;
- Facilidade de manutenção.

A utilização do `Pipeline` do Scikit-learn garante que todas as etapas de processamento sejam executadas de forma consistente tanto no treinamento quanto na inferência.

Além disso, o projeto foi estruturado de forma modular, separando claramente as etapas de treinamento, avaliação e disponibilização do modelo através da API.

#  Limitações e Trabalhos Futuros

Embora o modelo desenvolvido apresente uma solução funcional e adequada para o problema proposto, diversas melhorias podem ser exploradas em trabalhos futuros.

Entre elas destacam-se:

- Otimização dos hiperparâmetros utilizando Grid Search;
- Utilização de validação cruzada;
- Comparação entre diferentes algoritmos de classificação;
- Utilização de modelos baseados em Transformers, como BERTimbau;
- Utilização de embeddings contextualizados;

