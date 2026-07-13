from fastapi import FastAPI
from app.predictor import predict
from app.schemas import NewsRequest, NewsResponse

app = FastAPI(
    title="News Classifier API",
    description="API para classificação automática de notícias.",
    version="1.0"
)

@app.get("/",
    summary="Verifica o status da API",
    description="Endpoint utilizado para verificar se a aplicação está em execução."
)
def health() -> dict:
    """
    Verifica se a API está em execução

    Returns:
        dict: dicionário contendo o status da aplicação.
    """

    return {
        "status": "running"
    }

    return {
        "status": "running"
    }

@app.post( "/predict",
    response_model=NewsResponse,
    summary="Classifica uma notícia",
    description="Recebe o texto de uma notícia e retorna a categoria prevista pelo modelo treinado.",
    responses={
        200: {"description": "Predição realizada com sucesso."},
        422: {"description": "Requisição inválida."},
        500: {"description": "Erro interno do servidor."},
    }
)
def classify(news:NewsRequest) -> NewsResponse:
    """
    Classifica uma notícia em uma das categorias disponíveis

    Args:
        news (NewsRequest): objeto contendo o texto da notícia a ser classificada

    Returns:
        dict: dicionário contendo a categoria prevista pelo modelo.

    Raises:
        HTTPException: Retorna erro HTTP 500 caso ocorra uma falha durante a
                       classificação da notícia.
    """

    try:
        category = predict(news.text)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Erro interno durante a classificação."
        )

    return NewsResponse(category=category)