import joblib
from src.utils import get_best_model

MODEL_PATH = get_best_model()

pipeline = joblib.load(MODEL_PATH)

def predict(text: str) -> str:
    """
    Classifica uma notícia utilizando o modelo previamente treinado

    Args:
        text (str): texto da notícia a ser classificada.

    Returns:
        str: categoria prevista pelo modelo.
    """

    if not text or not text.strip():
        raise ValueError("O texto da notícia não pode ser vazio.")

    prediction = pipeline.predict([text])[0]

    return prediction