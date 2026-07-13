import pandas as pd
import argparse
import joblib
import sys

from src.utils import file_exists, get_best_model
from typing import Iterable

def parse_args() -> argparse.Namespace:
    """
    Lê os parâmetros fornecidos pela linha de comando

    Returns:
        argparse.Namespace: objeto contendo os parâmetros informados pelo usuário
    """

    parser = argparse.ArgumentParser(description='Parâmetros para avaliar um modelo.')
    parser.add_argument("--model_path", type=str, 
                        default='./models/news_classifier_f1_0.49_20260711_104852.pkl', 
                        help="Caminho para o modelo treinado para classificar categorias de notícias.")

    return parser.parse_args()

def predict_batch(texts: Iterable[str], model_path:str=None) -> list[str]:
    """
    Classifica um conjunto de notícias utilizando um modelo previamente treinado.
    Caso nenhum caminho para o modelo seja informado, o modelo com maior
    F1-score presente no diretório de modelos é carregado automaticamente.

    Args:
        texts (Iterable[str]): coleção contendo os textos das notícias a serem classificadas.
        model_path (str, optional): caminho para o modelo treinado. Se None, o modelo com maior
                                    F1-score é selecionado automaticamente.
    Returns:
        numpy.ndarray:
            Vetor contendo as categorias previstas para cada texto informado.

    Raises:
        SystemExit: caso o arquivo do modelo não seja encontrado.
    """
    
    if model_path is None:
        model_path = get_best_model()
    
    if file_exists(file_path=model_path):
        pipeline = joblib.load(model_path)
        predictions = pipeline.predict(texts)
    else:
        sys.exit(f"Arquivo não encontrado: {model_path}")

    return predictions.tolist()

if __name__ == "__main__":

    args = parse_args()
    model_path = args.model_path

    """
    texts = [
            'Samara caiu de bicicleta hoje', 
            'Hoje o dia é de chuva', 
            'O Brasil vence a copa do mundo e conquita o hexa'
            ]

    predictions = predict_batch(model_path=model_path, texts=texts)

    for text, pred in zip(texts, predictions):
        print(f"Texto: {text}")
        print(f"Categoria prevista: {pred}")
        print("-" * 50)
    """

    while(1):
        text = input("Digite um texto: ")
        if not text.strip():
            print("O texto da notícia não pode ser vazio.")
        else:
            prediction = predict_batch(model_path=model_path, texts=[text])
            print(f"Categoria prevista: {prediction[0]}")
        print("-" * 50)
        print()




