import os
import json
import re

from pathlib import Path


def file_exists(file_path:str) -> bool:
    """
    Verifica se um arquivo ou diretório existe

    Args:
        file_path (str): caminho do arquivo ou diretório

    Returns:
        bool: True caso o caminho exista, False caso contrário.
    """
    
    if os.path.exists(file_path):
        return True
    else:
        return False

def create_directory(dir_path='./models') -> None:
    """
    Cria um diretório caso ele ainda não exista

    Args:
        dir_path (str, optional): caminho do diretório a ser criado.

    Returns:
        None
    """

    if not file_exists(file_path=dir_path):
        os.makedirs(dir_path)

def save_metrics(metrics:dict, output_path:str) -> None:
    """
    Salva as métricas do modelo em um arquivo JSON

    Args:
        metrics (dict): dicionário contendo as métricas calculadas.
        output_path (str):Caminho do arquivo JSON de saída.

    Returns:
        None
    """

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

def get_best_model(models_dir:str="./models") -> str:
    """
    Retorna o caminho do modelo com maior F1-score encontrado no diretório

    Os modelos devem seguir o padrão de nomenclatura: news_classifier_f1_<f1>_<timestamp>.pkl
    
    Exemplo: news_classifier_f1_0.91_20260711_193649.pkl

    Args:
        models_dir (str, optional): diretório contendo os modelos treinados

    Returns:
        str: caminho do modelo com maior F1-score.

    Raises:
        FileNotFoundError: caso nenhum modelo compatível seja encontrado.
    """

    models = Path(models_dir).glob("news_classifier_f1_*.pkl")
    best_model = None
    best_f1 = -1.0
    pattern = re.compile(r"news_classifier_f1_([0-9]*\.?[0-9]+)_")

    for model in models:
        match = pattern.search(model.name)

        if match:
            f1 = float(match.group(1))
            if f1 > best_f1:
                best_f1 = f1
                best_model = model

    if best_model is None:
        raise FileNotFoundError(f"Modelo não encontrado em: {models_dir}")

    return str(best_model)

if __name__ == "__main__":

    best_model = get_best_model()
    print(best_model)