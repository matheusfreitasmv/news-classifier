import pandas as pd
import argparse
import os
import joblib

from src.utils import file_exists, create_directory, save_metrics
from src.preprocess import pre_process
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

def create_pipeline(random_state:int=42) -> Pipeline:
    """
    Cria o pipeline de treinamento utilizado para classificação de notícias

    O pipeline é composto por duas etapas:
        1. Vetorização dos textos utilizando TF-IDF.
        2. Classificação utilizando LinearSVC.

    Args:
        random_state (int, optional): seed utilizada para garantir a reprodutibilidade do treinamento

    Returns:
        Pipeline: Pipeline do Scikit-learn contendo o vetor TF-IDF e o classificador
    """

    # Preparando as configurações da vetorização e do modelo
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1,1),
                min_df=3,
                max_df=0.95,
                max_features=20000
            )
        ),
        (
            "classifier",
            LinearSVC( 
                random_state=random_state,
                max_iter=200
            )
        )
    ])

    return pipeline

def print_metrics(metrics:dict, y_true:list, y_pred:list) -> None:

    """
    Exibe no terminal as métricas de desempenho do modelo

    Args:
        metrics (dict): dicionário contendo as métricas agregadas do modelo
        y_true (list): lista com os rótulos reais do conjunto de teste
        y_pred (list): Lista com as categorias previstas pelo modelo

    Returns:
        None
    """

    print("\nMédias macro:")
    print("-" * 50)
    print(f"Acurácia:  {metrics['accuracy']:.2f}")   
    print(f"Precisão:  {metrics['precision_macro']:.2f}") 
    print(f"Revocação: {metrics['recall_macro']:.2f}")    
    print(f"F1-score:  {metrics['f1_macro']:.2f}")
        
    print("\nMétricas para cada classe:")
    print("-" * 50)
    print( classification_report(y_true, y_pred, zero_division=0) )
    print()

def parse_args() -> argparse.Namespace:
    """
    Lê os parâmetros fornecidos pela linha de comando

    Returns:
        argparse.Namespace: objeto contendo os parâmetros informados pelo usuário
    """

    parser = argparse.ArgumentParser(description='Parâmetros para treinar o modelo')
    parser.add_argument("--df_filtered_path", type=str, default="./data/filtered_articles.csv", 
                        help="Caminho para o dadaset filtrado")
    parser.add_argument("--train_size", type=float, default=0.2, 
                        help="Percentual dos dados de treino")
    parser.add_argument("--random_state", type=int, default=42, 
                        help="Seed padrão utilizada")

    return parser.parse_args()

def train_model(df_filtered_path:str="./data/filtered_articles.csv",
                train_size:int=0.8, random_state:int=42
                ) -> None:
    """
    Treina um modelo para classificação de categorias de notícias.
    Caso o conjunto de dados filtrado não exista, o pré-processamento é
    executado automaticamente.

    Args:
        df_filtered_path (str, optional): caminho para o dataset pré-processado
        train_size (float, optional): proporção de amostras utilizada para treinamento.
        random_state (int, optional): seed utilizada para garantir reprodutibilidade.

    Returns:
        None
    """

    if not file_exists(file_path=df_filtered_path): 
        pre_process()

    df_filtered = pd.read_csv(df_filtered_path)

    df_filtered["input"] = (
        df_filtered["title"].fillna("") + ". " +
        df_filtered["text"].fillna("")
    ).str.strip()

    X = df_filtered["input"]
    y = df_filtered['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        stratify=df_filtered["category"],
        random_state=random_state
    )

    print(f"\nQuantidade de amostras de treino: {len(X_train)}")
    print(f"Quantidade de amostras de teste: {len(X_test)}")
    print("\nCategorias de notícias utilizadas para treino:")
    print("-" * 50)
    for category in sorted(df_filtered['category'].unique()):
        print(category)
    
    pipeline = create_pipeline(random_state=random_state) 
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    print_metrics(metrics=metrics, y_true=list(y_test), y_pred=list(y_pred))

    # Salvando o modelo treinado
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_to_save = "./models"
    create_directory(dir_path=dir_to_save)
    filename_model = f"{dir_to_save}/news_classifier_f1_{metrics['f1_macro']:.2f}_{timestamp}.pkl"
    filename_metrics = f"{dir_to_save}/news_classifier_f1_{metrics['f1_macro']:.2f}_{timestamp}.json"
    joblib.dump(pipeline, filename_model)

    save_metrics(metrics=metrics, output_path=filename_metrics)

    print(f"Modelo salvo em: {filename_model}\n")
    print(f"Métricas salvas em: {filename_metrics}\n")

if __name__ == "__main__":
    args = parse_args()
    df_filtered_path = args.df_filtered_path
    train_size = args.train_size
    random_state = args.random_state

    train_model(df_filtered_path=df_filtered_path, train_size=train_size, 
                random_state=random_state)
    
