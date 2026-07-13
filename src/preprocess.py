import pandas as pd
import argparse
import sys

from src.utils import file_exists

def parse_args() -> argparse.Namespace:
    """
    Lê os parâmetros fornecidos pela linha de comando

    Returns:
        argparse.Namespace: Objeto contendo os parâmetros informados pelo usuário
    """

    parser = argparse.ArgumentParser(description='Parâmetros de pré-processamnto.')
    parser.add_argument("--min_samples_by_category", "-m", type=int, default=8, help="Quantidade mínima de amostras por categoria")

    return parser.parse_args()

def pre_process(df_path:str="./data/articles.csv", min_samples_by_category:int=8) -> None:
    """
    Realiza o pré-processamento do conjunto de dados utilizado para o
    treinamento do classificador.

    Args:
        df_path (str, optional): caminho para o dataset original.
        min_samples_by_category (int, optional): Número mínimo de notícias que uma categoria deve possuir para
                                                 permanecer no conjunto de dados
    Returns:
        None
    """

    if not file_exists(file_path=df_path):
        sys.exit(f"Arquivo {df_path} não encontrado. Adicione o arquivo para continuar")
    
    df = pd.read_csv(df_path)
    df_filtered = df.drop(columns=['date', 'subcategory', 'link'])
    min_samples = min_samples_by_category
    # Remove categorias com poucas amostras para reduzir o impacto de classes
    # extremamente raras durante o treinamento.
    category_counts = df_filtered['category'].value_counts()
    categories_to_keep = category_counts[category_counts >= min_samples].index
    df_filtered = df_filtered[df_filtered['category'].isin(categories_to_keep)]

    path_to_save = "./data/filtered_articles.csv"
    df_filtered.to_csv(path_to_save, index=False)
    print(f"Dataset pré-processado salvo em: {path_to_save}")
    

if __name__ == "__main__":

    args = parse_args()

    data_path = "./data/articles.csv"
    min_samples_by_category = args.min_samples_by_category

    pre_process(min_samples_by_category=min_samples_by_category)