import pandas as pd
import json

def load_word_data(word_list_path, corpus_path):
    """Carga listas de palabras y el corpus normativo."""
    words_df = pd.read_csv(word_list_path)
    corpus_df = pd.read_csv(corpus_path)

    merged_data = words_df.merge(corpus_df, on="palabra", how="left")
    print(f"Corpus combinado: {merged_data.shape}")
    return merged_data

def save_to_json(data, output_file):
    """Guarda el corpus en formato JSON."""
    data.to_json(output_file, orient="records", indent=4)
    print(f"Corpus guardado en {output_file}")

if __name__ == "__main__":
    word_list_file = "data/raw/word_lists.csv"
    corpus_file = "data/processed/corpus_normativo.csv"

    corpus_data = load_word_data(word_list_file, corpus_file)
    save_to_json(corpus_data, "data/processed/corpus_final.json")
