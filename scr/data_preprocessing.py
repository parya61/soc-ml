import os
import pandas as pd

def load_and_combine_data(data_dir):
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    df_list = []

    for file in all_files:
        file_path = os.path.join(data_dir, file)
        print(f"Загружаю: {file}")
        df = pd.read_csv(file_path)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def preprocess_data(df):
    # Пример простой предобработки
    df.drop_duplicates(inplace=True)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    return df

if __name__ == "__main__":
    data_path = os.path.join("..", "data", "first_data")
    df = load_and_combine_data(data_path)
    df = preprocess_data(df)

    output_path = os.path.join("..", "data", "eda", "combined.csv")
    df.to_csv(output_path, index=False)
    print(f"Сохранено в: {output_path}")
