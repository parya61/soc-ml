import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Относительные пути к файлам
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "soc_model_10f.pkl"
FEATURE_LIST_PATH = BASE_DIR / "models" / "feature_list_10.txt"



# Загрузка списка фич 
with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
    feature_list = [col.strip() for col in f.read().split(",")]

# Редактируем название фичей через словарь
feature_names_map = {
    "Packet Length Std": "Стандартное отклонение длины пакетов",
    "Packet Length Variance": "Дисперсия длины пакетов",
    "Avg Bwd Segment Size": "Средний размер сегмента (Bwd)",
    "Max Packet Length": "Максимальная длина пакета",
    "Bwd Packet Length Max": "Макс. длина пакета (Bwd)",
    "Bwd Packet Length Std": "Ст. отклонение длины пакета (Bwd)",
    "Average Packet Size": "Средний размер пакета",
    "Total Length of Bwd Packets": "Суммарная длина пакетов (Bwd)",
    "Total Length of Fwd Packets": "Суммарная длина пакетов (Fwd)",
    "Subflow Bwd Bytes": "Байт в подпотоке (Bwd)"
}

# Загрузка фичей
model = joblib.load(MODEL_PATH)
model_classes = list(model.classes_)  # Классы, на которых обучалась модель

st.title("🔍 SOC Анализ трафика")
st.write("Введите данные вручную, загрузите CSV или используйте примеры для теста.")

# Делаем примеры для тестового раздела
example_normal = {f: 0.0 for f in feature_list}
example_attack = {
    "Packet Length Std": 0.5,
    "Packet Length Variance": 1.2,
    "Avg Bwd Segment Size": 300.0,
    "Max Packet Length": 1500,
    "Bwd Packet Length Max": 1200,
    "Bwd Packet Length Std": 200.0,
    "Average Packet Size": 1000.0,
    "Total Length of Bwd Packets": 5000,
    "Total Length of Fwd Packets": 7000,
    "Subflow Bwd Bytes": 4500,
}

# Функция предсказания
def predict_and_show(input_df):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    if "BENIGN" in model_classes:
        benign_index = model_classes.index("BENIGN")
        attack_proba = 1 - proba[benign_index]
    else:
        # Бинарная модель
        attack_proba = proba[1] if len(model_classes) == 2 else None

    if pred != "BENIGN" and pred != 0:
        if attack_proba is not None:
            st.error(f"⚠ Обнаружена Атака: **{pred}** (Вероятность {attack_proba*100:.2f}%)")
        else:
            st.error(f"⚠ Обнаружена Атака: **{pred}**")
    else:
        if attack_proba is not None:
            st.success(f"✅ Трафик нормальный (Вероятность {proba[benign_index]*100:.2f}%)")
        else:
            st.success(f"✅ Трафик нормальный")

# Выбор ввода
mode = st.radio("Выберите режим ввода", ["Ввести вручную", "Загрузить CSV", "Примеры для теста"])

if mode == "Ввести вручную":
    input_data = {}
    cols = st.columns(2)
    for i, feature in enumerate(feature_list):
        col = cols[i % 2]
        label = feature_names_map.get(feature, feature)
        input_data[feature] = col.number_input(label, value=0.0)

    if st.button("Сделать предсказание"):
        input_df = pd.DataFrame([input_data])
        predict_and_show(input_df)

elif mode == "Загрузить CSV":
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        missing = [f for f in feature_list if f not in df.columns]
        if missing:
            st.error(f"В файле нет нужных колонок: {missing}")
        else:
            df = df[feature_list].apply(pd.to_numeric, errors="coerce").fillna(0)
            if st.button("Сделать предсказание для файла"):
                preds = model.predict(df)
                probs = model.predict_proba(df)
                
                if "BENIGN" in model_classes:
                    benign_index = model_classes.index("BENIGN")
                    attack_probs = 1 - probs[:, benign_index]
                else:
                    attack_probs = probs[:, 1] if len(model_classes) == 2 else np.nan

                result_df = df.copy()
                result_df["Prediction"] = preds
                result_df["Attack Probability"] = attack_probs
                st.write(result_df)
                st.download_button(
                    "Скачать результаты",
                    result_df.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv"
                )

else:  # Примеры
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Пример нормального трафика"):
            df_ex = pd.DataFrame([example_normal])
            st.write(df_ex)
            predict_and_show(df_ex)
    with col2:
        if st.button("Пример атаки"):
            df_ex = pd.DataFrame([example_attack])
            st.write(df_ex)
            predict_and_show(df_ex)
