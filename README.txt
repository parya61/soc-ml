SOC ML — Детекция сетевых атак с помощью машинного обучения

Проект демонстрирует, как можно использовать машинное обучение для анализа сетевого трафика и обнаружения аномалий в корпоративных сетях. Подходит как демонстрация навыков data science, feature engineering и развертывания модели.

---
Стек технологий
- Python 3.10+
- Pandas / NumPy / Scikit-learn — обработка данных и обучение модели  
- Matplotlib / Seaborn — визуализация  
- Streamlit** — веб-интерфейс для демонстрации модели  
- Git LFS** — хранение больших файлов (модель и датасеты)

---

Что внутри
- notebooks/ — ноутбуки с EDA, отбором признаков и обучением модели  
- models/ — сохраненные модели и списки признаков  
- scr/ — скрипты предобработки данных  
- app.py — веб-интерфейс на Streamlit  
- requirements.txt — зависимости проекта  

---

 Как запустить локально
bash
# 1. Клонируйте репозиторий
git clone https://github.com/parya61/soc-ml.git
cd soc-ml

# 2. Создайте и активируйте виртуальное окружение
python -m venv venv
venv\Scripts\activate      # для Windows
source venv/bin/activate   # для Linux/Mac

# 3. Установите зависимости
pip install -r requirements.txt

# 4. Запустите веб-приложение
streamlit run app.py
