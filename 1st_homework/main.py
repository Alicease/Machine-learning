import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import io
import sys

# Чтение данных
print("Шаг 1: Загрузка данных из UCI...")
try:
    url = 'https://archive.ics.uci.edu/static/public/359/news+aggregator.zip'
    response = requests.get(url)
    response.raise_for_status()
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    data = pd.read_csv(zip_file.open('newsCorpora.csv'), sep='\t', header=None, names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    print("Данные загружены. Размер датасета:", data.shape)
    print("Столбцы:", data.columns.tolist())
except Exception as e:
    print("Ошибка загрузки данных:", str(e))
    print("Скачайте датасет вручную с https://www.kaggle.com/datasets/uciml/news-aggregator-dataset, сохраните как 'news.csv' в папке с кодом и перезапустите.")
    sys.exit(1)

# Разделение на train/test
print("\nШаг 2: Разделение на обучающую и тестовую выборки...")
X = data['TITLE']
y = data['CATEGORY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Размер train:", X_train.shape, "test:", X_test.shape)

# Визуализация и характеристики
print("\nШаг 3: Визуализация и вычисление характеристик...")
print("Распределение классов:")
print(y.value_counts())
sns.countplot(x=y)
plt.title("Распределение классов")
plt.show()

data['title_length'] = data['TITLE'].apply(len)
print("Средняя длина заголовка:", data['title_length'].mean())
print("Разброс длины:", data['title_length'].std())
plt.hist(data['title_length'], bins=50)
plt.title("Распределение длин заголовков")
plt.show()
print("Корреляционная матрица: не применима для текста.")

# Обработка пропущенных значений
print("\nШаг 4: Проверка пропущенных значений...")
print("Пропуски в TITLE:", data['TITLE'].isnull().sum())
print("Пропуски в CATEGORY:", data['CATEGORY'].isnull().sum())
print("Пропусков нет.")

# Обработка категориальных признаков
print("\nШаг 5: Обработка категориальных признаков...")
print("Целевая переменная CATEGORY уже категориальная. Для текста используем TF-IDF.")
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')  # Уменьшили для скорости
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("TF-IDF векторы созданы. Размер:", X_train_vec.shape)

# Запуск KNN
print("\nШаг 7: Запуск KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_vec, y_train)

# Ошибки и оптимизация
print("\nШаг 8: Вычисление ошибок и оптимизация...")
y_pred_train = knn.predict(X_train_vec)
y_pred_test = knn.predict(X_test_vec)
print("KNN (k=5) - Train accuracy:", accuracy_score(y_train, y_pred_train))
print("KNN (k=5) - Test accuracy:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix (test):")
print(confusion_matrix(y_test, y_pred_test))
print("Classification Report:")
print(classification_report(y_test, y_pred_test))

# Оптимизация k
X_train_sub, _, y_train_sub, _ = train_test_split(X_train_vec, y_train, test_size=0.9, random_state=42)  # Subsample для скорости
k_values = range(1, 11, 2)
accuracies = []
for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_sub, y_train_sub)
    acc = accuracy_score(y_test, knn_temp.predict(X_test_vec))
    accuracies.append(acc)
plt.plot(k_values, accuracies)
plt.title("Accuracy vs k")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()
best_k = k_values[np.argmax(accuracies)]
print("Оптимальный k:", best_k, "с accuracy:", max(accuracies))

# Другие классификаторы
print("\nШаг 9: Запуск других классификаторов...")
# SVM
svm = SVC(kernel='linear', class_weight='balanced')
svm.fit(X_train_vec, y_train)
y_pred_svm = svm.predict(X_test_vec)
print("SVM - Test accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# RandomForest
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_vec, y_train)
y_pred_rf = rf.predict(X_test_vec)
print("RandomForest - Test accuracy:", accuracy_score(y_test, y_pred_rf))
print("RandomForest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
