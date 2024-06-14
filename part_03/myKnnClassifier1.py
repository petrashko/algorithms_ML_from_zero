import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import random

sep = os.sep
# Добавить в путь до родительской папки
#sys.path.append(os.path.join(sys.path[0], f'..{sep}'))
#sys.path.append(os.path.join(os.getcwd(), f'..{sep}'))

# Метод K-ближайших соседей (классификация)
# Для двух классов с метками: 0 и 1
class MyKNNClf():
    # metric: Принимает значение: 'euclidean', 'manhattan', 'chebyshev', 'cosine'
    #     default = 'euclidean'
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        #
        self.X = None  # DataFrame
        self.y = None  # Series
        # Размер тренировочной выборки 'X' в виде кортежа
        # (количество_строк, количество_столбцов)
        self.train_size = None
        #
        self.X_train = None  # ndarray
        # ndarray => вектор столбец c целевыми значениями с метками: 0 и 1
        self.y_train = None
        #
        self.eps = 1e-15
    
    #****************************************************************
    
    # Метод возвращает Евклидово расстояние между двумя векторами
    def euclidean_distance(self, a, b):
        sum2 = np.sum((a - b) ** 2)
        result = np.sqrt(sum2)
        return result
    
    #****************************************************************
    
    # Метод возвращает Манхэттенское расстояние между двумя векторами
    def manhattan_distance(self, a, b):
        absDif = np.abs(a - b)
        result = np.sum(absDif)
        return result
    
    #****************************************************************
    
    # Метод возвращает расстояние Чебышева между двумя векторами
    def chebyshev_distance(self, a, b):
        absDif = np.abs(a - b)
        result = np.max(absDif)
        return result

    #****************************************************************
    
    # Метод возвращает Косинусное расстояние между двумя векторами
    def cosine_distance(self, a, b):
        numerator = np.sum(a * b)
        normA = np.sqrt( np.sum(a ** 2) )
        normB = np.sqrt( np.sum(b ** 2) )
        denominator = normA * normB
        
        result = 1 - (numerator / denominator)
        return result
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    # y: pd.Series с целевыми значениями
    def fit(self, X, y):
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        
        self.X_train = self.X.to_numpy()
        # Вектор столбец
        self.y_train = self.y.values.reshape(y.shape[0], 1)
        
        # Запомнить размер тренировочной выборки
        self.train_size = self.X_train.shape
    
    #****************************************************************
    
    # X_test: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def help_predict(self, X_test):
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X_test = X_test.reset_index(drop=True)
        X_features = X_test.to_numpy()
        
        classListForX_test = []
        
        # Цикл по наблюдениям (объектам) из тестового набора
        for row in X_features:
            # Создать массив: Количество строк = числу строк в тренировочном наборе
            # С двумя колонками: первая колонка: расстояние от 'row'
            # до конкретного объекта из тренировочной выборки
            # вторая колонка: метка класса соответствующего объекта
            rowsDistance = np.zeros((self.train_size[0], 2))
            
            # Цикл по строкам из тренировочного набора
            for i in np.arange(0, self.train_size[0]):
                # Вычислить Косинусное расстояние
                if self.metric == 'cosine':
                    dist = self.cosine_distance(row, self.X_train[i])
                # Вычислить расстояние Чебышева
                elif self.metric == 'chebyshev':
                    dist = self.chebyshev_distance(row, self.X_train[i])
                # Вычислить Манхэттенское расстояние
                elif self.metric == 'manhattan':
                    dist = self.manhattan_distance(row, self.X_train[i])
                # Иначе, вычислить Евклидово расстояние
                else:
                    dist = self.euclidean_distance(row, self.X_train[i])
                
                rowsDistance[i, 0] = dist
                # Запомнить метку класса
                rowsDistance[i, 1] = self.y_train[i]
            
            # Сортировка по расстоянию (по убыванию)
            idx = np.lexsort([rowsDistance[:, 0]])
            rowsDistance = rowsDistance[idx]
            
            # Выбрать первых self.k ближайих классов, к
            # которым возможно принадлежит объект 'row'
            classList = []
            for r in np.arange(0, self.k):
                classList.append(rowsDistance[r, 1])
            
            # Запомнить и вернуть список первых self.k ближайих классов
            # для каждого наблюдения из тестового набора 'X_test'
            classListForX_test.append(classList)
        
        return np.array(classListForX_test, dtype=np.int32)
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict_proba(self, X):
        y_proba = []
        
        # Получить список первых self.k ближайих классов для
        # каждого наблюдения из тестового набора 'X_test'
        clasesForX = self.help_predict(X_test)
        
        for clases in clasesForX:
            # Опредилить с какой вероятностью каждое наблюдение из
            # тестового набора 'X_test' принадлежит к классу с меткой 1
            # Здесь пользуемся тем, что метки классов имеют значения: 0 и 1
            percent100 = clases.size
            proba = clases.sum() / percent100
            
            y_proba.append(proba)
        
        return np.array(y_proba, dtype=np.float64)
    
    #****************************************************************
    
    # X_test: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict(self, X_test):
        y_predict = []
        
        # Получить список первых self.k ближайих классов для
        # каждого наблюдения из тестового набора 'X_test'
        clasesForX = self.help_predict(X_test)
        
        for clases in clasesForX:
            # Опредилить к какому классу принадлежит каждое
            # наблюдение из тестового набора 'X_test'
            # Здесь пользуемся тем, что метки классов имеют значения: 0 и 1
            classProba = np.array(clases, dtype=np.int32).mean()
            if classProba < 0.5:
                y_predict.append(0)
            else:
                y_predict.append(1)
        
        return np.array(y_predict, dtype=np.int32)
    
    #****************************************************************
    
    def __str__(self):
        return f'MyKNNClf class: k={self.k}'
    
    def __repr__(self):
        return f'MyKNNClf(k={self.k})'

#********************************************************************

if __name__ == '__main__':
    # Для классификации
    df = pd.read_csv(f'..{sep}data{sep}banknote+authentication.zip', header=None)
    df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    X, y = df.iloc[:,:4], df['target']
    #print(y)

    myKnnClf = MyKNNClf(k=3, metric='chebyshev')
    # Обучить модель
    myKnnClf.fit(X, y)
    print()
    print( myKnnClf.train_size )
        
    # Проверка
    X_test = X.iloc[757:767, :]
    print()
    print( myKnnClf.predict_proba(X_test) )
    print( myKnnClf.predict_proba(X_test).sum() )
    print()
    print( myKnnClf.predict(X_test) )
    print( myKnnClf.predict(X_test).sum() )
    
    print(end='\n\n')
    print('END!!!');
    input();