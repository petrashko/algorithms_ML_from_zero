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

# Метод K-ближайших соседей (регрессия)
class MyKNNReg():
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
        # ndarray => вектор столбец c целевыми значениями
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
        
        dotsMatrixForX_test = []
        
        # Цикл по наблюдениям (объектам) из тестового набора
        for row in X_features:
            # Создать массив: Количество строк = числу строк в тренировочном наборе
            # С двумя колонками: первая колонка: расстояние от 'row'
            # до конкретного объекта из тренировочной выборки
            # вторая колонка: целевое значение соответствующего объекта
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
                
                if dist >= self.eps:
                    rowsDistance[i, 0] = dist
                else:
                    # Избавиться от возможных 0 в знаменателе
                    rowsDistance[i, 0] = self.eps

                # Запомнить целевое значение
                rowsDistance[i, 1] = self.y_train[i]
            
            # Сортировка по расстоянию (по убыванию)
            idx = np.lexsort([rowsDistance[:, 0]])
            rowsDistance = rowsDistance[idx]
            
            # Выбрать первых self.k ближайих точек, до объекта 'row'
            dotsMatrix = []
            for r in np.arange(0, self.k):
                # Запоминаем: ранг r+1, расстояние rowsDistance[r, 0]
                # и целевое значение rowsDistance[r, 1]
                dotDescribe = [r+1, rowsDistance[r, 0], rowsDistance[r, 1]]
                dotsMatrix.append(dotDescribe)
            
            # Запомнить и вернуть список первых self.k ближайих точек
            # с их рангом, расстоянием и целевым значением
            # для каждого наблюдения из тестового набора 'X_test'
            dotsMatrixForX_test.append(dotsMatrix)
        
        return np.array(dotsMatrixForX_test, dtype=np.float64)
    
    #****************************************************************
    
    # X_test: pd.DataFrame с признаками. Каждая строка - отдельное наблюдение
    #     Каждая колонка - конкретный признак
    def predict(self, X_test):
        y_predict = []
        
        # Получить список первых self.k ближайих точек для
        # каждого наблюдения из тестового набора 'X_test'
        dotsForX = self.help_predict(X_test)
        
        # Вычислить предсказанные значения для каждого наблюдения
        # из тестового набора 'X_test'
        for dots in dotsForX:
            # Для каждых первых self.k точек возвращается:
            # ранг, расстояние и целевое значение
            #rankList  = dots[:, 0].flatten()
            #distList  = dots[:, 1].flatten()
            valueList = dots[:, 2].flatten()
            
            # Вычислить среднее значение по всем self.k ближайшим точкам
            # к тестовому объекту
            y_predict.append(valueList.mean())
        
        return np.array(y_predict, dtype=np.float64)
    
    #****************************************************************
    
    def __str__(self):
        return f'MyKNNReg class: k={self.k}'
    
    def __repr__(self):
        return f'MyKNNReg(k={self.k})'

#********************************************************************

if __name__ == '__main__':
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=1000, n_features=14, n_informative=5, noise=10, random_state=101)
    X = pd.DataFrame(X)
    y = pd.Series(y)
        
    myKnnReg = MyKNNReg(k=3, metric='euclidean')
    # Обучить модель
    myKnnReg.fit(X, y)
    print(myKnnReg.train_size)
    print()
    # Проверка
    X, y = make_regression(n_samples=400, n_features=14, n_informative=5, noise=5, random_state=101)
    X = pd.DataFrame(X)
    print()
    print( myKnnReg.predict(X).sum() )
    
    
    print(end='\n\n')
    print('END!!!');
    input();
