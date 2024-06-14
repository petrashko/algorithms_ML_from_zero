import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

sep = os.sep
# Добавить в путь до родительской папки
#sys.path.append(os.path.join(sys.path[0], f'..{sep}'))
#sys.path.append(os.path.join(os.getcwd(), f'..{sep}'))

# Метод главных компонент (PCA)
class MyPCA():
    # n_components: Количество главных компонент,
    #     которое следует оставить. default=3
    def __init__(self, n_components=3):
        self.n_components = n_components
        
        # Количество объектов (строк) в исходном наборе
        self.countRows = 0
        # Количество признаков колонок в исходном наборе
        self.countCols = 0
        
        #
        self.eps = 1e-15

    #****************************************************************
    
    # Перемножение двумерных массивов numpy
    @staticmethod
    def dot(A, B):
        # В итоговой матрице:
        # число строк = число строк в A; число колонок = число колонок в B
        res = np.zeros((A.shape[0], B.shape[1]))
        
        # Цикл по строкам матрицы A
        for r in np.arange(0, A.shape[0]):
            # Цикл по колонкам матрицы B
            for c in np.arange(0, B.shape[1]):
                # Цикл по строкам матрицы B
                for t in np.arange(0, B.shape[0]):
                    res[r, c] += A[r, t] * B[t, c]
        return res
    
    #****************************************************************
    
    # Метод вычисляет дисперсию
    # vec: вектор numpy
    @staticmethod
    def calculateVar(vec):
        average = vec.mean()
        v2 = (vec - average) ** 2
        sum2 = v2.sum()
        
        result = sum2 / (len(vec)-1)
        return result
    
    #****************************************************************
    
    # Метод вычисляет ковариацию межде векторами
    # vec1, vec2: два вектора numpy
    @staticmethod
    def calculateCov(vec1, vec2):
        average1 = vec1.mean()
        average2 = vec2.mean()
        v1 = vec1 - average1
        v2 = vec2 - average2
        
        result = np.sum(v1 * v2) / (len(vec1)-1)
        return result
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def fit_transform(self, X):
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        
        X_original = X.to_numpy()
        
        # Запомнить кол-во объектов в исходном наборе
        self.countRows = X_original.shape[0]
        # Запомнить кол-во признаков в исходном наборе
        self.countCols = X_original.shape[1]
        
        # Шаг 1. Нормировка признаков
        X_norm = np.zeros((self.countRows, self.countCols), dtype=np.float64)
        
        # Цикл по признакам (колонкам) из набора признаков
        for col in np.arange(0, self.countCols):
            column = X_original[:, col]
            # Среднее по колонке
            columnAverage = column.mean()
            # Цикл по строкам в колонке n
            for row in np.arange(0, self.countRows):
                X_norm[row, col] = X_original[row, col] - columnAverage
        
        # Шаг 2. Вычисление матрицы ковариации (квадратная матрица)
        covMatrix = np.zeros((self.countCols, self.countCols), dtype=np.float64)
        
        # Вычисляем элементы главной диагонали для матрицы ковариации
        for row in np.arange(0, self.countCols):
            for col in np.arange(0, self.countCols):
                if row == col:
                    column = X_norm[:, col]
                    # Дисперсия по колонке
                    covMatrix[row, col] = self.calculateVar(column)
        
        # Вычисляем остальные элементы матрицы ковариации
        for col1 in np.arange(0, self.countCols-1):
            for col2 in np.arange(col1+1, self.countCols):
                column1 = X_norm[:, col1]
                column2 = X_norm[:, col2]
                # Ковариация между двумя колонками признаков
                cov = self.calculateCov(column1, column2)
                covMatrix[col1, col2] = cov
                covMatrix[col2, col1] = cov
            
        # Шаг 3. Разложение матрицы ковариации на собственные вектора и собственные числа
        eigenValues, eigenVectors = np.linalg.eigh(covMatrix)
        
        # Шаг 4. Сортировка собственных чисел и соответствующих
        # собственных векторов по убыванию
        sortedKey = np.argsort(eigenValues)[::-1]  # Вернуть индексы (типа) отсортированного массива
        
        # Взять первые self.n_components индексов
        sortedKey = sortedKey[0:self.n_components]
        # Взять первые self.n_components собственных чисел и соответствующих
        # и соответствующих собственных векторов (вектор это колонка)
        eigenValues, eigenVectors = eigenValues[sortedKey], eigenVectors[:, sortedKey]
        
        # Шаг 5. Снижение размерности матрицы с признаками
        X_components = self.dot(X_norm, eigenVectors)
        
        # Шаг 6. Вернуть в виде DataFrame'а
        colNames = []
        for k in np.arange(0, self.n_components):
            colNames.append(f'col_{k}')
        
        X_pca = pd.DataFrame(X_components, columns=colNames)
        return X_pca
    
    #****************************************************************
    
    def __str__(self):
        return f'MyPCA class: n_components={self.n_components}'
    
    def __repr__(self):
        return f'MyPCA(n_components={self.n_components})'
    
#********************************************************************

if __name__ == '__main__':
    # Для кластеризации
    from sklearn.datasets import make_blobs
    
    #X, _ = make_blobs(n_samples=5, centers=2, n_features=4, cluster_std=2.5, random_state=42)
    X, _ = make_blobs(n_samples=100, centers=5, n_features=20, cluster_std=2.5, random_state=42)
    X = pd.DataFrame(X)
    X.columns = [f'col_{col}' for col in X.columns]
    
    print(X); print()
    
    myPCA = MyPCA(n_components=2)
    #print(myPCA)
    
    X_pca = myPCA.fit_transform(X)
    print(X_pca); print()
    print(round(X_pca.median().median(), 10)); print()
    
    print(end='\n\n')
    print('END!!!');
    input();