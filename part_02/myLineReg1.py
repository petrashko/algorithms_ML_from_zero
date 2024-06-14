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

# Линейная регрессия
class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
    
    # Перемножение двумерных массивов numpy
    def dot(self, A, B):
        # В итоговой матрице:
        # число строк = число строк в A; число колонок = число колонок в
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
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    # y: pd.Series с целевыми значениями
    def fit(self, X, y, verbose=False):
        if not verbose: # verbose=False
            verbose = -1
        else:  # verbose: целое число
            verboseInc = verbose
    
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Для 'b0' добавить слева в матрицу признаков колонку с единицами
        X = self.addColumnForB0(X)
        
        X_features = X.to_numpy()
        # Вектор столбец
        y_target = y.to_numpy().reshape(y.shape[0], 1)

        # Создать весовые коэфициенты - вектор столбец из единиц
        # Число строк = количество колонок в X_features (каждому признаку свой вес)
        self.weights = np.ones((X_features.shape[1], 1), dtype=np.float64)
        
        # Матрица: Каждая строка - конкретный признак
        #     Каждая колонка - отдельное наблюдение
        X_transpose = X_features.T
        
        #
        for n in np.arange(0, self.n_iter):
            # Вычислить предсказанные значения
            y_predict = self.dot(X_features, self.weights)
            # Ошибки по каждому вектору наблюдений
            errors = y_predict - y_target
            # Среднеквадратичная ошибка, вычисленная по методу наименьших квадратов
            # Функция потерь
            mse = np.mean(errors ** 2)
            
            # Градиент вычисляется чтобы минимзировать ошибку вычисленную
            # по методу наименьших квадратов
            
            # Вычисляем градиент для каждого весового коэффициента
            # Сумма произведений каждой ошибки на все признаки для конкретного веса
            gradients = self.dot(X_transpose, errors)
            # X_transpose.shape[1]: Число наблюдений
            gradients = 2 * gradients / X_transpose.shape[1]
            
            # Изменяем весовые коэффициенты
            self.weights -= self.learning_rate * gradients

            # Вывод отладочной информации            
            if verbose > 0:
                if n == 0:
                    print(f'start | loss: {mse}')
                elif n+1 == verboseInc:
                    print(f'{verboseInc} | loss: {mse}')
                    verboseInc += verbose
        # END: for
    #****************************************************************
    
    def get_coef(self):
        # Отбросить первый вес 'b0' и развернуть в одномерный массив
        weights = self.weights[1:].flatten()
        return weights
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict(self, X):
        # Для 'b0' добавить слева в матрицу признаков колонку с единицами
        X = self.addColumnForB0(X)
        
        X_features = X.to_numpy()
        # Вычислить предсказанные значения
        y_predict = self.dot(X_features, self.weights)
        return y_predict.flatten()
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def addColumnForB0(self, X):
        # Для 'b0' добавить слева в матрицу признаков колонку с единицами
        onesDf = pd.DataFrame(np.ones((X.shape[0], 1), dtype=np.int32), columns=['for B0'])
        X_new = pd.concat([onesDf, X], axis='columns')
        return X_new
    
    #****************************************************************
    
    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def __repr__(self):
        return f'MyLineReg(n_iter={self.n_iter}, learning_rate={self.learning_rate})'

#********************************************************************

if __name__ == '__main__':
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=1000, n_features=14, n_informative=5, noise=10, random_state=101)
    X = pd.DataFrame(X)
    y = pd.Series(y)
        
    myLineReg = MyLineReg(n_iter=50, learning_rate=0.1)
    # Обучить модель
    myLineReg.fit(X, y, 5)
    print()
    print( myLineReg.get_coef() )
    print( myLineReg.get_coef().mean() )
    
    # Проверка
    X, y = make_regression(n_samples=400, n_features=14, n_informative=5, noise=5, random_state=101)
    X = pd.DataFrame(X)
    print()
    print( myLineReg.predict(X).sum() )
    
    print(end='\n\n')
    print('END!!!');
    input();
