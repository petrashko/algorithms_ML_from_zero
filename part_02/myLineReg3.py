import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

#sep = os.sep
# Добавить в путь до родительской папки
#sys.path.append(os.path.join(sys.path[0], f'..{sep}'))
#sys.path.append(os.path.join(os.getcwd(), f'..{sep}'))

import types

# Линейная регрессия (регуляризация)
class MyLineReg():
    # learning_rate: Или число float или lambda-функция
    # например: lambda iter: 0.5 * (0.85 ** iter)
    #     default = 0.1
    # metric: Принимает значение: 'r2', 'mape', 'mae', 'rmse', 'mse'
    #     default = None
    # reg: Принимает значение: 'l1', 'l2', 'elasticnet'
    #     default = None
    # l1_coef: Принимает значение в интервале: [0.0, 1.0]
    #     default = 0
    # l2_coef: Принимает значение в интервале: [0.0, 1.0]
    #     default = 0
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None,
                 reg=None, l1_coef=0, l2_coef=0):
        self.verboseInc = None    # Нужно для вывода логов
        self.bestScore = None     # Хранит значение метрики (оценку), обученной модели
        #
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        #
        self.metric = metric
        #
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
    
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
            self.verboseInc = verbose
        
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Для 'b0' добавить слева в матрицу признаков колонку с единицами
        X_new = self.addColumnForB0(X)
        
        X_features = X_new.to_numpy()
        # Вектор столбец
        y_target = y.to_numpy().reshape(y.shape[0], 1)

        # Создать весовые коэфициенты - вектор столбец из единиц
        # Число строк = количество колонок в X_features (каждому признаку свой вес)
        self.weights = np.ones((X_features.shape[1], 1), dtype=np.float64)
        
        # Матрица: Каждая строка - конкретный признак
        #     Каждая колонка - отдельное наблюдение
        X_transpose = X_features.T
        
        #
        learningSpeed = self.learning_rate
        
        #
        for n in np.arange(1, self.n_iter+1):
            # Если self.learning_rate: lambda-функция
            if type(self.learning_rate) is types.LambdaType:
                # Чем больше номер итеррации, тем меньше скорость обучения
                learningSpeed = self.learning_rate(n)

            # Вычислить предсказанные значения
            y_predict = self.dot(X_features, self.weights)
            # Ошибки по каждому вектору наблюдений
            errors = y_predict - y_target
            
            # Градиент вычисляется чтобы минимзировать ошибку вычисленную
            # по методу наименьших квадратов
            
            # Вычисляем градиент для каждого весового коэффициента
            # Сумма произведений каждой ошибки на все признаки для конкретного веса
            gradients = self.dot(X_transpose, errors)
            # X_transpose.shape[1]: Число наблюдений
            gradients = 2 * gradients / X_transpose.shape[1]
            
            # L1 регуляризация (Lasso регрессия)
            if self.reg == 'l1':
                gradients += self.l1_coef * np.sign(self.weights)
            # L2 регуляризация (Ridge регрессия)
            if self.reg == 'l2':
                gradients += 2 * self.l2_coef * self.weights
            # L1 и L2 регуляризация (ElasticNet)
            if self.reg == 'elasticnet':
                l1_reg = self.l1_coef * np.sign(self.weights)
                l2_reg = 2 * self.l2_coef * self.weights
                gradients += l1_reg + l2_reg
            
            # Изменяем весовые коэффициенты
            self.weights -= learningSpeed * gradients

            # Вывод метрик обучения
            self.show_estimators(n, verbose, y_target, y_predict)
        # END: for
        
        # Запомнить метрику (оценку), обученной модели
        self.bestScore = self.get_metric(y_target.flatten(), self.predict(X), self.metric)

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
    
    def get_coef(self):
        # Отбросить первый вес 'b0' и развернуть в одномерный массив
        weights = self.weights[1:].flatten()
        return weights
    
    #****************************************************************
    
    def get_best_score(self):
        return self.bestScore
    
    #****************************************************************
    
    # Функция возвращает одну из метрик оценки работы модели
    def get_metric(self, y_target, y_predict, metric_name):
        result = None
        
        errors = y_target - y_predict
        yMean = y_target.mean()
        
        # Коэффициент детерминации
        # R^2
        if metric_name == 'r2':
            numerator   = np.mean(errors ** 2)
            denominator = np.mean((y_target - yMean) ** 2)
            result = 1 - (numerator / denominator)
        # Средняя абсолютная процентная ошибка
        # Mean Absolute Percentage Error (MAPE)
        elif metric_name == 'mape':
            result = np.mean( np.abs(errors / y_target) ) * 100
        # Средняя абсолютная ошибка
        # Mean Absolute Error (MAE)
        elif metric_name == 'mae':
            result = np.mean( np.abs(errors) )
        # Квадратный корень из среднеквадратичной ошибки
        # Root Mean Squared Error (RMSE)
        elif metric_name == 'rmse':
            result = np.sqrt( np.mean(errors ** 2) )
        # Иначе,
        # Среднеквадратичная ошибка
        # Mean Squared Error (MSE)
        else:
            result = np.mean(errors ** 2)
        
        return result
    
    #****************************************************************
    
    # Вывод метрик обучения
    def show_estimators(self, n, verbose, y_target, y_predict):
        if (verbose > 0) and (n == 1 or n == self.verboseInc):
            # Среднеквадратичная ошибка, вычисленная по методу наименьших квадратов
            # Функция потерь
            mse = np.mean((y_target - y_predict) ** 2)
            
            # Вычислить нужную метрику
            if (self.metric != None) and (n == 1 or n == self.verboseInc):
                metric_value = self.get_metric(y_target.flatten(), y_predict.flatten(), self.metric)
            
            if n == 1:
                if self.metric == None:
                    print(f'start | loss: {mse}')
                else:
                    print(f'start | loss: {mse} | {self.metric}: {metric_value}')
            elif n == self.verboseInc:
                if self.metric == None:
                    print(f'{self.verboseInc} | loss: {mse}')
                else:
                    print(f'{self.verboseInc} | loss: {mse} | {self.metric}: {metric_value}')
                self.verboseInc += verbose
    
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
        
    myLineReg = MyLineReg(
        n_iter=50, learning_rate=lambda x: 0.5 * (0.85 ** x),
        weights=None, metric='mae',
        reg='elasticnet', l1_coef=0.5, l2_coef=0.5
    )
    # Обучить модель
    myLineReg.fit(X, y, 5)
    print()
    print( myLineReg.get_coef() )
    print( myLineReg.get_coef().mean() )
    
    # Проверка
    '''
    X, y = make_regression(n_samples=400, n_features=14, n_informative=5, noise=5, random_state=101)
    X = pd.DataFrame(X)
    print()
    print( myLineReg.predict(X).sum() )
    '''
    
    print(end='\n\n')
    print('END!!!');
    input();
