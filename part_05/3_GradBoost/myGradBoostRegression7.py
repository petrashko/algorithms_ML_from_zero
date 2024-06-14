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

import types

from myTreeRegression import MyTreeReg

# Градиентный бустинг - регрессия (скорость обучения)
class MyBoostReg():
    # random_state: Для воспроизводимости результата.
    #     default=42
    # learning_rate: Скорость обучения. Или число float или lambda-функция
    #     например: lambda iter: 0.5 * (0.85 ** iter)
    #     default=0.1
    # metric: Принимает значение: 'R2', 'MAPE', 'MAE', 'RMSE', 'MSE'
    #     default=None
    # loss: Функция потерь для расчета градиента, принимает значение: 'MAE', 'MSE'
    #     default='MSE'
    
    # Параметры исключительно для леса:
    # n_estimators: Количество деревьев в лесу.
    #     default=10
    # max_features: Доля фичей, которая будет случайным образом выбираться
    #     для каждого дерева. От 0.0 до 1.0. default=0.5
    # max_samples: Доля объектов, которая будет случайным образом выбираться
    #     из датасета для каждого дерева. От 0.0 до 1.0. default=0.5
    
    # Параметры отдельного дерева:
    # max_depth: Максимальная глубина.
    #     default=5
    # min_samples_split: Количество объектов в листе, чтобы его можно было
    #     разбить и превратить в узел. default=2
    # max_leafs: Максимальное количество листьев разрешенное для дерева.
    #     default=20
    # bins: Количество бинов для вычисления гистограммы значений,
    #     по каждому признаку. default=16
    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5,
                 max_depth=5, min_samples_split=2, max_leafs=20, bins=16,
                 loss='MSE', metric=None, learning_rate=0.1, random_state=42):
        self.verboseInc = None    # Нужно для вывода логов
        self.best_score = None    # Хранит значение метрики (оценку), обученной модели
        self.learningRate = learning_rate
        self.metric = metric
        self.loss = loss
        
        # Параметры для леса
        self.n_estimators = n_estimators
        self.maxFeatures = max_features
        self.maxSamples = max_samples
        
        # Подготовить список из n_estimators решающих деревьев (пока пустых)
        self.trees = [None] * self.n_estimators
        for k in np.arange(0, self.n_estimators):
            treeReg = MyTreeReg(
                max_depth=max_depth, min_samples_split=min_samples_split,
                max_leafs=max_leafs, bins=bins
            )
            self.trees[k] = treeReg
        
        # Словарь, номер колонки: название признака
        self.colsToFeatureName = {}
        # Суммарное кол-во листьев в лесу (для всех деревьев)
        self.leafs_cnt = 0
        
        # Кол-во объектов (строк) в исходном тренировочном наборе,
        # который подается на вход алгоритму обучения
        self.countSamplesTotal = 0
        # Кол-во признаков (колонок) в исходном тренировочном наборе,
        # который подается на вход алгоритму обучения
        self.countFeaturesTotal = 0
        
        # Первое предсказание - среднее по всем целевым
        # значениям из тренировочного набора
        self.pred_0 = None
        
        # Строка матрицы - это предсказаные значения для
        # каждого объекта из тестового набора, от одного дерева.
        # (кол-во строк = кол-во деревьев)
        self.predictsFromTrees = None
        
        # Словарь содержит важность каждого признака
        # в лесу (для всех деревьев)
        self.fi = {}
        
        #
        self.randomState = random_state
        self.testSum = 0
        self.eps = 1e-15

    #****************************************************************
    
    # Метод заменяет target'ы в листьях дерева
    # y_target: pd.Series с целевыми значениями
    # y_predict: вектор numpy с предсказанными значениями
    def replaceTargetInLeafs(self, tree, y_target, y_predict):
        tree.testSum = 0
        
        # Цикл по листам дерева, чтобы подменить в них target'ы
        for leaf in tree.leafList:
            # Получить номера объектов (наблюдений), попавших в лист дерева
            rows = leaf.data.get('rows')
            # Получить целевые значения для объектов, попавших в лист
            y_leafTarget = y_target[rows]
            # Получить предсказания (полученные в начале итерации)
            # для объектов, попавших в лист
            y_leafPredict = y_predict[rows]
            # Разница
            diff = y_leafTarget - y_leafPredict
            # Нужное нам предсказание в листе
            if self.loss == 'MAE':
                newTarget = diff.median()
            # Иначе, 'MSE'
            else:
                newTarget = diff.mean()
            
            # ...и заменить его в листе
            leaf.data['target'] = newTarget
            tree.testSum += newTarget
    
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
        
        for col, featureName in enumerate(X.columns.tolist()):
            # Создать словарь, номер колонки: название признака
            self.colsToFeatureName[col] = featureName
            # Каждому признаку присвоить важность 0
            self.fi[featureName] = 0
        
        # Запомнить кол-во объектов в тренировочном наборе
        self.countSamplesTotal = X.shape[0]
        # Запомнить кол-во признаков в тренировочном наборе
        self.countFeaturesTotal = X.shape[1]
        
        # Для воспроизводимости результатов (тестирования)
        random.seed(self.randomState)
        
        # Кол-во объектов (строк), которые используются
        # для построения каждого отдельного дерева
        #     (self.maxSamples перевести в целое число)
        countRows = int( np.round(self.maxSamples * self.countSamplesTotal, 0) )
        
        # Кол-во признаков (колонок), которые используются
        # для построения каждого отдельного дерева
        #     (self.maxFeatures перевести в целое число)
        countCols = int( np.round(self.maxFeatures * self.countFeaturesTotal, 0) )
        
        if self.loss == 'MAE':
            # Вычислить медиану по всем целевым
            # значениям из тренировочного набора
            self.pred_0 = y.median()
        # Иначе, 'MSE'
        else:
            # Вычислить среднее по всем целевым
            # значениям из тренировочного набора
            self.pred_0 = y.mean()
        
        # Предсказания для обучения 1-го дерева
        y_predictTotal = np.full(self.countSamplesTotal, self.pred_0)
        
        #
        learningSpeed = self.learningRate
        
        # Построить (обучить) n_estimators решающих деревьев
        for k in np.arange(0, self.n_estimators):
            tree = self.trees[k]
            # Передать в дерево ОБЩЕЕ кол-во объектов в тренировочном наборе
            tree.countSamplesTotal = self.countSamplesTotal
            
            # Если self.learningRate: lambda-функция
            if type(self.learningRate) is types.LambdaType:
                # Чем больше номер итеррации, тем меньше скорость обучения
                learningSpeed = self.learningRate(k+1)
            
            # Случайно выбираем countCols названий признаков
            nameFeatures = random.sample(X.columns.tolist(), countCols)
            X_random = X[nameFeatures]
            
            # Случайно выбираем countRows индексов строк (наблюдений)
            randomRows = random.sample(list(X.index), countRows)
            
            # Оставили только случайно выбранные колонки и строки
            X_random = X_random.iloc[randomRows]
            y_random = y.iloc[randomRows]
            
            X_random = X_random.reset_index(drop=True)
            y_random = y_random.reset_index(drop=True)
            
            y_predict = y_predictTotal[randomRows]
            
            gradient = self.getGradient(y_random.values, y_predict)
            # Построить (обучить) k-ое дереро на антиградиенте
            tree.fit(X_random, pd.Series(-1 * gradient))
            
            # Заменить target'ы в листьях дерева
            self.replaceTargetInLeafs(tree, y_random, y_predict)
            
            # После маниипуляции с листьями дерева.
            # Вычислить для это дерева, предсказаные значения
            # для каждого объекта из тренировочного набора
            y_predictForTree = tree.predict(X)
            
            # Обновить предсказания с учетом уже обученных
            # деревьев и скорости обучения
            y_predictTotal = y_predictTotal + (learningSpeed * y_predictForTree)
            
            # Считаем общее кол-во листьев в лесу (для всех деревьев)
            self.leafs_cnt += tree.lastLeaf
            self.testSum += tree.testSum
            
            # Вывод метрик обучения
            self.showMetricsValue(k, verbose, y_random.values, y_predict)
        # END: for k in np.arange(0, self.n_estimators):
        
        # Запомнить метрику (оценку), обученной модели
        if self.metric is None:
            # Если метрика не задана - в self.best_score записать последнее
            # значение функции потерь, от которой считали векторы градиентов
            lossVector = self.getVectorLoss(y.values, self.predict(X))
            self.best_score = lossVector.mean()
        else:
            self.best_score = self.getMetric(y.values, self.predict(X))
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict(self, X):
        y_predict = np.zeros(X.shape[0])
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        
        #
        learningSpeed = self.learningRate
        
        # Вычислить предсказаные значения для каждого объекта
        # из тестового набора, от каждого дерева
        for k in np.arange(0, self.n_estimators):
            tree = self.trees[k]
            
            # Если self.learningRate: lambda-функция
            if type(self.learningRate) is types.LambdaType:
                # Чем больше номер итеррации, тем меньше скорость обучения
                learningSpeed = self.learningRate(k+1)
            
            # Первое дерево
            if k == 0:
                if type(self.learningRate) is types.LambdaType:
                    # Создать матрицу с одной строкой
                    self.predictsFromTrees = learningSpeed * tree.predict(X)
                # self.learningRate - это число
                else:
                    # Создать матрицу с одной строкой
                    self.predictsFromTrees = tree.predict(X)
            else:
                if type(self.learningRate) is types.LambdaType:
                    predictList = learningSpeed * tree.predict(X)
                # self.learningRate - это число
                else:
                    predictList = tree.predict(X)
                
                # Добавить строку с предсказаниями в матрицу
                self.predictsFromTrees = np.vstack((self.predictsFromTrees, predictList))
        
        # Цикл по объектам (строкам) из тестового набора признаков
        for n in np.arange(0, X.shape[0]):
            # Предсказания для одного объекта от каждого дерева
            predicts = self.predictsFromTrees[:, n]
            
            if type(self.learningRate) is types.LambdaType:
                y_predict[n] = predicts.sum() + self.pred_0
            # self.learningRate - это число
            else:
                y_predict[n] = learningSpeed * predicts.sum() + self.pred_0
        
        return  y_predict
    
    #****************************************************************
    
    # Метод возвращает вектор градиента для одной из функций потерь
    def getGradient(self, y_target, y_predict):
        grad = None
        
        errors = y_predict - y_target
        
        # Градиент для
        # Средней абсолютной ошибки
        # Mean Absolute Error (MAE)
        if self.loss == 'MAE':
            grad = np.sign(errors)
        # Градиент для
        # Среднеквадратичной ошибки
        # Mean Squared Error (MSE)
        else:
            grad = 2 * errors
        
        return grad
    
    #****************************************************************
    
    # Метод возвращает вектор ошибок для одной из функции потерь
    def getVectorLoss(self, y_target, y_predict):
        result = None
        
        errors = y_target - y_predict
        
        # Средняя абсолютная ошибка
        # Mean Absolute Error (MAE)
        if self.loss == 'mae':
            result = np.abs(errors)
        # Среднеквадратичная ошибка
        # Mean Squared Error (MSE)
        else:
            result = errors ** 2
        
        return result
    
    #****************************************************************
    
    # Функция возвращает одну из метрик оценки работы модели
    def getMetric(self, y_target, y_predict):
        result = None
        
        errors = y_target - y_predict
        yMean = y_target.mean()
        
        # Коэффициент детерминации
        # R^2
        if self.metric == 'R2':
            numerator   = np.mean(errors ** 2)
            denominator = np.mean((y_target - yMean) ** 2)
            result = 1 - (numerator / denominator)
        # Средняя абсолютная процентная ошибка
        # Mean Absolute Percentage Error (MAPE)
        elif self.metric == 'MAPE':
            result = np.mean( np.abs(errors / y_target) ) * 100
        # Средняя абсолютная ошибка
        # Mean Absolute Error (MAE)
        elif self.metric == 'MAE':
            result = np.mean( np.abs(errors) )
        # Квадратный корень из среднеквадратичной ошибки
        # Root Mean Squared Error (RMSE)
        elif self.metric == 'RMSE':
            result = np.sqrt( np.mean(errors ** 2) )
        # Иначе,
        # Среднеквадратичная ошибка
        # Mean Squared Error (MSE)
        else:
            result = np.mean(errors ** 2)
        
        return result
    
    #****************************************************************
    
    # Вывод метрик обучения
    def showMetricsValue(self, k, verbose, y_target, y_predict):
        if (verbose > 0) and (k % self.verboseInc == 0):
            # Вычислить значение функции потерь, от которой считали
            # векторы градиентов для обучения деревьев
            lossVector = self.getVectorLoss(y_target, y_predict)
            lossValue = lossVector.mean()
            
            # Вычислить нужную метрику
            if self.metric is not None:
                metricValue = self.getMetric(y_target, y_predict)
            
            if self.metric is None:
                print(f'{k}. Loss[{self.loss}]: {lossValue}')
            else:
                print(f'{k}. Loss[{self.loss}]: {lossValue}|{self.metric}: {metricValue}')
    
    #****************************************************************
    
    # Метод по названию признака возвращает номер
    # колонки для матрицы с признаками
    def getColNumberByName(self, featureName):
        colNumber = None
        for key, name in self.colsToFeatureName.items():
            if featureName == name:
                colNumber = key
                break
        return colNumber
    
    #****************************************************************
    
    def __str__(self):
        tree = self.trees[0]
        return f'MyBoostReg class: n_estimators={self.n_estimators}, learning_rate={self.learningRate}, max_depth={tree.maxDepth}, min_samples_split={tree.minSamplesSplit}, max_leafs={tree.maxLeafs}, bins={tree.bins}'
    
    def __repr__(self):
        tree = self.trees[0]
        return f'MyBoostReg(n_estimators={self.n_estimators}, learning_rate={self.learningRate}, max_depth={tree.maxDepth}, min_samples_split={tree.minSamplesSplit}, max_leafs={tree.maxLeafs}, bins={tree.bins})'

#********************************************************************

if __name__ == '__main__':
    # Для регрессии
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)
    X_train, y_train = data['data'], data['target']
    #print(X_train); print()
    #print(y_train); print()
    
    X_test = X_train.iloc[211:231, :]
    y_test = y_train.iloc[211:231].values
    #print(X_test); print()
    
    myBoostReg = MyBoostReg(n_estimators=10, max_features=0.95, max_samples=0.95,
                max_depth=5, min_samples_split=2, max_leafs=40, bins=16,
                loss='MSE', metric='RMSE', learning_rate=lambda x: 0.5 * (0.85 ** x),
                random_state=42)
    #print(myBoostReg)
    
    # Обучение
    myBoostReg.fit(X_train, y_train, verbose=2)
    print()
    for k in np.arange(0, myBoostReg.n_estimators):
        tree = myBoostReg.trees[k]
        #tree.printTree(tree.root); print()
    print(myBoostReg.pred_0, myBoostReg.leafs_cnt, myBoostReg.testSum); print()
    
    # Предсказание
    print(y_test); print()
    predictList = myBoostReg.predict(X_test)
    print( predictList ); print()
    print( predictList.sum() ); print()
    
    # Параметры обученной модели
    print(f'{myBoostReg.metric}: {myBoostReg.best_score}'); print()
    
    print(end='\n\n')
    print('END!!!');
    input();