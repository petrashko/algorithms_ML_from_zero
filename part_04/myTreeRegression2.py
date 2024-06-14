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

from decisionTree import Tree

# Дерево решений - регрессия (поиск наилучшего разбиения)
class MyTreeReg():
    # max_depth: Максимальная глубина.
    #     default=5
    # min_samples_split: Количество объектов в листе, чтобы его можно было
    #     разбить и превратить в узел. default=2
    # max_leafs: Максимальное количество листьев разрешенное для дерева.
    #     default=20
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.maxDepth = max_depth
        self.minSamplesSplit = min_samples_split
        self.maxLeafs = max_leafs
        
        # Словарь содержит пороговые значения для каждого признака
        self.dictFeaturesThreshold = {}
        
        #
        self.tree = Tree()
        self.targetName = 'My-Target999'  # Название колонки с целевой переменной
        #
        self.eps = 1e-15
    
    #****************************************************************
    
    # Метод возвращает среднеквадратичную ошибку
    # Mean Squared Error (MSE) для вектора 'y'
    #
    # y: Series с целевыми значениями
    def get_mse(self, y):
        y = y.values.reshape(y.shape[0], 1)
        y = y.flatten()
        
        yMean = np.mean(y)
        errors = y - yMean
        
        mse = np.mean(errors ** 2)
        return mse
    
    #****************************************************************
    
    # Метод возвращает кортеж для найлучшего разбиения
    # (0: название признка, 1: пороговое значение, по которому проводилось разбиение, 2: прирост информаций)
    #
    # X_train: pd.DataFrame с признаками. Каждая строка - отдельное наблюдение
    #     Каждая колонка - конкретный признак
    # y_train: pd.Series с целевыми значениями
    def get_best_split(self, X_train, y_train):
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X_train = X.reset_index(drop=True)
        y_train = y.reset_index(drop=True)
        #
        y_train.name = self.targetName
        # Собрать матрицу признаков и целевую переменную в один DataFrame
        dfTrain = pd.concat([X_train, pd.DataFrame(y_train)], axis='columns')
        
        # Запомнить общее количество объектов в тренировочном наборе
        countTrainTotal = y_train.shape[0]
        # Получить названия признаков
        nameFeatures = X_train.columns
        
        dictFeatures = {}
        dictFeaturesUniq = {}
        for colName in nameFeatures:
            dictFeatures[colName] = X_train[colName].values.flatten()
            # Уникальные значения признаков
            dictFeaturesUniq[colName] = np.sort(np.unique( dictFeatures.get(colName) ))
        
        dictFeaturesThreshold = {}
        
        # Найти пороговые значения для каждого признака,
        # по которым будем пытаться разбивать признаки в узле
        for colName in nameFeatures:
            uniqList = dictFeaturesUniq.get(colName)
            thresholdList = []
            for n in np.arange(1, len(uniqList)):
                threshold = np.mean([uniqList[n-1], uniqList[n]])
                thresholdList.append(threshold)
            dictFeaturesThreshold[colName] = np.array(thresholdList)
        
        # Вычислить среднеквадратичную ошибку (MSE) для всего вектора y_train
        mseTotal = self.get_mse(y_train)
        
        # Присвоить заведомо нереально малое чило
        informationGainMax = float('-inf')
        
        # Результат будем возвращать в виде кортежа для найлучшего разбиения
        # (0: название признка, 1: пороговое значение, по которому проводилось разбиение, 2: прирост информаций)
        result = (None, None, None)
        
        # Проверить все пороговые значения до всем признакам
        
        # Цикл по признакам
        for colName in nameFeatures:
            thresholdList = dictFeaturesThreshold.get(colName)
            # Цикл по пороговым значениям в каждом признакам
            for threshold in thresholdList:
                # Получить список объектов, которые оказались в левой ветке
                mask = dfTrain[colName] <= threshold
                dfLeft = dfTrain[mask]
                # Кол-во объектов, которые попали в левую ветку
                countTrainLeft = dfLeft.shape[0]
                
                if countTrainLeft > 0:
                    # Получить целевые значения, которые оказались в левой ветке
                    y_left = dfLeft[self.targetName]
                    # Вычислить среднеквадратичную ошибку (MSE) левой ветки
                    mseLeft = self.get_mse(y_left)
                # Иначе, в левую ветку не попало ни одного объекта
                else:
                    mseLeft = 0
                
                # Получить список объектов, которые оказались в правой ветке
                mask = dfTrain[colName] > threshold
                dfRight = dfTrain[mask]
                # Кол-во объектов, которые попали в правую ветку
                countTrainRight = dfRight.shape[0]
                
                if countTrainRight > 0:
                    # Получить целевые значения, которые оказались в правой ветке
                    y_right = dfRight[self.targetName]
                    # Вычислить среднеквадратичную ошибку (MSE) правой ветки
                    mseRight = self.get_mse(y_right)
                # Иначе, в правую ветку не попало ни одного объекта
                else:
                    mseRight = 0
                
                # Вычислить прирост информаций
                # На сколько уменьшилась Entropy или GiniImpurity
                informationGain = mseTotal - \
                    ((countTrainLeft / countTrainTotal * mseLeft) + \
                    (countTrainRight / countTrainTotal * mseRight))
                
                if informationGain > informationGainMax:
                    result = (colName, threshold, informationGain)
                    informationGainMax = informationGain
        
        return result[0], result[1], result[2]
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельное наблюдение
    #     Каждая колонка - конкретный признак
    # y: pd.Series с целевыми значениями
    def fit(self, X, y):
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
    #****************************************************************
    
    # X_test: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict(self, X_test):
        y_predict = []
        
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X_test = X_test.reset_index(drop=True)
        
        return  y_predict
    
    #****************************************************************
    
    def __str__(self):
        return f'MyTreeReg class: max_depth={self.maxDepth}, min_samples_split={self.minSamplesSplit}, max_leafs={self.maxLeafs}'
    
    def __repr__(self):
        return f'MyTreeReg(max_depth={self.maxDepth}, min_samples_split={self.minSamplesSplit}, max_leafs={self.maxLeafs})'

#********************************************************************

if __name__ == '__main__':
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=1000, n_features=4, n_informative=5, noise=10, random_state=101)
    X_train = pd.DataFrame(X)
    y_train = pd.Series(y)
    
    myTreeReg = MyTreeReg(max_depth=5, min_samples_split=2, max_leafs=20)
    res = myTreeReg.get_best_split(X_train, y_train)
    print(res)
    # Обучить модель
    #myTreeReg.fit(X, y)
    
    
    print(end='\n\n')
    print('END!!!');
    input();
