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

from myTreeRegression import MyTreeReg

# Градиентный бустинг - регрессия
class MyBoostReg():
    # n_estimators: Количество деревьев в лесу.
    #     default=10
    # learning_rate Скорость обучения
    #     default=0.1
    # Параметры отдельного дерева:
    # max_depth: Максимальная глубина.
    #     default=5
    # min_samples_split: Количество объектов в листе, чтобы его можно было
    #     разбить и превратить в узел. default=2
    # max_leafs: Максимальное количество листьев разрешенное для дерева.
    #     default=20
    # bins: Количество бинов для вычисления гистограммы значений,
    #     по каждому признаку. default=16
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=5,
                 min_samples_split=2, max_leafs=20, bins=16):
        self.n_estimators = n_estimators
        self.learningRate = learning_rate
        
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
        
        # Словарь содержит важность каждого признака
        # в лесу (для всех деревьев)
        self.fi = {}
        
        #
        self.testSum = 0
        self.eps = 1e-15

    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    # y: pd.Series с целевыми значениями
    def fit(self, X, y):
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
    #print(X_test); print()
    
    myBoostReg = MyBoostReg(n_estimators=10, learning_rate=0.1, max_depth=5,
                 min_samples_split=2, max_leafs=20, bins=16)
    print(myBoostReg)
    
    print(end='\n\n')
    print('END!!!');
    input();