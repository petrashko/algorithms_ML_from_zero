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

from myTreeRegression import MyTreeReg

# Случайный лес - регрессия
class MyForestReg():
    # Параметры исключительно для леса:
    # n_estimators: Количество деревьев в лесу.
    #     default=10
    # max_features: Доля фичей, которая будет случайным образом выбираться
    #     для каждого дерева. От 0.0 до 1.0. default=0.5
    # max_samples: Доля объектов, которая будет случайным образом выбираться
    #     из датасета для каждого дерева. От 0.0 до 1.0. default=0.5
    # random_state: Для воспроизводимости результата.
    #     default=42
    
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
                 max_depth=5, min_samples_split=2, max_leafs=20,
                 bins=16, random_state=42):
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
        
        #
        self.randomState = random_state
        self.testSum = 0
        self.eps = 1e-15

    #****************************************************************
    
    def __str__(self):
        tree = self.trees[0]
        return f'MyForestReg class: n_estimators={self.n_estimators}, max_features={self.maxFeatures}, max_samples={self.maxSamples}, max_depth={tree.maxDepth}, min_samples_split={tree.minSamplesSplit}, max_leafs={tree.maxLeafs}, bins={tree.bins}, random_state={self.randomState}'
    
    def __repr__(self):
        tree = self.trees[0]
        return f'MyForestReg(n_estimators={self.n_estimators}, max_features={self.maxFeatures}, max_samples={self.maxSamples}, max_depth={tree.maxDepth}, min_samples_split={tree.minSamplesSplit}, max_leafs={tree.maxLeafs}, bins={tree.bins}, random_state={self.randomState})'
    
#********************************************************************

if __name__ == '__main__':
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)
    X_train, y_train = data['data'], data['target']
    #print(X_train); print()
    #print(y_train); print()
    
    myForestReg = MyForestReg(n_estimators=10, max_features=0.5, max_samples=0.5,
                 max_depth=5, min_samples_split=2, max_leafs=20,
                 bins=16, random_state=42)
    print(myForestReg)
    
    print(end='\n\n')
    print('END!!!');
    input();