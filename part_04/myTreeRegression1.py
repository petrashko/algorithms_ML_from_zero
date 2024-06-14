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

# Дерево решений - регрессия
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
        #
        self.eps = 1e-15
    
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
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    myTreeReg = MyTreeReg(max_depth=5, min_samples_split=2, max_leafs=20)
    print(myTreeReg)
    # Обучить модель
    #myTreeReg.fit(X, y)
    
    
    print(end='\n\n')
    print('END!!!');
    input();
