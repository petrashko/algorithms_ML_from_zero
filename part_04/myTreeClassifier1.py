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

# Дерево решений - классификация
# Для двух классов с метками: 0 и 1
class MyTreeClf():
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
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    # y: pd.Series с целевыми значениями
    def fit(self, X, y):
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        X_train = X.to_numpy()
        # Вектор столбец
        y_train = y.values.reshape(y.shape[0], 1)
    
    #****************************************************************
    
    # X_test: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def help_predict(self, X_test):
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X_test = X_test.reset_index(drop=True)
        X_features = X_test.to_numpy()

    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict_proba(self, X):
        y_proba = []
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict(self, X):
        y_predict = []
    
    #****************************************************************
    
    def __str__(self):
        return f'MyTreeClf class: max_depth={self.maxDepth}, min_samples_split={self.minSamplesSplit}, max_leafs={self.maxLeafs}'
    
    def __repr__(self):
        return f'MyTreeClf(max_depth={self.maxDepth}, min_samples_split={self.minSamplesSplit}, max_leafs={self.maxLeafs})'

#********************************************************************

if __name__ == '__main__':
    # Для классификации
    df = pd.read_csv(f'..{sep}data{sep}banknote+authentication.zip', header=None)
    df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    X, y = df.iloc[:,:4], df['target']
    #print(y)

    myTreeClf = MyTreeClf(max_depth=5, min_samples_split=2, max_leafs=20)
    
    X_train = X.iloc[562:962, :]
    y_train = y.iloc[562:962]
    #print(X_train.shape); print(y_train.shape)
    #print(X_train); print(y_train)
    '''
    # Обучить модель
    myTreeClf.fit(X_train, y_train)
    print()
    '''
    
    # Проверка
    X_test = X.iloc[757:767, :]
    '''
    print()
    print( myTreeClf.predict_proba(X_test) )
    print( myTreeClf.predict_proba(X_test).sum() )
    '''
    
    '''
    print()
    print( myTreeClf.predict(X_test) )
    print( myTreeClf.predict(X_test).sum() )
    '''
    
    print(end='\n\n')
    print('END!!!');
    input();
