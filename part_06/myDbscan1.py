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

# DBSCAN - кластеризация
class MyDBSCAN():
    # eps: Радиус, в котором будут искаться соседи.
    #     default=3
    # min_samples: Минимальное количество соседей, которое должно быть
    #     в радиусе eps (при поиске соседей) default=3
    def __init__(self, eps=3, min_samples=3):
        self.eps = eps
        self.minSamples = min_samples
        
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def fit(self, X):
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        
        X_train = X.to_numpy()
        
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict(self, X):
        y_predict = np.zeros(X.shape[0], dtype=np.int32)
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        
        X_test = X.to_numpy()
        
        # Цикл по объектам (строкам) из тестового набора
        for n in np.arange(0, X_test.shape[0]):
            pass
        
        return y_predict
    
    #****************************************************************
    
    def __str__(self):
        return f'MyDBSCAN class: eps={self.eps}, min_samples={self.minSamples}'
    
    def __repr__(self):
        return f'MyDBSCAN(eps={self.eps}, min_samples={self.minSamples})'

#********************************************************************

if __name__ == '__main__':
    # Для кластеризации
    from sklearn.datasets import make_blobs
    
    X, _ = make_blobs(n_samples=100, centers=5, n_features=5, cluster_std=2.5, random_state=42)
    X = pd.DataFrame(X)
    X.columns = [f'col_{col}' for col in X.columns]
    
    #print(X)
    
    myDBSCAN = MyDBSCAN(eps=10, min_samples=11)
    print(myDBSCAN)
    
    print(end='\n\n')
    print('END!!!');
    input();
