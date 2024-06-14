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

# Иерархическая агломеративная кластеризация
class MyAgglomerative():
    # n_clusters: Количество кластеров, которые нужно сформировать.
    #     default=3
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        
        #
        self.eps = 1e-15

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
        
        return y_predict
    
    #****************************************************************
    
    def __str__(self):
        return f'MyAgglomerative class: n_clusters={self.n_clusters}'
    
    def __repr__(self):
        return f'MyAgglomerative(n_clusters={self.n_clusters})'

#********************************************************************

if __name__ == '__main__':
    # Для кластеризации
    from sklearn.datasets import make_blobs
    
    X, _ = make_blobs(n_samples=100, centers=5, n_features=5, cluster_std=2.5, random_state=42)
    X = pd.DataFrame(X)
    X.columns = [f'col_{col}' for col in X.columns]
    
    #print(X)
    
    X_test = X.iloc[41: 61, :]
    #print(X_test); print()
    
    myAgglomerative = MyAgglomerative(n_clusters=5)
    print(myAgglomerative)
    
    print(end='\n\n')
    print('END!!!');
    input();
