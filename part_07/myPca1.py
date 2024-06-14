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

# Метод главных компонент (PCA)
class MyPCA():
    # n_components: Количество главных компонент,
    #     которое следует оставить. default=3
    def __init__(self, n_components=3):
        self.n_components = n_components
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
    
    def __str__(self):
        return f'MyPCA class: n_components={self.n_components}'
    
    def __repr__(self):
        return f'MyPCA(n_components={self.n_components})'
    
#********************************************************************

if __name__ == '__main__':
    # Для кластеризации
    from sklearn.datasets import make_blobs
    
    X, _ = make_blobs(n_samples=100, centers=5, n_features=20, cluster_std=2.5, random_state=42)
    X = pd.DataFrame(X)
    X.columns = [f'col_{col}' for col in X.columns]
    
    #print(X)
    
    myPCA = MyPCA(n_components=5)
    print(myPCA)
    
    print(end='\n\n')
    print('END!!!');
    input();