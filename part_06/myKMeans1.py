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

# Метод k-средних - кластеризация
class MyKMeans():
    # n_clusters: Количество кластеров, которые нужно сформировать.
    #     default=3
    # max_iter: Количество итераций алгоритма.
    #     default=10
    # n_init: Сколько раз прогоняется алгоритм k-средних.
    #     default=3
    # random_state: Для воспроизводимости результата.
    #     default=42
    def __init__(self, n_clusters=3, max_iter=10, n_init=3, random_state=42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.maxIter = max_iter
        
        #
        self.randomState = random_state
        self.eps = 1e-15

    #****************************************************************
    
    def __str__(self):
        return f'MyKMeans class: n_clusters={self.n_clusters}, max_iter={self.maxIter}, n_init={self.n_init}, random_state={self.randomState}'
    
    def __repr__(self):
        return f'MyKMeans(n_clusters={self.n_clusters}, max_iter={self.maxIter}, n_init={self.n_init}, random_state={self.randomState})'

#********************************************************************

if __name__ == '__main__':
    # Для кластеризации
    from sklearn.datasets import make_blobs
    
    X, _ = make_blobs(n_samples=100, centers=5, n_features=5, cluster_std=2.5, random_state=42)
    X = pd.DataFrame(X)
    X.columns = [f'col_{col}' for col in X.columns]
    
    #print(X)
    
    myKMeans = MyKMeans(n_clusters=5, max_iter=10, n_init=3, random_state=42)
    print(myKMeans)
    
    print(end='\n\n')
    print('END!!!');
    input();
