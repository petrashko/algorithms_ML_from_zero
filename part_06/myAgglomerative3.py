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

# Иерархическая агломеративная кластеризация (метрики расстояния)
class MyAgglomerative():
    # n_clusters: Количество кластеров, которые нужно сформировать.
    #     default=3
    # metric: Принимает значение: 'euclidean', 'manhattan', 'chebyshev', 'cosine'
    #     default='euclidean'
    def __init__(self, n_clusters=3, metric='euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric
        
        self.X_train = None
        
        # Словарь со списками номеров объектов, попавших в один кластер
        # Ключ словаря: номер кластера
        self.clusterObjectsDict = {}
        
        # Словарь с центрами кластеров (среднее по объектам попавших в один кластер)
        # Ключ словаря: номер кластера
        self.clusterCentersDict = {}
        
        #
        self.eps = 1e-15

    #****************************************************************
    
    # Метод возвращает Евклидово расстояние между двумя векторами
    def euclideanDistance(self, a, b):
        sum2 = np.sum((a - b) ** 2)
        result = np.sqrt(sum2)
        return result
    
    #****************************************************************
    
    # Метод возвращает Манхэттенское расстояние между двумя векторами
    def manhattanDistance(self, a, b):
        absDif = np.abs(a - b)
        result = np.sum(absDif)
        return result
    
    #****************************************************************
    
    # Метод возвращает расстояние Чебышева между двумя векторами
    def chebyshevDistance(self, a, b):
        absDif = np.abs(a - b)
        result = np.max(absDif)
        return result

    #****************************************************************
    
    # Метод возвращает Косинусное расстояние между двумя векторами
    def cosineDistance(self, a, b):
        numerator = np.sum(a * b)
        normA = np.sqrt( np.sum(a ** 2) )
        normB = np.sqrt( np.sum(b ** 2) )
        denominator = normA * normB
        
        result = 1 - (numerator / denominator)
        return result
    
    #****************************************************************
    
    # Метод вычисляет центр кластера с номером cluster
    def calculateCenter(self, cluster):
        center = None
        # Получить все номера объектов, которые попали в кластер
        rows = self.clusterObjectsDict.get(cluster, [])
        
        # Центр кластера это вектор, длина которого равна
        # кол-ву признаков в тренировочном наборе
        tmpVec = np.zeros(self.X_train.shape[1])
        for row in rows:
            obj = self.X_train[row, :]
            tmpVec += obj
        
        if len(rows) > 0:
            center =  tmpVec / len(rows)
        self.clusterCentersDict[cluster] = center

    #****************************************************************
    
    def help_fit(self):
        currCluster1 = None
        currCluster2 = None
        
        # Присвоить заведомо большое число
        minDistance = float('inf')
        
        clusters = sorted(self.clusterCentersDict.keys())
        
        # Попарно считаем расстояние между всеми центрами кластеров
        for num1, numCluster1 in enumerate(clusters):
            for numCluster2 in clusters[num1+1 : len(clusters)]:
                if numCluster1 == numCluster2:
                    continue
                center1 = self.clusterCentersDict.get(numCluster1)
                center2 = self.clusterCentersDict.get(numCluster2)
                
                # Вычислить Косинусное расстояние
                if self.metric == 'cosine':
                    distance = self.cosineDistance(center1, center2)
                # Вычислить расстояние Чебышева
                elif self.metric == 'chebyshev':
                    distance = self.chebyshevDistance(center1, center2)
                # Вычислить Манхэттенское расстояние
                elif self.metric == 'manhattan':
                    distance = self.manhattanDistance(center1, center2)
                # Иначе, вычислить Евклидово расстояние
                else:
                    distance = self.euclideanDistance(center1, center2)
                
                if distance < minDistance:
                    currCluster1 = numCluster1
                    currCluster2 = numCluster2
                    minDistance = distance
        
        # Два ближайщих кластера объединяем в один
        rows1 = self.clusterObjectsDict.get(currCluster1)
        rows2 = self.clusterObjectsDict.get(currCluster2)
        # Номера объектов из кластера currCluster2 добавить
        # в список с номерами объектов кластера currCluster1
        self.clusterObjectsDict[currCluster1] = rows1 + rows2
        # Пересчитать центр кластера currCluster1
        self.calculateCenter(currCluster1)
        # Удалить все данные о кластере currCluster2 он больше не нужен
        del self.clusterCentersDict[currCluster2]
        del self.clusterObjectsDict[currCluster2]
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def fit_predict(self, X):
        y_predict = np.zeros(X.shape[0], dtype=np.int32)
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        
        self.X_train = X.to_numpy()
        
        # Изначально каждый объект это отдельный кластер 
        for row in np.arange(0, self.X_train.shape[0]):
            self.clusterObjectsDict[row+1] = [row]
            # Каждый объект это центр кластера 
            self.calculateCenter(row+1)
        
        # Пока количество кластеров больше self.n_clusters
        while len(self.clusterCentersDict.keys()) > self.n_clusters:
            self.help_fit()
        
        # Получить и отсортировать номера кластеров
        clusters = sorted(self.clusterCentersDict.keys())
        
        # Здесь num начинается с 0 [0, 1, 2, ... self.n_clusters-1]
        # cluster: ключ словаря
        for num, cluster in enumerate(clusters):
            # Список номеров объектов, которые попали в cluster
            rows = self.clusterObjectsDict.get(cluster)
            for row in rows:
                y_predict[row] = num+1
        
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
    
    myAgglomerative = MyAgglomerative(n_clusters=5, metric='cosine')
    #print(myAgglomerative)
    
    # Обучение и предсказание
    print()
    predictList = myAgglomerative.fit_predict(X)
    print( predictList ); print()
    #predictList = myAgglomerative.fit_predict(X_test)
    #print( predictList ); print()
    
    print(end='\n\n')
    print('END!!!');
    input();
