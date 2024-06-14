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

# DBSCAN - кластеризация (метрики расстояния)
class MyDBSCAN():
    # eps: Радиус, в котором будут искаться соседи.
    #     default=3
    # min_samples: Минимальное количество соседей, которое должно быть
    #     в радиусе eps (при поиске соседей) default=3
    # metric: Принимает значение: 'euclidean', 'manhattan', 'chebyshev', 'cosine'
    #     default='euclidean'
    def __init__(self, eps=3, min_samples=3, metric='euclidean'):
        self.distance = eps
        self.minSamples = min_samples
        self.metric = metric
        
        self.X_train = None
        
        # Номер кластера
        self.numCluster = 0
        
        # Словарь со списками номеров объектов, попавших в один кластер
        # Ключ словаря: номер кластера
        self.clusterObjectsDict = {}
        
        # Список с номерами объектов, которые остались как выбросы
        self.outlierList = []
        
        # Номера объектов, которые уже попали в какой-то кластер.
        # (без разницы в какой)
        self.alreadyInClustersSet = set()
        
        # Номера объектов, которые уже посетили
        self.visitedObjectsSet = set()
        
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
    
    # Метод возвращает номера соседей объекта с номером num,
    # расстояние до которых <= self.distance
    def getNeighbourList(self, num):
        obj = self.X_train[num, :]
        neighbourList = []
        
        # Цикл по объектам из тренировочного набора
        for row in np.arange(0, self.X_train.shape[0]):
            if row == num:
                continue
            obj2 = self.X_train[row, :]
            
            # Вычислить Косинусное расстояние
            if self.metric == 'cosine':
                distance = self.cosineDistance(obj, obj2)
            # Вычислить расстояние Чебышева
            elif self.metric == 'chebyshev':
                distance = self.chebyshevDistance(obj, obj2)
            # Вычислить Манхэттенское расстояние
            elif self.metric == 'manhattan':
                distance = self.manhattanDistance(obj, obj2)
            # Иначе, вычислить Евклидово расстояние
            else:
                distance = self.euclideanDistance(obj, obj2)

            if distance < self.distance:
                neighbourList.append(row)
        
        return neighbourList
    
    #****************************************************************
    
    # num: Номер объекта, который помечаем как корневой
    # neighbourList: Список с номерами найденных соседей, для корневого объекта
    def createCluster(self, num, neighbourList):
        self.numCluster += 1
        
        if self.numCluster not in self.clusterObjectsDict.keys():
            # Создаем новый кластер
            self.clusterObjectsDict[self.numCluster] = []
        
        self.clusterObjectsDict.get(self.numCluster).append(num)
        
        # Запомнить, что объект уже попал в какой-то кластер
        self.alreadyInClustersSet.add(num)
        
        # Обрабатываем соседей
        # - поиск всех объетов, которые попадают в кластер self.numCluster
        while neighbourList:
            # Получить номер объекта (соседа) из тренировочного набора
            row = neighbourList.pop()
            #neighbour = self.X_train[row, :]
            
            # Если в ГЛАВНОМ цикле еще не добрались до объекта.
            # (Объект с номером row НЕ корневой и НЕ выброс)
            if row not in self.visitedObjectsSet:
                # ...теперь посетили объект с номером row
                self.visitedObjectsSet.add(row)
                # Поиск соседей для соседа
                neighbourNextList = self.getNeighbourList(row)
                if len(neighbourNextList) >= self.minSamples:
                    # Расширить список соседей
                    # (объектов которые возможно попадают в кластер self.numCluster)
                    neighbourList.extend(neighbourNextList)
            
            # Если объект еще не попал в кластер
            if row not in self.alreadyInClustersSet:
                self.alreadyInClustersSet.add(row)
                self.clusterObjectsDict.get(self.numCluster).append(row)
                # Если объект был в списке выбросов
                if row in self.outlierList:
                    # ...удалить его от туда
                    self.outlierList.remove(row)
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def fit_predict(self, X):
        y_predict = np.zeros(X.shape[0], dtype=np.int32)
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        
        self.X_train = X.to_numpy()
        
        # ГЛАВНЫЙ цикл по объектам из тренировочного набора
        for row in np.arange(0, self.X_train.shape[0]):
            # Если объект уже проходили
            if row in self.visitedObjectsSet:
                continue
            
            # Посетили объект с номером row
            self.visitedObjectsSet.add(row)
            # Поиск соседей для объекта
            neighbourList = self.getNeighbourList(row)
            
            if len(neighbourList) < self.minSamples:
                # Добавить объект в список выбросов
                self.outlierList.append(row)
            else:
                # Делаем объект корневым и создаем новый кластер
                self.createCluster(row, neighbourList)
    
        # Получить и отсортировать номера кластеров
        clusters = sorted(self.clusterObjectsDict.keys())
        
        # Здесь num начинается с 0 [0, 1, 2, ... self.n_clusters-1]
        # cluster: ключ словаря
        for num, cluster in enumerate(clusters):
            # Список номеров объектов, которые попали в cluster
            rows = self.clusterObjectsDict.get(cluster)
            for row in rows:
                y_predict[row] = num+1
        
        # Перебрать номера объектов, которые остались как выбросы
        # (они пойдут как кластер с номером 0)
        for row in self.outlierList:
            y_predict[row] = 0

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
    #X_test = X.iloc[41: 61, :]
    #X_test = X[['col_1', 'col_2']]
    #print(X_test); print()
    
    myDBSCAN = MyDBSCAN(eps=5, min_samples=4, metric='euclidean')
    #print(myDBSCAN)
    
    # Обучение и предсказание
    print()
    predictList = myDBSCAN.fit_predict(X)
    print(myDBSCAN.clusterObjectsDict); print()
    print( predictList ); print()
    #predictList = myDBSCAN.fit_predict(X_test)
    #print( predictList ); print()
    
    print(end='\n\n')
    print('END!!!');
    input();
