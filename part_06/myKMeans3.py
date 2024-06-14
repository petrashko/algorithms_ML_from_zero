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

import copy

# Метод k-средних - кластеризация (предсказание)
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
        
        # Координаты всех центроидов (список списков)
        self.cluster_centers_ = []
        # Лучшее значение WCSS (within-cluster sum of squares)
        # - cумма квадратов внутрикластерных расстояний до центроидов
        self.inertia_ = None
        
        #
        self.randomState = random_state
        self.eps = 1e-15

    #****************************************************************
    
    # Метод возвращает Евклидово расстояние между двумя векторами
    def euclideanDistance(self, a, b):
        sum2 = np.sum((a - b) ** 2)
        result = np.sqrt(sum2)
        return result
    
    #****************************************************************
    
    # Метод возвращает cумму квадратов внутрикластерных расстояний до центров
    #
    # centerMatrix: Матрица с центрами для каждого кластера
    # clusterObjectsDict: Словарь со списками объектов, попавших в один кластер
    #     Ключ словаря: номер кластера
    def getWCSS(self, centerMatrix, clusterObjectsDict):
        wcss = 0
        
        # Цикл по кластерам
        for k in clusterObjectsDict.keys():
            # Объекты, которые попали в k-ый кластер
            objectsInCluster = np.array(clusterObjectsDict.get(k))
            
            # Цикл по объектам (строкам), которые попали в k-ый кластер
            for row in np.arange(0, objectsInCluster.shape[0]):
                obj = objectsInCluster[row, :]
                # Вычислить расстояние от объекта до
                # центра кластера, в который он попал
                distance = self.euclideanDistance(obj, centerMatrix[k])
                # Считаем cумму ВСЕХ квадратов внутрикластерных расстояний до центров
                wcss += distance ** 2
        
        return wcss
    
    #****************************************************************
    
    # X_train: Матрица numpy с признаками из тренировочного набора
    # Каждая строка - отдельный объект. Каждая колонка - конкретный признак
    def help_fit(self, X_train):
        # Матрица с центрами для каждого кластера
        # Каждая строка - это центр одного кластера
        centerMatrix = [None] * self.n_clusters
        
        # Для каждого кластера генерируем случайные центры
        for k in np.arange(0, self.n_clusters):
            # Центр для k-го кластера
            centerList = []
            # Цикл по признакам (колонкам) из тренировочного набора
            for col in np.arange(0, X_train.shape[1]):
                feature = X_train[:, col]
                # Получить случайное число в пределах возможных значений признака
                randomValue = np.random.uniform(feature.min(), feature.max())
                centerList.append(randomValue)
            centerMatrix[k] = centerList
        
        # Словарь со списками объектов, попавших в один кластер
        # Ключ словаря: номер кластера
        clusterObjectsDict = {}
        
        # self.maxIter раз выполняем разнесение объектов
        # по кластерам и пересчет их центров
        for z in np.arange(0, self.maxIter):
            changedCenters = False
        
            # Очистить словарь для следующей итерации
            clusterObjectsDict.clear()
            
            # Цикл по объектам (строкам) из тренировочного набора
            for row in np.arange(0, X_train.shape[0]):
                obj = X_train[row, :]
                # Список расстояний для одного объекта до центров кластеров
                distanceForObj = np.zeros(self.n_clusters)
                
                # Вычислить расстояние от объекта до центра каждого кластера
                for k in np.arange(0, self.n_clusters):
                    distanceForObj[k] = self.euclideanDistance(obj, centerMatrix[k])
    
                # Номер кластера с минимальным растоянием
                numCluster = distanceForObj.argmin()
                # Добавили текущий объект в ближайщий кластер
                clusterObjectsDict.setdefault(numCluster, []).append(obj)
            
            # Пересчитать центры кластеров
            for k in clusterObjectsDict.keys():
                # Центр для k-го кластера
                centerList = []
                objectsInCluster = np.array(clusterObjectsDict.get(k))
                
                # Цикл по признакам (колонкам) объектов, которые попали в k-ый кластер
                for col in np.arange(0, objectsInCluster.shape[1]):
                    feature = objectsInCluster[:, col]
                    # Берем среднее значение признака
                    average = feature.mean()
                    centerList.append(average)
                    
                # Сравнить центры кластеров на предыдущем шаге с текущими
                if np.allclose(centerList, centerMatrix[k], atol=1e-8) == False:
                    changedCenters = True
                # Новый центр для k-го кластера
                centerMatrix[k] = centerList
            
            # Если ничего не изменилось
            if changedCenters == False:
                break
        # END: for z in np.arange(0, self.maxIter):
        
        # Если это первая попытка разбить на кластеры 
        if self.inertia_ is None:
            # Запомнить результаты
            self.inertia_ = self.getWCSS(centerMatrix, clusterObjectsDict)
            self.cluster_centers_ = copy.deepcopy(centerMatrix)
        else:
            # Считаем общую cумму квадратов
            # внутрикластерных расстояний до центров
            wcss = self.getWCSS(centerMatrix, clusterObjectsDict)
            # Если результат получился лучше - запоминаем его
            if wcss < self.inertia_:
                self.inertia_ = wcss
                self.cluster_centers_ = copy.deepcopy(centerMatrix)
        
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def fit(self, X):
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        
        X_train = X.to_numpy()
        
        # Для воспроизводимости результатов (тестирования)
        np.random.seed(self.randomState)

        # Делаем self.n_init попыток разбить тренировочный
        # набор на кластеры и выбираем лучший результат
        for n in np.arange(0, self.n_init):
            self.help_fit(X_train)
        
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
            obj = X_test[n, :]
            # Список расстояний для одного объекта до центров кластеров
            distanceForObj = np.zeros(self.n_clusters)
            
            # Вычислить расстояние от объекта до центра каждого кластера
            for k in np.arange(0, self.n_clusters):
                distanceForObj[k] = self.euclideanDistance(obj, self.cluster_centers_[k])

            # Номер кластера с минимальным растоянием
            numCluster = distanceForObj.argmin()
            y_predict[n] = numCluster
        
        return y_predict
    
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
    
    #print(X); print()
    
    X_test = X.iloc[41: 61, :]
    #print(X_test); print()
    
    myKMeans = MyKMeans(n_clusters=5, max_iter=10, n_init=3, random_state=42)
    #print(myKMeans)
    
    # Обучение
    myKMeans.fit(X)
    #print()
    sumTotal = 0
    for cluster in myKMeans.cluster_centers_:
        print(cluster); print()
        sumTotal += sum(cluster)
    print(myKMeans.inertia_, sumTotal); print()
    
    # Предсказание
    predictList = myKMeans.predict(X_test)
    print( predictList ); print()
    
    print(end='\n\n')
    print('END!!!');
    input();
