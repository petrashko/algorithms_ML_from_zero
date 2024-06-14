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

from myTreeClassifier import MyTreeClf

# Случайный лес - классификация (предсказание - спуск по каждому дереву)
class MyForestClf():
    # Параметры для леса:
    # n_estimators: Количество деревьев в лесу.
    #     default=10
    # max_features: Доля признаков, которая будет случайным образом выбираться
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
    # criterion: Как вычислять прирост информации при поиске наилучшего
    #     разбиения. Возможные значения: 'entropy', 'gini'.
    #     default='entropy'
    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5,
                 max_depth=5, min_samples_split=2, max_leafs=20,
                 bins=16, criterion='entropy', random_state=42):
        # Параметры для леса
        self.n_estimators = n_estimators
        self.maxFeatures = max_features
        self.maxSamples = max_samples
        
        # Подготовить список из n_estimators решающих деревьев (пока пустых)
        self.trees = [None] * self.n_estimators
        for k in np.arange(0, self.n_estimators):
            treeClf = MyTreeClf(
                max_depth=max_depth, min_samples_split=min_samples_split,
                max_leafs=max_leafs, criterion=criterion, bins=bins
            )
            self.trees[k] = treeClf
        
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
        
        # Словарь содержит важность каждого признака
        # в лесу (для всех деревьев)
        self.fi = {}
        
        # Вычисленная оценка out-of-bag Error
        self.oob_score_ = 0
        
        #
        self.randomState = random_state
        self.testSum = 0
        self.eps = 1e-15

    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    # y: pd.Series с целевыми значениями
    def fit(self, X, y):
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        for col, featureName in enumerate(X.columns.tolist()):
            # Создать словарь, номер колонки: название признака
            self.colsToFeatureName[col] = featureName
            # Каждому признаку присвоить важность 0
            self.fi[featureName] = 0
        
        # Запомнить кол-во объектов в тренировочном наборе
        self.countSamplesTotal = X.shape[0]
        # Запомнить кол-во признаков в тренировочном наборе
        self.countFeaturesTotal = X.shape[1]
        
        # Для воспроизводимости результатов (тестирования)
        random.seed(self.randomState)
        
        # Кол-во объектов (строк), которые используются
        # для построения каждого отдельного дерева
        #     (self.maxSamples перевести в целое число)
        countRows = int( np.round(self.maxSamples * self.countSamplesTotal, 0) )
        
        # Кол-во признаков (колонок), которые используются
        # для построения каждого отдельного дерева
        #     (self.maxFeatures перевести в целое число)
        countCols = int( np.round(self.maxFeatures * self.countFeaturesTotal, 0) )
        
        # Построить (обучить) n_estimators решающих деревьев
        for k in np.arange(0, self.n_estimators):
            tree = self.trees[k]
            # Передать в дерево ОБЩЕЕ кол-во объектов в тренировочном наборе
            tree.countSamplesTotal = self.countSamplesTotal
            
            # Случайно выбираем countCols названий признаков
            nameFeatures = random.sample(X.columns.tolist(), countCols)
            X_random = X[nameFeatures]
            
            # Случайно выбираем countRows индексов строк (наблюдений)
            idxRows = random.sample(list(X.index), countRows)
            
            # Оставили только случайно выбранные колонки и строки
            X_random = X_random.loc[idxRows]
            y_random = y.loc[idxRows]
            
            # Построить (обучить) k-ое дереро
            tree.fit(X_random, y_random)
            
            # Считаем общее кол-во листьев в лесу (для всех деревьев)
            self.leafs_cnt += tree.lastLeaf
            self.testSum += tree.testSum
        # END: for k in np.arange(0, self.n_estimators):
    
    #****************************************************************
    
    # x_test: вектор numpy с признаками
    # numTree: Номер дерева
    def help_predict(self, x_test, numTree):
        tree = self.trees[numTree]
        # Корень дерева
        node = tree.root
        # В листьях дерева находится значение вероятности для класса с меткой 1
        proba1 = node.data.get('proba1', None)
        
        # Спускаемся по узлам дерева, пока не достигнем листа
        while proba1 is None:
            # Получить название признака и порогове значение для него
            # из текущего узла дерева
            featureName = node.data.get('featureName')
            threshold = node.data.get('threshold')
            
            # По названию признака, получить номер колонки
            col = self.getColNumberByName(featureName)
            # Получить значение этого признака у объекта
            value = x_test[col]
            # Если текущее значение <= порога - спускаемся по левому поддереву
            if value <= threshold:
                node = node.left
            # Иначе, спускаемся по правому поддереву
            else:
                node = node.right
            
            # Если достигли листа proba1 будет числом
            proba1 = node.data.get('proba1', None)
        
        return proba1
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict_proba(self, X):
        y_proba = np.zeros(X.shape[0])
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        X_test = X.to_numpy()
        
        # Цикл по объектам (строкам) из тестового набора признаков
        for n in np.arange(0, X_test.shape[0]):
            row = X_test[n, :]
            # Сумма вероятностей для класса 1
            # в найденных листах каждого дерева
            proba1 = 0
            
            # Поиск нужного узла в каждом дереве
            for k in np.arange(0, self.n_estimators):
                # После прохода по дереву решений получить
                # вероятность для класса 1
                proba1 += self.help_predict(row, k)
            
            # Взять среднее значение по всем деревьям
            y_proba[n] = proba1 / self.n_estimators
        
        return y_proba
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    # type: Способ усреднения результата. Принимает значение: 'mean', 'vote'
    def predict(self, X, type='mean'):
        y_predict = np.zeros(X.shape[0], dtype=np.int32)
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        X_test = X.to_numpy()
        
        # Цикл по объектам (строкам) из тестового набора признаков
        for n in np.arange(0, X_test.shape[0]):
            row = X_test[n, :]
            # Список вероятностей, того что объект принадлежит
            # классу 1, по каждому дереву в лесу
            proba1List = np.zeros(self.n_estimators)
            
            # Поиск нужного узла в каждом дереве
            for k in np.arange(0, self.n_estimators):
                # После прохода по дереву решений получить
                # вероятность для класса 1
                proba1List[k] = self.help_predict(row, k)
            
            # Усреднить вероятности
            if type == 'mean':
                proba1 = proba1List.sum() / self.n_estimators
                y_predict[n] = 1 if proba1 > 0.5 else 0
            # Иначе, голосование
            else:
                # От каждого дерева получить метку класса
                labels = list(map(
                    lambda probability: 1 if probability > 0.5 else 0,
                    proba1List
                ))
                labels = np.array(labels, dtype=np.int8)
                # Кол-во 0
                count_0 = np.count_nonzero(labels == 0)
                # Кол-во 1
                count_1 = np.count_nonzero(labels == 1)
                y_predict[n] = 1 if count_1 >= count_0 else 0
        
        return y_predict
    
    #****************************************************************
    
    # Метод по названию признака возвращает номер
    # колонки для матрицы с признаками
    def getColNumberByName(self, featureName):
        colNumber = None
        for key, name in self.colsToFeatureName.items():
            if featureName == name:
                colNumber = key
                break
        return colNumber
    
    #****************************************************************
    
    def __str__(self):
        tree = self.trees[0]
        return f'MyForestClf class: n_estimators={self.n_estimators}, max_features={self.maxFeatures}, max_samples={self.maxSamples}, max_depth={tree.maxDepth}, min_samples_split={tree.minSamplesSplit}, max_leafs={tree.maxLeafs}, bins={tree.bins}, criterion={tree.criterion}, random_state={self.randomState}'
    
    def __repr__(self):
        tree = self.trees[0]
        return f'MyForestClf(n_estimators={self.n_estimators}, max_features={self.maxFeatures}, max_samples={self.maxSamples}, max_depth={tree.maxDepth}, min_samples_split={tree.minSamplesSplit}, max_leafs={tree.maxLeafs}, bins={tree.bins}, criterion={tree.criterion}, random_state={self.randomState})'

#********************************************************************

if __name__ == '__main__':
    # Для классификации
    df = pd.read_csv(f'..{sep}data{sep}banknote+authentication.zip', header=None)
    df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    X, y = df.iloc[:,:4], df['target']
    
    X_train = X
    y_train = y
    
    #X_train = X.iloc[757:767, :]
    #y_train = y.iloc[757:767]
    #print(X_train); print(y_train)
    
    myForestClf = MyForestClf(n_estimators=6, max_features=0.5, max_samples=0.5,
                 max_depth=4, min_samples_split=2, max_leafs=20,
                 bins=16, criterion='entropy', random_state=42)
    #print(myForestClf)
    
    # Обучение
    myForestClf.fit(X_train, y_train)
    #print()
    for k in np.arange(0, myForestClf.n_estimators):
        tree = myForestClf.trees[k]
        tree.printTree(tree.root); print()
    print(myForestClf.leafs_cnt, myForestClf.testSum)
    
    # Предсказание
    X_test = X.iloc[757:767, :]
    print()
    probaList = myForestClf.predict_proba(X_test)
    print( probaList ); print()
    print( probaList.sum() ); print()
    predictList = myForestClf.predict(X_test, type='mean')
    print( predictList ); print()
    print( predictList.sum() ); print()
    predictList = myForestClf.predict(X_test, type='vote')
    print( predictList ); print()
    print( predictList.sum() ); print()
    
    # Параметры обученной модели
    #print(myForestClf.fi); print()
    #print(f'{myForestClf.oobScore}: {myForestClf.oob_score_}'); print()
    
    print(end='\n\n')
    print('END!!!');
    input();
        