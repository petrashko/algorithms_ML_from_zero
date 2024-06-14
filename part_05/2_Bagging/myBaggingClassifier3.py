import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import random
import copy

sep = os.sep
# Добавить в путь до родительской папки
#sys.path.append(os.path.join(sys.path[0], f'..{sep}'))
#sys.path.append(os.path.join(os.getcwd(), f'..{sep}'))

from myLogClassifier import MyLogReg
from myKnnClassifier import MyKNNClf
from myTreeClassifier import MyTreeClf

# Бэггинг - классификация (предсказание - обход всех n_estimators моделей)
class MyBaggingClf():
    # estimator: Содержит экземпляр одного из базового класса
    #     регрессии: MyLineReg, MyKNNReg или MyTreeReg
    #     default=None
    # n_estimators: Количество базовых экземпляров, которые будут обучены.
    #     default=10
    # max_samples: Доля объектов, которая будет случайным образом выбираться
    #     из датасета для обучения каждого экземпляра модели. От 0.0 до 1.0.
    #     От 0.0 до 1.0. default=1.0
    # random_state: Для воспроизводимости результата.
    #     default=42
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0,
                 random_state=42):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.maxSamples = max_samples
        
        # Подготовить список из n_estimators моделей (пока пустых)
        self.estimators = [None] * self.n_estimators
        for k in np.arange(0, self.n_estimators):
            self.estimators[k] = copy.deepcopy(self.estimator)
        
        # Кол-во объектов (строк) в исходном тренировочном наборе,
        # который подается на вход алгоритму обучения
        self.countSamplesTotal = 0
        
        #
        self.randomState = random_state
        self.eps = 1e-15

    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    # y: pd.Series с целевыми значениями
    def fit(self, X, y):
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Запомнить кол-во объектов в тренировочном наборе
        self.countSamplesTotal = X.shape[0]
        
        # Кол-во объектов (строк), которые используются
        # для обучения каждого экземпляра модели
        #     (self.maxSamples перевести в целое число)
        countRows = int( np.round(self.maxSamples * self.countSamplesTotal, 0) )
        
        # Для воспроизводимости результатов (тестирования)
        random.seed(self.randomState)
        
        # Из-за странного поведения модуля random - индексы строк
        # для обучения каждой модели, формируем в отдельном цикле
        idxMatrix = [None] * self.n_estimators
        for k in np.arange(0, self.n_estimators):
            # Случайно выбираем countRows индексов строк
            # (наблюдений) для обучения k-ой модели
            idxRows = random.choices(list(X.index), k=countRows)
            idxMatrix[k] = idxRows
        
        # Обучить n_estimators моделей
        for k in np.arange(0, self.n_estimators):
            model = self.estimators[k]
            
            if hasattr(model, 'countSamplesTotal'):
                # Передать в модель ОБЩЕЕ кол-во объектов в тренировочном наборе
                model.countSamplesTotal = self.countSamplesTotal
            
            idxRows = idxMatrix[k]
            # Оставили только случайно выбранные строки
            X_random = X.loc[idxRows]
            y_random = y.loc[idxRows]
            
            # Обучить k-ую модель
            model.fit(X_random, y_random)

    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict_proba(self, X):
        y_proba = np.zeros(X.shape[0])
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        
        # Суммируем вероятности для класса
        # с меткой 1, от каждой модели
        for model in self.estimators:
            y_proba += model.predict_proba(X)
        
        # Взять средную вероятнность по всем моделям
        y_proba /= self.n_estimators
        
        return  y_proba
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    # type: Способ усреднения результата. Принимает значение: 'mean', 'vote'
    def predict(self, X, type='mean'):
        y_predict = np.zeros(X.shape[0], dtype=np.int32)
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        
        # В строках будут вероятности для
        # тестового объекта от каждой модели
        Y_proba = np.zeros((X.shape[0], self.n_estimators))
        
        # Цикл по моделям
        for k, model in enumerate((self.estimators)):
            # Вероятности для объектов из тестового набора от модели
            proba1FromModel = model.predict_proba(X)
            Y_proba[:, k] = proba1FromModel
            
        
        # Цикл по объектам (строкам) из тестового набора признаков
        for n in np.arange(0, X.shape[0]):
            # Вероятности (от каждой модели) того что
            # n-ый объект принадлежит классу с меткой 1
            proba1List = Y_proba[n, :]
            
            # Усреднить вероятности
            if type == 'mean':
                proba1 = proba1List.mean()
                y_predict[n] = 1 if proba1 > 0.5 else 0
            # Иначе, голосование
            else:
                # От каждого модели получить метку класса
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
    
    def __str__(self):
        return f'MyBaggingClf class: estimator={self.estimator}, n_estimators={self.n_estimators}, max_samples={self.maxSamples}, random_state={self.randomState}'
    
    def __repr__(self):
        return f'MyBaggingClf(estimator={self.estimator}, n_estimators={self.n_estimators}, max_samples={self.maxSamples}, random_state={self.randomState})'

#********************************************************************

if __name__ == '__main__':
    # Для классификации
    df = pd.read_csv(f'..{sep}..{sep}data{sep}banknote+authentication.zip', header=None)
    df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    X, y = df.iloc[:,:4], df['target']
    
    X_train = X
    y_train = y
    
    #X_train = X.iloc[757:767, :]
    #y_train = y.iloc[757:767]
    #print(X_train); print(y_train)
    
    X_test = X.iloc[757:767, :]
    
    myBaggingClf = MyBaggingClf(
        estimator=None, n_estimators=10, max_samples=1.0, random_state=42
    )
    #print(myBaggingClf)
    
    # Обучение
    
    print('# Логистическая регрессия')
    myLogReg = MyLogReg(
        n_iter=50, learning_rate=lambda x: 0.5 * (0.85 ** x),
        weights=None, metric='accuracy',
        reg='elasticnet', l1_coef=0.5, l2_coef=0.5,
        sgd_sample=0.5, random_state=42
    )
    
    myBaggingClf = MyBaggingClf(
        estimator=myLogReg, n_estimators=5, max_samples=0.5, random_state=42
    )
    myBaggingClf.fit(X_train, y_train)
    #print()
    meanCoef = 0
    for model in myBaggingClf.estimators:
        meanCoef += model.get_coef().mean()
        #print( model.get_coef() )
    print( meanCoef ); print()
    
    print('# Предсказание')
    probaList = myBaggingClf.predict_proba(X_test)
    predictList = myBaggingClf.predict(X_test, type='mean')
    print( probaList ); print()
    print( predictList ); print()
    print( probaList.sum(), predictList.sum() ); print()
    
    # Параметры обученной модели
    #print(f'{myBaggingClf.oobScore}: {myBaggingClf.oob_score_}'); print()
    
    # Обучение
    
    print('# Метод K-ближайших соседей')
    myKnnClf = MyKNNClf(k=3, metric='chebyshev', weight='distance')
    
    myBaggingClf = MyBaggingClf(
        estimator=myKnnClf, n_estimators=5, max_samples=0.5, random_state=42
    )
    myBaggingClf.fit(X_train, y_train)
    #print()
    for model in myBaggingClf.estimators:
        print(model.train_size, end='; ')
    print(end='\n\n')
    
    print('# Предсказание')
    probaList = myBaggingClf.predict_proba(X_test)
    predictList = myBaggingClf.predict(X_test, type='mean')
    print( probaList ); print()
    print( predictList ); print()
    print( probaList.sum(), predictList.sum() ); print()
        
    # Параметры обученной модели
    #print(f'{myBaggingClf.oobScore}: {myBaggingClf.oob_score_}'); print()
    
    # Обучение
    
    print('# Дерево решений')
    myTreeClf = MyTreeClf(max_depth=4, min_samples_split=2, max_leafs=17, criterion='gini', bins=16)
    
    myBaggingClf = MyBaggingClf(
        estimator=myTreeClf, n_estimators=5, max_samples=0.5, random_state=42
    )
    myBaggingClf.fit(X_train, y_train)
    #print()
    lastLeaf = 0; testSum = 0
    for model in myBaggingClf.estimators:
        lastLeaf += model.lastLeaf; testSum += model.testSum
        #model.printTree(model.root); print()
    print(lastLeaf, testSum); print()
    
    print('# Предсказание')
    probaList = myBaggingClf.predict_proba(X_test)
    predictList = myBaggingClf.predict(X_test, type='vote')
    print( probaList ); print()
    print( predictList ); print()
    print( probaList.sum(), predictList.sum() ); print()
        
    # Параметры обученной модели
    #print(f'{myBaggingClf.oobScore}: {myBaggingClf.oob_score_}'); print()
    
    print(end='\n\n')
    print('END!!!');
    input();