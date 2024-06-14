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

from myLineRegression import MyLineReg
from myKnnRegression import MyKNNReg
from myTreeRegression import MyTreeReg

# Бэггинг - регрессия (обучение - создание n_estimators экземпляров модели)
class MyBaggingReg():
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
    
    def __str__(self):
        return f'MyBaggingReg class: estimator={self.estimator}, n_estimators={self.n_estimators}, max_samples={self.maxSamples}, random_state={self.randomState}'
    
    def __repr__(self):
        return f'MyBaggingReg(estimator={self.estimator}, n_estimators={self.n_estimators}, max_samples={self.maxSamples}, random_state={self.randomState})'

#********************************************************************

if __name__ == '__main__':
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)
    X_train, y_train = data['data'], data['target']
    #print(X_train); print()
    #print(y_train); print()
    
    myBaggingReg = MyBaggingReg(
        estimator=None, n_estimators=10, max_samples=1.0, random_state=42
    )
    #print(myBaggingReg)
    
    # Обучение
    
    # Линейная регрессия
    myLineReg = MyLineReg(
        n_iter=50, learning_rate=lambda x: 0.5 * (0.85 ** x),
        weights=None, metric='mae',
        reg='elasticnet', l1_coef=0.5, l2_coef=0.5,
        sgd_sample=0.5, random_state=42
    )
    
    myBaggingReg = MyBaggingReg(
        estimator=myLineReg, n_estimators=5, max_samples=0.5, random_state=42
    )
    myBaggingReg.fit(X_train, y_train)
    #print()
    meanCoef = 0
    for model in myBaggingReg.estimators:
        meanCoef += model.get_coef().mean()
        #print( model.get_coef() )
    print( meanCoef ); print()
    
    # Метод K-ближайших соседей
    myKnnReg = MyKNNReg(k=3, metric='euclidean', weight='distance')
    
    myBaggingReg = MyBaggingReg(
        estimator=myKnnReg, n_estimators=5, max_samples=0.5, random_state=42
    )
    myBaggingReg.fit(X_train, y_train)
    #print()
    for model in myBaggingReg.estimators:
        print(model.train_size, end='; ')
    print(end='\n\n')
    
    # Дерево решений
    myTreeReg = MyTreeReg(max_depth=4, min_samples_split=50, max_leafs=17, bins=16)
    myBaggingReg = MyBaggingReg(
        estimator=myTreeReg, n_estimators=5, max_samples=0.5, random_state=42
    )
    myBaggingReg.fit(X_train, y_train)
    #print()
    lastLeaf = 0; testSum = 0
    for model in myBaggingReg.estimators:
        lastLeaf += model.lastLeaf; testSum += model.testSum
        #model.printTree(model.root); print()
    print(lastLeaf, testSum); print()
    
    # Предсказание
    '''
    X_test = X_train.iloc[211:231, :]
    predictList = myBaggingReg.predict(X_test)
    print( predictList ); print()
    print( predictList.sum() ); print()
    '''
    
    print(end='\n\n')
    print('END!!!');
    input();