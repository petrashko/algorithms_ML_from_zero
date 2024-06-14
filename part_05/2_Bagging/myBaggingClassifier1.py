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

from myLogClassifier import MyLogReg
from myKnnClassifier import MyKNNClf
from myTreeClassifier import MyTreeClf

# Бэггинг - классификация
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
        
        #
        self.randomState = random_state
        self.eps = 1e-15

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
    print(myBaggingClf)
    
    # Обучение
    '''
    print('# Логистическая регрессия')
    myLogReg = MyLogReg(
        n_iter=50, learning_rate=lambda x: 0.5 * (0.85 ** x),
        weights=None, metric='accuracy',
        reg='elasticnet', l1_coef=0.5, l2_coef=0.5,
        sgd_sample=0.5, random_state=42
    )
    
    myBaggingClf = MyBaggingClf(
        estimator=myLogReg, n_estimators=5, max_samples=0.5, oob_score='mae', random_state=42
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
    predictList = myBaggingClf.predict(X_test)
    #print( probaList ); print()
    #print( predictList ); print()
    print( probaList.sum(), predictList.sum() ); print()
    
    # Параметры обученной модели
    print(f'{myBaggingClf.oobScore}: {myBaggingClf.oob_score_}'); print()
    
    # Обучение
    
    print('# Метод K-ближайших соседей')
    myKnnClf = MyKNNClf(k=3, metric='chebyshev', weight='distance')
    
    myBaggingClf = MyBaggingClf(
        estimator=myKnnClf, n_estimators=5, max_samples=0.5, oob_score='mae', random_state=42
    )
    myBaggingClf.fit(X_train, y_train)
    #print()
    for model in myBaggingClf.estimators:
        print(model.train_size, end='; ')
    print(end='\n\n')
    
    print('# Предсказание')
    probaList = myBaggingClf.predict_proba(X_test)
    predictList = myBaggingClf.predict(X_test)
    #print( probaList ); print()
    #print( predictList ); print()
    print( probaList.sum(), predictList.sum() ); print()
        
    # Параметры обученной модели
    print(f'{myBaggingClf.oobScore}: {myBaggingClf.oob_score_}'); print()
    
    # Обучение
    
    print('# Дерево решений')
    myTreeClf = MyTreeClf(max_depth=4, min_samples_split=2, max_leafs=17, criterion='gini', bins=16)
    
    myBaggingClf = MyBaggingClf(
        estimator=myTreeClf, n_estimators=5, max_samples=0.5, oob_score='mae', random_state=42
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
    predictList = myBaggingClf.predict(X_test)
    #print( probaList ); print()
    #print( predictList ); print()
    print( probaList.sum(), predictList.sum() ); print()
        
    # Параметры обученной модели
    print(f'{myBaggingClf.oobScore}: {myBaggingClf.oob_score_}'); print()
    '''
    print(end='\n\n')
    print('END!!!');
    input();