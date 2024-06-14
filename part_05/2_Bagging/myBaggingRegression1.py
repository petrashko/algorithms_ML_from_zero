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


# Бэггинг - регрессия
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
        
        #
        self.randomState = random_state
        self.eps = 1e-15

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
    print(myBaggingReg)
    
    # Обучение
    '''
    myBaggingReg.fit(X_train, y_train)
    #print()
    for k in np.arange(0, myBaggingReg.n_estimators):
        tree = myBaggingReg.trees[k]
        tree.printTree(tree.root); print()
    print(myBaggingReg.leafs_cnt, myBaggingReg.testSum); print()
    '''
    
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