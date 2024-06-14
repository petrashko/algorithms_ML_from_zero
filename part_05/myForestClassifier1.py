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

from myTreeClassifier import MyTreeClf

# Случайный лес - классификация
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
    print(myForestClf)
    
    # Обучение
    '''
    myForestClf.fit(X_train, y_train)
    #print()
    for k in np.arange(0, myForestClf.n_estimators):
        tree = myForestClf.trees[k]
        tree.printTree(tree.root); print()
    print(myForestClf.leafs_cnt, myForestClf.testSum)
    '''
    
    # Предсказание
    '''
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
    '''

    # Параметры обученной модели
    #print(myForestClf.fi); print()
    #print(f'{myForestClf.oobScore}: {myForestClf.oob_score_}'); print()
    
    print(end='\n\n')
    print('END!!!');
    input();
        