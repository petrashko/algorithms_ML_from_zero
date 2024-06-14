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

from myTreeForClf import MyTreeReg

# Градиентный бустинг - классификация (предсказание)
# Для двух классов с метками: 0 и 1
# (предсказание - сумма предсказаний от каждого дерева + self.pred_0)
class MyBoostClf():
    # learning_rate Скорость обучения
    #     default=0.1
    
    # Параметры для леса:
    # n_estimators: Количество деревьев в лесу.
    #     default=10
    
    # Параметры отдельного дерева:
    # max_depth: Максимальная глубина.
    #     default=5
    # min_samples_split: Количество объектов в листе, чтобы его можно было
    #     разбить и превратить в узел. default=2
    # max_leafs: Максимальное количество листьев разрешенное для дерева.
    #     default=20
    # bins: Количество бинов для вычисления гистограммы значений,
    #     по каждому признаку. default=16
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2,
                 max_leafs=20, bins=16, learning_rate=0.1):
        self.verboseInc = None        # Нужно для вывода логов
        self.learningRate = learning_rate
        
        # Параметры для леса
        self.n_estimators = n_estimators
        
        # Подготовить список из n_estimators решающих деревьев (пока пустых)
        self.trees = [None] * self.n_estimators
        for k in np.arange(0, self.n_estimators):
            treeReg = MyTreeReg(
                max_depth=max_depth, min_samples_split=min_samples_split,
                max_leafs=max_leafs, bins=bins
            )
            self.trees[k] = treeReg
        
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
        
        # Первое предсказание - логарифм шансов ln(p / (1-p))
        # где p: среднее по всем целевым значениям из
        # тренировочного набора - вероятность класса 1
        self.pred_0 = None
        
        # Строка матрицы - это предсказаные логарифмы шансов для
        # каждого объекта из тестового набора, от одного дерева.
        # (кол-во строк = кол-во деревьев)
        self.oddsFromTrees = None
        
        # Словарь содержит важность каждого признака
        # в лесу (для всех деревьев)
        self.fi = {}
        
        #
        self.testSum = 0
        self.eps = 1e-15

    #****************************************************************
    
    # Метод заменяет target'ы в листьях дерева
    # y_target: pd.Series с целевыми значениями (метки 0 и 1)
    # y_proba: вектор numpy с предсказанными вероятностями
    def replaceTargetInLeafs(self, tree, y_target, y_proba):
        tree.testSum = 0
        
        # Цикл по листам дерева, чтобы подменить в них target'ы
        for leaf in tree.leafList:
            # Получить номера объектов (наблюдений), попавших в лист дерева
            rows = leaf.data.get('rows')
            # Получить целевые значения для объектов, попавших в лист
            y_leafTarget = y_target[rows]
            # Получить вероятности (полученные в начале итерации)
            # для объектов, попавших в лист
            y_leafProba = y_proba[rows]
            
            # Вычислить нужное предсказание в листе, которое минимизирует
            # логистическую функцию потерь (см. метод getVectorLoss)
            numerator = np.sum(y_leafTarget - y_leafProba)
            denominator = np.sum(y_leafProba * (1 - y_leafProba))
            
            newTarget = numerator / (denominator + self.eps)
            
            # ...и заменить его в листе
            leaf.data['target'] = newTarget
            tree.testSum += newTarget
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    # y: pd.Series с целевыми значениями
    def fit(self, X, y, verbose=False):
        if not verbose: # verbose=False
            verbose = -1
        else:  # verbose: целое число
            self.verboseInc = verbose
        
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
        
        # Вычислить логарифм шанса для класса с меткой 1
        # ln(p / (1-p))
        p = y.mean()
        self.pred_0 = np.log(p / (1-p))
        
        # Предсказания для обучения 1-го дерева
        # y_odds: логарифм шансов
        y_odds = np.full(shape=self.countSamplesTotal, fill_value=self.pred_0,
                            dtype=np.float64)
        
        # Построить (обучить) n_estimators решающих деревьев
        for k in np.arange(0, self.n_estimators):
            tree = self.trees[k]
            # Передать в дерево ОБЩЕЕ кол-во объектов в тренировочном наборе
            tree.countSamplesTotal = self.countSamplesTotal
            
            # Вернуться от логарифма шансов к вероятностям
            y_proba = self.oddsToProba(y_odds)
            
            # Вычислить антиградиент
            antiGradient = y.values - y_proba
            # Построить (обучить) k-ое дереро на антиградиенте
            tree.fit(X, pd.Series(antiGradient))
            
            # Заменить target'ы в листьях дерева
            self.replaceTargetInLeafs(tree, y, y_proba)
            
            # После маниипуляции с листьями дерева.
            # Вычислить для это дерева, предсказаные значения (логарифм шансов)
            # для каждого объекта из тренировочного набора
            y_oddsForTree = tree.predict(X)
            
            # Обновить предсказания (логарифм шансов) с учетом уже
            # обученных деревьев и скорости обучения
            y_odds = y_odds + (self.learningRate * y_oddsForTree)
            
            # Считаем общее кол-во листьев в лесу (для всех деревьев)
            self.leafs_cnt += tree.lastLeaf
            self.testSum += tree.testSum
            
            # Вывод метрик обучения
            self.showLossValue(k, verbose, y.values, y_odds)
        # END: for k in np.arange(0, self.n_estimators):
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict_proba(self, X):
        y_odds = np.zeros(X.shape[0])
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        
        # Вычислить предсказаные логрифмы шансов для каждого
        # объекта из тестового набора, от каждого дерева
        for k in np.arange(0, self.n_estimators):
            tree = self.trees[k]
            # Первое дерево
            if k == 0:
                # Создать матрицу с одной строкой
                self.oddsFromTrees = tree.predict(X)
            else:
                oddsList = tree.predict(X)
                # Добавить строку с предсказаниями в матрицу
                self.oddsFromTrees = np.vstack((self.oddsFromTrees, oddsList))
        
        # Цикл по объектам (строкам) из тестового набора признаков
        for n in np.arange(0, X.shape[0]):
            # Предсказания (логрифмы шансов) для
            # одного объекта от каждого дерева
            odds = self.oddsFromTrees[:, n]
            y_odds[n] = self.learningRate * odds.sum() + self.pred_0
        
        # Перевести логарифмы шансов в вероятности
        y_proba = self.oddsToProba(y_odds)
        return  y_proba
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict(self, X):
        y_predict = np.zeros(X.shape[0], dtype=np.int32)
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        
        y_proba = self.predict_proba(X)
        
        for idx, proba in enumerate(y_proba):
            if proba > 0.5:
                y_predict[idx] = 1
            else:
                y_predict[idx] = 0
        
        return y_predict
    
    #****************************************************************
    
    # Метод возвращает вектор ошибок для логистической функции потерь
    # y_target: вектор numpy
    # y_proba: вектор numpy с вероятностями
    def getVectorLoss(self, y_target, y_proba):
        result = None
        
        # + self.eps: Чтобы в аргументе логарифма не было нуля
        tmp1 = y_target * np.log(y_proba + self.eps)
        tmp2 = (1 - y_target) * np.log(1 - y_proba + self.eps)
        average = np.mean(tmp1 + tmp2)
        
        result = -1 * average
        return result
    
    #****************************************************************
    
    # Метод переводит вектор (логарифм шансов) в вектор вероятностей
    def oddsToProba(self, y_odds):
        numerator = np.exp(y_odds)
        denominator = 1 + numerator
        
        y_proba = numerator / denominator
        return y_proba
    
    #****************************************************************
    
    # Вывод метрик обучения
    # y_odds: numpy вектор (логарифм шансов)
    def showLossValue(self, k, verbose, y_target, y_odds):
        if (verbose > 0) and (k % self.verboseInc == 0):
            # Вернуться от логарифма шансов к вероятностям
            y_proba = self.oddsToProba(y_odds)
            
            # Вычислить вектор ошибок для логистической функции потерь
            lossVector = self.getVectorLoss(y_target, y_proba)
            lossValue = lossVector.mean()
            
            print(f'{k}. [loss]: {lossValue}')
    
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
        return f'MyBoostClf class: n_estimators={self.n_estimators}, learning_rate={self.learningRate}, max_depth={tree.maxDepth}, min_samples_split={tree.minSamplesSplit}, max_leafs={tree.maxLeafs}, bins={tree.bins}'
    
    def __repr__(self):
        tree = self.trees[0]
        return f'MyBoostClf(n_estimators={self.n_estimators}, learning_rate={self.learningRate}, max_depth={tree.maxDepth}, min_samples_split={tree.minSamplesSplit}, max_leafs={tree.maxLeafs}, bins={tree.bins})'
    
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
    
    #print(X_train.shape); print(y_train.shape)
    #print(X_train); print(y_train)
    
    myBoostClf = MyBoostClf(n_estimators=10, max_depth=3, min_samples_split=2,
                            max_leafs=40, bins=16, learning_rate=0.5)
    #print(myBoostClf)
    
    # Обучение
    myBoostClf.fit(X_train, y_train, verbose=2)
    print()
    for k in np.arange(0, myBoostClf.n_estimators):
        tree = myBoostClf.trees[k]
        #tree.printTree(tree.root); print()
    print(myBoostClf.pred_0, myBoostClf.leafs_cnt, myBoostClf.testSum); print()
    
    # Предсказание
    X_test = X.iloc[757:767, :]
    y_test = y.iloc[757:767]
    print(y_test.values); print()
    
    probaList = myBoostClf.predict_proba(X_test)
    print(probaList); print()
    print(probaList.sum()); print()
    
    predictList = myBoostClf.predict(X_test)
    print( predictList ); print()
    print( predictList.sum() ); print()
    
    # Параметры обученной модели
    #print(myBoostClf.fi); print()
    #print(f'{myBoostClf.metric}: {myBoostClf.best_score}'); print()
    
    print(end='\n\n')
    print('END!!!');
    input();