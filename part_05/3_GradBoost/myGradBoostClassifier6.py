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

import types

from myTreeForClf import MyTreeReg

# Градиентный бустинг - классификация (скорость обучения)
# Для двух классов с метками: 0 и 1
class MyBoostClf():
    # random_state: Для воспроизводимости результата.
    #     default=42
    # learning_rate: Скорость обучения. Или число float или lambda-функция
    #     например: lambda iter: 0.5 * (0.85 ** iter)
    #     default=0.1
    # metric: Принимает значение: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
    #     default=None
    
    # Параметры для леса:
    # n_estimators: Количество деревьев в лесу.
    #     default=10
    # max_features: Доля фичей, которая будет случайным образом выбираться
    #     для каждого дерева. От 0.0 до 1.0. default=0.5
    # max_samples: Доля объектов, которая будет случайным образом выбираться
    #     из датасета для каждого дерева. От 0.0 до 1.0. default=0.5
    
    # Параметры отдельного дерева:
    # max_depth: Максимальная глубина.
    #     default=5
    # min_samples_split: Количество объектов в листе, чтобы его можно было
    #     разбить и превратить в узел. default=2
    # max_leafs: Максимальное количество листьев разрешенное для дерева.
    #     default=20
    # bins: Количество бинов для вычисления гистограммы значений,
    #     по каждому признаку. default=16
    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5,
                 max_depth=5, min_samples_split=2, max_leafs=20, bins=16,
                 metric=None, learning_rate=0.1, random_state=42):
        self.verboseInc = None        # Нужно для вывода логов
        self.best_score = None    # Хранит значение метрики (оценку), обученной модели
        self.metric = metric
        self.learningRate = learning_rate
        
        # Параметры для леса
        self.n_estimators = n_estimators
        self.maxFeatures = max_features
        self.maxSamples = max_samples
        
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
        self.randomState = random_state
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
        
        # Вычислить логарифм шанса для класса с меткой 1
        # ln(p / (1-p))
        p = y.mean()
        self.pred_0 = np.log(p / (1-p))
        
        # Предсказания для обучения 1-го дерева
        # y_odds: логарифм шансов
        y_oddsTotal = np.full(shape=self.countSamplesTotal,
                              fill_value=self.pred_0, dtype=np.float64)
        
        #
        learningSpeed = self.learningRate
        
        # Построить (обучить) n_estimators решающих деревьев
        for k in np.arange(0, self.n_estimators):
            tree = self.trees[k]
            # Передать в дерево ОБЩЕЕ кол-во объектов в тренировочном наборе
            tree.countSamplesTotal = self.countSamplesTotal
            
            # Если self.learningRate: lambda-функция
            if type(self.learningRate) is types.LambdaType:
                # Чем больше номер итеррации, тем меньше скорость обучения
                learningSpeed = self.learningRate(k+1)
            
            # Случайно выбираем countCols названий признаков
            nameFeatures = random.sample(X.columns.tolist(), countCols)
            X_random = X[nameFeatures]
            
            # Случайно выбираем countRows индексов строк (наблюдений)
            randomRows = random.sample(list(X.index), countRows)
            
            # Оставили только случайно выбранные колонки и строки
            X_random = X_random.iloc[randomRows]
            y_random = y.iloc[randomRows]
            
            X_random = X_random.reset_index(drop=True)
            y_random = y_random.reset_index(drop=True)
            
            y_odds = y_oddsTotal[randomRows]
            
            # Вернуться от логарифма шансов к вероятностям
            y_proba = self.oddsToProba(y_odds)
            
            # Вычислить антиградиент
            antiGradient = y_random.values - y_proba
            # Построить (обучить) k-ое дереро на антиградиенте
            tree.fit(X_random, pd.Series(antiGradient))
            
            # Заменить target'ы в листьях дерева
            self.replaceTargetInLeafs(tree, y_random, y_proba)
            
            # После маниипуляции с листьями дерева.
            # Вычислить для это дерева, предсказаные значения (логарифм шансов)
            # для каждого объекта из тренировочного набора
            y_oddsForTree = tree.predict(X)
            
            # Обновить предсказания (логарифм шансов) с учетом уже
            # обученных деревьев и скорости обучения
            y_oddsTotal = y_oddsTotal + (learningSpeed * y_oddsForTree)
            
            # Считаем общее кол-во листьев в лесу (для всех деревьев)
            self.leafs_cnt += tree.lastLeaf
            self.testSum += tree.testSum
            
            # Вывод метрик обучения
            self.showLossValue(k, verbose, y.values, y_oddsTotal)
        # END: for k in np.arange(0, self.n_estimators):
        
        # Запомнить метрику (оценку), обученной модели
        if self.metric is None:
            # Если метрика не задана - в self.best_score записать
            # последнее значение логистической функции потерь,
            # для которой считалии векторы антиградиентов
            lossVector = self.getVectorLoss(y.values, self.predict_proba(X))
            self.best_score = lossVector.mean()
        else:
            self.best_score = self.getMetric(y.values, self.predict(X), self.predict_proba(X))
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict_proba(self, X):
        y_odds = np.zeros(X.shape[0])
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        
        #
        learningSpeed = self.learningRate
        
        # Вычислить предсказаные логрифмы шансов для каждого
        # объекта из тестового набора, от каждого дерева
        for k in np.arange(0, self.n_estimators):
            tree = self.trees[k]
            
            # Если self.learningRate: lambda-функция
            if type(self.learningRate) is types.LambdaType:
                # Чем больше номер итеррации, тем меньше скорость обучения
                learningSpeed = self.learningRate(k+1)
                
            # Первое дерево
            if k == 0:
                if type(self.learningRate) is types.LambdaType:
                    # Создать матрицу с одной строкой
                    self.oddsFromTrees = learningSpeed * tree.predict(X)
                # self.learningRate - это число
                else:
                    # Создать матрицу с одной строкой
                    self.oddsFromTrees = tree.predict(X)
            else:
                if type(self.learningRate) is types.LambdaType:
                    oddsList = learningSpeed * tree.predict(X)
                # self.learningRate - это число
                else:
                    oddsList = tree.predict(X)
                
                # Добавить строку с предсказаниями в матрицу
                self.oddsFromTrees = np.vstack((self.oddsFromTrees, oddsList))
        
        # Цикл по объектам (строкам) из тестового набора признаков
        for n in np.arange(0, X.shape[0]):
            # Предсказания (логрифмы шансов) для
            # одного объекта от каждого дерева
            odds = self.oddsFromTrees[:, n]
            
            if type(self.learningRate) is types.LambdaType:
                y_odds[n] = odds.sum() + self.pred_0
            # self.learningRate - это число
            else:
                y_odds[n] = learningSpeed * odds.sum() + self.pred_0
        
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
    
    # Метод возвращает одну из метрик оценки работы модели
    def getMetric(self, y_target, y_predict, y_proba):
        result = None
        
        # Количество положительных классов (с меткой 1),
        # которые правильно определены как положительные
        truePositive = 0
        # Количество отрицательных классов (с меткой 0),
        # которые правильно определены как отрицательные
        trueNegative = 0
        
        # Количество отрицательных классов (с меткой 0),
        # которые неправильно определены как положительные
        # Ошибка 1-го рода
        falsePositive = 0
        # Количество положительных классов (с меткой 1),
        # которые неправильно определены как отрицательны
        # Ошибка 2-го рода
        falseNegative = 0
        
        # Количество наблюдений положительного класса (с меткой 1)
        countPositive = 0
        # Количество наблюдений отрицательного класса (с меткой 0)
        countNegative = 0
        
        for k in np.arange(0, len(y_target)):
            if y_target[k] == 1 and y_predict[k] == 1:
                truePositive += 1
                countPositive +=1
            if y_target[k] == 0 and y_predict[k] == 0:
                trueNegative += 1
                countNegative += 1
            # Ошибка 1-го рода
            if y_target[k] == 0 and y_predict[k] == 1:
                falsePositive += 1
                countNegative += 1
            # Ошибка 2-го рода
            if y_target[k] == 1 and y_predict[k] == 0:
                falseNegative += 1
                countPositive +=1
        
        # roc_auc:
        if self.metric == 'roc_auc':
            proba = np.round(y_proba, 10)
            proba = np.append(proba.reshape(-1, 1), y_target.reshape(-1, 1), axis=1)
            # Отсортировать по вероятности (по убыванию)
            proba = pd.DataFrame(proba, columns=['p', 'y'])
            proba = proba.sort_values(by='p', ascending=False)
            proba['y'] = proba['y'].apply(int)
            proba = proba.reset_index(drop=True)
            
            startIndex = 0
            sumPositiveTotal = 0
            tmpList = []
            for k in np.arange(0, proba.shape[0]):
                # Если класс имеет метку 0
                if proba.iloc[k]['y'] == 0:
                    sumPositiveClass = 0
                    sumPositiveXZ = 0
                    currProba = proba.iloc[k]['p']
                    tmpDF = proba.iloc[startIndex:k]

                    for m in np.arange(0, tmpDF.shape[0]):
                        if tmpDF.iloc[m]['y'] == 1 and tmpDF.iloc[m]['p'] != currProba:
                            sumPositiveClass += 1
                        if tmpDF.iloc[m]['y'] == 1 and tmpDF.iloc[m]['p'] == currProba:
                            sumPositiveXZ += 1
                    
                    startIndex = k
                    sumPositiveXZ += sumPositiveXZ / 2
                    sumPositiveTotal += sumPositiveClass
                    tmpList.append(sumPositiveTotal)
                    tmpList.append(sumPositiveXZ)
            
            result = sum(tmpList) / (countPositive * countNegative)
        # f1: Метрика, которая одновременно отвечает за улучшение и Precision и Recall
        elif self.metric == 'f1':
            precision = truePositive / (truePositive + falsePositive)
            recall = truePositive / (truePositive + falseNegative)
            result = 2 * precision * recall / (precision + recall)
        # recall - Доля объектов положительного класса (с меткой 1)
        # из всех объектов положительного класса, которую  нашел алгоритм
        elif self.metric == 'recall':
            result = truePositive / (truePositive + falseNegative)
        # precision - Доля объектов, названных классификатором положительными
        # (с меткой 1) и при этом действительно являющимися положительными
        elif self.metric == 'precision':
            result = truePositive / (truePositive + falsePositive)
        # Иначе,
        # accuracy: Доля правильных ответов алгоритма
        else:
            result = (truePositive + trueNegative) / len(y_predict)
        
        return result
    
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
            
            # Вычислить нужную метрику
            if self.metric is not None:
                y_predict = np.zeros(X.shape[0], dtype=np.int32)
                for idx, proba in enumerate(y_proba):
                    if proba > 0.5:
                        y_predict[idx] = 1
                    else:
                        y_predict[idx] = 0
                
                metricValue = self.getMetric(y_target, y_predict, y_proba)
            
            if self.metric is None:
                print(f'{k}. Loss: {lossValue}')
            else:
                print(f'{k}. Loss: {lossValue}|{self.metric}: {metricValue}')
    
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
    
    myBoostClf = MyBoostClf(n_estimators=10, max_features=0.9, max_samples=0.9,
                            max_depth=3, min_samples_split=2, max_leafs=40, bins=16,
                            metric='accuracy', learning_rate=lambda x: 0.5 * (0.85 ** x),
                            random_state=42)
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
    #print(probaList.sum()); print()
    
    predictList = myBoostClf.predict(X_test)
    print( predictList ); print()
    #print( predictList.sum() ); print()
    
    # Параметры обученной модели
    #print(myBoostClf.fi); print()
    print(f'{myBoostClf.metric}: {myBoostClf.best_score}'); print()
    
    print(end='\n\n')
    print('END!!!');
    input();