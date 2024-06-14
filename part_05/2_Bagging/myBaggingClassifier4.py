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

# Бэггинг - классификация (out-of-bag (OOB) Error)
class MyBaggingClf():
    # estimator: Содержит экземпляр одного из базового класса
    #     регрессии: MyLineReg, MyKNNReg или MyTreeReg
    #     default=None
    # n_estimators: Количество базовых экземпляров, которые будут обучены.
    #     default=10
    # max_samples: Доля объектов, которая будет случайным образом выбираться
    #     из датасета для обучения каждого экземпляра модели. От 0.0 до 1.0.
    #     От 0.0 до 1.0. default=1.0
    # oob_score: Принимает значение: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
    #     default=None
    # random_state: Для воспроизводимости результата.
    #     default=42
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0,
                 oob_score=None, random_state=42):
        self.oobScore = oob_score
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
        
        # Вычисленная оценка out-of-bag Error
        self.oob_score_ = 0
        # Словарь содержит список вероятностей от каждой модели, для
        # объектов из тренировочного набора, для вычисления oob-оценок
        self.dictSamplesOobProbability = {}
        
        
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
            
            if self.oobScore is not None:
                idxRowsWithoutDuplicates = list(set(idxRows))
                # Выбрать объекты с признаками из тренировочного набора,
                # которые НЕ участвовали в создание k-ой модели
                X_oob = X.drop(labels=idxRowsWithoutDuplicates, axis='rows')
                # Вычислить вероятности по каждому объекту из X_oob для k-го дерева
                self.calculateOobProbabilityForModel(X_oob, k)
        # END: for k in np.arange(0, self.n_estimators):
        
        # out-of-bag Error для случайного леса
        self.calculateOobScore(y)

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
    
    # Метод вычисляет вероятности по каждому
    # объекту из X_oob для одной модели
    #
    # X_oob: Объекты из тренировочного набора, которые
    # НЕ участвовали в обучении k-ой модели
    # k: Номер модели
    def calculateOobProbabilityForModel(self, X_oob, k):
        if self.oobScore is None:
            return
        
        model = self.estimators[k]
        # Получить вероятности принадлежности объектов
        # к классу с меткой 1, от k-ой модели
        oob_proba = model.predict_proba(X_oob)
        
        p = 0
        for n in X_oob.index:
            # Для n-го объекта добавили вероятность полученную от k-ой модели
            self.dictSamplesOobProbability.setdefault(n, []).append(oob_proba[p])
            p += 1
    
    #****************************************************************
    
    # Метод вычисляет out-of-bag Error для Бэггинга
    # y_train: pd.Series с целевыми значениями из всего тренировочного набора
    def calculateOobScore(self, y_train):
        if self.oobScore is None:
            return
        targetList = []
        probaList =  []
        predictList = []
        for n, proba1_list in self.dictSamplesOobProbability.items():
            if len(proba1_list) != 0:
                # Значение целевой переменной для n-го объекта
                # из тренировочного набора
                targetList.append(y_train.loc[n])
                
                # Среднее значение вероятностей для n-го объекта
                meanProba1 = sum(proba1_list) / len(proba1_list)
                probaList.append(meanProba1)
                # Вероятности перевести в метки класса 0 или 1
                predictList.append(1 if meanProba1 > 0.5 else 0)
        
        self.oob_score_ = self.getMetric(
            np.array(targetList), np.array(predictList), np.array(probaList)
        )

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
        if self.oobScore == 'roc_auc':
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
        elif self.oobScore == 'f1':
            precision = truePositive / (truePositive + falsePositive)
            recall = truePositive / (truePositive + falseNegative)
            result = 2 * precision * recall / (precision + recall)
        # recall - Доля объектов положительного класса (с меткой 1)
        # из всех объектов положительного класса, которую  нашел алгоритм
        elif self.oobScore == 'recall':
            result = truePositive / (truePositive + falseNegative)
        # precision - Доля объектов, названных классификатором положительными
        # (с меткой 1) и при этом действительно являющимися положительными
        elif self.oobScore == 'precision':
            result = truePositive / (truePositive + falsePositive)
        # Иначе,
        # accuracy: Доля правильных ответов алгоритма
        else:
            result = (truePositive + trueNegative) / len(y_predict)
        
        return result
    
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
        estimator=myLogReg, n_estimators=5, max_samples=0.5, oob_score='roc_auc', random_state=42
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
    #rint( probaList ); print()
    print( predictList ); print()
    #print( probaList.sum(), predictList.sum() ); print()
    
    # Параметры обученной модели
    print(f'{myBaggingClf.oobScore}: {myBaggingClf.oob_score_}'); print()
    
    # Обучение
    
    print('# Метод K-ближайших соседей')
    myKnnClf = MyKNNClf(k=3, metric='chebyshev', weight='distance')
    
    myBaggingClf = MyBaggingClf(
        estimator=myKnnClf, n_estimators=5, max_samples=0.5, oob_score='precision', random_state=42
    )
    myBaggingClf.fit(X_train, y_train)
    #print()
    for model in myBaggingClf.estimators:
        print(model.train_size, end='; ')
    print(end='\n\n')
    
    print('# Предсказание')
    probaList = myBaggingClf.predict_proba(X_test)
    predictList = myBaggingClf.predict(X_test, type='mean')
    #print( probaList ); print()
    print( predictList ); print()
    #print( probaList.sum(), predictList.sum() ); print()
        
    # Параметры обученной модели
    print(f'{myBaggingClf.oobScore}: {myBaggingClf.oob_score_}'); print()
    
    # Обучение
    
    print('# Дерево решений')
    myTreeClf = MyTreeClf(max_depth=4, min_samples_split=2, max_leafs=17, criterion='gini', bins=16)
    
    myBaggingClf = MyBaggingClf(
        estimator=myTreeClf, n_estimators=5, max_samples=0.5, oob_score='accuracy', random_state=42
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
    #print( probaList ); print()
    print( predictList ); print()
    #print( probaList.sum(), predictList.sum() ); print()
        
    # Параметры обученной модели
    print(f'{myBaggingClf.oobScore}: {myBaggingClf.oob_score_}'); print()
    
    print(end='\n\n')
    print('END!!!');
    input();