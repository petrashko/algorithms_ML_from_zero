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

from myTreeRegression import MyTreeReg

# Случайный лес - регрессия (out-of-bag (OOB) Error)
class MyForestReg():
    # oob_score: Принимает значение: 'r2', 'mape', 'mae', 'rmse', 'mse'
    #     default=None
    # Параметры исключительно для леса:
    # n_estimators: Количество деревьев в лесу.
    #     default=10
    # max_features: Доля фичей, которая будет случайным образом выбираться
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
    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5,
                 max_depth=5, min_samples_split=2, max_leafs=20,
                 bins=16, oob_score=None, random_state=42):
        self.oobScore = oob_score
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
        
        # Словарь содержит важность каждого признака
        # в лесу (для всех деревьев)
        self.fi = {}
        
        # Вычисленная оценка out-of-bag Error
        self.oob_score_ = 0
        # Словарь содержит список предсказаний от каждого дерева, для
        # объектов из тренировочного набора, для вычисления oob-оценок
        self.dictSamplesOobPredict = {}
        
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
            
            if self.oobScore is not None:
                # Выбрать объекты с признаками из тренировочного набора,
                # которые НЕ участвовали в создание k-го дерева
                X_oob = X.drop(labels=idxRows, axis='rows')
                # Вычислить предсказания по каждому объекту из X_oob для k-го дерева
                self.calculateOobPredictForTree(X_oob, k)
            
            # Суммируем важности каждого признака, по всем деревьям
            for key in tree.fi.keys():
                self.fi[key] += tree.fi.get(key)
            
            # Считаем общее кол-во листьев в лесу (для всех деревьев)
            self.leafs_cnt += tree.lastLeaf
            self.testSum += tree.testSum
        # END: for k in np.arange(0, self.n_estimators):
        
        # out-of-bag Error для случайного леса
        self.calculateOobScore(y)
    
    #****************************************************************
    
    # x_test: вектор numpy с признаками
    # numTree: Номер дерева
    def help_predict(self, x_test, numTree):
        tree = self.trees[numTree]
        # Корень дерева
        node = tree.root
        # В листьях дерева находится среднее значение
        # целевой переменной всех объектов в листе
        target = node.data.get('target', None)
        
        # Спускаемся по узлам дерева, пока не достигнем листа
        while target is None:
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
            
            # Если достигли листа target будет числом
            target = node.data.get('target', None)
        
        return target
            
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict(self, X):
        y_predict = np.zeros(X.shape[0])
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        X_test = X.to_numpy()
        
        # Цикл по объектам (строкам) из тестового набора признаков
        for n in np.arange(0, X_test.shape[0]):
            row = X_test[n,:]
            # Сумма значений целевой переменной
            # в найденных листах каждого дерева
            target = 0
            
            # Поиск нужного узла в каждом дереве
            for k in np.arange(0, self.n_estimators):
                # После прохода по дереву решений получить среднее значение
                # целевой переменной всех объектов в листе
                target += self.help_predict(row, k)
            
            # Взять среднее значение по всем деревьям
            y_predict[n] = target / self.n_estimators
        
        return  y_predict
    
    #****************************************************************
    
    # Метод вычисляет предсказания по каждому
    # объекту из X_oob для одного дерева
    #
    # X_oob: Объекты из тренировочного набора, которые
    # НЕ участвовали в создание k-го дерева
    # k: Номер дерева
    def calculateOobPredictForTree(self, X_oob, k):
        if self.oobScore is None:
            return
        
        for n in X_oob.index:
            row = X_oob.loc[n].values
            # Получить предсказание целевой переменной
            # для объекта row в k-ом дереве
            predict = self.help_predict(row, k)

            # Для n-го объекта добавили предсказанное значение k-ым деревом
            self.dictSamplesOobPredict.setdefault(n, []).append(predict)
    
    #****************************************************************
    
    # Метод вычисляет out-of-bag Error для случайного леса
    # y_train: pd.Series с целевыми значениями из всего тренировочного набора
    def calculateOobScore(self, y_train):
        if self.oobScore is None:
            return
        targetList = []
        predictList = []
        for n, predicts in self.dictSamplesOobPredict.items():
            if len(predicts) != 0:
                # Значение целевой переменной для n-го объекта
                # из тренировочного набора
                targetList.append(y_train.loc[n])
                
                # Среднее значение предсказаний для n-го объекта
                meanPredict = sum(predicts) / len(predicts)
                predictList.append(meanPredict)
        
        self.oob_score_ = self.getMetric( np.array(targetList), np.array(predictList) )
    
    #****************************************************************
    
    # Метод возвращает одну из метрик оценки
    # работы случайного леса
    def getMetric(self, y_target, y_predict):
        result = None
        
        errors = y_target - y_predict
        yMean = y_target.mean()
        
        # Коэффициент детерминации
        # R^2
        if self.oobScore == 'r2':
            numerator   = np.mean(errors ** 2)
            denominator = np.mean((y_target - yMean) ** 2)
            result = 1 - (numerator / denominator)
        # Средняя абсолютная процентная ошибка
        # Mean Absolute Percentage Error (MAPE)
        elif self.oobScore == 'mape':
            result = np.mean( np.abs(errors / y_target) ) * 100
        # Средняя абсолютная ошибка
        # Mean Absolute Error (MAE)
        elif self.oobScore == 'mae':
            result = np.mean( np.abs(errors) )
        # Квадратный корень из среднеквадратичной ошибки
        # Root Mean Squared Error (RMSE)
        elif self.oobScore == 'rmse':
            result = np.sqrt( np.mean(errors ** 2) )
        # Иначе,
        # Среднеквадратичная ошибка
        # Mean Squared Error (MSE)
        else:
            result = np.mean(errors ** 2)
        
        return result
    
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
        return f'MyForestReg class: n_estimators={self.n_estimators}, max_features={self.maxFeatures}, max_samples={self.maxSamples}, max_depth={tree.maxDepth}, min_samples_split={tree.minSamplesSplit}, max_leafs={tree.maxLeafs}, bins={tree.bins}, random_state={self.randomState}'
    
    def __repr__(self):
        tree = self.trees[0]
        return f'MyForestReg(n_estimators={self.n_estimators}, max_features={self.maxFeatures}, max_samples={self.maxSamples}, max_depth={tree.maxDepth}, min_samples_split={tree.minSamplesSplit}, max_leafs={tree.maxLeafs}, bins={tree.bins}, random_state={self.randomState})'
    
#********************************************************************

if __name__ == '__main__':
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)
    X_train, y_train = data['data'], data['target']
    #print(X_train); print()
    #print(y_train); print()
    
    myForestReg = MyForestReg(n_estimators=10, max_features=0.5, max_samples=0.5,
                 max_depth=5, min_samples_split=2, max_leafs=20,
                 bins=16, oob_score='mae', random_state=42)
    #print(myForestReg)
    
    # Обучение
    myForestReg.fit(X_train, y_train)
    #print()
    for k in np.arange(0, myForestReg.n_estimators):
        tree = myForestReg.trees[k]
        tree.printTree(tree.root); print()
    print(myForestReg.leafs_cnt, myForestReg.testSum); print()
    
    # Предсказание
    '''
    X_test = X_train.iloc[211:231, :]
    predictList = myForestReg.predict(X_test)
    print( predictList ); print()
    print( predictList.sum() ); print()
    '''
    
    # Параметры обученной модели
    #print(myForestReg.fi); print()
    print(f'{myForestReg.oobScore}: {myForestReg.oob_score_}'); print()
    
    print(end='\n\n')
    print('END!!!');
    input();