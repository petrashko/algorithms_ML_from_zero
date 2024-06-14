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

# Логистическая регрессия (метрики)
# Для двух классов с метками: 0 и 1
class MyLogReg():
    # metric: Принимает значение: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
    #     default = None
    def __init__(self, n_iter=10, learning_rate=0.1, weights=None, metric=None):
        self.verboseInc = None        # Нужно для вывода логов
        self.countSampleAll = None    # Общее число наблюдений в обучающей выборке
        self.bestScore = None         # Хранит значение метрики (оценку), обученной модели
        #
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        #
        self.metric = metric
    
    #****************************************************************
    
    # Перемножение двумерных массивов numpy
    def dot(self, A, B):
        # В итоговой матрице:
        # число строк = число строк в A; число колонок = число колонок в
        res = np.zeros((A.shape[0], B.shape[1]))
        
        # Цикл по строкам матрицы A
        for r in np.arange(0, A.shape[0]):
            # Цикл по колонкам матрицы B
            for c in np.arange(0, B.shape[1]):
                # Цикл по строкам матрицы B
                for t in np.arange(0, B.shape[0]):
                    res[r, c] += A[r, t] * B[t, c]
        return res
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    # y: pd.Series с целевыми значениями (метки класса: 0 и 1)
    def fit(self, X, y, verbose=False):
        if not verbose: # verbose=False
            verbose = -1
        else:  # verbose: целое число
            self.verboseInc = verbose
        
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
                
        # Для 'b0' добавить слева в матрицу признаков колонку с единицами
        X_new = self.addColumnForB0(X)
        
        X_features = X_new.to_numpy()
        # Вектор столбец
        y_target = y.values.reshape(y.shape[0], 1)
        
        # Нужно для ...
        X_features_all = X_new.to_numpy()
        y_target_all = y.values.reshape(y.shape[0], 1)
        # Общее число наблюдений в обучающей выборке
        self.countSampleAll = X_features_all.shape[0]
         
        # Создать весовые коэфициенты - вектор столбец из единиц
        # Число строк = количество колонок в X_features (каждому признаку свой вес)
        self.weights = np.ones((X_features.shape[1], 1), dtype=np.float64)
        
        # Матрица: Каждая строка - конкретный признак
        #     Каждая колонка - отдельное наблюдение
        X_transpose = X_features.T
        
        # Количество наблюдений
        countSamples = X_features.shape[0]
        
        #
        learningSpeed = self.learning_rate
        
        #
        for n in np.arange(1, self.n_iter+1):
            '''
            Предсказанные значения для классов (как непрерывная величина)
            y_predict = dot(X_features, weights)
            
            Вероятности для классов в интервале (0, 1)
            y_proba = 1 / (1 + e^-z)
            где:
            z = X * W => y_predict
            
            Для каждого наблюдения:
            Чем больше y_predict, тем y_proba ближе к 1
            При y_predict = 0, y_proba = 0.5
            Чем менше y_predict, тем y_proba ближе к 0
            '''
            
            # Вычислить предсказанные значения
            y_predict = self.dot(X_features, self.weights)
            
            # Вычислить предсказанные вероятности для классов
            # В интервале (0, 1)
            y_proba = 1 / (1 + np.exp(-1*y_predict))
            
            # Ошибки по каждому вектору наблюдений
            errors = y_proba - y_target
            
            # Вычисляем градиент для каждого весового коэффициента
            # Сумма произведений каждой ошибки на все признаки для конкретного веса
            gradients = self.dot(X_transpose, errors)
            # X_transpose.shape[1]: Число наблюдений
            gradients = gradients / countSamples
            
            # Изменяем весовые коэффициенты
            self.weights -= learningSpeed * gradients
            
            # Вывод метрик обучения
            self.show_estimators(
                n, verbose, y_target_all.flatten(),
                self.predict(X), self.predict_proba(X)
            )
        # END: for
    
        # Запомнить метрику (оценку), обученной модели
        self.bestScore = self.get_metric(
            y_target_all.flatten(),
            self.predict(X), self.predict_proba(X)
        )
    
    #****************************************************************
    
    # Метод возвращает вероятности для класса с меткой 1. Чтобы получить
    # вероятность для класса с меткой 0 нужно: 1 - predict_proba
    #
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict_proba(self, X):
        # Для 'b0' добавить слева в матрицу признаков колонку с единицами
        X = self.addColumnForB0(X)
        
        X_features = X.to_numpy()
        # Вычислить предсказанные значения
        y_predict = self.dot(X_features, self.weights)
        # Вычислить предсказанные вероятности для классов
        y_proba = 1 / (1 + np.exp(-1*y_predict))
        return y_proba.flatten()
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def predict(self, X):
        # Вычислить предсказанные вероятности для классов
        y_poba = self.predict_proba(X)
        
        y_predict = y_poba.copy()
        # Перевести вероятности в бинарные классы по порогу
        # 1 если вероятность > 0.5
        y_predict[y_predict > 0.5] = 1
        # 0 если вероятность <= 0.5
        y_predict[y_predict <= 0.5] = 0
        
        y_predict = y_predict.astype(np.int32)
        return y_predict.flatten()
    
    #****************************************************************
    
    def get_coef(self):
        # Отбросить первый вес 'b0' и развернуть в одномерный массив
        weights = self.weights[1:].flatten()
        return weights
    
    #****************************************************************
    
    def get_best_score(self):
        return self.bestScore
    
    #****************************************************************
    
    # Функция потерь для логистической регрессии
    # y_proba: предсказанные вероятности для классов
    def getLogLoss(self, y_target_all, y_proba_all):
        # Добавочный коэффициент, чтобы избавиться от нуля в аргументе логарифма
        eps = 1e-15
        
        tmp1 = y_target_all * np.log(y_proba_all + eps)
        tmp2 = (1 - y_target_all) + np.log(1 - y_proba_all + eps)
        summa = np.sum(tmp1 + tmp2)
        logLoss = -1 * (summa / self.countSampleAll)
        
        return logLoss
    
    #****************************************************************
    
    # Функция возвращает одну из метрик оценки работы модели
    def get_metric(self, y_target, y_predict, y_proba):
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
    def show_estimators(self, n, verbose, y_target_all, y_predict_all, y_proba_all):
        if (verbose > 0) and (n == 1 or n == self.verboseInc):
            # Вычислить функцию потерь для логистической регрессии
            logLoss = self.getLogLoss(y_target_all, y_proba_all)

            # Вычислить нужную метрику            
            if (self.metric != None) and (n == 1 or n == self.verboseInc):
                metric_value = self.get_metric(
                    y_target_all.flatten(),
                    y_predict_all.flatten(),
                    y_proba_all.flatten()
                )
                
            if n == 1:
                if self.metric == None:
                    print(f'start | loss: {logLoss}')
                else:
                    print(f'start | loss: {logLoss} | {self.metric}: {metric_value}')
            elif n == self.verboseInc:
                if self.metric == None:
                    print(f'{self.verboseInc} | loss: {logLoss}')
                else:
                    print(f'{self.verboseInc} | loss: {logLoss} | {self.metric}: {metric_value}')
                self.verboseInc += verbose
    
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    def addColumnForB0(self, X):
        # Для 'b0' добавить слева в матрицу признаков колонку с единицами
        onesDf = pd.DataFrame(np.ones((X.shape[0], 1), dtype=np.int32), columns=['for B0'])
        X_new = pd.concat([onesDf, X], axis='columns')
        return X_new
    
    #****************************************************************
    
    def __str__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def __repr__(self):
        return f'MyLogReg(n_iter={self.n_iter}, learning_rate={self.learning_rate})'

#********************************************************************

if __name__ == '__main__':
    # Для классификации
    df = pd.read_csv(f'..{sep}data{sep}banknote+authentication.zip', header=None)
    df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    X, y = df.iloc[:,:4], df['target']
    #print(y)

    myLogReg = MyLogReg(n_iter=50, learning_rate=0.1, weights=None, metric='accuracy')
    # Обучить модель
    myLogReg.fit(X, y, verbose=10)
    print()
    #print( myLogReg.get_coef() )
    #print( myLogReg.get_coef().mean() )
    
    # Проверка
    print()
    print( myLogReg.predict_proba(X) )
    print( myLogReg.predict_proba(X).mean() )
    print()
    print( myLogReg.predict(X) )
    print( myLogReg.predict(X).sum() )
    
    print(end='\n\n')
    print('END!!!');
    input();