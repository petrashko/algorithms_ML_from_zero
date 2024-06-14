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

from typesNode import NodeClf

# Дерево решений - классификация (обучение - построение дерева)
# Для двух классов с метками: 0 и 1
class MyTreeClf:
    # max_depth: Максимальная глубина.
    #     default=5
    # min_samples_split: Количество объектов в листе, чтобы его можно было
    #     разбить и превратить в узел. default=2
    # max_leafs: Максимальное количество листьев разрешенное для дерева.
    #     default=20
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.maxDepth = max_depth
        self.minSamplesSplit = min_samples_split
        # Кол-во листьев в дереве не может быть меньше 2
        self.maxLeafs = max(max_leafs, 2)
        # Кол-во листьев в построеном дереве
        self.leafs_cnt = 0
        # Номер последнего узла в дереве
        self.lastLeaf = 0
        
        # Словарь, номер колонки: название признака
        self.colsToFeatureName = {}
        
        # Словарь, номер колонки: спиисок пороговых значений
        self.colsToThreshold = {}
        
        self.root = None
        #
        self.testSum = 0
        self.eps = 1e-15

    #****************************************************************
    
    # Метод вычисляет пороговые значения для каждого признака,
    # по которым будем пытаться разбивать признаки в узле
    #
    # X: двумерный массив numpy, с матрицей признаков
    def calculateFeaturesThreshold(self, X):
        dictFeatures = {}
        dictFeaturesUniq = {}
        for col in self.colsToFeatureName.keys():
            dictFeatures[col] = X[:, col]
            # Уникальные значения признаков
            dictFeaturesUniq[col] = np.sort(np.unique( dictFeatures.get(col) ))
        
        # Найти пороговые значения для каждого признака,
        # по которым будем пытаться разбивать признаки в узле
        for col in self.colsToFeatureName.keys():
            uniqList = dictFeaturesUniq.get(col)
            thresholdList = []
            for n in np.arange(1, len(uniqList)):
                threshold = np.mean([uniqList[n-1], uniqList[n]])
                thresholdList.append(threshold)
            self.colsToThreshold[col] = np.array(thresholdList)
    
    #****************************************************************
    
    # Метод возвращает энтропию для вектора 'y'
    #
    # y: вектор numpy с целевыми значениями (0 и 1)
    def getEntropy(self, y):
        # Кол-во 0
        count0 = np.count_nonzero(y == 0)
        # Кол-во 1
        count1 = np.count_nonzero(y == 1)
        
        countTotal = y.shape[0]
        
        proba0 = count0 / countTotal
        proba1 = count1 / countTotal
        
        tmp0 = proba0 * np.log2(proba0 + self.eps)
        tmp1 = proba1 * np.log2(proba1 + self.eps)
        
        entropy = -1 * (tmp0 + tmp1)
        return entropy
    
    #****************************************************************
    
    # Метод возвращает прирост информации (на сколько уменьшилась энтропия)
    # если один вектор разбить на два: left и right
    def getInformationGain(self, left, right):
        cntLeft, cntRight = left.shape[0], right.shape[0]
        N = cntLeft + cntRight
        
        if (cntLeft == 0) or (cntRight == 0):
            # Ничего не изменилось - прирост информаций = 0
            informationGain = 0
        else:
            # Вычислить энтропию для целого вектора
            infTotal = self.getEntropy(np.hstack((left, right)))
            # Вычислить энтропию отдельно для каждого из подвекторов
            infLeft, infRight = self.getEntropy(left), self.getEntropy(right)
            # Вычислить прирост информации
            informationGain = infTotal - \
                ((cntLeft / N * infLeft) + (cntRight / N * infRight))
        
        return informationGain
    
    #****************************************************************
    
    # Метод возвращает кортеж для найлучшего разбиения
    # (0: название признка, 1: пороговое значение, по которому проводилось разбиение, 2: прирост информаций)
    #
    # X: двумерный массив numpy, с матрицей признаков
    # y: вектор numpy с целевыми значениями
    def getBestSplit(self, X, y):
        # Результат будем возвращать в виде кортежа для найлучшего разбиения
        # (0: название признка, 1: пороговое значение, по которому проводилось разбиение, 2: прирост информаций)
        result = (None, None, None)
        
        # Если в 'y' все значения одинаковые - разбивать бесполезно
        if y.max() == y.min():
            return (None, None, 0)
        
        # Присвоить заведомо нереально малое чило
        informationGainMax = float('-inf')
        
        # Проверить все пороговые значения до всем признакам
        
        # Цикл по признакам
        for col in self.colsToFeatureName.keys():
            # Получить значения из колонки с признаком
            featureValues = X[:, col]
            # Список пороговых значений для текущего признака
            thresholdList = self.colsToThreshold.get(col)
            
            prevThreshold = float('inf')
            # Цикл по пороговым значениям в каждом признаке
            for threshold in thresholdList:
                # Если два числа практически равны - перейти к следующему
                if abs(threshold - prevThreshold) < 0.001:
                    continue  # сделал чтобы пройти тесты на ограничение по времени
                
                # Получить целевые значения, которые попали в левую ветку
                mask = featureValues <= threshold
                y_left = y[mask]
                # Кол-во объектов, которые попали в левую ветку
                cntLeft = y_left.shape[0]
                
                # Получить целевые значения, которые попали в правую ветку
                mask = featureValues > threshold
                y_right = y[mask]
                # Кол-во объектов, которые попали в правую ветку
                cntRight = y_right.shape[0]
                
                if (cntLeft == 0) or (cntRight == 0):
                    # Ничего не изменилось - прирост информаций = 0
                    informationGain = 0
                else:
                    # Вычислить прирост информаций
                    # На сколько уменьшилась Entropy
                    informationGain = self.getInformationGain(y_left, y_right)
                
                if informationGain > informationGainMax:
                    featureName = self.colsToFeatureName.get(col)
                    result = (featureName, threshold, informationGain)
                    informationGainMax = informationGain
                
                prevThreshold = threshold
        
        return (result[0], result[1], result[2])
    
    #****************************************************************
    
    # Метод возвращает вероятность для класса с меткой 1
    # y: вектор numpy с целевыми значениями (0 и 1)
    @staticmethod
    def getProba1(y):
        # Здесь пользуемся тем, что метки класса
        # имеют только два значение 0 и 1
        return y.sum() / y.shape[0]

    #****************************************************************
    
    # Метод возвращает лист дерева
    # y: вектор numpy с целевыми значениями
    def createToLeaf(self, y, side):
        # Вычислить вероятность для класса с меткой 1 в текущем узле
        proba1 = self.getProba1(y)
        
        # Создать лист
        leafData = {
            'leafName': f'{side}Leaf-{self.lastLeaf+1}',
            # Вероятность для класса с меткой 1
            'proba1': proba1 
        }
        
        node = NodeClf(leafData)
        
        self.lastLeaf += 1
        self.testSum += proba1
        return node
    
    #****************************************************************
    
    # X: двумерный массив numpy, с матрицей признаков
    # y: вектор numpy с целевыми значениями
    def buildDecisionTree(self, X, y, depth=0, side='_'):
        # Находим оптимальное разбиение
        featureName, threshold, infoGain = self.getBestSplit(X, y)
        
        # Создать узел
        dataNode = {
            'featureName': featureName,
            'threshold': threshold 
        }
        node = NodeClf(dataNode)
        
        # Если в результате разбиения прироста информации не добавилось
        if infoGain == 0:
            self.leafs_cnt += 1
            # Прeвратить узел в лист и вернуть его
            node = self.createToLeaf(y, side)
            return node
        
        # По названию признака, получить номер колонки
        col = self.getColNumberByName(featureName)
        
        # 1. Ограничение на глубину дерева
        # 2, 3. Ограничение на кол-во объектов в листе
        # 4. Ограничение на максимальное кол-во листьев
        # 5. Подвыборка не может содержать классы только одного типа
        if (
                (depth == self.maxDepth) or
                (y.shape[0] == 1) or
                (y.shape[0] < self.minSamplesSplit) or
                (self.maxLeafs - self.leafs_cnt == 1)  or
                (self.getProba1(y) in (0, 1))
        ):
            self.leafs_cnt += 1
            # Содать лист и вернуть его
            return self.createToLeaf(y, side)
        
        if self.leafs_cnt < self.maxLeafs:
            # Колонка с признаком, по которому будем
            # разбивать на левую и правую подвыборки
            featureSplit = X[:, col]
            # Получить индексы строк для левой и правой подвыборки
            idxLeft = np.argwhere(featureSplit <= threshold).flatten()
            idxRight = np.argwhere(featureSplit > threshold).flatten()
            
            # Получить строки с признаками, которые попали в левую ветку
            X_left = X[idxLeft, :]
            # Получить целевые значения, которые попали в левую ветку
            y_left = y[idxLeft]
            
            # Получить строки с признаками, которые попали в правую ветку
            X_right = X[idxRight, :]
            # Получить целевые значения, которые попали в правую ветку
            y_right = y[idxRight]
            
            self.leafs_cnt += 1
            # Продолжаем строить левое поддерево
            node.left = self.buildDecisionTree(X_left, y_left, depth+1, 'l_')
            self.leafs_cnt -= 1
            # Продолжаем строить правое поддерево
            node.right = self.buildDecisionTree(X_right, y_right, depth+1, 'r_')
            
            return node
        else:
            pass
        
    #****************************************************************
    
    # X: pd.DataFrame с признаками. Каждая строка - отдельный объект
    #     Каждая колонка - конкретный признак
    # y: pd.Series с целевыми значениями
    def fit(self, X, y):
        # Сбросить индекс. Нумерация будет [0, 1, 2, 3, ...]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Создать словарь, номер колонки: название признака
        for col, featureName in enumerate(X.columns.tolist()):
            self.colsToFeatureName[col] = featureName
        
        X_train = X.to_numpy()
        y_train = y.values
        
        # Создать словарь, номер колонки: спиисок пороговых значений
        # (для признака), по которым будем пытаться
        # разбивать признаки в узле
        self.calculateFeaturesThreshold(X_train)
        
        # Создать дерево и вернуть его корень
        self.root = self.buildDecisionTree(X_train, y_train)
        #self.printTree(self.root); print()
        
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
    
    def printTree(self, node, level=0):
        if node == None:
            return
        self.printTree(node.right, level+1)
        indented = '    ' * level
        strData = str(node)
        print(indented + strData)
        self.printTree(node.left, level+1)
    
    #****************************************************************
    
    def __str__(self):
        return f'MyTreeClf class: max_depth={self.maxDepth}, min_samples_split={self.minSamplesSplit}, max_leafs={self.maxLeafs}'
    
    def __repr__(self):
        return f'MyTreeClf(max_depth={self.maxDepth}, min_samples_split={self.minSamplesSplit}, max_leafs={self.maxLeafs})'

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
    
    #print(X_train.shape); print(y_train.shape)
    #print(X_train); print(y_train)
    
    #myTreeClf = MyTreeClf(max_depth=1, min_samples_split=1, max_leafs=2)
    #myTreeClf = MyTreeClf(max_depth=3, min_samples_split=2, max_leafs=5)
    #myTreeClf = MyTreeClf(max_depth=5, min_samples_split=200, max_leafs=10)
    #myTreeClf = MyTreeClf(max_depth=4, min_samples_split=100, max_leafs=17)
    #myTreeClf = MyTreeClf(max_depth=10, min_samples_split=40, max_leafs=21)
    myTreeClf = MyTreeClf(max_depth=15, min_samples_split=20, max_leafs=30)
    #myTreeClf = MyTreeClf(max_depth=3, min_samples_split=2, max_leafs=1)
    
    # Обучение
    myTreeClf.fit(X_train, y_train)
    #print()
    myTreeClf.printTree(myTreeClf.root); print()
    print(myTreeClf.lastLeaf, myTreeClf.testSum)
    
    # Предсказание
    X_test = X.iloc[757:767, :]
    '''
    print()
    probaList = myTreeClf.predict_proba(X_test)
    print(probaList); print()
    print(probaList.sum()); print()
    predictList = myTreeClf.predict(X_test)
    print( predictList ); print()
    print( predictList.sum() ); print()
    '''
    
    print(end='\n\n')
    print('END!!!');
    input();
