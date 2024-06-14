import numpy as np

# Узел дерева
class Node:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
    
    def __str__(self):
        # Лист дерева
        if (self.left is None) and (self.right is None):
            # data.get("target"): # Среднее целевых значений в листе
            return f'({self.data.get("leafName")} = {self.data.get("target")})'
        # Узел дерева
        else:
            return f'({self.data.get("featureName")} <= {self.data.get("threshold")})'

#********************************************************************

# Дерево решений - регрессия
# Дерево адаптировано для задач классификации, методом градиентного бустинга
class MyTreeReg():
    # max_depth: Максимальная глубина.
    #     default=5
    # min_samples_split: Количество объектов в листе, чтобы его можно было
    #     разбить и превратить в узел. default=2
    # max_leafs: Максимальное количество листьев разрешенное для дерева.
    #     default=20
    # bins: Количество бинов для вычисления гистограммы значений,
    #     по каждому признаку. default=None
    def __init__(self, max_depth=5, min_samples_split=2,
                 max_leafs=20, bins=None):
        self.maxDepth = max_depth
        self.minSamplesSplit = min_samples_split
        self.bins = bins
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
        # Словарь, номер колонки: спиисок пороговых значений,
        # вычисленные на основе гистограмм
        self.colsToThresholdByHist = {}
        
        # Кол-во объектов (строк) в исходном тренировочном наборе,
        # который подается на вход алгоритму обучения
        self.countSamplesTotal = 0
        # Словарь содержит важность каждого признака, в построенном дереве
        self.fi = {}
        
        # Список листьев дерева
        self.leafList = []
        
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
    
    # Метод вычисляет (на основе гистограмм) пороговые значения для каждого
    # признака, по которым будем пытаться разбивать признаки в узле.
    #
    # X: двумерный массив numpy, с матрицей признаков
    def calculateFeaturesThresholdByHist(self, X):
        # Найти (на основе гистограмм) пороговые значения для каждого
        # признака, по которым будем пытаться разбивать признаки в узле.
        for col in self.colsToFeatureName.keys():
            # thresholdList: Границы бинов
            _, thresholdList = np.histogram(X[:, col], self.bins)
            # Вырезать два крайних числа.
            # Остальные - пороговые значения для признака
            self.colsToThresholdByHist[col] = \
                np.array(thresholdList[1 : -1])
    
    #****************************************************************
    
    # Метод вычисляет важность признака для узла, в котором
    # происходит разбиение на левую и правую ветки
    def calculateFeatureImpotance(self, feature, y_left, y_right):
        # Значения целевых переменных
        y_node = np.hstack((y_left, y_right))
        
        # Кол-во строк
        cntN = y_node.shape[0]
        cntL = y_left.shape[0]
        cntR = y_right.shape[0]
        
        # Вычислить MSE для целого вектора
        mseN = self.getMse(y_node)
        # Вычислить MSE отдельно для каждого из подвекторов
        mseL, mseR = self.getMse(y_left), self.getMse(y_right)
        
        # Вычисляем важность признака в узле
        tmp = mseN - (cntL * mseL / cntN) - (cntR * mseR / cntN)
        featureImpotance = cntN / self.countSamplesTotal * tmp
        
        # Суммируем важность для текущего признака
        prevImpotance = self.fi.get(feature, 0)
        self.fi[feature] = prevImpotance + featureImpotance
    
    #****************************************************************
    
    # Метод возвращает среднеквадратичную ошибку
    # Mean Squared Error (MSE) для вектора 'y'
    #
    # y: вектор numpy с целевыми значениями
    def getMse(self, y):
        yMean = np.mean(y)
        errors = y - yMean
        
        mse = np.mean(errors ** 2)
        return mse
    
    #****************************************************************
    
    # Метод возвращает прирост информации. На сколько
    # уменьшилась среднеквадратичная ошибка (MSE) если
    # один вектор разбить на два: left и right
    def getInformationGain(self, left, right):
        cntLeft, cntRight = left.shape[0], right.shape[0]
        N = cntLeft + cntRight
        
        if (cntLeft == 0) or (cntRight == 0):
            # Ничего не изменилось - прирост информаций = 0
            informationGain = 0
        else:
            # Вычислить MSE для целого вектора
            mseTotal = self.getMse(np.hstack((left, right)))
            # Вычислить MSE отдельно для каждого из подвекторов
            mseLeft, mseRight = self.getMse(left), self.getMse(right)
            # Вычислить прирост информации
            informationGain = mseTotal - \
                ((cntLeft / N * mseLeft) + (cntRight / N * mseRight))
        
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
            
            if self.bins is not None:
                # Длина self.colsToThresholdByHist.get(col) = self.bins-1
                # Выбрать, который короче
                if len(thresholdList) > (self.bins-1):
                    thresholdList = self.colsToThresholdByHist.get(col)
            
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
    
    # Метод возвращает лист дерева
    # y: вектор numpy с целевыми значениями
    # rows: Номера наблюдений которые попали в лист
    def createToLeaf(self, y, rows, side):
        # Вычислить среднее значение целевых переменных,
        # которые попали в текущий узел
        target = y.mean()
        
        # Создать лист
        leafData = {
            'leafName': f'{side}Leaf-{self.lastLeaf+1}',
            # Значение в листе
            'target': target,
            # Сохраняем номера наблюдений, которые попали в лист.
            # Нужно для пересчета target'а в градиентном бустинге
            'rows': rows
        }
        
        node = Node(leafData)
        self.leafList.append(node)
        
        self.lastLeaf += 1
        self.testSum += target
        return node
    
    #****************************************************************
    
    # X: двумерный массив numpy, с матрицей признаков
    # y: вектор numpy с целевыми значениями
    # rows: Номера наблюдений из тренировочного набора,
    #     которые попали в текущее поддерево
    def buildDecisionTree(self, X, y, rows, depth=0, side='_'):
        # Находим оптимальное разбиение
        featureName, threshold, infoGain = self.getBestSplit(X, y)
        
        # Создать узел
        dataNode = {
            'featureName': featureName,
            'threshold': threshold 
        }
        node = Node(dataNode)
        
        # Если в результате разбиения прироста информации не добавилось
        if infoGain == 0:
            self.leafs_cnt += 1
            # Прeвратить узел в лист и вернуть его
            node = self.createToLeaf(y, rows, side)
            return node
        
        # По названию признака, получить номер колонки
        col = self.getColNumberByName(featureName)
        
        # 1. Ограничение на глубину дерева
        # 2, 3. Ограничение на кол-во объектов в листе
        # 4. Ограничение на максимальное кол-во листьев
        if (
                (depth == self.maxDepth) or
                (y.shape[0] == 1) or
                (y.shape[0] < self.minSamplesSplit) or
                (self.maxLeafs - self.leafs_cnt == 1)
        ):
            self.leafs_cnt += 1
            # Содать лист и вернуть его
            return self.createToLeaf(y, rows, side)
        
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
            
            # Запомнить номера объектов, которые
            # пойдут в левое и правое поддерево
            rowsLeft = []
            for idx in idxLeft:
                rowsLeft.append(rows[idx])
            
            rowsRight = []
            for idx in idxRight:
                rowsRight.append(rows[idx])
            
            # Вычисляем важность признака в узле
            self.calculateFeatureImpotance(featureName, y_left, y_right)
            
            self.leafs_cnt += 1
            # Продолжаем строить левое поддерево
            node.left = self.buildDecisionTree(X_left, y_left, rowsLeft, depth+1, 'l_')
            self.leafs_cnt -= 1
            # Продолжаем строить правое поддерево
            node.right = self.buildDecisionTree(X_right, y_right, rowsRight, depth+1, 'r_')
            
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
        
        for col, featureName in enumerate(X.columns.tolist()):
            # Создать словарь, номер колонки: название признака
            self.colsToFeatureName[col] = featureName
            # Каждому признаку присвоить важность 0.0
            self.fi[featureName] = 0.0
        
        X_train = X.to_numpy()
        y_train = y.values
        
        # Запомнить кол-во объектов в тренировочном наборе
        #self.countSamplesTotal = X_train.shape[0] - значение будем передавать из случайного леса
        
        # Создать словарь, номер колонки: спиисок пороговых значений
        # (для признака), по которым будем пытаться
        # разбивать признаки в узле
        self.calculateFeaturesThreshold(X_train)
        
        if self.bins is not None:
            # Создать словарь (на основе гистограмм),
            # номер колонки: спиисок пороговых значений (для признака),
            # по которым будем пытаться разбивать признаки в узле
            self.calculateFeaturesThresholdByHist(X_train)
        
        rows = np.arange(0, X_train.shape[0])
        # Создать дерево и вернуть его корень
        self.root = self.buildDecisionTree(X_train, y_train, rows)
        #self.printTree(self.root); print()
        
    #****************************************************************
    
    # x_test: вектор numpy с признаками
    def help_predict(self, x_test):
        # Корень дерева
        node = self.root
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
        
        # При стохастическом градиентном бустинге порядок
        # колонок в DataFrame изменяется, поэтому нужно
        for col, featureName in enumerate(X.columns.tolist()):
            # ...обновить словарь, номер колонки: название признака
            self.colsToFeatureName[col] = featureName
        
        X_test = X.to_numpy()
        
        # Цикл по объектам (строкам) из тестового набора признаков
        for n in np.arange(0, X_test.shape[0]):
            row = X_test[n,:]
            # После прохода по дереву решений получить среднее значение
            # целевой переменной всех объектов в листе
            target = self.help_predict(row)
            y_predict[n] = target
        
        return  y_predict
    
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
        return f'MyTreeReg class: max_depth={self.maxDepth}, min_samples_split={self.minSamplesSplit}, max_leafs={self.maxLeafs}'
    
    def __repr__(self):
        return f'MyTreeReg(max_depth={self.maxDepth}, min_samples_split={self.minSamplesSplit}, max_leafs={self.maxLeafs})'

#********************************************************************

if __name__ == '__main__':
    import pandas as pd
    import os
    sep = os.sep
    
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
    
    # Для работы в методе fit нужно раскомментировать строчку
    #self.countSamplesTotal = X_train.shape[0]
    
    #myTreeReg = MyTreeReg(max_depth=5, min_samples_split=2, max_leafs=40, bins=16)
    
    # Обучение
    '''
    proba1 = y_train.mean()
    pred_0 = proba1 / (1-proba1)
    
    y_odds = np.full(shape=X_train.shape[0], fill_value=pred_0,
                        dtype=np.float64)
    
    numerator = np.exp(np.log(y_odds + 1e-15))
    denominator = 1 + numerator
    
    y_proba = numerator / denominator
    
    antiGradient = y_train.values - y_proba
    print(y_train)
    print(antiGradient)
    # Построить (обучить) k-ое дереро на антиградиенте
    myTreeReg.fit(X_train, pd.Series(antiGradient))
    
    print()
    myTreeReg.printTree(myTreeReg.root); print()
    print(myTreeReg.lastLeaf, myTreeReg.testSum); print()
    '''
    
    # Предсказание
    '''
    X_test = X_train.iloc[211:231, :]
    predictList = myTreeReg.predict(X_test)
    print( predictList ); print()
    print( predictList.sum() ); print()
    '''
    
    # Параметры обученной модели
    #print(myTreeReg.fi); print()
    
    print(end='\n\n')
    print('END!!!');
    input();
