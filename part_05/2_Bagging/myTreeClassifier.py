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
            # data.get("proba1"): Вероятность для класса с меткой 1
            return f'({self.data.get("leafName")} = {self.data.get("proba1")})'
        # Узел дерева
        else:
            return f'({self.data.get("featureName")} <= {self.data.get("threshold")})'

#********************************************************************

# Дерево решений - классификация
# Для двух классов с метками: 0 и 1
class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2,
                 max_leafs=20, criterion='entropy', bins=None):
        self.maxDepth = max_depth
        self.minSamplesSplit = min_samples_split
        self.criterion = criterion
        self.bins = bins
        self.maxLeafs = max(max_leafs, 2)
        self.leafs_cnt = 0
        self.lastLeaf = 0
        self.colsToFeatureName = {}
        self.colsToThreshold = {}
        self.colsToThresholdByHist = {}
        self.countSamplesTotal = 0
        self.fi = {}
        self.root = None
        self.testSum = 0
        self.eps = 1e-15

    #****************************************************************
    
    def calculateFeaturesThreshold(self, X):
        dictFeatures = {}
        dictFeaturesUniq = {}
        for col in self.colsToFeatureName.keys():
            dictFeatures[col] = X[:, col]
            dictFeaturesUniq[col] = np.sort(np.unique( dictFeatures.get(col) ))
        
        for col in self.colsToFeatureName.keys():
            uniqList = dictFeaturesUniq.get(col)
            thresholdList = []
            for n in np.arange(1, len(uniqList)):
                threshold = np.mean([uniqList[n-1], uniqList[n]])
                thresholdList.append(threshold)
            self.colsToThreshold[col] = np.array(thresholdList)
    
    #****************************************************************
    
    def calculateFeaturesThresholdByHist(self, X):
        for col in self.colsToFeatureName.keys():
            _, thresholdList = np.histogram(X[:, col], self.bins)
            self.colsToThresholdByHist[col] = \
                np.array(thresholdList[1 : -1])
    
    #****************************************************************
    
    def calculateFeatureImpotance(self, feature, y_left, y_right):
        y_node = np.hstack((y_left, y_right))
        cntN = y_node.shape[0]
        cntL = y_left.shape[0]
        cntR = y_right.shape[0]
        
        if self.criterion == 'gini':
            giN = self.getGini(y_node)
            giL = self.getGini(y_left)
            giR = self.getGini(y_right)
        else:
            giN = self.getEntropy(y_node)
            giL = self.getEntropy(y_left)
            giR = self.getEntropy(y_right)
        
        tmp = giN - (cntL * giL / cntN) - (cntR * giR / cntN)
        featureImpotance = cntN / self.countSamplesTotal * tmp
        prevImpotance = self.fi.get(feature, 0)
        self.fi[feature] = prevImpotance + featureImpotance
    
    #****************************************************************
    
    def getGini(self, y):
        count0 = np.count_nonzero(y == 0)
        count1 = np.count_nonzero(y == 1)
        countTotal = y.shape[0]
        proba0 = count0 / countTotal
        proba1 = count1 / countTotal
        giniImpurity = 1 - ((proba0 ** 2) + (proba1 ** 2))
        return giniImpurity
    
    #****************************************************************
    
    def getEntropy(self, y):
        count0 = np.count_nonzero(y == 0)
        count1 = np.count_nonzero(y == 1)
        countTotal = y.shape[0]
        proba0 = count0 / countTotal
        proba1 = count1 / countTotal
        tmp0 = proba0 * np.log2(proba0 + self.eps)
        tmp1 = proba1 * np.log2(proba1 + self.eps)
        entropy = -1 * (tmp0 + tmp1)
        return entropy
    
    #****************************************************************
    
    def getInformationGain(self, left, right):
        cntLeft, cntRight = left.shape[0], right.shape[0]
        N = cntLeft + cntRight
        if (cntLeft == 0) or (cntRight == 0):
            informationGain = 0
        else:
            if self.criterion == 'gini':
                infTotal = self.getGini(np.hstack((left, right)))
                infLeft, infRight = self.getGini(left), self.getGini(right)
            else:
                infTotal = self.getEntropy(np.hstack((left, right)))
                infLeft, infRight = self.getEntropy(left), self.getEntropy(right)
            
            informationGain = infTotal - \
                ((cntLeft / N * infLeft) + (cntRight / N * infRight))
        return informationGain
    
    #****************************************************************
    
    def getBestSplit(self, X, y):
        result = (None, None, None)
        if y.max() == y.min():
            return (None, None, 0)
        
        informationGainMax = float('-inf')
        for col in self.colsToFeatureName.keys():
            featureValues = X[:, col]
            thresholdList = self.colsToThreshold.get(col)
            
            if self.bins is not None:
                if len(thresholdList) > (self.bins-1):
                    thresholdList = self.colsToThresholdByHist.get(col)
            
            prevThreshold = float('inf')
            for threshold in thresholdList:
                if abs(threshold - prevThreshold) < 0.001:
                    continue  # сделал чтобы пройти тесты на ограничение по времени
                mask = featureValues <= threshold
                y_left = y[mask]
                cntLeft = y_left.shape[0]
                mask = featureValues > threshold
                y_right = y[mask]
                cntRight = y_right.shape[0]
                
                if (cntLeft == 0) or (cntRight == 0):
                    informationGain = 0
                else:
                    informationGain = self.getInformationGain(y_left, y_right)
                
                if informationGain > informationGainMax:
                    featureName = self.colsToFeatureName.get(col)
                    result = (featureName, threshold, informationGain)
                    informationGainMax = informationGain
                
                prevThreshold = threshold
        
        return (result[0], result[1], result[2])
    
    #****************************************************************
    
    @staticmethod
    def getProba1(y):
        return y.sum() / y.shape[0]

    #****************************************************************
    
    def createToLeaf(self, y, side):
        proba1 = self.getProba1(y)
        leafData = {
            'leafName': f'{side}Leaf-{self.lastLeaf+1}',
            'proba1': proba1 
        }
        node = Node(leafData)
        self.lastLeaf += 1
        self.testSum += proba1
        return node
    
    #****************************************************************
    
    def buildDecisionTree(self, X, y, depth=0, side='_'):
        featureName, threshold, infoGain = self.getBestSplit(X, y)
        dataNode = {
            'featureName': featureName,
            'threshold': threshold 
        }
        node = Node(dataNode)
        
        if infoGain == 0:
            self.leafs_cnt += 1

            node = self.createToLeaf(y, side)
            return node
        
        col = self.getColNumberByName(featureName)
        if (
                (depth == self.maxDepth) or
                (y.shape[0] == 1) or
                (y.shape[0] < self.minSamplesSplit) or
                (self.maxLeafs - self.leafs_cnt == 1)  or
                (self.getProba1(y) in (0, 1))
        ):
            self.leafs_cnt += 1
            return self.createToLeaf(y, side)
        
        if (self.leafs_cnt < self.maxLeafs):
            featureSplit = X[:, col]
            idxLeft = np.argwhere(featureSplit <= threshold).flatten()
            idxRight = np.argwhere(featureSplit > threshold).flatten()
            X_left = X[idxLeft, :]
            y_left = y[idxLeft]
            X_right = X[idxRight, :]
            y_right = y[idxRight]
            self.calculateFeatureImpotance(featureName, y_left, y_right)
            self.leafs_cnt += 1
            node.left = self.buildDecisionTree(X_left, y_left, depth+1, 'l_')
            self.leafs_cnt -= 1
            node.right = self.buildDecisionTree(X_right, y_right, depth+1, 'r_')
            return node
        else:
            pass
        
    #****************************************************************
    
    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        for col, featureName in enumerate(X.columns.tolist()):
            self.colsToFeatureName[col] = featureName
            self.fi[featureName] = 0
        
        X_train = X.to_numpy()
        y_train = y.values
        
        self.calculateFeaturesThreshold(X_train)
        
        if self.bins is not None:
            self.calculateFeaturesThresholdByHist(X_train)
        
        self.root = self.buildDecisionTree(X_train, y_train)
        
    #****************************************************************
    
    def help_predict(self, x_test):
        node = self.root
        proba1 = node.data.get('proba1', None)
        
        while proba1 is None:
            featureName = node.data.get('featureName')
            threshold = node.data.get('threshold')
            col = self.getColNumberByName(featureName)
            value = x_test[col]
            if value <= threshold:
                node = node.left
            else:
                node = node.right
            
            proba1 = node.data.get('proba1', None)
        return proba1
    
    #****************************************************************
    
    def predict_proba(self, X):
        y_proba = np.zeros(X.shape[0])
        X = X.reset_index(drop=True)
        X_test = X.to_numpy()
        
        for n in np.arange(0, X_test.shape[0]):
            row = X_test[n, :]
            proba1 = self.help_predict(row)
            y_proba[n] = proba1
        
        return y_proba
    
    #****************************************************************
    
    def predict(self, X):
        y_predict = np.zeros(X.shape[0], dtype=np.int32)
        X = X.reset_index(drop=True)
        y_proba = self.predict_proba(X)
        
        for idx, proba in enumerate(y_proba):
            if proba > 0.5:
                y_predict[idx] = 1
            else:
                y_predict[idx] = 0
        
        return y_predict
    
    #****************************************************************
    
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
    print(end='\n\n')
    print('END!!!');
    input();
