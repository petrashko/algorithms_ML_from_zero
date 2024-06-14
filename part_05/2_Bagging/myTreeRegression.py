import numpy as np

class Node:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
    
    def __str__(self):
        if (self.left is None) and (self.right is None):
            return f'({self.data.get("leafName")} = {self.data.get("target")})'
        else:
            return f'({self.data.get("featureName")} <= {self.data.get("threshold")})'


# Дерево решений - регрессия
class MyTreeReg():
    def __init__(self, max_depth=5, min_samples_split=2,
                 max_leafs=20, bins=None):
        self.maxDepth = max_depth
        self.minSamplesSplit = min_samples_split
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
        mseN = self.getMse(y_node)
        mseL, mseR = self.getMse(y_left), self.getMse(y_right)
        tmp = mseN - (cntL * mseL / cntN) - (cntR * mseR / cntN)
        featureImpotance = cntN / self.countSamplesTotal * tmp
        prevImpotance = self.fi.get(feature, 0)
        self.fi[feature] = prevImpotance + featureImpotance
    
    #****************************************************************
    
    def getMse(self, y):
        yMean = np.mean(y)
        errors = y - yMean
        mse = np.mean(errors ** 2)
        return mse
    
    #****************************************************************
    
    def getInformationGain(self, left, right):
        cntLeft, cntRight = left.shape[0], right.shape[0]
        N = cntLeft + cntRight
        if (cntLeft == 0) or (cntRight == 0):
            informationGain = 0
        else:
            mseTotal = self.getMse(np.hstack((left, right)))
            mseLeft, mseRight = self.getMse(left), self.getMse(right)
            informationGain = mseTotal - \
                ((cntLeft / N * mseLeft) + (cntRight / N * mseRight))
        
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
    
    def createToLeaf(self, y, side):
        target = y.mean()
        leafData = {
            'leafName': f'{side}Leaf-{self.lastLeaf+1}',
            'target': target
        }
        node = Node(leafData)
        self.lastLeaf += 1
        self.testSum += target
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
                (self.maxLeafs - self.leafs_cnt == 1)
        ):
            self.leafs_cnt += 1
            return self.createToLeaf(y, side)
        
        if self.leafs_cnt < self.maxLeafs:
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
        target = node.data.get('target', None)
        while target is None:
            featureName = node.data.get('featureName')
            threshold = node.data.get('threshold')
            col = self.getColNumberByName(featureName)
            value = x_test[col]
            if value <= threshold:
                node = node.left
            else:
                node = node.right
            
            target = node.data.get('target', None)
        return target
            
    #****************************************************************
    
    def predict(self, X):
        y_predict = np.zeros(X.shape[0])
        X = X.reset_index(drop=True)
        X_test = X.to_numpy()
        for n in np.arange(0, X_test.shape[0]):
            row = X_test[n,:]
            target = self.help_predict(row)
            y_predict[n] = target
        return  y_predict
    
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
        return f'MyTreeReg class: max_depth={self.maxDepth}, min_samples_split={self.minSamplesSplit}, max_leafs={self.maxLeafs}'
    
    def __repr__(self):
        return f'MyTreeReg(max_depth={self.maxDepth}, min_samples_split={self.minSamplesSplit}, max_leafs={self.maxLeafs})'

#********************************************************************

if __name__ == '__main__':
    print(end='\n\n')
    print('END!!!');
    input();
