import numpy as np
import pandas as pd
import random

import types

# Логистическая регрессия
# Для двух классов с метками: 0 и 1
class MyLogReg():
    def __init__(self, n_iter=10, learning_rate=0.1, weights=None, metric=None,
                 reg=None, l1_coef=0, l2_coef=0,
                 sgd_sample=None, random_state=42):
        self.verboseInc = None
        self.countSampleAll = None
        self.bestScore = None
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
    
    #****************************************************************
    
    def dot(self, A, B):
        res = np.zeros((A.shape[0], B.shape[1]))
        
        for r in np.arange(0, A.shape[0]):
            for c in np.arange(0, B.shape[1]):
                for t in np.arange(0, B.shape[0]):
                    res[r, c] += A[r, t] * B[t, c]
        return res
    
    #****************************************************************
    
    def fit(self, X, y, verbose=False):
        if not verbose:
            verbose = -1
        else:
            self.verboseInc = verbose
        
        random.seed(self.random_state)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        X_new = self.addColumnForB0(X)
        X_features = X_new.to_numpy()
        y_target = y.values.reshape(y.shape[0], 1)
        X_features_all = X_new.to_numpy()
        y_target_all = y.values.reshape(y.shape[0], 1)
        self.countSampleAll = X_features_all.shape[0]
        self.weights = np.ones((X_features.shape[1], 1), dtype=np.float64)
        X_transpose = X_features.T
        countSamples = X_features.shape[0]
        
        if self.sgd_sample != None:
            if self.sgd_sample < 1:
                countSamples = int( np.round(self.sgd_sample * X_features.shape[0], 0) )
            else:
                countSamples = self.sgd_sample
        
        learningSpeed = self.learning_rate
        for n in np.arange(1, self.n_iter+1):
            if type(self.learning_rate) is types.LambdaType:
                learningSpeed = self.learning_rate(n)
            
            if self.sgd_sample != None:
                rowsInd = random.sample(list(np.arange(0, X_features_all.shape[0])), countSamples)
                X_features = X_features_all[rowsInd, :]
                y_target = y_target_all[rowsInd, :]
                X_transpose = X_features.T

            y_predict = self.dot(X_features, self.weights)
            y_proba = 1 / (1 + np.exp(-1*y_predict))
            errors = y_proba - y_target
            gradients = self.dot(X_transpose, errors)
            gradients = gradients / countSamples
            
            if self.reg == 'l1':
                gradients += self.l1_coef * np.sign(self.weights)
            if self.reg == 'l2':
                gradients += 2 * self.l2_coef * self.weights
            if self.reg == 'elasticnet':
                l1_reg = self.l1_coef * np.sign(self.weights)
                l2_reg = 2 * self.l2_coef * self.weights
                gradients += l1_reg +l2_reg
            
            self.weights -= learningSpeed * gradients
            self.show_estimators(
                n, verbose, y_target_all.flatten(),
                self.predict(X), self.predict_proba(X)
            )

        self.bestScore = self.get_metric(
            y_target_all.flatten(),
            self.predict(X), self.predict_proba(X)
        )
    
    #****************************************************************
    
    def predict_proba(self, X):
        X = X.reset_index(drop=True)
        X = self.addColumnForB0(X)
        X_features = X.to_numpy()
        y_predict = self.dot(X_features, self.weights)
        y_proba = 1 / (1 + np.exp(-1*y_predict))
        return y_proba.flatten()
    
    #****************************************************************
    
    def predict(self, X):
        y_poba = self.predict_proba(X)
        y_predict = y_poba.copy()
        y_predict[y_predict > 0.5] = 1
        y_predict[y_predict <= 0.5] = 0
        y_predict = y_predict.astype(np.int32)
        return y_predict.flatten()
    
    #****************************************************************
    
    def get_coef(self):
        weights = self.weights[1:].flatten()
        return weights
    
    #****************************************************************
    
    def get_best_score(self):
        return self.bestScore
    
    #****************************************************************
    
    def getLogLoss(self, y_target_all, y_proba_all):
        eps = 1e-15
        tmp1 = y_target_all * np.log(y_proba_all + eps)
        tmp2 = (1 - y_target_all) + np.log(1 - y_proba_all + eps)
        summa = np.sum(tmp1 + tmp2)
        logLoss = -1 * (summa / self.countSampleAll)
        return logLoss
    
    #****************************************************************
    
    def get_metric(self, y_target, y_predict, y_proba):
        result = None
        truePositive = 0
        trueNegative = 0
        falsePositive = 0
        falseNegative = 0
        countPositive = 0
        countNegative = 0
        
        for k in np.arange(0, len(y_target)):
            if y_target[k] == 1 and y_predict[k] == 1:
                truePositive += 1
                countPositive +=1
            if y_target[k] == 0 and y_predict[k] == 0:
                trueNegative += 1
                countNegative += 1
            if y_target[k] == 0 and y_predict[k] == 1:
                falsePositive += 1
                countNegative += 1
            if y_target[k] == 1 and y_predict[k] == 0:
                falseNegative += 1
                countPositive +=1
        
        if self.metric == 'roc_auc':
            proba = np.round(y_proba, 10)
            proba = np.append(proba.reshape(-1, 1), y_target.reshape(-1, 1), axis=1)
            proba = pd.DataFrame(proba, columns=['p', 'y'])
            proba = proba.sort_values(by='p', ascending=False)
            proba['y'] = proba['y'].apply(int)
            proba = proba.reset_index(drop=True)
            
            startIndex = 0
            sumPositiveTotal = 0
            tmpList = []
            for k in np.arange(0, proba.shape[0]):
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
        elif self.metric == 'f1':
            precision = truePositive / (truePositive + falsePositive)
            recall = truePositive / (truePositive + falseNegative)
            result = 2 * precision * recall / (precision + recall)
        elif self.metric == 'recall':
            result = truePositive / (truePositive + falseNegative)
        elif self.metric == 'precision':
            result = truePositive / (truePositive + falsePositive)
        else:
            result = (truePositive + trueNegative) / len(y_predict)
        
        return result
    
    #****************************************************************
    
    def show_estimators(self, n, verbose, y_target_all, y_predict_all, y_proba_all):
        if (verbose > 0) and (n == 1 or n == self.verboseInc):
            logLoss = self.getLogLoss(y_target_all, y_proba_all)

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
    
    def addColumnForB0(self, X):
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
    print(end='\n\n')
    print('END!!!');
    input();
