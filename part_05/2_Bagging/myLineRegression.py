import numpy as np
import pandas as pd
import random

import types

# Линейная регрессия
class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None,
                 reg=None, l1_coef=0, l2_coef=0,
                 sgd_sample=None, random_state=42):
        self.verboseInc = None
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
        self.weights = np.ones((X_features.shape[1], 1), dtype=np.float64)
        X_transpose = X_features.T
        countSample = X_features.shape[0]

        if self.sgd_sample != None:
            if self.sgd_sample < 1:
                countSample = int( np.round(self.sgd_sample * X_features.shape[0], 0) )
            else:
                countSample = self.sgd_sample
        
        learningSpeed = self.learning_rate
        for n in np.arange(1, self.n_iter+1):
            if type(self.learning_rate) is types.LambdaType:
                learningSpeed = self.learning_rate(n)

            if self.sgd_sample != None:
                rowsInd = random.sample(list(np.arange(0, X_features_all.shape[0])), countSample)
                X_features = X_features_all[rowsInd, :]
                y_target = y_target_all[rowsInd, :]
                X_transpose = X_features.T

            y_predict = self.dot(X_features, self.weights)
            errors = y_predict - y_target
            gradients = self.dot(X_transpose, errors)
            gradients = 2 * gradients / countSample
            
            if self.reg == 'l1':
                gradients += self.l1_coef * np.sign(self.weights)
            if self.reg == 'l2':
                gradients += 2 * self.l2_coef * self.weights
            if self.reg == 'elasticnet':
                l1_reg = self.l1_coef * np.sign(self.weights)
                l2_reg = 2 * self.l2_coef * self.weights
                gradients += l1_reg + l2_reg
            
            self.weights -= learningSpeed * gradients
            self.show_estimators(n, verbose, y_target_all.flatten(), self.predict(X))

        self.bestScore = self.get_metric(y_target_all.flatten(), self.predict(X), self.metric)
    
    #****************************************************************
    
    def predict(self, X):
        X = X.reset_index(drop=True)
        X = self.addColumnForB0(X)
        X_features = X.to_numpy()
        y_predict = self.dot(X_features, self.weights)
        return y_predict.flatten()
    
    #****************************************************************
    
    def get_coef(self):
        weights = self.weights[1:].flatten()
        return weights
    
    #****************************************************************
    
    def get_best_score(self):
        return self.bestScore
    
    #****************************************************************
    
    def get_metric(self, y_target, y_predict, metric_name):
        result = None
        
        errors = y_target - y_predict
        yMean = y_target.mean()
        
        if metric_name == 'r2':
            numerator   = np.mean(errors ** 2)
            denominator = np.mean((y_target - yMean) ** 2)
            result = 1 - (numerator / denominator)
        elif metric_name == 'mape':
            result = np.mean( np.abs(errors / y_target) ) * 100
        elif metric_name == 'mae':
            result = np.mean( np.abs(errors) )
        elif metric_name == 'rmse':
            result = np.sqrt( np.mean(errors ** 2) )
        else:
            result = np.mean(errors ** 2)
        
        return result
    
    #****************************************************************
    
    def show_estimators(self, n, verbose, y_target, y_predict):
        if (verbose > 0) and (n == 1 or n == self.verboseInc):
            mse = np.mean((y_target - y_predict) ** 2)
            
            if (self.metric != None) and (n == 1 or n == self.verboseInc):
                metric_value = self.get_metric(y_target.flatten(), y_predict.flatten(), self.metric)
            
            if n == 1:
                if self.metric == None:
                    print(f'start | loss: {mse}')
                else:
                    print(f'start | loss: {mse} | {self.metric}: {metric_value}')
            elif n == self.verboseInc:
                if self.metric == None:
                    print(f'{self.verboseInc} | loss: {mse}')
                else:
                    print(f'{self.verboseInc} | loss: {mse} | {self.metric}: {metric_value}')
                self.verboseInc += verbose
    
    #****************************************************************
    
    def addColumnForB0(self, X):
        onesDf = pd.DataFrame(np.ones((X.shape[0], 1), dtype=np.int32), columns=['for B0'])
        X_new = pd.concat([onesDf, X], axis='columns')
        return X_new
    
    #****************************************************************
    
    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def __repr__(self):
        return f'MyLineReg(n_iter={self.n_iter}, learning_rate={self.learning_rate})'

#********************************************************************

if __name__ == '__main__':
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)
    X_train, y_train = data['data'], data['target']
    
    idxs = [1, 2, 3, 1, 2]
    
    #print(X_train); print()
    #print(X_train.loc[idxs]); print()
    
    #print(y_train); print()
    #print(y_train.loc[idxs]); print()
        
    print(end='\n\n')
    print('END!!!');
    input();
