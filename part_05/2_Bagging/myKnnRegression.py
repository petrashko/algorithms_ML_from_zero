import numpy as np

# Метод K-ближайших соседей - регрессия
class MyKNNReg():
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.X = None  # DataFrame
        self.y = None  # Series
        self.train_size = None
        self.X_train = None  # ndarray
        self.y_train = None
        self.eps = 1e-15
    
    #****************************************************************
    
    def euclidean_distance(self, a, b):
        sum2 = np.sum((a - b) ** 2)
        result = np.sqrt(sum2)
        return result
    
    #****************************************************************
    
    def manhattan_distance(self, a, b):
        absDif = np.abs(a - b)
        result = np.sum(absDif)
        return result
    
    #****************************************************************
    
    def chebyshev_distance(self, a, b):
        absDif = np.abs(a - b)
        result = np.max(absDif)
        return result

    #****************************************************************
    
    def cosine_distance(self, a, b):
        numerator = np.sum(a * b)
        normA = np.sqrt( np.sum(a ** 2) )
        normB = np.sqrt( np.sum(b ** 2) )
        denominator = normA * normB
        result = 1 - (numerator / denominator)
        return result
    
    #****************************************************************
    
    def fit(self, X, y):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.X_train = self.X.to_numpy()
        self.y_train = self.y.values.reshape(y.shape[0], 1)
        self.train_size = self.X_train.shape
    
    #****************************************************************
    
    def help_predict(self, X):
        X = X.reset_index(drop=True)
        X_features = X.to_numpy()
        dotsMatrixForX_test = []
        for row in X_features:
            rowsDistance = np.zeros((self.train_size[0], 2))
            for i in np.arange(0, self.train_size[0]):
                if self.metric == 'cosine':
                    dist = self.cosine_distance(row, self.X_train[i])
                elif self.metric == 'chebyshev':
                    dist = self.chebyshev_distance(row, self.X_train[i])
                elif self.metric == 'manhattan':
                    dist = self.manhattan_distance(row, self.X_train[i])
                else:
                    dist = self.euclidean_distance(row, self.X_train[i])
                
                if dist >= self.eps:
                    rowsDistance[i, 0] = dist
                else:
                    rowsDistance[i, 0] = self.eps

                rowsDistance[i, 1] = self.y_train[i]
            
            idx = np.lexsort([rowsDistance[:, 0]])
            rowsDistance = rowsDistance[idx]
            dotsMatrix = []
            for r in np.arange(0, self.k):
                dotDescribe = [r+1, rowsDistance[r, 0], rowsDistance[r, 1]]
                dotsMatrix.append(dotDescribe)
            
            dotsMatrixForX_test.append(dotsMatrix)
        return np.array(dotsMatrixForX_test, dtype=np.float64)
    
    #****************************************************************
    
    def predict(self, X_test):
        y_predict = []
        dotsForX = self.help_predict(X_test)
        for dots in dotsForX:
            rankList  = dots[:, 0].flatten()
            distList  = dots[:, 1].flatten()
            valueList = dots[:, 2].flatten()
            
            if self.weight == 'rank':
                rankBack = (1 / rankList)
                denominator = np.sum(rankBack)
                dotsWeight = rankBack / denominator
                y_predict.append( (dotsWeight * valueList).sum() )
            elif self.weight == 'distance':
                distBack = (1 / distList)
                denominator = np.sum(distBack)
                dotsWeight = distBack / denominator
                y_predict.append( (dotsWeight * valueList).sum() )
            else:
                y_predict.append(valueList.mean())
        
        return np.array(y_predict, dtype=np.float64)
    
    #****************************************************************
    
    def __str__(self):
        return f'MyKNNReg class: k={self.k}'
    
    def __repr__(self):
        return f'MyKNNReg(k={self.k})'

#********************************************************************

if __name__ == '__main__':
    print(end='\n\n')
    print('END!!!');
    input();
