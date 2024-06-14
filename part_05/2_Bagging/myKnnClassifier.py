import numpy as np

# Метод K-ближайших соседей - классификация
# Для двух классов с метками: 0 и 1
class MyKNNClf():
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.X = None
        self.y = None
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
    
    def help_predict(self, X_test):
        X_test = X_test.reset_index(drop=True)
        X_features = X_test.to_numpy()
        classListForX_test = []

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
            classList = []
            for r in np.arange(0, self.k):
                classDescribe = [r+1, rowsDistance[r, 0], rowsDistance[r, 1]]
                classList.append(classDescribe)
            
            classListForX_test.append(classList)
        return np.array(classListForX_test)
    
    #****************************************************************
    
    def predict_proba(self, X):
        X_test = X.reset_index(drop=True)
        y_proba = []
        dotsForX = self.help_predict(X_test)
        
        for dots in dotsForX:
            rankList  = dots[:, 0].flatten()
            distList  = dots[:, 1].flatten()
            labelList = dots[:, 2].flatten()
            
            dots1 = []
            for n in np.arange(0, self.k):
                if dots[n, 2] == 1:
                    dots1.append(dots[n])
            dots1 = np.array(dots1)
            
            if self.weight == 'rank':
                if len(dots1) == 0:
                    nominator = 0
                else:
                    nominator = np.sum(1 / dots1[:, 0].flatten())
                denominator = np.sum(1 / rankList)
                proba = nominator / denominator
            elif self.weight == 'distance':
                if len(dots1) == 0:
                    nominator = 0
                else:
                    nominator = np.sum(1 / dots1[:, 1].flatten())
                denominator = np.sum(1 / distList)
                proba = nominator / denominator
            else:
                proba = labelList.sum() / self.k
            
            y_proba.append(proba)
        return np.array(y_proba, dtype=np.float64)
    
    #****************************************************************
    
    def predict(self, X_test):
        y_predict = np.ones(X_test.shape[0])
        y_proba = self.predict_proba(X_test)
        
        for n in np.arange(0, y_proba.size):
            if y_proba[n] < 0.5:
                y_predict[n] = 0
        
        return np.array(y_predict, dtype=np.int32)
    
    #****************************************************************
    
    def __str__(self):
        return f'MyKNNClf class: k={self.k}'
    
    def __repr__(self):
        return f'MyKNNClf(k={self.k})'

#********************************************************************

if __name__ == '__main__':
    print(end='\n\n')
    print('END!!!');
    input();
