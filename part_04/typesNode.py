
# Узел для задач классификации
class NodeClf:
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


# Узел для задач регрессии
class NodeReg:
    def __init__(self, data):
        self.data = data
        #
        self.left = None
        self.right = None
    
    def __str__(self):
        # Лист дерева
        if (self.left is None) and (self.right is None):
            # data.get("target"): # Среднее целевых значений в листе
            return f'({self.data.get("leafName")} = {self.data.get("target")})'
        # Узел дерева
        else:
            return f'({self.data.get("featureName")} <= {self.data.get("threshold")})'

