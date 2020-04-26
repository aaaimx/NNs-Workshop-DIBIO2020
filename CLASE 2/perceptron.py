import numpy as np

class perceptron:
    """
    \tx = input len
    """
    def __init__(self, x):
        self.f = 'linear'
        self.inputLen = x
        self.weights = np.random.rand(self.inputLen, 1)
        self.bias = np.zeros((1, 1))

    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))

    def linear(self, s, deriv=False):
        if (deriv == True):
            return 1
        return s
    
    def function(self, f, s, deriv):
        if f == 'sigmoid':
            return self.sigmoid(s, deriv)
        else:
            return self.linear(s, deriv)

    def predict(self, x):
        """
        x = input vector
        """
        return self.function(self.f, np.dot(np.array(x), self.weights) + self.bias, False)

    def train(self, x, y, lr, epochs):
        """
        \tx = input vector
        \ty = out vector
        \tlr = learning rate
        """
        self.input = np.array(x)
        self.out = np.array(y)
                
        for _ in range(epochs):
            output = self.predict(self.input)
            error = -(self.out - output) # derivada del error cuadratico medio
            delta0 = error*self.function(self.f, output, deriv=True)
            delta1 = np.dot(self.input.T, delta0)
            
            self.weights -= lr*delta1
            self.bias -= lr*np.mean(delta0)