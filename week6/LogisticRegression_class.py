import numpy as np

#시그모이드 함수
def sigmoid(x):
        return 1 / (1+np.exp(-x))

#편미분 함수
def numerical_derivative(f, x):
    delta_x = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x) # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad

#========================================================================================#

class LogisticRegression_cls:
    def __init__(self, X_train, y_train, lr = 1e-2, W=np.random.rand(1,1), b = np.random.rand(1)):
        #lr,W,b -> default value 설정

        self.X_train = X_train
        self.y_train = y_train
        self.lr = lr #learning rate
        self.W = W #Weight
        self.b = b #bias

    #손실함수
    def loss_func(self):

        delta = 1e-7 # log 무한대 발산 방지

        z = np.dot(self.X_train,self.W) + self.b
        y = sigmoid(z)

        # cross-entropy
        return -np.sum(self.y_train*np.log(y+delta) + (1-self.y_train)*np.log((1-y)*delta))


    #손실 값 계산 함수
    def error_val(self):

        delta = 1e-7    # log 무한대 발산 방지

        z = np.dot(self.X_train, self.W) + self.b
        y = sigmoid(z)

        # cross-entropy
        return  -np.sum( self.y_train*np.log(y + delta) + (1-self.y_train)*np.log((1 - y)+delta ) )



    #예측 함수
    def predict(self, test): #test: 예측에 사용될 데이터

        result = []
        for x in test:
            z = np.dot(x, self.W) + self.b
            y = sigmoid(z)

            if y > 0.5:
                result.append(1)
            else:
                result.append(0)

        return result

    #학습 함수
    def train(self):
        print("Initial error value = ", self.error_val())
        f = lambda x: self.loss_func()
        for step in range(10001):
            self.W -= self.lr * numerical_derivative(f, self.W)

            self.b -= self.lr * numerical_derivative(f, self.b)

            if (step % 400 == 0):
                print("step = ",step, "error value = ", self.error_val())
