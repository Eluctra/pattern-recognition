import numpy as np

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    s1 = np.exp(x - np.max(x, axis, keepdims=True))
    s2 = np.sum(s1, axis, keepdims=True)
    return s1 / s2

class SigmoidRegression(object):

    def __init__(self, input_dim, scale=0.1) -> None:
        self.w = scale * np.random.randn(input_dim)
        self.b = scale * np.random.randn()
    
    def predict(self, data, logit=True):
        y_pred = np.matmul(data, self.w)
        y_pred = y_pred + self.b
        if logit:
            y_pred = sigmoid(y_pred)
        else:
            y_pred = y_pred > 0
        return y_pred
    
    def fit(
            self, 
            data, 
            label, 
            lr, 
            iterations, 
            epsilon=0.001
    ): 
        history = dict()
        history['loss'] = []
        history['acc']  = []
        n = data.shape[0]
        for i in range(iterations):
            y_pred = self.predict(data)
            positive = label * (1 / (y_pred + epsilon))
            negative = (1 - label) * (1 / (y_pred - 1 - epsilon))
            grad = -(positive + negative)
            grad = grad * y_pred * (1 - y_pred)
            grad_w = np.matmul(data.T, grad) / n
            grad_b = np.sum(grad) / n
            self.w -= lr * grad_w
            self.b -= lr * grad_b
            positive = label * np.log(y_pred)
            negative = (1 - label) * np.log(1 - y_pred)
            loss = -np.sum(positive + negative) / n
            y_pred = y_pred > 0.5
            acc = np.sum(y_pred == label) / n
            history['loss'].append(loss)
            history['acc'].append(acc)
        return history

class SoftmaxRegression(object):

    def __init__(
            self, 
            input_dim, 
            output_dim, 
            scale=0.1
    ):
        self.w = scale * np.random.randn(
            input_dim, 
            output_dim
        )
        self.b = scale * np.random.randn(
            output_dim
        )
        self.c = output_dim
    
    def predict(self, data, logit=True):
        y_pred = np.matmul(data, self.w)
        y_pred = y_pred + self.b
        if logit:
            y_pred = softmax(y_pred)
        else:
            y_pred = np.argmax(y_pred, -1)
        return y_pred
    
    def fit(
            self, 
            data, 
            label, 
            lr, 
            iterations, 
            epsilon=0.001
    ):
        history = dict()
        history['loss'] = []
        history['acc']  = []
        y_true = np.eye(self.c)[label]
        n = data.shape[0]
        for i in range(iterations):
            y_pred = self.predict(data)
            grad = -y_true * (1 / (y_pred + epsilon))
            grad = grad * y_pred * (1 - y_pred)
            grad_w = np.matmul(data.T, grad) / n
            grad_b = np.sum(grad, 0) / n
            self.w -= lr * grad_w
            self.b -= lr * grad_b
            loss = -y_true * np.log(y_pred)
            loss = np.sum(loss) / n
            y_pred = np.argmax(y_pred, -1)
            acc = np.sum(y_pred == label) / n
            history['loss'].append(loss)
            history['acc'].append(acc)
        return history

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    data = np.random.randn(100, 2)
    data1 = data + np.array([2, 2])
    data2 = data + np.array([-2, 2])
    data3 = data + np.array([-2, -2])
    data4 = data + np.array([2, -2])
    data = np.concatenate([
        data1, data2, data3, data4
    ], axis=0)
    label = np.ones(100)
    label1 = (label * 0).astype(np.int32)
    label2 = (label * 1).astype(np.int32)
    label3 = (label * 2).astype(np.int32)
    label4 = (label * 3).astype(np.int32)
    label = np.concatenate([
        label1, label2, label3, label4
    ], axis=0)
    model = SoftmaxRegression(2, 4)
    history = model.fit(data, label, 0.01, 80)
    fig = plt.figure(figsize=(13, 6))
    axs = plt.subplot(121), plt.subplot(122)
    axs[0].set_title('loss', fontsize=20)
    axs[1].set_title('acc', fontsize=20)
    axs[0].plot(history['loss'], color='darkcyan')
    axs[1].plot(history['acc'], color='orange')
    plt.show()