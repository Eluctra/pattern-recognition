import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from knn import KNN

args = dict()
args['dataroot'] = r'./data/'

class IrisData(object):

    def __init__(self):
        self.train = args['dataroot'] + 'train.csv'
        self.val = args['dataroot'] + 'val.csv'
        self.test = args['dataroot'] + 'test_data.csv'
        self.train = pd.read_csv(self.train).to_numpy()
        self.val = pd.read_csv(self.val).to_numpy()
        self.test = pd.read_csv(self.test).to_numpy()
    
    def euclidean(self, k):
        model = KNN('euclidean')
        data = self.train[:, :-1]
        label = self.train[:, -1].astype(np.int32)
        model.fit(data, label)
        data = self.val[:, :-1]
        label = self.val[:, -1].astype(np.int32)
        acc = 0
        for x, y_true in zip(data, label):
            y_pred = model.predict(x, k)
            if y_pred == y_true: acc += 1
        acc /= data.shape[0]
        return model, acc
    
    def mahalanobis(
            self, 
            k, 
            transform_dim, 
            lr, 
            decay, 
            iterations
    ):
        model = KNN('mahalanobis')
        data = self.train[:, :-1]
        label = self.train[:, -1].astype(np.int32)
        history = model.fit(
            data, 
            label, 
            transform_dim=transform_dim, 
            lr=lr, 
            decay=decay, 
            iterations=iterations
        )
        data = self.val[:, :-1]
        label = self.val[:, -1].astype(np.int32)
        acc = 0
        for x, y_true in zip(data, label):
            y_pred = model.predict(x, k)
            if y_pred == y_true: acc += 1
        acc /= data.shape[0]
        return model, acc, history
    
    def render_mahalanobis_history(self, history):
        fig = plt.figure(figsize=(12, 12))
        axs = plt.subplot(1, 1, 1)
        axs.set_title('similarity', fontsize=20)
        axs.set_xlabel('iterations', fontsize=20)
        axs.plot(history)
        plt.show()

    def render_mahalanobis_model(self, model):
        data = self.train[:, :-1]
        transformed_data = model.transform(data)
        if transformed_data.shape[-1] != 2:
            return
        label = self.train[:, -1]
        c = int(np.max(label)) + 1
        data_split = [
            transformed_data[np.where(label == i)]
            for i in range(c)
        ]
        fig = plt.figure(figsize=(12, 12))
        axs = plt.subplot(1, 1, 1)
        axs.set_title('decomposed sample', fontsize=20)
        colors = cm.get_cmap('tab10')
        colors = colors(np.linspace(0, 1, c))
        for i in range(c):
            data_i = transformed_data[
                np.where(label == i)
            ]
            axs.scatter(
                data_i[:, 0], 
                data_i[:, 1], 
                c=colors[i], 
                label='category {}'.format(i)
            )
        axs.legend(fontsize=20)
        plt.show()

if __name__ == '__main__':
    iris = IrisData()
    model, acc, history = iris.mahalanobis(
        5, 2, 0.1, 0.95, 64
    )
    plt.style.use('seaborn')
    iris.render_mahalanobis_history(history)
    iris.render_mahalanobis_model(model)
    print(acc)