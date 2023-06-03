import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from logistic import SigmoidRegression
from logistic import SoftmaxRegression
from decompose import PCA
from decompose import LDA

args = dict()
args['dataroot'] = r'./data/'

class WineData(object):

    def __init__(self, wine_type:str):
        filename = 'winequality-{}.csv'.format(wine_type)
        filename = args['dataroot'] + filename
        self.data = pd.read_csv(filename, sep=';')
        self.data = self.data.to_numpy()
        self.label = self.data[:, -1]
        self.data = self.data[:, :-1]
        self.label -= np.min(self.label)
        self.label = self.label.astype(np.int32)
        self.c = len(set(self.label))
        self.dim = self.data.shape[-1]

    def decompose(self, solver, n_components):
        model = solver(n_components)
        if solver is PCA:
            model.fit(self.data)
        elif solver is LDA:
            model.fit(
                self.data, 
                self.label, 
                self.c
            )
        self.data = model.transform(self.data)
        self.dim = n_components
        return model

    def classify(
            self, 
            lr, 
            decay, 
            iterations, 
            patience
    ):
        model = SoftmaxRegression(
            self.dim, self.c
        )
        history = model.fit(
            self.data, 
            self.label, 
            lr, 
            decay, 
            iterations, 
            patience
        )
        return model, history
    
    def render_history(self, history):
        fig = plt.figure(figsize=(13, 6))
        axs = [
            plt.subplot(1, 2, 1), 
            plt.subplot(1, 2, 2)
        ]
        axs[0].set_title('loss', fontsize=20)
        axs[1].set_title('accuracy', fontsize=20)
        axs[0].plot(history['loss'], color='darkcyan')
        axs[1].plot(history['acc'],  color='orange')
        plt.show()

    def render_data(self, title, xlim=None, ylim=None):
        if self.dim != 2:
            return
        fig = plt.figure(figsize=(12, 12))
        axs = plt.subplot(1, 1, 1)
        axs.set_title(title, fontsize=20)
        if xlim is not None:
            axs.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            axs.set_ylim(ylim[0], ylim[1])
        colors = cm.get_cmap('tab10')
        colors = colors(np.linspace(0, 1, self.c))
        for i in range(self.c):
            data_i = self.data[
                np.where(self.label == i)
            ]
            axs.scatter(
                data_i[:, 0], 
                data_i[:, 1], 
                s=10, 
                c=colors[i], 
                label='category {}'.format(i)
            )
        axs.legend(fontsize=20)
        plt.show()

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    plt.style.use('seaborn')
    
    # *************** origin ******************** #

    wine_data = dict()
    wine_data['red'] = WineData('red')
    wine_data['white'] = WineData('white')
    
    model, history = wine_data['red'].classify(
        lr=0.001, 
        decay=0.97,
        iterations=64, 
        patience=5
    )
    wine_data['red'].render_history(history)

    model, history = wine_data['white'].classify(
        lr=0.001, 
        decay=0.97,
        iterations=64, 
        patience=5
    )
    wine_data['white'].render_history(history)
    
    # ****************** PCA ******************** #

    wine_data = dict()
    wine_data['red'] = WineData('red')
    wine_data['white'] = WineData('white')

    wine_data['red'].decompose(PCA, 2)
    wine_data['red'].render_data(
        'red', 
        (-150, 50), 
        (-30, 50)
    )
    model, history = wine_data['red'].classify(
        lr=0.001, iterations=120
    )
    wine_data['red'].render_history(history)

    wine_data['white'].decompose(PCA, 2)
    wine_data['white'].render_data(
        'white', 
        (-200, 150), 
        (-50, 100)
    )
    model, history = wine_data['white'].classify(
        lr=0.001, iterations=120
    )
    wine_data['white'].render_history(history)
    
    # ****************** LDA ******************** #

    wine_data = dict()
    wine_data['red'] = WineData('red')
    wine_data['white'] = WineData('white')

    wine_data['red'].decompose(LDA, 2)
    wine_data['red'].render_data('red')
    model, history = wine_data['red'].classify(
        lr=0.1, 
        decay=0.97, 
        iterations=64, 
        patience=5
    )
    wine_data['red'].render_history(history)

    wine_data['white'].decompose(LDA, 2)
    wine_data['white'].render_data('white')
    model, history = wine_data['white'].classify(
        lr=0.1, 
        decay=0.97, 
        iterations=64, 
        patience=5
    )
    wine_data['white'].render_history(history)