import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from logistic import SigmoidRegression
from logistic import SoftmaxRegression
from decompose import PCA
from decompose import LDA
from data import WineData

args = dict()
args['dataroot'] = r'./data/'
args['modelroot'] = r'./model/'

class MyWineData(WineData):

    def __init__(
            self, 
            dataroot:str, 
            wine_type:str
    ):
        super().__init__(
            dataroot, 
            wine_type
        )

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

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    plt.style.use('seaborn')
    
    # *************** origin ******************** #

    wine_data = dict()
    wine_data['red'] = MyWineData(
        args['dataroot'], 'red'
    )
    wine_data['white'] = MyWineData(
        args['dataroot'], 'white'
    )
    
    model, history = wine_data['red'].classify(
        lr=0.001, 
        decay=0.97,
        iterations=64, 
        patience=3
    )
    wine_data['red'].render_history(history)

    model, history = wine_data['white'].classify(
        lr=0.001, 
        decay=0.97,
        iterations=64, 
        patience=3
    )
    wine_data['white'].render_history(history)
    
    # ****************** PCA ******************** #

    wine_data = dict()
    wine_data['red'] = MyWineData(
        args['dataroot'], 'red'
    )
    wine_data['white'] = MyWineData(
        args['dataroot'], 'white'
    )

    wine_data['red'].decompose(PCA, 2)
    wine_data['red'].render_data(
        'red', 
        (-150, 50), 
        (-30, 50)
    )
    model, history = wine_data['red'].classify(
        lr=0.01, 
        decay=0.97,
        iterations=64, 
        patience=2
    )
    model.save_model(
        args['modelroot'] + 'red_pca'
    )
    wine_data['red'].render_history(history)

    wine_data['white'].decompose(PCA, 2)
    wine_data['white'].render_data(
        'white', 
        (-200, 150), 
        (-50, 100)
    )
    model, history = wine_data['white'].classify(
        lr=0.01, 
        decay=0.97,
        iterations=64, 
        patience=2
    )
    model.save_model(
        args['modelroot'] + 'white_pca'
    )
    wine_data['white'].render_history(history)
    
    # ****************** LDA ******************** #

    wine_data = dict()
    wine_data['red'] = MyWineData(
        args['dataroot'], 'red'
    )
    wine_data['white'] = MyWineData(
        args['dataroot'], 'white'
    )

    wine_data['red'].decompose(LDA, 2)
    wine_data['red'].render_data('red')
    model, history = wine_data['red'].classify(
        lr=0.1, 
        decay=0.97, 
        iterations=64, 
        patience=5
    )
    model.save_model(
        args['modelroot'] + 'red_lda'
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
    model.save_model(
        args['modelroot'] + 'white_lda'
    )
    wine_data['white'].render_history(history)