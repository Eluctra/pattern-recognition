import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import\
LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from data import WineData

args = dict()
args['dataroot'] = r'./data/'

class SciWineData(WineData):

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
        model = solver(n_components=n_components)
        model.fit(self.data, self.label)
        self.data = model.transform(self.data)
        self.dim = n_components
        return model
    
    def classify(self):
        model = LogisticRegression()
        model.fit(self.data, self.label)
        y_pred = model.predict(self.data)
        acc = np.sum(y_pred == self.label)
        acc /= self.data.shape[0]
        return model, acc
    
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    plt.style.use('seaborn')

    # *************** origin ******************** #

    wine_data = dict()
    wine_data['red'] = SciWineData(
        args['dataroot'], 'red'
    )
    wine_data['white'] = SciWineData(
        args['dataroot'], 'white'
    )

    model, acc = wine_data['red'].classify()
    print('red origin data accuracy: {}'.format(acc))

    model, acc = wine_data['white'].classify()
    print('white origin data accuracy: {}'.format(acc))
    
    # ****************** PCA ******************** #

    wine_data = dict()
    wine_data['red'] = SciWineData(
        args['dataroot'], 'red'
    )
    wine_data['white'] = SciWineData(
        args['dataroot'], 'white'
    )

    model = wine_data['red'].decompose(PCA, 2)
    wine_data['red'].render_data(
        'red', 
        (-50, 150), 
        (-30, 40)
    )
    model, acc = wine_data['red'].classify()
    print('red PCA data accuracy: {}'.format(acc))

    model = wine_data['white'].decompose(PCA, 2)
    wine_data['white'].render_data(
        'white', 
        (-150, 200), 
        (-50, 100)
    )
    model, acc = wine_data['white'].classify()
    print('white PCA data accuracy: {}'.format(acc))

    # ****************** LDA ******************** #

    wine_data = dict()
    wine_data['red'] = SciWineData(
        args['dataroot'], 'red'
    )
    wine_data['white'] = SciWineData(
        args['dataroot'], 'white'
    )

    model = wine_data['red'].decompose(LDA, 2)
    wine_data['red'].render_data('red')
    model, acc = wine_data['red'].classify()
    print('red LDA data accuracy: {}'.format(acc))

    model = wine_data['white'].decompose(LDA, 2)
    wine_data['white'].render_data('white')
    model, acc = wine_data['white'].classify()
    print('white LDA data accuracy: {}'.format(acc))