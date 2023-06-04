import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class WineData(object):

    def __init__(
            self, 
            dataroot:str, 
            wine_type:str
    ):
        filename = 'winequality-{}.csv'.format(wine_type)
        filename = dataroot + filename
        self.data = pd.read_csv(filename, sep=';')
        self.data = self.data.to_numpy()
        self.label = self.data[:, -1]
        self.data = self.data[:, :-1]
        self.label -= np.min(self.label)
        self.label = self.label.astype(np.int32)
        self.c = len(set(self.label))
        self.dim = self.data.shape[-1]

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