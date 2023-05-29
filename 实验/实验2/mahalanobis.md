马式距离
$$
\begin{align*}
    d_{M}(\boldsymbol{x},\ \boldsymbol{y}) &= (\mathbf{A}\boldsymbol{x} - \mathbf{A} \boldsymbol{y})^{2} \\ \\
    &= (\boldsymbol{x} - \boldsymbol{y})^{\mathrm{T}} \mathbf{A}^{\mathrm{T}} \mathbf{A} (\boldsymbol{x} - \boldsymbol{y}) \\ \\
    &= (\boldsymbol{x} - \boldsymbol{y})^{\mathrm{T}} \mathbf{M} (\boldsymbol{x} - \boldsymbol{y})
\end{align*}
$$
其中$\mathbf{A}$为正交非单位矩阵

度量学习损失函数
$$
\ell(\mathbf{A}) = \sum_{k = 1}^{c} \sum_{\boldsymbol{x}_{i} \in D_{k}} \left[ \sum_{\boldsymbol{x}_{j} \in D_{k}/\boldsymbol{x}_{i}} \exp(-d_{M}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j})) \right] \bigg/ \left[ \sum_{x \in D / \boldsymbol{x}_{i}} \exp(-d_{M}(\boldsymbol{x}_{i},\ \boldsymbol{x})) \right]
$$
对$\mathbf{A}$的导数
$$
\frac{\partial \ell}{\partial \mathbf{A}} = \sum_{k = 1}^{c} \sum_{\boldsymbol{x}_{i} \in D_{k}} 2 \mathbf{A} \frac{\left[ \sum_{\boldsymbol{x}_{j} \in D_{k}/\boldsymbol{x}_{i}} \exp(-d_{M}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j})) \right]}{}
$$
$$
\frac{\partial d_{M}(\boldsymbol{x}, \boldsymbol{y})}{\partial \mathbf{A}} = 2 \mathbf{A} (\boldsymbol{x} - \boldsymbol{y}) (\boldsymbol{x} - \boldsymbol{y})^{\mathrm{T}}
$$