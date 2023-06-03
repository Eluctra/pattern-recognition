# 马式距离

样本的散布矩阵
$$
\mathbf{S} = \frac{1}{n} \sum_{i = 1}^{n} (\boldsymbol{x}_{i} - \boldsymbol{\mu}) (\boldsymbol{x} - \boldsymbol{\mu})^{\mathrm{T}}
$$
马式距离
$$
d_{M}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \sqrt{(\boldsymbol{x_{i}} - \boldsymbol{x}_{j})^{\mathrm{T}} \mathbf{S}^{-1} (\boldsymbol{x}_{i} - \boldsymbol{x}_{j})}
$$
可学习的马式距离（伪马式距离）
$$
d_{M}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \sqrt{(\boldsymbol{x_{i}} - \boldsymbol{x}_{j})^{\mathrm{T}} \mathbf{M} (\boldsymbol{x}_{i} - \boldsymbol{x}_{j})}
$$
其中，$\mathbf{M}$是半正定矩阵，重写以上距离数学形式
$$
d_{M}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j}) = \sqrt{(\boldsymbol{x}_{i} - \boldsymbol{x}_{j})^{\mathrm{T}} \mathbf{A}^{\mathrm{T}} \mathbf{A} (\boldsymbol{x}_{i} - \boldsymbol{x}_{j})}
$$
其中变换矩阵$\mathbf{A} \in \mathbb{R}^{k \times d}$，并且$k \ge rank(\mathbf{M})$

# $\mathrm{NCA}$算法

样本$\boldsymbol{x}_{i}$的近邻分布
$$
\begin{gather*}
    p_{ij} = \exp\{-d^{2}_{M}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j})\} \bigg/ Z_{i} \\ \\
    Z_{i} = \sum_{k \ne i} \exp\{ -d^{2}_{M}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{k}) \}
\end{gather*}
$$
代表样本$\boldsymbol{x}_{j}$与样本$\boldsymbol{x}_{i}$属于相同类别的概率

针对变换矩阵$\mathbf{A}$最大化同类样本相似程度
$$
\max_{\mathbf{A}} f(\mathbf{A}) = \max_{\mathbf{A}} \sum_{i = 1}^{n} \sum_{j\ \in\ \Omega_{i}} p_{ij}
$$
其中$\Omega_{i}$表示和样本$\boldsymbol{x}_{i}$属于同一类别的其它样本的集合

令
$$
d_{ij} = d^{2}_{M}(\boldsymbol{x}_{i},\ \boldsymbol{x}_{j})
$$
准则函数对变换矩阵$\mathbf{A}$的导数
$$
\begin{gather*}
    \frac{\partial p_{ij}}{\partial d_{ik}} = \left\{
    \begin{matrix}
        -p_{ij} (1 - p_{ij}) & k = j \\ \\
        p_{ij} p_{ik} & k \ne j
    \end{matrix}
    \right. \\ \\
    \frac{\partial d_{ik}}{\partial \mathbf{A}} = 2\mathbf{A}(\boldsymbol{x}_{i} - \boldsymbol{x}_{k}) (\boldsymbol{x}_{i} - \boldsymbol{x}_{k})^{\mathrm{T}} \\ \\
    \frac{\partial p_{ij}}{\partial \mathbf{A}} = 2 p_{ij} \mathbf{A} \left[ \sum_{k \ne i} p_{ik} (\boldsymbol{x}_{i} - \boldsymbol{x}_{k}) (\boldsymbol{x}_{i} - \boldsymbol{x}_{k})^{\mathrm{T}} - (\boldsymbol{x}_{i} - \boldsymbol{x}_{j}) (\boldsymbol{x}_{i} - \boldsymbol{x}_{j})^{\mathrm{T}} \right]
     \\ \\
    \frac{\partial f}{\partial \mathbf{A}} = \sum_{i = 1}^{n} \sum_{j\ \in\ \Omega_{i}} \frac{\partial p_{ij}}{\partial \mathbf{A}}
\end{gather*}
$$
