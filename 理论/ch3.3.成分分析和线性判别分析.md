# 主成分分析（$\mathrm{PCA}$）

为了降低样本的特征维度，将原特征空间$\mathbb{R}^{m}$中的样本通过线性变换降维到特征空间$\mathbb{R}^{p}$
$$
\boldsymbol{y} = \mathbf{A} \boldsymbol{x} \Rightarrow \mathbb{R}^{p} = \mathbb{R}^{p \times m} \mathbb{R}^{m}
$$
降维的同时要求保留原特征空间的尺度，线性变换矩阵$\mathbf{A}$的$p$个行向量需要满足单位模长且相互正交
$$
\mathbf{A} \mathbf{A}^{\mathrm{T}} = \mathbf{I}_{p}
$$
样本包含的信息量与样本的分散程度正相关，一般通过样本的散布矩阵来度量样本的散布情况
$$
\mathbf{S}_{t} = \sum_{i = 1}^{n} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{x})(\boldsymbol{x}_{i} - \boldsymbol{\mu}_{x})^{\mathrm{T}}
$$
为了最大化降维后的样本包含的信息量，需要最大化降维样本的分散程度
$$
\begin{gather*}
\max_{\mathbf{A}} | \mathbf{S}_{t} | =  \left| \sum_{i = 1}^{n} (\boldsymbol{y}_{i} - \boldsymbol{\mu}_{y})(\boldsymbol{y}_{i} - \boldsymbol{\mu}_{y})^{\mathrm{T}} \right| \\ \\
s.t.\quad \mathbf{A} \mathbf{A}^{\mathrm{T}} = \mathbf{I}_{p}
\end{gather*}
$$
构造拉格朗日函数
$$
\begin{align*}
\mathcal{L}(\mathbf{A},\ \mathbf{\Lambda}) &= |\mathbf{S}_{t}| + tr\bigg[\mathbf{\Lambda}(\mathbf{A} \mathbf{A}^{\mathrm{T}} - \mathbf{I}_{p})\bigg] \\ \\
&= \left| \sum_{i = 1}^{n} (\boldsymbol{y}_{i} - \boldsymbol{\mu}_{y})(\boldsymbol{y}_{i} - \boldsymbol{\mu}_{y})^{\mathrm{T}} \right| + tr\bigg[\mathbf{\Lambda}(\mathbf{A} \mathbf{A}^{\mathrm{T}} - \mathbf{I}_{p})\bigg] \\ \\
&= \left| \sum_{i = 1}^{n} \mathbf{A} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{x})(\boldsymbol{x}_{i} - \boldsymbol{\mu}_{x})^{\mathrm{T}} \mathbf{A}^{\mathrm{T}} \right| + tr\bigg[\mathbf{\Lambda}(\mathbf{A} \mathbf{A}^{\mathrm{T}} - \mathbf{I}_{p})\bigg] \\ \\
\end{align*}
$$
