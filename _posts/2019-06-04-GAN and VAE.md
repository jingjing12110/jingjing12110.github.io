---
title: 'GAN和VAE的数理基础'
date: 2019-03-04
permalink: /posts/2019/06/GAN和VAE的数理基础/
tags:
  - GAN
  - VAE
---

生成模型一般具有两个基本功能：密度估计和样本生成 - **题记**

## Basic Knowledge

### Mathematics Background

- 隐变量

- 范数norm
  - 向量范数：
    $$||\boldsymbol v||=(\sum\limits_{i=1}^{n}v_{i}^2)^{1/2} = \sqrt{\langle\boldsymbol v, \boldsymbol v \rangle}$$
    向量内积；范数来衡量一个向量的长度，将一个向量变换为单位长度的操作叫做标准化。
  - 矩阵范数： 
    $$||\textbf{A}||_F = \left( \sum\limits_{i=1}^m\sum\limits_{j=1}^n|a_{ij}|^2 \right)^{1/2} = \langle\textbf{A}, \textbf{A} \rangle$$
    矩阵内积；对应位置上的元素最终相加，结果为标量。
  - 张量范数：
    $$||\mathcal{T}||_F = \left(\sum\limits_{i=1}^I\sum\limits_{j=1}^J\sum\limits_{k=1}^K|t_{ijk}|^2 \right)^{1/2} = \sqrt{\langle \mathcal{T}, \mathcal{T}}\rangle$$
    张量内积，

- 条件概率的链式规则(chain rule)  
$$P(x^{(1)},...,x^{(n)}) = P(x^{(1)}) \prod_{i=2}^{n}P(x^{(i)}|x^{1},...,x^{(i-1)})$$

- 边缘分布（Marginal Distribution）
指在概率论和统计学的多维随机变量中，只包含其中部分变量的概率分布。

- 隐空间latent space
隐变量模型将高维数据映射到低维紧致的子空间，不仅可以发现数据潜在的结构，还尽可能有效地保留输入数据的有用信息（暂无理论保证一定可以得到可解释性的mapping）。
eg: 潜在狄利克雷分配模型: Latent Dirichlet Allocation，LDA；

- 概率生成模型generative model
指一系列用于随机生成可观测数据的模型：
高维空间$\mathcal{X}$，存在随机向量${X}$服从一个未知的数据分布$p_{r}(x), x \in \mathcal{X}$；generative model就是根据一些可观测的样本$x^{1}, x^{2},...,x^{N}$来学习一个参数化模型$p_{\theta}(x)$来近似未知分布$P_{r}(X)$，并可以用这个模型来生成一些样本，使得“生成”的样本和“真实”的样本尽可能地相似。
**深度生成模型就是利用深层神经网络可以近似任意函数的能力来建模一个复杂的分布$p_{r}(x)$**。

- KL散度
在信息论中，**用生成的概率分布Q来拟合逼近真实的概率分布P时，所产生的信息损耗**，即描述两个概率分布的差异，其本身是非对称的；

- GM应用于监督学习
监督学习的目标是建模输出标签的条件概率密度函数$p(y|\textbf{x})$，根据贝叶斯公式：
$$
p(y|\textbf{x}) = \frac{p(\textbf{x}, y)}{\sum_{y}p(\textbf{x}, y)}
$$
将监督学习问题转换成联合概率密度函数$p(\textbf{x}, y)$的密度估计问题。

- 判别模型discriminative model
和生成模型相对应的另一类监督学习模型是判别模型，判别式模型直接建模条件概率密度函数$p(y|x)$，并不建模其联合概率密度函数$p(x, y)$。
由生成模型可以得到判别模型，但由判别模型得不到生成模型。

- 生成样本
给定一个概率密度函数为$p_{model}(\textbf{x})$的分布，生成一些服从这个分布的样本，也称为采样。

### 信息论-Information Theory

- 熵-Entropy

在信息论中，熵用来衡量一个随机事件的不确定性。假设对一个随机变量X（取值集合为$\mathcal{X}$，概率分布为$p(x), x\in \mathcal{X}$）进行编码，自信息（**较不可能发生的事件具有更高的信息量，因此用负对数来表示一个事件的信息量**）$I(x)$是变量$X = x$时的信息量或编码长度，定义为$I(x) = − \log(p(x))$；那么随机变量X的平均编码长度(**自信息只处理单个输出，so，用熵来对整个概率分布中的不确定性总量进行量化，自信息的数学期望**)，即熵定义为:
$$H(x)=\mathbb{E}_{x \sim{P}}[I(x)] = - \mathbb{E}_{x \sim{P}}[\log (p(x))] = - \sum\limits_{x \in \mathcal{X}} p(x) \log p(x)$$

- 联合熵和条件熵

对于两个离散随机变量X 和Y ，假设X取值集合为$\mathcal{X}$； Y取值集合为$\mathcal{Y}$，其联合概率分布满足为$p(x, y)$，则:
X和Y的联合熵（Joint Entropy）为：  
$$H(X, Y) = − \sum\limits_{x \in \mathcal{X}}\sum \limits_{y \in \mathcal{Y}} p(x, y) \log p(x, y)$$；

X和Y的条件熵（Conditional Entropy）为：
$$H(X|Y) = − \sum\limits_{x \in \mathcal{X}}\sum \limits_{y \in \mathcal{Y}} p(x, y) \log p(x|y) = − \sum\limits_{x \in \mathcal{X}}\sum \limits_{y \in \mathcal{Y}} p(x, y) \log \frac{p(x,y)}{p(y)}$$  

即: 
$$H(X|Y) = H(X, Y) - H(Y)$$

- 互信息

互信息（Mutual Information）是衡量已知一个变量时，另一个变量不确定性的减少程度。两个离散随机变量X 和Y 的互信息定义为：  
$$I(X;Y) = − \sum\limits_{x \in \mathcal{X}}\sum \limits_{y \in \mathcal{Y}} p(x, y) \log \frac{p(x,y)}{p(x)p(y)}$$

即:
$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

- 交叉熵

对应分布为p(x)的随机变量，**熵H(p)表示其最优编码长度**。交叉熵（Cross Entropy）是按照**概率分布q**的最优编码对**真实分布为p**的信息进行编码的长度，定义为：  
$$H(p,q) = \mathbb{E}_{p}[-\log q(x)] = -\sum\limits_{x}p(x)\log q(x)$$  
则，在给定p的情况下，如果q 和p越接近，交叉熵越小；如果q 和p越远，交叉熵就越大。

- KL散度

KL散度（Kullback-Leibler Divergence），也叫KL距离或相对熵(Relative Entropy)，**是用概率分布q来近似p时所造成的信息损失量**。 KL散度是按照概率分布q的最优编码对真实分布为p的信息进行编码，其平均编码长度H(p, q)-交叉熵，和p的最优平均编码长度H(p) - 熵，之间的差异。对于离散概率分布p和q，从q 到p的KL散度定义为：  
$$D_{KL}(p∥q) = H(p, q) − H(p) = \sum\limits_x p(x)\log \frac{p(x)}{q(x)}$$

- JS散度

JS散度（Jensen–Shannon Divergence）是一种**对称的**衡量两个分布相似度的度量方式，定义为：  
$$D_{JS}(p∥q) = \frac{1}{2} D_{KL}(p∥m) + \frac{1}{2} D_{KL}(q∥m)$$ 
其中 $m = \frac{1}{2}(p+q)$  
JS散度是KL散度一种改进。但两种散度有存在一个问题，即如果两个分布 p, q 个分布没有重叠(常数)，或者重叠非常少时， KL散度和JS散度都很难衡量两个分布的距离。

- **Wasserstein 距离**

Wasserstein距离也是用于衡量两个分布之间的距离；
对于两个分布q1, q2， $p^{th}$-Wasserstein距离定义为:  
$$W_{p}(q_1, q_2) = (\inf\limits_{\gamma(x,y)\in \Gamma(q_1, q_2)} \mathbb{E}_{(x, y) \sim \gamma(x, y)}[d(x, y)^p])^{\frac{1}{p}}$$  
其中，$\Gamma(q_1, q_2)$是**边际分布**(也叫边缘分布或边沿分布)为q1 和q2 的所有可能的**联合分布集合**， d(x, y)为x和y的距离度量，比如$\ell_p$ 距离等，联合分布$\gamma(x, y)$, inf 上确界。

**理解**：  
如果将两个分布看作是两个土堆，联合分布$\gamma(x, y)$看作是从土堆q1的位置x到土堆q2的位置y的搬运土的**数量**，并有:  
$$\sum\limits_x \gamma(x,y)=q_2(y); \sum\limits_y \gamma(x,y)=q_1(x)$$  
其中, q1 和q2 为$\gamma(x, y)$的两个边际分布；  
$\mathbb{E}_{(x, y) \sim \gamma(x, y)}[d(x, y)^p]$可以理解为在联合分布$\gamma(x, y)$下把形状为 q1 的土堆搬运到形状为q2 的土堆所需的**工作量**:  
$$\mathbb{E}_{(x, y) \sim \gamma(x, y)}[d(x, y)^p] = \sum\limits_{(x,y)}\gamma(x, y)d(x,y)^p$$  
其中从土堆q1中的点x到土堆q2中的点y的**移动土的数量和距离**分别为$\gamma(x, y)$和$d(x, y)^p$。因此， Wasserstein距离可以理解为**搬运土堆的最小工作量**，也称为推土机距离（Earth-Mover’s Distance， EMD）。

### 最优传输定理

- Optimal Transport（OT）问题

给定欧式空间中的一个区域$\Omega \subset \mathbb{R}^n$，有两个概率测度$\mu$和$\nu$，满足$\int_{\Omega}d\mu = \int_{\Omega}d\nu$；寻找一个区域到自身的同胚映（diffeomorphism），$T:(\Omega, \mu) \rightarrow (\Omega, \nu)$，满足两个条件：**保持测度和极小化传输代价**:    
1). **保持测度**：对于一切波莱尔集(波莱尔集，在一个拓扑空间中，从所有的开集出发，通过取补集，可数并，可数交等运算，构造出来的所有集合，统称为这一个空间中的波莱尔集) $E\subset \Omega$，$\int_{E}d\nu=\int_{T^{-1}(E)}d\mu$；换句话说映射 $T$ 将概率分布 $\mu$ 映射成了概率分布 $\nu$，记成 $T^{*\mu} = \nu$ 直观上，自映射 $T:\Omega \rightarrow \Omega$ 带来体积元的变化，因此改变了概率分布。我们用$\mu$和$\nu$来表示概率密度函数，用$J_T$来表示映射的雅克比矩阵（Jacobian matrix），那么保持测度的微分方程应该是:  
$$\forall x \in \Omega$$，$$J_T(x)\mu(x)=\nu(T(x))$$     
2). **最优传输映射**:自映射 $T:\Omega \rightarrow \Omega$ 的传输代价(Transportation Cost)定义为:  
$$E(T):= \frac{1}{2}\int_{\Omega}|x-T(x)|^2d\mu(x)$$  
在所有保持测度的自映射中，传输代价最小者被称为是最优传输映射（Optimal Mass Transportation Map），亦即:  
$$T^* = \arg min_{T*\mu=\nu}E(T)$$  
**最优传输映射的传输代价被称为是概率测度$\mu$和概率测度$\nu$之间的Wasserstein距离**，记为$d_{W}(\mu, \nu)$。

[OT in GAN.](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650723168&idx=3&sn=41fcf2fb0408c7b6a9b82d55d91c2b9c&chksm=871b171eb06c9e082c4083ff32748104a617e5cb1e6bd4d296b4db431358b8a41f40908ea8a5&scene=21#wechat_redirect)

## 变分自编码器VAE- Variational Autoencoder

思想是利用神经网络来分别建模两个复杂的条件概率密度函数：

1. 用神经网络来产生变分分布$q(\textbf{z}\|\phi)$称为推断网络；理论上$q(\textbf{z}\|\phi)$ 可以不依赖x。但由于$q(\textbf{z}\|\phi)$ 的目标是近似后验分布 $q(\textbf{z}\|\textbf{x}, \theta)$ , 其和x相关，因此变分密度函数一般写为 $q(\textbf{z}\|\textbf{x}, \phi)$ ；推断网络的输入是x，输出为变分分布 $q(\textbf{z}\|\textbf{x}, \phi)$ ；

2. 用神经网络来产生概率分布$p(\textbf{x}\|\textbf{z}, \theta)$，称为生成网络，生成网络的输入是隐变量z，输出是概率分布$p(\textbf{x}\|\textbf{z}, \theta)$；

将推断网络和生成网络合并在一起就得到了变分自编码的整个网络结构：
![vae](/images/blog/vae.png)

### [VAE的科学解释](https://zhuanlan.zhihu.com/p/27549418)

**自动编码器**(AutoEncoder)最开始作为一种数据的压缩方法, 现在自动编码器主要应用有数据去噪，进行可视化降维和生成数据.
![AE](https://pic2.zhimg.com/v2-e5745659cd57562c1dcfc3de7e2a4229_r.jpg)
AE由两部分组成, 第一个部分是编码器(Encoder)，第二个部分是解码器(Decoder)，编码器和解码器都可以是任意的模型，通常我们使用神经网络模型作为编码器和解码器。**输入的数据经过神经网络降维到一个编码(code)，接着又通过另外一个神经网络去解码得到一个与输入原数据一模一样的生成数据**，然后通过去比较这两个数据，最小化他们之间的差异来训练这个网络中编码器和解码器的参数。

**变分编码器**(variational autoencoder)是自动编码器的升级版本，其结构跟自动编码器是类似的，也由编码器和解码器构成。
<span style="color:red;">我们没有办法自己去构造隐藏向量，需要通过一张图片输入编码才知道得到的隐含向量是什么, 这导致AE无法生成任意图像, 所以提出VAE。</span>
**VAE在编码过程中增加一些限制,使其生成的隐含向量能够粗略的遵循一个标准正态分布**, 这样在生成图像时,只需要给他一个标准正态分布的随机latent vector,就能通过decoder生成任意的图像。
![VAE](https://pic4.zhimg.com/v2-8769151d6bd61bceead581d4aa0c2b37_r.jpg)

**VAE使用“重新参数化 - reparametrization”的技巧来解决KL divergence的计算问题。** 不再是每次产生一个隐含向量，而是生成两个向量，一个表示均值，一个表示标准差，然后通过这两个统计量来合成隐含向量(即, 用一个标准正态分布先乘上标准差再加上均值就行了，这里我们默认编码之后的隐含向量是服从一个正态分布的)。

[Note: reparemerization 的技巧](https://blog.csdn.net/JackyTintin/article/details/53641885), latent vector $z \sim N(\mu, \sigma)$, 我们应该从$N(μ,σ)$采样，但这个采样操作对 $\mu$ 和 $\sigma$ 是不可导的，导致常规的通过误差反传的梯度下降法（GD）不能使用, 因此通过 reparemerization, 首先从 $N(0,1) $上采样 $\epsilon$, 然后$z= \sigma \cdot \epsilon + \mu$, 这样 $z \sim N(\mu, \sigma)$, 而且从encoder输出到z, 只涉及线性操作, 因此可以正常使用GD进行优化.

**AutoEncoder的缺点:** autoencoder要求encoder和decoder的能力不能太强, 极端情况下, 他们有能力完全记住训练样本。（ **如何判断, 如何解决?** - CVAE-条件变分自编码器）

## 生成对抗网络GAN-Generative Adversarial Network

VAE等显示地构建出样本的密度函数，通过最大似然估计来求解参数，是显示密度模型（explicit density model）； - **限制了神经网络的能力？**
但如果只是希望有一个模型能生成符合数据分布$p_{r}(x)$的样本，那么可以不显示地估计出数据分布的密度函数，即不显示地建模$p_{r}(x)$，而是建模生成过程本身，隐式密度模型（implicit density model）。

**Then**，如何确保生成网络产生的样本一定是服从真实的数据分布？
GAN是通过**对抗训练的方式**使得生成网络产生的样本服从真实数据分布；那还有其他的解决思路吗？

如何设置G和D的网络结构？MLP的层数和hidden units的个数？

### 相关介绍

1. [GAN & Conditional GAN](https://blog.csdn.net/taoyafan/article/details/81229466)
所谓条件，就是指我们现在生成的网络不仅仅需要逼真，而且还要有一定的条件。

2. Adversarial Attack - 对抗攻击
**给定一个训练后的分类器，我们能生成一个能骗过该网络的样本吗？**
对抗样本（adversarial example）是指经过精心计算得到的旨在误导分类器的样本。

3. Wasserstein距离和Lipschitz连续
- Wasserstein Distance
Wasserstein 距离又叫 Earth-Mover（EM）距离，1-th的Wasserstein距离为：
$$W(P_r, P_g)= \inf\limits_{\gamma\in \Pi(P_r, P_g)} \mathbb{E}_{(x,y) \sim \gamma} [||x-y||]$$
<!-- 解释如下：$\Pi (P_r, P_g)$是$P_r$和$P_g$组合起来的所有可能的联合分布的集合，反过来说，$\Pi (P_r, P_g)$中每一个分布的边缘分布都是$P_r$和$P_g$。对于每一个可能的联合分布$\gamma$而言，可以从中采样$(x, y) \sim \gamma$得到一个真实样本x和一个生成样本y，并算出这对样本的距离$||x-y||$，所以可以计算该联合分布$\gamma$下，样本对距离的期望值$\mathbb{E}_{(x, y) \sim \gamma} [||x - y||]$。在所有可能的联合分布中能够对这个期望值取到的下界$\inf\limits_{\gamma \sim \Pi (P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma} [||x - y||]$，就定义为Wasserstein距离。 -->
- Lipschitz 连续
有函数$f(x)$，如果存在一个常量$K$，使得对$f(x)$定义域上（可为实数也可以为复数）的任意两个值满足如下条件：
$$
|f(x_{1} - f(x_{2}))| \leq |x_{1} - x_{2}|*K
$$
那么称函数$f(x)$满足Lipschitz连续条件，并称$K$为$f(x)$的Lipschitz常数；Lipschitz连续比一致连续要强，它限制了函数的局部变动幅度不能超过某个常量。

### 典型GAN

#### [1]. [WGAN - Wasserstein Generative Adversarial Networks](https://arxiv.org/pdf/1701.07875.pdf)

facebook[论文版](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf)的[方法解读](https://zhuanlan.zhihu.com/p/25071913)。
[LiI'Log](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)
WGAN系列（解决判别器太强的问题）中critic本质就是GAN中的discriminator，但是，在WGAN中并不是用于区分是真实样本还是生成样本, 而是**用于拉近真是样本和生成样本的深度特征的差异**，因此不好再用discriminator,改用了critic。

#### [2]. [WGAN-GP - Improved Training of Wasserstein GANs](.)

[方法解读](https://zhuanlan.zhihu.com/p/52799555)，添加gradient penalty梯度惩罚；
![L](https://pic1.zhimg.com/80/v2-43bd56599fffa85d1f52b15093b75d7c_hd.jpg)

- review
WGAN-GP相对于WGAN的改进很小，除了增加了一个正则项，其他部分都和WGAN一样。 这个正则项就是WGAN-GP中GP（gradient penalty），即梯度约束。这个约束的意思是：critic相对于原始输入的梯度的L2范数要约束在1附近（双边约束）。

- 算法
![alg](https://pic2.zhimg.com/v2-11c621c88b18de5a703d98dd4bd3c8a1_r.jpg)
可以看出跟WGAN不同的主要有几处：1）用gradient penalty取代weight clipping；2）在生成图像上增加高斯噪声；3）优化器用Adam取代RMSProp。

#### [3]. [DCGAN - Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

DCGAN将传统的mlp构成的G和D换成了CNN，并对卷积神经网络的结构做了一些改变，以提高样本的质量和收敛的速度，这些改变有：

- 取消所有pooling层，G网络中使用转置卷积（transposed convolutional layer）进行上采样，D网络中用加入stride的卷积代替pooling
- 在D和G中均使用batch normalization
- 去掉FC层，使网络变为全卷积网络
- G网络中使用ReLU作为激活函数，最后一层使用tanh
- D网络中使用LeakyReLU作为激活函数

![DCGAN](/images/blog/DCGAN.png)

### GAN的训练

1. BatchNorm: 一个mini-batch里面必须保证只有Real样本或者Fake样本，不要把他们混起来训练;尽可能使用batchnorm，如果限制了不能用，则用instance normalization;

2. Discriminator 太好, 导致Generator的性能越来越差?

3. new idea in GAN
- self-attention
- BigGAN - TPU
