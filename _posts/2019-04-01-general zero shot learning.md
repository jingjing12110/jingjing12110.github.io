---
title: 'Generalized Zero-Shot Learning文献阅读与总结'
date: 2019-06-04
permalink: /posts/2019/06/Generalized Zero-Shot Learning/
tags:
  - Generalized Zero-Shot Learning
---

Generalized Zero-Shot Learning

问题设定： ZSL是一个强假设的问题（Test on the unseen class），但是真实情况是<span style="color:red;">Test on both seen and unseen classes。</span> 

## 存在的问题

- 强偏（strong bias problem）
- extremely imbalance samples-样本的极端不平衡问题-> generative model
- image classification systems do not have access to whether a novel image belongs to a seen or unseen class in advance.

**思考**:

1. 在GZSL中, 识别出的unseen classes, 当置信度高时,可反过来用于训练吗?

---

## Paper Reading

#### [1]. [An Empirical Study and Analysis of Generalized Zero-Shot Learning for Object Recognition in the Wild - ECCV2018](https://arxiv.org/pdf/1605.04253v2.pdf) 

--- two-stage的方法

[Wei-Lun Chao 个人主页](http://www-scf.usc.edu/~weilunc/publication.html)


review之前的方法：

- two stage: 首先，确定test data是seen还是unseen(准确率高，但并不绝对正确，对于multiple unseen classes分类性能理论上会更差)，然后应用对应的分类器。

- direct stacking: **$\hat{y} = arg \max\limits_{c \in \mathcal{T}} \mathcal{f}_{c}(x)$**，其中[stacking](https://blog.csdn.net/wstcjf/article/details/77989963)过程如下：
![stacking](https://img-blog.csdn.net/20170915114447314?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd3N0Y2pm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
结果，几乎所有unseen classes的test data被误分类为seen classes，因为训练时seen classes的scoring函数（训练时只见得到seen classes）支配了unseen classes，导致了strong bias。

train set: $\mathcal{D}=(x_{n}\in \mathbb{R}^{D}, y_{n})_{n=1}^{N}$

label space of seen classes: $y_{n} \in \mathcal{S} = \{ 1, 2, ..., S\}$

label space of unseen classes: $\mathcal{U} = \{S+1, S+2, ..., S+U \}$

the union of the two classes: $\mathcal{T} = \mathcal{S} \cup \mathcal{U}$

**本文的观点：** 将object recognition in wild problem定义为GZSL问题，因为直接使用传统的性能很好的classifiers通常会将unseen classes分为seen classes； 所以提出calibrated stacking的方法：
calibrated stacking：**$\hat{y} = arg \max\limits_{c \in \mathcal{T}} \mathcal{f}_{c}(x) - \gamma II[c \in \mathcal{S}]$**，其中，指示符indicator$II[\cdot] \in \{0,1 \}$表示c是seen classes还是unseen classes， $\gamma$是calibrated factor，暴力地降低seen classes的scores；
分析calibrated factor $\gamma$，$\gamma \rightarrow + \infty, -\infty, 0$

提出使用AUSUC(area under seen-unseen accurancy curve： $A_{\mathcal{S} \rightarrow \mathcal{T}}$ - $A_{\mathcal{U} \rightarrow \mathcal{T}}$)作为评价指标来进行model selection和超参调整（hyperparameter tuning）；

## 现有解决思路

### Inductive ZSL(归纳) & Transductive ZSL(转导)

- inductive ZSL： 在训练的阶段，只有训练集模型是可见的，训练好了一个模型后，再用它来预测没有见过的类别。
- transductive ZSL： 在训练阶段，标注过的原始数据类别和未标注过的测试数据类别都在训练过程中可见的。

#### [2]. [QFSL: Transductive Unbiased Embedding for Zero-Shot Learning - CVPR2018](https://blog.csdn.net/cp_oldy/article/details/81661468)

浙大和阿里合作论文, 利用transductive解决ZSL中的强偏(strong bias)问题。  
**_In the semantic embedding space, the labeled source images are mapped to several fixed points specified by the source categories, and the unlabeled target images are forced to be mapped to other points specified by the target categories._**  
在语义嵌入空间中，被标记的源图像被映射到由源类别指定的若干个嵌入点，并且未标记的目标图像被强制映射到由目标类别指定的其他点。  
QFSL模型:
![QFSL](/images/blog/QFSL.png)

---

### 基于生成的方法(GAN、VAE - Popular Now)

基于生成的方法(即, 将ZSL任务视为数据增强/data-augmentation任务)按生成的数据可以分为:

- 生成图像
- 生成图像特征(visual features): 最常用
- 生成低维的latent space features: 如 [3]

#### [3]. [CADA-VAE : Generalized Zero-and Few-Shot Learning via Aligned Variational Autoencoders - Bosch AI博世 - CVPR2019](https://arxiv.org/pdf/1812.01784v4.pdf)

Cross-Model learning跨模态学习, 跨模态的embedding models： 基于autoencoder，

**本文的观点：** 在基于生成的方法中,基于GAN的loss, 训练时极其不稳定, 因此考虑条件VAE。GZSL是一种典型的从seen classes到unseen classes知识迁移，通过利用 side information进行跨模态（cross model：指数据存在的形式）生成。

**Model:**

_training example_: $S=\{(x,y,c(y)) \|x \in X, y\in Y^S, c(y) \in C \}$

_an auxiliary training set_: $U = \{(u, c(u))\| u\in Y^u, c(u)\in C \}$, u来自于unseen classes $Y^u$.

_class-embedding of unseen classes_: $C(U) = \{c(u_1),\cdots, c(u_L) \}$

![CADA-VAE](/images/blog/CADA-VAE.png)

- VAE - basic building block  
$$\mathcal{L} = \mathbb{E}_{q_{\phi}(z\|x)}[\log p_{\theta}(x\|z)] - D_{KL}(q_{\phi}(z\|x)||p_{\theta}(z))$$  
其中, first term 是reconstruction error, 第二项是inference model $q(z\|x)$ 和latent vector $z$ 的先验分布(该prior 一般选择多元高斯分布-multivariate standard gaussian distribution)之间的KL散度。

- Cross and Distribution Aligned VAE  
CADA-VAE的目的就是在common space中学习M个模态数据的表示(representation), 所以 CADA-VAE有M个encoder(一个modality 数据一个encoder将其mapping到表示空间), 为了最小化信息的loss, 需要decoder重构原始数据。模型的VAE loss即为M个VAE loss的和:  
$$\mathcal{L}_{VAE} = \sum\limits_{i}^M \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x^{(i)}|z)] - \beta D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta }(z) )$$  
在匹配images features和class embedding时, M=2, $x^{(1)} \in X, x^{(2)} \in C(Y^S)$, 然而要学到跨模态的相似的表示, 就需要正则项(latent distribution匹配项):  
**Cross-Alignment Loss**: 通过decoding 来自另一模态但同一类样本的latent 编码来重建数据, 即每个模态的decoder的训练数据也来源于其他模态的latent vector, so, cross-reconstruction loss为:
$$\mathcal{L}_{CA} = \sum\limits_i^M \sum\limits_{j \neq i}^M |x^{(j)} - D_j(E_i(x^{(i)}))|$$  
其中, $E_i$ 是第 $i^{th}$个modality 特征的encoder, $D_j$ 是第 $j^{th}$个modality 同一类特征的decoder。  
**Distribution-Alignment Loss**: 图像和类别表示也可以通过最小化他们间的距离度量进行匹配, so, 本文选用minimize 不同latent 多元高斯分布的2-Wasserstein 距离。  
$$ W_{ij} = [||\mu_i -\mu_j||^2_2 + Tr(\Sigma_i) + Tr(\Sigma_j) - 2(\Sigma_i^{\frac{1}{2}} \Sigma_i \Sigma_j^{\frac{1}{2}})^{\frac{1}{2}}]^{\frac{1}{2}}$$  
又因为encoder predict 了可交换顺序的对角协方差矩阵(diagonal covariance matrices), 所以$W_{ij}$ 可简化为:  
$$W_{ij} = (||\mu_i -\mu_j||^2_2 + ||\Sigma_i^{\frac{1}{2}} - \Sigma_j^{\frac{1}{2}}||^2_{Frobenius})^{\frac{1}{2}}$$  
其中, Frobenius范数, 简称F-范数，是一种矩阵范数(矩阵A的Frobenius范数定义为矩阵A各项元素的绝对值平方的总和).
so, M个模态数据的Distribution Alignment Loss可写为:  
$$\mathcal{L}_{DA} = \sum\limits_i^{M} \sum\limits_{j \neq i}^{M}W_{ij}$$

- Cross- and Distribution Alignment(CADA-VAE) Loss  
$$\mathcal{L}_{CADA-VAE} = L_{VAE} + \gamma \mathcal{L}_{CA} + \delta \mathcal{L}_{DA}$$

**训练**: 训练的过程中采用了学习率预热的方法(warmup)。 Warm up是在ResNet论文中提到的一种学习率预热的方法, 由于刚开始训练时模型的权重(weights)是随机初始化的(全部置为0是一个坑)，此时选择一个较大的学习率，可能会带来模型的不稳定; 学习率预热就是在刚开始训练的时候先使用一个较小的学习率，训练一些epoches或iterations，等模型稳定时再修改为预先设置的学习率进行训练。

#### [4]. [f-CLSWGAN: Feature Generating Networks for Zero-Shot Learning - CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_Feature_Generating_Networks_CVPR_2018_paper.pdf)

crux skill:<br>
1). WGAN-GP中梯度惩罚项gradient penalty的**代码实现和理论推导**；  
2). GZSL分类网络的处理（数据的读取与划分）；  
3). 多model网络的训练(pretrain 两个分类网络);  
4). torch1.0不论怎么训练, acc均为0 - 要用torch.div除法;  

作者利用Conditional GAN生成图像特征feature而不是图像像素pixel；提出3个Conditional GAN，如下：  
on train set $\mathcal{S}$，学习conditional生成器，$G:\mathcal{Z}\times \mathcal{C} \rightarrow \mathcal{X}$；即:  
输入为**随机高斯噪声$z\in \mathcal{Z} \subset \mathbb{R}^{d_z}$和类别的embedding $c(y) \in \mathcal{C}$；**
输出是**类别$y$的CNN图像特征$\tilde{x} \in \mathcal{X}$；**
一旦$G$利用seen classes的embedding $c(y) \in \mathcal{Y}^{\mathcal{s}}$学会生成真实图像的CNN特征$x$；
那么他也能利用$c(u)$生成任意unseen classes $u$的CNN feature $\tilde{x}$。

1). f-GAN  
$$
\min\limits_{G} \max\limits_{D}\mathcal{L}_{GAN} = E[\log{D(x, c(y))}] + E[\log(1-D(\tilde{x}, c(y)))]
$$  
where, $\tilde{x} = G(z, c(y))$；判别器$D: \mathcal{X}\times\mathcal{C}\rightarrow [0,1]$是多层感知机；

2). f-WGAN  
$$
\min\limits_{G} \max\limits_{D}\mathcal{L}_{WGAN} = E[{D(x, c(y))}] - E[D(\tilde{x}, c(y))] - \lambda E[(||\nabla_{\hat{x}} D(\hat{x}, c(y))||_{2}-1)^{2}]
$$  
where，$\tilde{x} = G(z, c(y)), \hat{x}=\alpha x+(1-\alpha)\tilde{x}$；去掉了f-GAN中的log；  
3). f-CLSWGAN  
f-WGAN不能保证生成的CNN特征适合训练判别器D，添加分类损失:
$$
\mathcal{L}_{CLS} = -E_{\tilde{x}\sim p_{\tilde{x}}}[\log P(y|\tilde{x};\theta)]
$$  
where，$P(y\|\tilde{x};\theta)$表示$\tilde{x}$被正确预测为类别y的概率;  
$$
\min\limits_{G} \max\limits_{D}\mathcal{L}_{WGAN} + \beta \mathcal{L}_{CLS}
$$

f-CLSWGAN如下图：  
<center>
<img src = "/images/blog/f-CLSWGAN.png" style="zoom:70%"/>
</center>

分类时，给定unseen classes $u \in \mathcal{Y^{u}}$的class embedding，resampling 噪声z，计算$\tilde{x} = G(z, c(u))$，得到生成的训练集 $\tilde{\mathcal{U}}={(\tilde{x}, u, c(u))}$；训练multimodel embedding Model或者softmax classifier实现分类：
prediction function:  
1). multimodel embedding  
$f(x) = \arg\max\limits_{y} F(x, c(y); W)$  
2). softmax classifier  
$f(x) = \arg\max\limits_{y} P(y|x; \theta)$  

#### [5]. [LisGAN: Leveraging the Invariant Side of Generative Zero-Shot Learning - CVPR2019](https://arxiv.org/pdf/1904.04092.pdf)

**Note: 基于CLSWGAN文章的改进版!**

**改进的出发点是**：语义描述生成视觉特征的过程（考虑到一种语义描述/semantic description可以对应各种合成的visual samples，而且，semantic description是unseen classes生成visual features的唯一依据）。
所以作者借鉴原型学习(prototype)，对每个class找一个mete-representation(本文叫soul sample)，来刻画同一个类别中每个样本的最具语义意义的层面。
**具体实现方法**：  
1). 对每个seen class进行聚类：$\{ X_1^c, X_2^c,..., X_k^c\}$表示类别c的k个clusters；  
对应的soul samples为：$S^c = \{ s_1^c, s_2^c,..., s_k^c\}$，其中单个soul sample定义为：  
$$s_k^c = \frac{1}{|X_k^c|}\sum\limits_{x_i \in X_k^c}x_i$$  
2). 对unseen classes同理，对生成的fake features有：  
$$\tilde{s_k^c} = \frac{1}{\tilde{|X_k^c|}}\sum\limits_{x_i \in \tilde{X_k^c}}\tilde{x_i}$$  
其中，$\tilde{x_i}=G(z,a)$即为生成的fake特征。考虑到生成的类别c的样本$\tilde{x}$至少应该和类别c中的一个soul sample靠近，建立individual regularization:  
$$L_{R1}=\frac{1}{n_1}\sum\limits_{i=1}^{n_1}\min\limits_{j \in [1,k]}||\tilde{x}-s_j^c||_2^2$$  
同时，soul sample可认为是类别的centroid，因此fake soul samples应该至少接近同一类的一个real soul sample，建立group regularization：  
$$L_{R2}= \frac{1}{C}\sum\limits_{c=1}^{C}\min\limits_{j \in [1,k]}||\tilde{s_j^c}-s_j^c||_2^2$$  
so, G的loss由CLSWGAN中的  
$$L_G = -\mathbb{E}[D(G(z,a))]-\lambda \mathbb{E}[\log P(y|G(z,a))]$$  
变为：  
$$L_G = -\mathbb{E}[D(G(z,a))]-\lambda \mathbb{E}[\log P(y|G(z,a))] + \lambda_1L_{R1} + \lambda_2L_{R2}$$

**预测unseen samples**：在传统的标准的softmax分类器的基础上进行改进，- cascade classifier.

![LisGAN](/images/blog/lisGAN.png)

#### [6]. [f-VAEGAN-D2: A Feature Generating Framework for Any-Shot Learning - CVPR2019](https://arxiv.org/abs/1903.10132)

Note：**MPII实验室，与CLSWGAN同作者**； 文章强调了visual feature的可解释性。
[主页](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/f-vaegan-d2-a-feature-generating-framework-for-any-shot-learning/)

文章的出发点是：visual features的生成；  
分析现在特征生成方法的shortcomings:  
1). G模型简单，难以捕获复杂的数据的分布；  
2). 多数情况下，并没有很好的理解要表征的类的semantic attributes；  
3). 生成features难以interpretable(可解释性).  
综合GAN和VAE的优势提出f-VAEGAN-D2.

![fVAEGAN](/images/blog/fVAEGAN.png)

**model介绍**：

- setting  
images： $X=\{x_1,...,x_l \} \cup \{ x_{l+1},...,x_{t}\}$，image feature space $\mathcal{X}$;  
seen class label set: $Y^s$  
novel label set: $Y^n$, a.k.a, unseen class label $Y^u$  
class embedding: $C=\{c(y)| \forall y \in Y^s \subset Y^n \}$ in embedding space $\mathcal{C}$  
**inductive setting**: train set contains only labeled samples of seen class images $\{x_1,...,x_l\}$  
**transductive setting**: train set contains both labeled and unlabeled samples $\{x_1,...,x_l,x_{l+1},...,x_t\}$  
二者inference过程相同。

- module  
  - f-WGAN: 同WGAN-GP
  - f-VAE: encoder $E(x,c)$ encodes visual feature $x$和condition $c$为latent变量$z$; decoder $Dec(z,c)$ 通过优化$\mathcal{L}_{VAE}^{s}$, 从latent $z$和condition $c$ 中重构 input $x$。  
  $$\mathcal{L}_{VAE}^{s} = KL(q(z\|x,c)||p(z\|c)) - \mathbb{E}_{q(z\|x,c)}[\log p(x\|z,c)]$$  
  其中,$q(z|x,c)$ 是encoder $E(x,c)$的model, $p(z|c)$ 服从正态分布 $\mathcal{N}(0,1)$,KL是Kullback-Leibler 散度, $p(x|z,c)$ 是decoder $Dec(z,c)$.
  - conditional discriminator $D_1$ 和nonconditional discriminator $D_2$ (学习novel class的manifold)

- loss设计  
$$\min\limits_{G,E}\max\limits_{D_1,D_2} \mathcal{L}_{VAEGAN}^s + \mathcal{L}_{WGAN}^n$$  
其中$$\mathcal{L}_{VAEGAN}^s=\mathcal{L}_{VAE}^s +\lambda \mathcal{L}_{WGAN}^s$$， WGAN的loss是$\mathcal{L}_{WGAN}^s $, 上标s表示loss是训练seen class的, n即novel classes.

#### [7]. [GDAN: Generative Dual Adversarial Network for Generalized Zero-shot Learning - CVPR2019](https://arxiv.org/abs/1811.04857)

- preknowledge

  **dual learning-对偶学习**: 很多人工智能的应用涉及两个互为对偶的任务，例如机器翻译中从中文到英文翻译和从英文到中文的翻译互为对偶、语音处理中语音识别和语音合成互为对偶、**图像理解中基于图像生成文本和基于文本生成图像互为对偶**、问答系统中回答问题和生成问题互为对偶，以及在搜索引擎中给检索词查找相关的网页和给网页生成关键词互为对偶。这些互为对偶的人工智能任务可以形成一个闭环，使从没有标注的数据中进行学习成为可能。
  对偶学习的最关键一点在于，给定一个原始任务模型，其对偶任务的模型可以给其提供反馈；同样的，给定一个对偶任务的模型，其原始任务的模型也可以给该对偶任务的模型提供反馈；从而这两个互为对偶的任务可以相互提供反馈，相互学习、相互提高。(在某种程度上与强化学习相同)

  **Hubness问题**，在高维空间中，一个语义向量嵌入点可能会成为很多数据点的最近邻，而这些数据点并不一定来自同一个类别(当我们使用语义空间 - semantic space, 作为嵌入空间时，需要将视觉特征映射到语义空间中，这样会使得空间发生萎缩，点与点之间更加稠密，从而加重hubness problem)。
  <span style="color:red;">在ZSL中，视觉特征空间作为嵌入空间要比语义空间作为嵌入空间对结果的影响好很多, 但仍然存在问题</span>, 本文解决hubness问题。

- 问题分析:  
  visual -> semantic mapping: 使用semantic space为latent空间,存在严重的hubness problem;  
  semantic -> visual mapping: 为了缓和(mitigate/alleviate)枢纽点问题, 使用了visual space为latent空间, 但仍然存在问题, 因为one class label有很多个相关的视觉特征。  
  所以作者针对此问题, 提出了一个unified framework：  
  <center>
  <img src="/images/blog/GDAN.png" width = "400" height = "200"/>
  </center>

  其中<span style="color:red;">Generator</span>采取(resort)CVAE, mapping noise z 从 $P(z|x,c)$到数据x的真实分布, 即encoder $P_E(z\|x,c)$ 和decoder $P_G(x\|z,c)$, loss若为CVAE的loss:  
  $$loss(\theta_E,\theta_G) = \mathbb{E}_{P_{data}(x,z), P_E(z\|x,c)}[\log P_G(x\|z,c)] - D_{KL}(P_E(z\|x,c)||P(z))$$  
  其中, $P(z)$是单位高斯分布。
  vanilla CVAE难train, 会出现posterior collapse问题, 所以disentangle z from c, loss为:  
  $$\mathcal{L}_{CVAE}(\theta_E, \theta_G) = \mathbb{E}_{P_{data}(v,s), P_E(z\|v)} [\log P_G(v\|z,s)] - D_{KL}(P_E(z\|v)||P(z))$$  
  <span style="color:red;">Regressor</span> 以两种images features(从train set中采样的real features $v$ 和CVAE生成的fake features $ = G(s, v)$)作为输入, loss为:  
  $$\mathcal{L}_{sup}(R) = \mathbb{E}_{P_{data}(v,s)}||s-R(v)||^2_2$$
  则, Regressor和CVAE之间的对偶学习的cyclic-consistency loss为:
  $$\mathcal{L}_{cyc}(\theta_G, \theta_E, \theta_R) = \mathbb{E}_{P_{data}(v,s), P_E(z|v)}[||v-G(R(v),z)||^2_2 + ||s-R(G(s,z))||_2^2]$$
  <span style="color:red">Discriminator</span> 用来better evaluate 特征的相似性, 以visual-semantic feature pair $(v,s)$为输入, 输出compatiblity score, (用network来学习 metric 而不是预先定义度量方式)。 为了区分不同类型的fake data, 用两种fake input ($G(s,z)$, s)和(v, $s^-$), $s^-$是随机采样的class embedding ($s \neq s^-$); 另外add由Regressor生成的第三类fake input $(v, R(v))$，Discriminator采用的是LSGAN, loss为:   
  $$\mathcal{L}_{adv}(\theta_D) = \mathbb{E}_{P_{data}(v,s)}[D(v,s)-1]^2 + \mathbb{E}_{P_{data}(v,s),P_E(z|v)}[D(G(s,z),s)^2] \\~~~~~~~~~~~~~~~~~~~~ + \mathbb{E}_{P_{data}(v)}[D(v,R(v))^2] + \mathbb{E}_{P_{data}(v,s), P_{data}(s^-|s)}[D(v,s^-)^2]$$

- train process
  - pretrain generator (CVAE)
  - train whole model (adversarial way)
  - generate unseen sample
  - train all classes classifier

#### [8]. [Gzsl-VSE: Generalized Zero-Shot Recognition based on Visually Semantic Embedding - CVPR2019](https://arxiv.org/pdf/1811.07993.pdf)

作者(Pengkai Zhu, Boston University, 也是18年ECCV zero shot detection的一作。)提出更general的GZSL问题是train阶段, unseen images和**unseen semantic vectors**都是agnostic.

---

### two-stage: 先区分seen和unseen classes，再各自分类

Then：
Q1： 如何准确的区分seen classes和unseen classes？
因为在训练时，并为见过unseen classes的样本（即二分类时，只有正样本，没有负样本）

#### [9]. [COCMO: Adaptive Confidence Smoothing for Generalized Zero-Shot Learning - CVPR2019](https://chechiklab.biu.ac.il/~gal/Papers/Atzmon_CVPR2019.pdf)

[作者主页](https://chechiklab.biu.ac.il/~yuvval/publications.html)和[其导师主页](https://chechiklab.biu.ac.il/~gal/pubs.html).
作者认为GZSL是一个类别不平衡分类问题的特例（extreme case of classification with unbalanced classes）；

对问题的分析：

First, when training a gating module it is hard to provide an accurate estimate of the probability that a sample is from the “unseen” classes, because by definition no samples have been observed from those classes.

Second, experts tend to behave in uncontrolled ways when presented with **out-of-distribution samples**, often producing confident-but-wrong predictions.

As a result, when using a soft combination of the two expert models, the “irrelevant” expert may overwhelm the decision of the correct expert.

网络结构如下:
![COSMO](/images/blog/cosmo.png)
**基于confidence的门控模块**： 将seen classes和unseen classes的分类问题当做是**Out-Of-Distribution detection**的任务;  
**自适应smoothing**： [Laplace smoothing](https://zyzypeter.github.io/2017/07/22/machine-learning-ch5-Naive-Bayes/).

#### [10]. [DA: Learning to Separate Domains in Generalized Zero-Shot and Open Set Learning: a probabilistic perspective - ICLR2019](https://arxiv.org/pdf/1810.07368.pdf)

- Preknowledge  
  [open set](https://zhuanlan.zhihu.com/p/31230331): 是迁移学习子问题-领域自适应中的一个概念；开放集（open set）与之前的数据集（通常是close set）主要区别在于 target 中是否包含 source 中不存在的类别；在实际场景中，模型可能会遇到训练集中所没有见过的类别，此时，模型依然会将其识别为已知类别当中的一类，反馈信息甚至根据判断执行下一步命令。  
  **Open Set Learning** judges whether the instances belong to known/seen classes or a novel unknown class.  
  **GZSL较OSL更严苛**：OSL的类别空间为seen classes + unknown，GZSL的类别空间为所有的类别；（OSL aims at discriminating the instances into seen classes and instances beyond these classes are categorized into a single novel class. Critically OSL **does not have the semantic prototypes of unseen classes** to further give the class label of those instances in the novel class.）

- review  
  作者认为GZSL和OSL（open set learning）一样，需要首先划分known domain和unknown domain（通过instances distribution），然后在每个domain中直接应用classifier(?可能是一个做过迁移学习的作者)；但这一过程存在几个关键的问题：  
  - 首先，单独的视觉特征不足以区分seen和unseen类别；  
  - 其次，基于seen classes训练的predictors可能并不可靠；  
  - 除此之外，每个domain中的分类器的performance对domain separate十分敏感，可能导 致： test instance一旦被错误的划分（domain），它将永远不会被正确的分类。  
  为了解决aforementioned issue，作者增加新的uncertain domains-考虑test instance和seen、unseen classes的重叠区域，并从统计概率的角度划分domain，先bootstrapping，然后fine-tuning by K-S test。

- method  
  **problem setting**： $$D_s = \left\{ \textbf{x}_{i}, \textbf{y}_i, l_i \right\}^{n_s}_{i=1}$$  
  其中$\textbf{x}_{i} \in \mathbb{R}^{n}$是instance的视觉特征，$l_i$是类别，$\textbf{y}_i$是instance的语义属性vector；open set recognition： $$c_i \in \left\{C_s, 'novel class' \right\}$$， GZSL: $$c_i \in \left\{C_s, C_t \right\}$$；  
  **极值分布(extreme value distributions)**：极值分布是指在概率论中极大值（或者极小值）的概率分布，从很多个彼此独立的值中挑出来的各个极大值应当服从的概率密度分布数f(x)。
