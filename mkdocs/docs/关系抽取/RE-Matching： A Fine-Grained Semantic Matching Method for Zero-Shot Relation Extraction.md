---
tags:
  - NLP
  - RelationExtraction
  - 语义匹配
  - zero-shot
---

Resource :: ACL2023-main



一句话描述 :: 使用细粒度语义匹配进行zero-shot的关系抽取，对输入和description进行实体匹配和忽略冗余上下文的文本匹配。

# abstract

Semantic matching is a mainstream paradigm of zero-shot relation extraction, which matches a given input with a corresponding label description. The entities in the input should exactly match their hypernyms in the description, while the irrelevant contexts should be ignored when matching. However, general matching methods lack explicit modeling of the above matching pattern. In this work, we propose a fine-grained semantic matching method tailored for zero-shot relation extraction. Following the above matching pattern, we decompose the sentence-level similarity score into entity and context matching scores. Due to the lack of explicit annotations of the redundant components, we design a feature distillation module to adaptively identify the relationirrelevant features and reduce their negative impact on context matching. Experimental results show that our method achieves higher matching F1 score and has an inference speed 10 times faster, when compared with the stateof-the-art methods.

# Motivation
语义匹配是关系抽取零样本学习的主流方法，将输入与对应的标签描述相匹配。实体应该和描述中的上位词相匹配，而匹配时应该忽略无关的上下文。一般的匹配方法缺少显式的匹配模式建模。

![[Pasted image 20231015031521.png]]

语义匹配的例子。`organization`是`Apple`的上位词，`city`是`California`的上位词，`is located in a`和 `situated at` 是匹配上下文，`is a great company` 是多余上下文。

- siamese ：分别编码输入和描述，再进行匹配，优势在于描述编码可以重复使用快速推理，但是交互不足限制匹配性能。
- full encodeing：使用自注意力机制编码，交互丰富，计算开销大

本文综合了两种方法

# 贡献

本文提出了一种用于零样本关系抽取细粒度语义匹配方法。将句子相似度分解为实体匹配得分和上下文匹配得分。
对于缺少明确标注的多余上下文，设计了一个特征蒸馏模块来自适应识别不相关的特征并减少他们对文本匹配的负面影响。
- 针对zeroRE的细粒度语义匹配方法，显式建模了关系数据的匹配模式
- 提出上下文蒸馏方法，减少冗余上下文的负面影响。
- 和SOTA比F1分数更高并且推理速度快10倍。


# 方法

将编码和匹配分为两个模块，编码采用siamese，即分别编码输入和描述，可以重复使用描述编码，而使用匹配模块进行细粒度的交互。
遵循关系数据的匹配模式，句子的相似度得分被分解为实体匹配和上下文匹配得分两个部分。
为了处理没有显式标注的冗余上下文，设计了一个特征蒸馏模块，最大化分类损失的上下文特征被识别为与关系无关的特征。


## 零样本关系抽取zero-shot RE

目标是从已有的关系 $R_s={r^{s}_{1},r^{s}_{2},\dots,r^{s}_{n}}$ 中学习，并推广到未训练过的关系$R_u=\{r^{u}_{1},r^{u}_{2},\dots,r^{u}_{n}\}$ ，这两个集合不相交。
对于包含N个样本的给定训练集$$D=\{(x_{i},e_{i1},e_{i2},y_{i},d_{i})|i=1,\dots，N\}$$
每个样本包含输入实例$x_{i}$ ，目标实体对$e_{i1},e_{i2}$ ，关系$y_{i}$ ，和关系描述$d_{i}$ 。
匹配模型$$M(x,e_{1},e_{2},d)\rightarrow s\in\mathbb{R}$$
s表示输入实例x和关系描述d之间的语义相似度得分。
测试时，将匹配模型M迁移，提取$R_{u}$ 中未见过的关系，即给定一个表示$R_{u}$ 中未见过的关系的样本$(x_{j},e_{j1},e_{j2})$，查找描述与输入有最高的相似度得分的关系$\hat{y}_{j}$ 


![[Pasted image 20230924152751.png]]

图2：提出的重匹配方法的概述。 输入实例和候选关系描述（左侧）分别编码以提高效率。 为了建模关系数据的匹配模式，我们按实体和上下文匹配（中间）计算相似性。 此外，我们设计了一个蒸馏模块，以减少无关组件（输入中的灰色部分）对上下文匹配（右侧）的影响。

## 编码模块
编码模块分为关系描述编码和输入实例编码，将实体和上下文信息编码为固定长度的表示，以供后续的细粒度匹配。
### 关系描述编码
每一个关系描述对应一种关系，如关系 `headquartered_in` 对应描述 `the headquarters of an organization is located in a place`

构建实体描述的方法：
keyword：使用上位词，如关系 `headquartered_in` ，$d^{h}$是`organization`，$d^{t}$是`place`
synonyms：使用同义词，从wikidata和Thesaurus两个数据库种提取的上位词含义完全相同或者几乎相同的单词。例然后将$d^h$ 拓展为 `organization, institution, company`

Rule-based Template Filling: 基于模板的规则填充，prompt learning的启发，可以将同义词扩展的上位词序列填充到一些有空位的模板上，如 `the head/tail entity types including [S], [S], .`然后将$d^h$ 扩展为 `the head entity types including organization, institution, company`。但是本文没有使用，留待以后的工作。

关系描述编码对描述$\{d\in d_{r_{i}^{s}}|i=1,\dots,n\}$编码,使用修改后的Sentence-BERT，编码头实体，尾实体和匹配文本为$d^{h},d^{t},d$

### 输入实例编码

输入实例编码使用BERT，编码头实体，尾实体和匹配文本为$x_{i}^{h},x_{i}^{t},x_{i}$ 

对于输入实例$$x_{i}=\{ w_{1},w_{2,}\dots,w_{n}\}$$
使用四个特殊tokens来标记实体$$[E_{h}],[\E_{h}],[E_{t}],[\E_{t}]$$
通过maxpool对应实体token的隐藏状态来获得实体表示$x_{i}^{h},x_{i}^{t}$ ，而对应的上下文信息由特殊token$[E_{h}],[E_{t}]$
的隐藏状态拼接得到。

$$
\begin{gathered}
\boldsymbol{h}_1, \ldots, \boldsymbol{h}_n=\operatorname{BERT}\left(w_1, \ldots, w_n\right) \\
\boldsymbol{x}_i^h=\operatorname{MaxPool}\left(\boldsymbol{h}_{b_h}, \ldots, \boldsymbol{h}_{e_h}\right) \\
\boldsymbol{x}_i^t=\operatorname{MaxPool}\left(\boldsymbol{h}_{b_t}, \ldots, \boldsymbol{h}_{e_t}\right) \\
\boldsymbol{x}_i=\phi\left(\left\langle\boldsymbol{h}_{\left[E_h\right]} \mid \boldsymbol{h}_{\left[E_t\right]}\right\rangle\right),
\end{gathered}
$$
$<\cdot|\cdot>$ 表示拼接符，$b_{h},e_{h},b_{t},e_{t}$给出头实体和尾实体的标记，$\boldsymbol{h}_{\left[E_h\right]} \mid \boldsymbol{h}_{\left[E_t\right]}$ 表示$[E_{h}],[E_{t}]$的隐藏状态，$\phi$ 是tanh线性激活层，将拼接后的维度由2n降维为n

### 上下文蒸馏

对于无关特征即多余上下文，给出输出$h_1,\dots,h_n$使用可训练的查询代码q
$$
\begin{gathered}
\left(\alpha_1, \ldots, \alpha_n\right)=\operatorname{Softmax}\left(\boldsymbol{q} \cdot \boldsymbol{h}_1, \ldots, \boldsymbol{q} \cdot \boldsymbol{h}_n\right) \\
\boldsymbol{x}_i^*=\sum_{j=1}^n \alpha_j \cdot \boldsymbol{h}_j
\end{gathered}
$$
关系分类器不能根据无关特征来区分输入实例的关系，因此引入Gradient Reverse Lyer （GRL）梯度反向层，和优化器q来欺骗关系分类器。
$$
\begin{gathered}
\text { prob }_i=\operatorname{Softmax}\left(\operatorname{GRL}\left(\boldsymbol{x}_i^*\right) \cdot W+b\right) \\
\mathcal{L}_{c e, i}=\operatorname{CrossEntropy}\left(y_i, \text { prob }_i\right)
\end{gathered}
$$
其中 W 和 b 是关系分类器的权重和偏差。 xi* 在被输入分类器之前经过 GRL 层。 GRL 不影响前向传播，但通过乘以 −λ 来改变反向传播期间的梯度符号。
也就是说，随着训练的进行，分类器通过梯度下降来优化以减少$\mathcal{L}_{c e, i}$，而查询代码q通过梯度上升来优化以增加$\mathcal{L}_{c e, i}$,i，直到xi中不包含关系特征. 


而对上下文的蒸馏通过投影到特征空间来完成
对于给定句子表示$x_{i}$ 和关系无关特征$x^{*}$ ,
$$
\begin{gathered}
\hat{\boldsymbol{x}}_i=\operatorname{Proj}\left(\boldsymbol{x}_i, \boldsymbol{x}_i^*\right) \\
\operatorname{Proj}(\boldsymbol{a}, \boldsymbol{b})=\frac{\boldsymbol{a} \cdot \boldsymbol{b}}{|\boldsymbol{b}|} \cdot \frac{\boldsymbol{b}}{|\boldsymbol{b}|},
\end{gathered}
$$
$$x^{p}_{i}=x_{i}-\hat{x}_{i}$$ 



因为描述编码可以重复使用，所以计算复杂度由O(mn)降低为O(m+n)。（m，n表示描述和输入实例的数量）

## 匹配模块
匹配模块负责输入$x_{i}$和描述$d$之间的交互

实体匹配的得分直接由余弦相似度计算
$$\cos(\mathbf{x}_{i}^{h},\mathbf{d}^{h}),\cos(\mathbf{x}_{i}^{t},\mathbf{d}^{t})$$

而为了减少输入实例$x_{i}$中多余文本信息的影响，匹配文本表示$x_{i}$ 将输入一个蒸馏模块。蒸馏模块将$x_{i}$投影到相关特征的正交空间中，以得到更新后的匹配文本表示$x_{i}^{p}$ ,然后计算匹配得分，仍然是余弦相似度$cos(x_{i}^{p},d)$


匹配分数为：
$$
\begin{aligned}
s\left(x_i, d\right)= & \alpha \cdot \cos \left(\boldsymbol{x}_i^h, \boldsymbol{d}^h\right)+\alpha \cdot \cos \left(\boldsymbol{x}_i^t, \boldsymbol{d}^t\right) \\
& +(1-2 \cdot \alpha) \cdot \cos \left(\boldsymbol{x}_i^p, \boldsymbol{d}\right)
\end{aligned}
$$

$\alpha$ 是超参数，
使用margin loss
$$
\begin{gathered}
\delta_i=s\left(x_i, d_{y_i}\right)-\max _{j \neq y_i}\left(s\left(x_i, d_j\right)\right) \\
\mathcal{L}_{m, i}=\max \left(0, \gamma-\delta_i\right),
\end{gathered}
$$
$\gamma>0$ 是超参数，表示正对的匹配分数必须高于最接近的负对的匹配分数。 
最终的训练目标函数为
$$
\mathcal{L}=\frac{1}{N} \sum_{i=1}^N\left(\mathcal{L}_{c e, i}+\mathcal{L}_{m, i}\right)
$$


# 实验 

## 数据集

FewRel (Han et al., 2018) 是从维基百科收集并由众包工作者进一步手工注释的few-shot关系分类数据集，其中包含 80 个关系，每个关系包含 700 个句子

Wiki-ZSL（Chen 和 Li，2021）源自维基数据知识库，由 113 种关系类型的 93,383 个句子组成。与FewRel数据集相比，Wiki-ZSL具有更丰富的关系信息，但由于它是由远程监督生成的，因此原始数据中不可避免地存在更多噪声

随机选择5个关系为验证集，5/10/15个关系为测试集的未见关系，

## 对比模型
- R-BERT（Wu 和 He，2019b）。一种 SOTA 监督 RE 方法。继 Chen 和 Li（2021）之后，我们通过使用句子表示来执行最近邻搜索并生成零样本预测，使其适应零样本设置。 
- ESIM（Levy 等人，2017）。一种经典的基于匹配的 ZeroRE 方法，它使用 Bi-LSTM 对输入和标签描述进行编码。 ZS-BERT（Chen 和 Li，2021）。一种基于SOTA siamese的ZeroRE方法，采用BERT作为编码器，对输入和关系描述分别进行编码。除了分类损失之外，还使用基于度量的损失来优化表示空间以改进最近邻搜索。 
- PromptMatch（Sainz 等人，2021）。一种基于 SOTA 完全编码的 ZeroRE 方法，采用 BERT 对输入对的串联进行编码并对其细粒度语义交互进行建模。 
- REPrompt（Chia 等人，2022）。该基线是一种基于 seq2seq 的竞争性 ZeroRE 方法。它使用 GPT-2 生成这些关系的伪数据来微调模型。我们使用 NoGen 来表示没有数据增强的结果。



# 结果

分类损失只关注已知关系的区分，R-bert等监督方法再ZeroRE效果差，而siames scheme限制了输入和关系描述的字级别的交互，导致性能不佳，而本文通过细粒度匹配进行显式建模来提高效果。

和使用promptmatch，REPrompt这种完整的隐式建模对比，效果仍然好，可能原因是relational matching pattern，
作为一种归纳偏差，减少了训练集种可见关系的过拟合，因此泛化性能更好。

实验结果：

![[Pasted image 20231015132843.png]]

消融实验
![[Pasted image 20231015132823.png]]

更改encoder对数据集上的表现的影响
![[Pasted image 20231015133110.png]]


使用上下文蒸馏和归因技术来减少无关上下文的影响，实体匹配分数提供更多的信息

![[Pasted image 20231015133350.png]]

超参数gamma
![[Pasted image 20231015133727.png]]

持续增大，效果不差，模型鲁棒性好




# 改进

对于抽象关系，没有明确实体类型很难识别。
P460:said_to_be_the_same 
F1：0.03



