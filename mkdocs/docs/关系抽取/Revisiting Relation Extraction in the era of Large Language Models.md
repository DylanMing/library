---
tags:
  - NLP
  - RelationExtraction
  - LLM
---

Resource :: ACL2023


一句话描述 :: 用**大模型**进行关系抽取，GPT3**思维链微调**后Flan-T5模型达到SOTA效果
# Abstract

Relation Extraction : 

**关系提取**（RE）是从文本中推断实体之间语义关系的核心NLP任务。 标准监督的RE技术需要将培训模块标记为包括实体跨度的代币，然后预测它们之间的关系。 相反，最近的工作将问题视为**seq2seq**任务，将实体之间的关系线性化为在输入上生成的目标字符串。 在这里，我们使用更大的语言模型（GPT-3和Flan-t5大）来推动这种方法的限制，该模型比先前的工作中所考虑的，并在不同的监督水平下评估其在标准RE任务上的绩效。 我们解决了通过进行人体评估来代替确切匹配的人类评估来评估生成方法固有的问题。 在这项精致的评估下，我们发现：（1）几乎没有GPT-3的射击促使SOTA性能，即大致相当于现有的完全监督的模型； （2）Flan-T5在几个弹射设置中没有能力，而是用**思想链（COT）** 样式的解释（通过GPT-3生成）对其进行监督和微调会产生SOTA结果。 我们将此模型作为RE Tasks1的新基线


# motivation

关系抽取：
句子或文章中抽出 triple：`<entity,relation,entity>`

传统方法：实体识别+关系抽取
pipeline方法或joint方法，先进行实体识别，然后在闭集上进行关系的分类，或者同时进行实体识别和关系分类，类似1-stage或2-stage

S2S方法：
后出现sequence2sequence，视为输入句子序列，生成triple序列，将句子和triple都序列化，变成gernerate的问题，这样可以使用一些预训练模型如BART进行关系抽取任务。

大型预训练模型的出现带来了提升关系抽取效果的契机。


# 主要贡献

- **few-shot**的GPT-3可以达到全监督模型下的sota
- Flan-T5即使经过微调效果也不如sota，但是使用GPT-3生成的**思维链CoT**风格，可以达到Sota效果
- 关系抽取的**严格评估**对生成式关系抽取不适用，原先是闭集的分类，如果对生成式模型进行严格标签验证，会很多错误，使用人工来评估
# 实验数据集

|         | entity types | relation types | No. of relation triples  train | No. of relation triples  val | No. of relation triples  test |
| ------- | ------------ | -------------- | ------------------------------ | ---------------------------- | ----------------------------- |
| ADE     | 2            | 1              | 4272                           |                              |                               |
| CoNLL04 | 4            | 5              | 922                            | 231                          | 288                           |
| NYT     | 4            | 24             | 56196                          | 5000                         | 5000                          |
| DocRED  | 6            | 96             | 3008                           | 300                          | 700

**ADE** Adverse Drug Events (Gurulingappa et al., 2012) contains binary relations of `(drug, adverse event)` pairs. Drugs and adverse events are the only two entity types. This dataset provides a 10-fold split. 

**CONLL04** The CoNLL04 consists of sentences from news articles that were annotated for the mentioned entities and relations between entities (Roth and Yih, 2004). It includes **four entity types** `(PER, ORG, LOC, OTH)` and **five possible relations** `(KILL, WORK_FOR, LIVE_IN, LOCATED_IN, ORG_BASED_IN)`. 

**NYT** The NYT comprises sentences sampled from New York Times news articles published between 1987 and 2007 (Riedel et al., 2010). The data **was distantly annotated with relations triplets from FreeBase**. We use a processed version of NYT (Zeng et al., 2018) containing **three overlapping entity types** `(LOC, PER, ORG)` and **24 relation types**. 

**DocRED** Originally designed as a relation classification task, DocRED (Yao et al., 2019) differs considerably from the other datasets considered in this work in two important ways: (1) It comprises **long texts** which feature relations between entities at a **document-level**; (2) It contains annotations for **6 entity types** and **96 relation types**, with **an average of 19.9 entities** and **19.5 relation instances per document**.

四个数据集从小到大
（文档级关系抽取？）

# 实验

实验主要分为两个部分：
- 在GPT-3上的**zero-shot**和**few-shot**，（使用prompt）
- 在Flan-T5上的**zero-shot**，**few-shot**（使用prompt）和使用GPT-3生成的CoT（思维链）进行微调


# GPT-3上的实验

在GPT-3上的实验通过**prompt微调**来实现，给模型输入一段上下文示例和输出的示例，用于prompt微调。
输入和输出都是序列合集：
**Input** Bill Nelson, NASA administrator announced the mars mission today. 
**Target** `[(Bill Nelson:Per, Work_For, NASA:Org)]`


下面是各个数据集的prompt示例：

>[!note] 
>这里的explanation属于CoT思维链部分，在GPT-3上的实验没有使用，而是在Flan-T5实验中使用。

## ADE数据集上的prompt

在ADE数据集上，给出一个**Example Instructional Prefix**，用于提示性描述任务。
然后从训练集中随机选择了12个例子，每个例子都跟在**Example Instructional Prefix**后面，包含输入和输出。
12个例子共755个tokens。

对于新序列的预测，我们使用相同的**Example Instructional Prefix** ，要求其生成目标序列。
sampleling temperature设置为0.5，最长输入序列为256个tokens。

**Example Instructional Prefix**: List all `[drug, adverse effects]` pairs in the TEXT provided below.

**TEXT**: CD4 T-lymphocyte depletion, myelosuppression, and subsequent severe infections are the major side effects of fludarabine phosphate therapy. 

**Relations**: `[[’fludarabine phosphate’, ’CD4 T-lymphocyte depletion’], [’fludarabine phosphate’, ’myelosuppression’], [’fludarabine phosphate’, ’severe infections’]]` 

**Explanation**: Following major side-effects are known of fludarabine phosphate therapy, CD4 T-lymphocyte depletion, myelosuppression, and severe

## CONLL04数据集上的prompt

在CONLL04上的promp和ADE上的类似，同样使用12个例子进行训练，prompt包含960 tokens

**Examplee Instructional Prefix**: List the relations of the types `[OrgBased In, Work For, Located In, Live In, Kill]` among the entities `[PERSON, LOCATION, ORGANIZATION, OTHER]` in the given text and provide a reasonable explanation.`<s>`

**TEXT**: Meanwhile, Shi Liming at the Institute of Zoology of Kunming found that pandas lack variety in their protein heredity, which may serve as one of the major reasons for pandas’ near extinction. 
**Relations**: `[[’Shi Liming:Per’, ’Work For’, ’Institute of Zoology:Org’], [’Institute of Zoology:Org’, ’OrgBased In’, ’Kunming:Loc’]]` 
**Explanation**: Shi Liming works for the Institute of Zoology, which is an organization based in Kunming.`<s>`

## NYT数据集上的prompt

因为NYT中包含24种关系，所以不能通过**Examplee Instructional Prefix**将所有关系列出来，因此在**Examplee Instructional Prefix**中不再罗列关系，然后使用20个例子进行微调。prompt包含2095个tokens

**TEXT**: Quebec, Canada’s second most populous province, after Ontario, has not decided to go that far. 
**Relations**: `[[’Ontario:Loc’, ’/location/administrative-division/country’, ’Canada:Loc’], [’Canada:Loc’, ’/location/location/contains’, ’Ontario:Loc’], [’Canada:Loc’, ’/location/country/administrative-divisions’, ’Ontario:Loc’]]` 
**Explanation**: Ontario is a place located in the administrative divisions of the country Canada. Quebec is Canada’s second most populous province and hence, Canada is a place that contains Quebec.`<s>`

## 实验结果

### 结果评估问题

错误问题：
GPT生成的paires/triples在实际上如果按照严格对应的关系进行评估，会出现一些错误现象。比如将原本为真阴性的例子被分类为假阴性（ADE例子），真阳性被分为了假阳性（NYT例子），或者是产生了一些在原先闭集外的关系，这种关系与示例triple中的关系有强关联，比如示例为kill，生成关系为 shot_by。这为评估带来了麻烦。

![[Pasted image 20230924050126.png]]


在ADE数据集上，51.67%的假阳性其实是真阳性，32.61%的假阴性实际上是真阴性
在CoNLL数据集上，50.27%的假阳性是真正分类正确的，36.6%的假阴性实际上是准确的

而在NYT数据集上，因为prompt长度的问题，将在CoT微调的Flan-T5上进行评估。36.9%的假阳性和22.97%的假阴性是准确的。

>[!question] 
>这里感觉他的写法有点怪，两个数据集上没有用同一种描述方法，不确定是否为笔误。


![[Pasted image 20230924050103.png]]


REBEL是将RE转化为S2S的生成任务的模型，使用BART作为基准，进行预训练和微调。为此前全监督RE的SOTA。
可以看到使用GPT-3进行prompt微调后的模型可以**在CONLL和ADE两个数据集上达到SOTA效果**，但是在**NYT上的效果比SOTA下降了近30个点**，因为在NYT上的prompt没有将可能的关系列出来，因为token长度过长。虽然多使用了20个train example，但是效果很差。同时产生了10.6%的无效序列和空序列，这也是prompt中没有足够信息的大模型生成式关系抽取的劣势。

>[!attention] 
>在长文本或者大量目标的情况下，大模型无法在prompt中进行结构化的问题描述



>[!note] 
>此处可能是可以改进的点，大模型在多关系分类中由于无法在prompt中将所有可能关系罗列出来，因此分类效果差，通过传统微调或者CoT思维链等方式改进其效果。


# Flan-T5上的实验


Flan-T5，（后简称T5），使用和GPT-3上相同的结构化提示和prompt，但是因为T5的参数量更少，可接收的prompt长度更短，因此减少了输入示例的数量。

ADE上的结果
示例数量由GPT-3的12个减少为7个，输出的不合格关系的数量增加了约13.9%
包含重复产生同一组token或者生成的关系序列中超过或少于两个实体。
F1分数比GPT减少20


CoNLL上的结果
CoNLL同样将示例减少为7个，不合格关系数量增加约12.5%。
同时会产生大量超过闭集中的关系，导致其评估产生困难。

NYT上无法输入结构化提示的prompt，因此无法类似实验。但是在后面使用微调解决了这个问题。


微调
使用可用数据集进行全监督微调，效果在表中1.e表示，可以达到SOTA效果
为了提升微调的效果，使用GPT3得到的CoT思维链进行微调，包括对关系抽取的explanation。
然后将思维链作为附加文本输入GPT和T5中进行生成。

![[Pasted image 20230930150545.png]]

使用思维链对GPT进行微调，如图所示，
GPT3在ADE和CoNLL上的micro-F1分数分别提升了3和2.2
同时，不合格关系的产生也大幅度减少。
ADE:13.9%-0.8%
CoNLL:12.5%-1.1%
同时，CoNLL上的超出闭集关系由120个减少到1个


Flan-T5上使用思维链进行微调，结果为表中f，效果超过了sota


上面的实验使用标准标签和GPT生成的思维链对Flan-T5进行微调，现在尝试使用GPT生成的标签和思维链对T5进行微调，来对实验进行进一步改进。而这样需要GPT在整个数据集上进行推理，因为仅在CoNLL进行实验。
使用12个例子对GPT进行微调，然后再CoNLL剩下的1557个例子上进行推理产生标签和explanation的思维链。

这种方法优于现有的全监督方法，但是不如使用标准标签的效果结果为表中 2.c

未来方向：使用模型进行自动评估？BERT分类器

限制：
- 避开了复杂RE数据集，比如n元关系的，也不能在长文本多关系的数据集上运行，因为prompt长度
- 没有评估生成的思维链解释的质量，可能对模型表现有影响
- 不能在GPT上微调使用思维链，权重无法保存本地运行。
- 仅在英文数据集上测试，没有在其他语言上测试。

