## Cora数据集

[TOC]

### 参考链接

+ [数据集下载链接](https://linqs.soe.ucsc.edu/data)

### 数据集介绍

Cora数据集由许多机器学习领域的paper构成，这些paper被分为7个类别：

+ Case_Based
+ Genetic_Algorithms
+ Neural_Networks
+ Probabilistic_Methods
+ Reinforcement_Learning
+ Rule_Learning
+ Theory

在该数据集中，每一篇论文至少引用了该数据集里面另外一篇论文或者被另外一篇论文所引用，数据集总共有2708篇papers。

在消除停词以及除去文档频率小于10的词汇，最终词汇表中有1433个词汇。

数据集文件夹中包含两个文件：

1. `.content`文件包含对paper的内容描述，格式为
   $$
   \text{<paper_id>  <word_attributes> <class_label>}
   $$
   其中

   + `<paper_id>`是paper的标识符，每一篇paper对应一个标识符。
   + `<word_attributes>`是词汇特征，为0或1，表示对应词汇是否存在。
   + `<class_label>`是该文档所述的类别。

2. `.cites`文件包含数据集的引用图(citation graph)，每一行格式如下
   $$
   \text{<ID of cited paper> <ID of citing paper>}
   $$
   其中

   + `<ID of cited paper>`是被引用的paper标识符。
   + `<ID of citing paper>`是引用的paper标识符。

   引用的方向是从右向左的，比如有一行为`paper1 paper2`，那么对应的连接关系是`paper2->paper1`。

