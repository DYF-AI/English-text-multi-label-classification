# English-text-multi-label-classification

Dear,   

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.   

In this assignment, you are challenged to build a multi-label classification model for detecting different types of toxic comments like threats, obscenity, insults, and identity-based hate. You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. Please send back the answers in 48 hours. The types of toxicity are:
toxic
severe_toxic
obscene
threat
insult
identity_hate

You must create a model which predicts a probability of each type of toxicity for each comment.


Good luck!   
Best regards,


1、项目分析
对于一个多标签分类问题，即一个样本可能会对应对个label。
我的第一个想法是将多个标签组合的形式转为one-hot形式，然后对one-hot后的label建立一个多分类模型，即将多标签分类模型转换为多分类模型，然而当标签的类别个数太多时，转成one-hot编码形式，需要的one-hot维数太大，这种方法有点行不通。
我的第二个想法是将多标签分类问题转换为多个二分类问题，即对每个标签进行一个二分类模型。对每个数据和label分别进行训练，最后跟多多个模型的结果组合成一个多标签分类模型，实验中我是基于第二种方法。

									图1 模型结构
2、样本的正负类别简要分析：总样本数量=159571
	0	1
toxic	144277（90.415%）	15294（9.585%）
severe_toxic	157976（99.000%）	1595（1.000%）
obscene	151122（94.705%）	8449（5.295%）
threat	159093（99.700%）	478（0.300%）
insult	151694（95.064%）	7877（4.936%）
identity_hate	158166（99.120%）	1405（0.880%）
结果分析：
从样本的正负类别上看，样本标签为0的数量所占的比例要远远高于标签为1的样本。对于一个二分类问题，正负样本比例较大，模型很容易被训练成预测较大占比的类别，因此在这种情况下，如果采用二分类的方法，必须上下采样平衡正负样本的比例。

3、文本特征分析
（1）对整个问题的样本都进行正则化处理，得到的单词的数量为149998个，也就是完整表达整个数据集所需的词向量个数为149998个。如果全部进行向量变换，所需的内存比较大，因此我们可以在训练前提取部分样本进行词向量转换，可以降低内存的需求。
（2）在正则化后，可以计算每段话所包含的词的个数。记过分析，一个句子包含的单词数量最多为4951个，即如果完整使用词向量表达所有单词，需要4951维的词向量。正对这个问题，我们可以过滤掉句子中一些介词、代词，可以大幅度降低句子中单词的数量。
我的第一个想法是可以通过判断词性来过滤掉一些介词、代词；
我的第二个想法比较简单，而且实现比较容易，即可以通过判断单词的字符长度来过渡掉一些单词，如我在实验中过滤长度小于等于3、或者单词的长度大于13的单词，其因为可以理解为小于等于3的单词很大概率是介词、代词等，类似于高斯分布，我们通过单词的长度对两端的数据进行一个截尾。
由于时间比较有限，实验中，我使用第二种方法。

4、实验过程及实验设备

实验流程：
（1）对数据集进行分析及预处理；
（2）对数据进行上下采样平衡正负样本比例，通过一些方法过滤一些无用词语，对得到的数据进行词向量转换；
（3）将处理好的数据划分数据集，数据集划分为训练集、验证集、测试集；
（4）分别对各个标签建立二分类模型，如图1所示，最后组合成一个多分类标签模型。
实验结果简要截图：


实验结果说明：
（1）由于时间比较紧，实验代码部分实现了关于label toxic的分类，其它label模型类似toxic分类。
（2）实验设备：使用了实验室电脑，配置如下：
	类别	型号
1	CPU	i7 8700
2	GPU	GTX 1080
3	内存	32G

实验文件：
1、Data_preprocess.py: 数据预处理
2、main.py: 模型训练（主程序）
3、text_cnn_rnn.py: 文本处理模型
