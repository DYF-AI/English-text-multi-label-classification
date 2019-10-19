#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import re
import sys
import json
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
import gensim as gs
import nltk, re, string
from pprint import pprint
from collections import Counter
from tensorflow.contrib import learn


# ## 一些想法
# 1、显然多分类样本的比例极不平衡，可能需要抽样；<br />
# 2、在文本中提取出一些关键词；<br />
# 3、利用关键词，进行词变量转换；<br />
# 4、问题的难点在于，一个样本对应多个标签，因此不能直接使用one-hot编码label；7个标签['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'other']， 共5040种排列方式，显然是不合理的 <br />
# 5、或者是做多次二分类操作，即每一个标签都做一个二分类，即对于toxic做一次二分类，判断其是否为toxic，以此类推做其它的便签的二分类，感觉可以，将多分类任务转换为多次的二分类和决策树有点类似 <br />

# ## 思路
# 1、对文本进行预处理、清理一些无用的标点符号；<br />
# 2、将文本数据转换为词向量；<br />
# 3、将处理好的数据划分数据集，训练集和验证集；<br />
# 4、建立模型进行训练，将多分类任务转为多个二分类的任务，避免由于分类任务个数太多，onehot编码的类别次数太多，如7个特征的组合个数为5040=7\*6\*5\*4\*3\*2\*1 <br />

# In[7]:


# Author:DYF2019-10-14-17:29

def clean_str(s):
	s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
	s = re.sub(r" : ", ":", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	return s.strip().lower()

def clean_text(text):
    ##print(text)

    # Remove puncuation 去除标点
    text = text.translate(string.punctuation)

    # Convert words to lower case and split them
    text = text.lower()

    # Remove stop words
    # text = text.split()
    # stops = set(stopwords.words("english"))
    # text = [w for w in text if not w in stops and len(w) >= 3]

    # text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)  # 除A-Za-z0-9(),!?'`外的字符，去除
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\!", "!", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\+", "+", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\-", "-", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"\:", ":", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # text = text.split()
    # stemmer = SnowballStemmer('english')
    # stemmed_words = [stemmer.stem(word) for word in text]
    # text = " ".join(stemmed_words)

    #print(text)
    #print("")
    return text

def load_embeddings(vocabulary):
    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, len(vocabulary))  ## 300 ？？？
    return word_embeddings

def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    """Pad setences during training or prediction""" ## 训练或预测期间的PAD设置，句子的长度
    if forced_sequence_length is None: # Train
        sequence_length = max(len(x) for x in sentences)
    else: # Prediction
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum length is {}'.format(sequence_length))

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        if num_padding < 0: # Prediction: cut off the sentence if it is longer than the sequence length
            logging.info('This sentence has to be cut off because it is longer than trained sequence length')
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences
    
def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def clean_str_length(str): # 删除x中，字符长度小于等于三的字符
    ## 如果字符长度小于三则过滤掉
    ans = []
    for i in range(len(str)):
        if len(str[i]) < 4 or len(str[i]) > 13:
            continue
        else:
            ans.append(str[i])
    return ans

def load_data(filename):
    #df = pd.read_csv('toxic_comments.csv')
    ## 不需要使用所有的数据数据
    data = pd.read_csv(filename)
    #data = data[:10000]   ## 一次性把所有数据放到内存，直接内存溢出。
    #data = data[:20000]

    #print(data.describe())
    
    column_sum = data.iloc[:,7].sum()
    print("cilumn_sum", column_sum)
    
    selected = ['comment_text','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    non_selected = list(set(data.columns) - set(selected))  # 没有被选中的，即无用数据
    
    data = data.drop(non_selected, axis=1)  ## 删除无用数据
    #print(data.describe())
    

    
    data = data.dropna(axis=0, how='any', subset=selected)  ## 过滤数据中的缺失数据
    #data = data.reindex(np.random.permutation(data.index))
    #labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'other']
    #print('labels', labels)


    ## 使得正负样本个数一样
    # data = data.loc[(data['toxic'] == 1)]  ## 找出标签为1的数据
    data1 = data.loc[(data['toxic'] == 1)]  ## 取值标签值为1的样本
    data_0 = data.loc[(data['toxic'] == 0)]
    data2 = data_0.sample(data1.shape[0])
    data_concat = [data1, data2]
    data = pd.concat(data_concat)
    print('data.shape', data1.shape)
    print('data.shape', data2.shape)
    print("#########", data[0:10])

    print('data.shape', data.shape)
    
    labels = ['Yes', 'No']  ## 我们是分别对各类进行处理，每个类的标签均为0,1
    # 多分类使用one-hot编码
    '''
    num_labels = 2#len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)  ## 将onehot对角元素设置为1
    label_dict = dict(zip(labels, one_hot))
    print('label_dict', label_dict)
    '''
    
    #x_raw= data[selected[0]].apply(lambda x: clean_str(x).split(' ')).tolist()  ## 对输入的字符串进行以“”空格划分
    x_raw= data[selected[0]].apply(lambda x: clean_text(x).split(' ')).tolist()

    x_raw_clean = []
    for i in range(len(x_raw)):
        x_raw_clean.append(clean_str_length(x_raw[i]))
    print('x_raw_clean', x_raw_clean[:10])
    print('x_raw', x_raw[:10])
    x_raw = x_raw_clean


    print('x_raw', x_raw)
    print('xraw',x_raw[:10])

    x_raw = x_raw_clean  ## x_raw_clean过滤了字符串长度小于等于3的字符，其它的和x_raw一样
    
    y_raw_toxic = data[selected[1]].apply(lambda y: [1,0] if y==1 else [0,1]).tolist() ##########
    #y_raw = data[selected[1]].apply(lambda y: label_dict[y]).tolist()   ## 对selected[0]进行onehot编码
    ## 给每个样本打上标签（一个样本可能有多个标签），从简单做起，先假设每个类对于一个标签。
    #y_raw_toxic = data[selected[1]].apply(lambda x: x if x==1 else 0)
    print('y_raw', y_raw_toxic[:7])
    y_raw_severe_toxic = data[selected[2]].apply(lambda x: x if x==1 else 0)
    y_raw_obscene = data[selected[3]].apply(lambda x: x if x==1 else 0)
    y_raw_threat = data[selected[4]].apply(lambda x: x if x==1 else 0)
    y_raw_insult = data[selected[4]].apply(lambda x: x if x==1 else 0)
    y_identity_hate = data[selected[4]].apply(lambda x: x if x==1 else 0)
    
    #y_raw_toxic = y_raw_toxic.tolist()
    y_raw_severe_toxic = y_raw_severe_toxic.tolist()
    y_raw_obscene = y_raw_obscene.tolist()
    y_raw_threat = y_raw_threat.tolist()
    y_raw_insult = y_raw_insult.tolist()
    y_identity_hate = y_identity_hate.tolist()
    
    x_raw = pad_sentences(x_raw, forced_sequence_length=1000)  # The maximum length is 4951,这也太大了吧，还需要提取关键词才行，下一步在说...
    vocabulary, vocabulary_inv = build_vocab(x_raw)
    print("vocabulary", vocabulary)  # 为每一个单词创建一个编号（共有149998个单词）
    print("vocabulary", vocabulary_inv)
    
    x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
    y = np.array(y_raw_toxic)
    print("x", x[0])
    print('x.shape', x.shape)

    return x, y, vocabulary, vocabulary_inv, data, labels ######## y
    #print('y_raw', y_raw_toxic)
    #print(data.loc[[2,3],['severe_toxic', 'obscene']]) 
    
datafile = 'toxic_comments.csv'
load_data(datafile)








# In[ ]:




