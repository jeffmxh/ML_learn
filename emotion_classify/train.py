# -*- coding:utf-8 -*-
'''
构建LSTM网络训练情感分类模型

@author: Jeffmxh
'''

from __future__ import absolute_import #导入3.x的特征函数
from __future__ import print_function
import pandas as pd #导入Pandas
import keras
import os
import numpy as np #导入Numpy
import jieba #导入结巴分词
jieba.enable_parallel(32)
import logging
import gensim
import sys
import time
reload(sys)
sys.setdefaultencoding('utf-8')

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras import backend as K
from keras import metrics
from pickle import dump
from collections import namedtuple

'''
参数设置
'''
class Params():
    def __init__(self, rnn_type):
        if rnn_type not in ['lstm', 'bilstm', 'gru']:
            raise KeyError
        self.num_classes = 8
        self.maxlen = 50
        self.batch_size = 16
        self.epochs = 10
        self.layer = rnn_type
        self.train_data_path = 'train_data/train_data.xlsx'
        self.word2vec_path = 'word2vec/word2vec_wx'
        self.model_name = 'lstm_seven_senti_prewordvec'
        self.embedding_train = False
        self.thread = 16
        self.dict_len = 0
    def get_dict_len(self, dict_len):
        self.dict_len = dict_len

Train_Set = namedtuple('Train_Set', 'x y xt yt xa ya')

'''
转化情感标记
'''
def trans_emo(emo):
    trans_dict = dict(zip(['none', 'disgust', 'like', 'happiness', 'sadness', 'surprise', 'anger', 'fear'], range(8)))
    return trans_dict[emo]

'''
通过gensim加载预训练的word2vec模型
'''
def load_word2vec(model_path):
    try:
        wordvec_model = gensim.models.word2vec.Word2Vec.load(model_path)
        wordvec_weight = wordvec_model.wv.syn0
    except:
        wordvec_model = ''
        wordvec_weight = ''
        print('Pretrained word2vec model not found, use fresh params...')
    return wordvec_model, wordvec_weight

'''
训练数据预处理过程
'''
def preprocess_data(wordvec_model, params, logger):
    if wordvec_model:
        raw_data = pd.read_excel(params.train_data_path)
        logger.info('Data loaded!')
        data = pd.DataFrame({'sent' : raw_data.sentence,
                             'mark' : raw_data.emotion_1 })
        data['mark'] = data['mark'].apply(trans_emo)
        logger.info('emotion_tag transformed!')
        cw = lambda x: list(jieba.cut(str(x))) #定义分词函数
        data['words'] = data['sent'].apply(cw)
        vocab = dict([(k, v.index) for k, v in wordvec_model.wv.vocab.items()])
        word_to_id = lambda word: not (vocab.get(word) is None) and vocab.get(word) or 0
        words_to_ids = lambda words: list(map(word_to_id, words))
        data['sent'] = data['words'].apply(words_to_ids)
        reverse_seq = lambda id_seq: id_seq[::-1]
        concat_seq = lambda a,b: list(np.hstack((a, b)))
        logger.info("Pad sequences (samples x time)...")
        data['sent_rev'] = list(sequence.pad_sequences(data['sent'], maxlen=params.maxlen))
        data['sent_rev'] = data['sent_rev'].apply(reverse_seq)
        data['sent'] = list(sequence.pad_sequences(data['sent'], maxlen=params.maxlen, padding='post', truncating='post'))
        data['sent'] = data['sent'].combine(data['sent_rev'], func=concat_seq)
    else:
        raw_data = pd.read_excel(params.train_data_path)
        logger.info('Data loaded!')
        data = pd.DataFrame({'sent' : raw_data.sentence,
                             'mark' : raw_data.emotion_1 })
        data['mark'] = data['mark'].apply(trans_emo)
        logger.info('emotion_tag transformed!')
        cw = lambda x: list(jieba.cut(str(x))) #定义分词函数
        data['words'] = data['sent'].apply(cw)
        w = [] #将所有词语整合在一起
        for i in data['words']:
            w.extend(i)
        dict_w = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
        dict_w['id']=list(range(1,len(dict_w)+1))
        dump(dict_w, open('dict_w.pickle', 'wb'))
        params.get_dict_len(len(dict_w)+1)
        get_sent = lambda x: list(dict_w['id'][x])
        data['sent'] = data['words'].apply(get_sent)
        logger.info("Pad sequences (samples x time)...")
        data['sent'] = list(sequence.pad_sequences(data['sent'], maxlen=params.maxlen, padding='post', truncating='post'))
    return data, params

'''
切分训练集和测试集
'''
def split_data(train_data, params):
    x = np.array(list(train_data['sent']))[::2] #训练集
    y = np.array(list(train_data['mark']))[::2]
    y = keras.utils.to_categorical(y, params.num_classes)
    xt = np.array(list(train_data['sent']))[1::2] #测试集
    yt = np.array(list(train_data['mark']))[1::2]
    yt = keras.utils.to_categorical(yt, params.num_classes)
    xa = np.array(list(train_data['sent'])) #全集
    ya = np.array(list(train_data['mark']))
    return Train_Set(x, y, xt, yt, xa, ya)

'''
构建keras模型
'''
def build_model(wordvec_weight, params, logger):
    if wordvec_weight!='':
        word_embedding_layer = Embedding(
            input_dim=wordvec_weight.shape[0],
            output_dim=wordvec_weight.shape[1],
            weights=[wordvec_weight],
            trainable=params.embedding_train)
    else:
        word_embedding_layer = Embedding(params.dict_len+1, 256)
    logger.info('Build model...')
    model = Sequential()
    model.add(word_embedding_layer)
    model.add(Dropout(0.1))
    if params.layer=='lstm':
        model.add(LSTM(128, return_sequences = False))
    if params.layer=='bilstm':
        model.add(Bidirectional(LSTM(128, return_sequences = False))) 
    if params.layer=='gru':
        model.add(GRU(128, return_sequences = False))
    model.add(Dropout(0.5))
    model.add(Dense(params.num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.mae, metrics.categorical_accuracy])
    return model
    
def main():
    params = Params('lstm')
	if not os.path.isdir('log'):
        os.mkdir('log')
    '''
    设定日志文件格式
    '''
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    # 建立一个filehandler来把日志记录在文件里，级别为debug以上
    fh = logging.FileHandler('log/' + params.model_name + "_train.log")
    fh.setLevel(logging.DEBUG)
    # 建立一个streamhandler来把日志打在CMD窗口上，级别为info以上
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 设置日志格式
    formatter = logging.Formatter('[%(levelname)-3s]%(asctime)s %(filename)s[line:%(lineno)d]:%(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    #将相应的handler添加在logger对象中
    logger.addHandler(ch)
    logger.addHandler(fh)

    '''
    开始训练
    '''
    logger.info('try loading pretrained word2vec model...')
    wordvec_model, wordvec_weight = load_word2vec(params.word2vec_path)
    data_all, params = preprocess_data(wordvec_model, params, logger)
    train_data = split_data(data_all, params)
    model = build_model(wordvec_weight, params, logger)
    model.fit(train_data.x, train_data.y, batch_size=params.batch_size, epochs=params.epochs, validation_data=(train_data.xt, train_data.yt))
    model_path = 'models/' + params.model_name + '_' + time.strftime('%m%d',time.localtime(time.time())) + '.json'
    weight_path = 'models/' + params.model_name + '_' + time.strftime('%m%d',time.localtime(time.time())) + '_weight.h5'
    logger.info('模型存储路径: ' + model_path)
    logger.info('模型权重存储路径: ' + weight_path)
    json_string = model.to_json()
    with open(model_path, 'wt') as f:
        f.write(json_string)
    model.save_weights(weight_path)

if __name__ == '__main__':
	main()