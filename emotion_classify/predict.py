# -*- coding:utf-8 -*-

"""
@author: Jeffmxh
"""

from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
import gensim
import argparse
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import jieba
jieba.enable_parallel(32)
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import Sequential, load_model, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras import backend as K
from pickle import dump, load
from train import load_word2vec

params = {
    'word2vec_path' : 'word2vec/word2vec_wx',
    'model_name':'lstm_seven_senti'
}

def load_predict_data(data_path, target_column = 'content'):
    comment = pd.read_excel(data_path)
    cw = lambda x: list(jieba.cut(str(x))) #定义分词函数
    maxlen = 50
    logger.info('try loading pretrained word2vec model...')
    wordvec_model, wordvec_weight = load_word2vec(params['word2vec_path'])
    vocab = dict([(k, v.index) for k, v in wordvec_model.wv.vocab.items()])
    word_to_id = lambda word: not (vocab.get(word) is None) and vocab.get(word) or 0
    words_to_ids = lambda words: list(map(word_to_id, words))
    reverse_seq = lambda id_seq: id_seq[::-1]
    concat_seq = lambda a,b: list(np.hstack((a, b)))
    comment = comment[comment[target_column].notnull()] #仅读取非空评论
    comment['words'] = comment[target_column].apply(cw) #评论分词
    comment['sent'] = comment['words'].apply(words_to_ids)
    comment['sent_rev'] = list(sequence.pad_sequences(comment['sent'], maxlen=maxlen, padding='pre', truncating='pre'))
    comment['sent_rev'] = comment['sent_rev'].apply(reverse_seq)
    comment['sent'] = list(sequence.pad_sequences(comment['sent'], maxlen=maxlen, padding='post', truncating='post'))
    comment['sent'] = comment['sent'].combine(comment['sent_rev'], func=concat_seq)
    return comment

def load_model(params):
    model_path = 'models/' + params['model_name'] + '_' + time.strftime('%m%d',time.localtime(time.time())) + '.json'
    weight_path = 'models/' + params['model_name'] + '_' + time.strftime('%m%d',time.localtime(time.time())) + '_weight.h5'
    with open(model_path, 'rt') as f:
        json_string=f.read()
    model = model_from_json(json_string)
    model.load_weights(weight_path)
    return model

def result_translate(prob_list):
    rev_trans_dict = dict(zip(range(8), ['none', 'disgust', 'like', 'happiness', 'sadness', 'surprise', 'anger', 'fear']))
    sign_dict = dict(zip(prob_list, rev_trans_dict.values()))
    result_emotion_code = sign_dict[max(prob_list)]
    return result_emotion_code

def main(file_path, column, outfile):
    '''
    定义logging格式
    '''
    logger = logging.getLogger('mylogger')  
    logger.setLevel(logging.INFO) 
    console = logging.StreamHandler()  
    console.setLevel(logging.INFO) 
    formatter = logging.Formatter('[%(levelname)-3s]%(asctime)s %(filename)s[line:%(lineno)d]:%(message)s')
    console.setFormatter(formatter)  
    logger.addHandler(console)  

    comment = load_predict_data(data_path = file_path, target_column = column)
    logger.info('Data ready, try loading pretrained model...')
    model = load_model(params)
    logger.info('Model loaded, begin predicting...')
    pred_x = np.array(list(comment['sent']))
    pred_y = model.predict(pred_x)
    comment['mark']=list(pred_y)    
    logger.info('Translating emotion mark...')
    comment['emotion_result'] = comment['mark'].apply(result_translate)
    comment = comment.loc[:,['content','emotion_result']]
    logger.info('save result to excel...')
    writer = pd.ExcelWriter(outfile)
    comment.to_excel(writer, sheet_name='senti', encoding = 'utf-8', index = False)
    writer.save()
    logger.info('Task done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='情感分析使用说明')
    parser.add_argument('-i', '--infile', dest='file_path', nargs='?', default='',
                        help='The absolute path to data.')
    parser.add_argument('-c', '--column', dest='column', nargs='?', default='content',
                        help='The column to deal with.')
    parser.add_argument('-o', '--outfile', dest='outfile', nargs='?', default='emotion_prediction.xlsx',
                        help='The name of output file, end with .xlsx.')
    args = parser.parse_args()
    main(file_path=args.file_path, target_column=args.column, args.outfile)