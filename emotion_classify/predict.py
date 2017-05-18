# -*- coding:utf-8 -*-

"""
@author: Jeffmxh
"""

from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
import gensim
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
import pymysql

def get_db_data(query_str):
        conn = pymysql.connect(host='127.0.0.1',port=3306,user='analyzer',password='analyzer@tbs2016',database='dp_relation',charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)
        cur=conn.cursor()
        doc = pd.read_sql_query(query_str, conn)
        doc = pd.DataFrame(doc)
        for column in doc.columns:
            column_null = pd.isnull(doc[column])
            column_null_true = column_null[column_null == True]
            if len(column_null_true) == len(doc):
                del doc[column]
        cur.close()
        conn.close()
        return doc

test_data = get_db_data("select * from weibo_raw_data where keyword_id='37_5'")
print('Data loaded from mysql...')
test_data.loc[0:10,'content']
comment = test_data

cw = lambda x: list(jieba.cut(str(x))) #定义分词函数
maxlen = 50
print('try loading pretrained word2vec model...')
wordvec_model = gensim.models.word2vec.Word2Vec.load('/home/da/nlp_repo/rnn_test/word2vec_wx/word2vec_wx')
wordvec_weight = wordvec_model.wv.syn0
vocab = dict([(k, v.index) for k, v in wordvec_model.wv.vocab.items()])
word_to_id = lambda word: not (vocab.get(word) is None) and vocab.get(word) or 0
words_to_ids = lambda words: list(map(word_to_id, words))
reverse_seq = lambda id_seq: id_seq[::-1]
concat_seq = lambda a,b: list(np.hstack((a, b)))
comment = comment[comment['content'].notnull()] #仅读取非空评论
comment['words'] = comment['content'].apply(cw) #评论分词
comment['sent'] = comment['words'].apply(words_to_ids)
comment['sent_rev'] = list(sequence.pad_sequences(comment['sent'], maxlen=maxlen, padding='pre', truncating='pre'))
comment['sent_rev'] = comment['sent_rev'].apply(reverse_seq)
comment['sent'] = list(sequence.pad_sequences(comment['sent'], maxlen=maxlen, padding='post', truncating='post'))
comment['sent'] = comment['sent'].combine(comment['sent_rev'], func=concat_seq)

print('Data ready, try loading pretrained model...')
	
with open('/home/jeffmxh/rnn_test/models/2gru_seven_senti_0509_train.json', 'rt') as f:
    json_string=f.read()
model = model_from_json(json_string)
model.load_weights('/home/jeffmxh/rnn_test/models/2gru_seven_senti_0509_train_weight.h5')

print('Model loaded, begin predicting...')
pred_x = np.array(list(comment['sent']))
pred_y = model.predict(pred_x)
comment['mark']=list(pred_y)
comment.loc[0:5,['content','mark']]

rev_trans_dict = dict(zip(range(8), ['none', 'disgust', 'like', 'happiness', 'sadness', 'surprise', 'anger', 'fear']))

def result_translate(prob_list):
    sign_dict = dict(zip(prob_list, rev_trans_dict.values()))
    result_emotion_code = sign_dict[max(prob_list)]
    return result_emotion_code

print('Translating emotion mark...')
comment['emotion_result'] = comment['mark'].apply(result_translate)
comment = comment.loc[:,['content','emotion_result']]

print('save result to excel...')
writer = pd.ExcelWriter('37_5predict_single_emotion.xlsx')
comment.to_excel(writer, sheet_name='senti', encoding = 'utf-8', index = False)
writer.save()
print('Task done.')