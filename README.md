[项目主页](https://jeffmxh.github.io/ML_learn/)

* 目录
  * [Emotion_Classify](#emotion_classify)

Emotion_Classify
-----------------------------
# 利用RNN进行情感分析

## 简介：

利用[Keras](https://keras.io/)框架搭建RNN神经网络对文本数据进行情感分析，注意此处情感分析并非单独的褒贬，而是包含以下8个方面：
+ none
+ disgust
+ like
+ happiness
+ sadness
+ surprise
+ anger
+ fear

所以该模型更适合与对描述类的文本进行分析，而非**评价类**。

***

## 依赖

+ keras
+ gensim
+ pandas
+ numpy
+ jieba
+ collections
+ argparse

***

## 使用方法：

### 训练集数据

此处的训练数据采用[NLPCC2013](http://tcci.ccf.org.cn/conference/2013/pages/page04_tdata.html)和[NLPCC2014](http://tcci.ccf.org.cn/conference/2014/pages/page04_sam.html)给出的微博情绪分析测试数据，原始数据格式为xml，解析后得到的``train_data.xlsx``位于``train_data``目录下，如需要用其他的数据进行训练只需保存成格式同的excel文件即可

### 训练模型

训练模型时，只需要在命令行运行
```
python train.py
```
模型的名称定义在``train.py``的``Params()``类中，在输入的名称后会自动补全一个时间戳，训练结束时会在命令行打印相应的模型名称以及存储路径。训练用的数据默认是``train/train_data.xlsx``，如需要使用其他的训练数据时请把数据reshape成相应格式，或调整``preprocess_data``函数读取相应的数据。当模型拟合完毕后会将生成的模型保存在``models/``目录下，同时在``log/``目录中生成一个日志文件记录程序运行情况

#### 参数修改

~~在模型的构建过程中RNN层可以选择**LSTM**，**BiLSTM**，**GRU**三种，可以通过修改``train.py``中初始化params的参数实现，例如：~~
```python
params = Params('lstm') # 采用LSTM层
```

***

**更新：**

扩展加入了多种新型模型结构，定义在``train.py``中的``build_model函数中``，包括传统的**RNN**模型和**CNN**模型，还有**ConvRnn**模型，以及**mutate**用于自定义模型结构。在最新的测试集中，**CNN**达到了75%的准确率，**GRU**达到了79%的准确率，由于扩充了训练集所以后期可以考虑进一步增加模型的复杂度，增加更多的层等等。

其他定义模型时可修改的参数都位于文件开始的``Params()``类中，可以根据需求进行修改

#### 关于词向量

在训练模型是推荐使用预训练好的词向量，一方面可以缩短训练所需时间，并且还可以防止出现未登录词的情况，这里我使用了[科学空间](http://spaces.ac.cn/archives/4304/)微信文章预训练的词向量，可以去相应链接下载解压后保存至``word2vec``目录下，当然如果使用了自行训练的词向量只需修改脚本中载入词向量的路径即可

### 进行预测

当训练好模型后，在``predict.py``中修改``params``字典中的模型路径为想调用的模型，即可使用此模型进行训练。在命令行下运行：

```
python predict.py -h
```
可以查看接受的各个参数
```
python predict.py -i '/home/jeffmxh/example.xlsx' -c 'content' -o 'emotion_output.xlsx'
```
+ ``-i/--infile``: 输入文件的路径
+ ``-c/--column``: 待处理的列名
+ ``-o/--outfile``: 输出文件的名

在``predict.py``的``params``字典中可以修改调用的模型路径，可以根据训练的情况调用需要的模型进行预测

***

## 注意：

以上所有任务均在**python2.7**和**Keras2.0.2**下完成，由于**Keras**升级过程中API变化较大建议升级到此版本
