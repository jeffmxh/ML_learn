[项目主页](https://jeffmxh.github.io/ML_learn/)

* 目录
  * [Emotion_Classify](#emotion_classify)
  * [MNIST_GAN](#mnist_gan)

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

***

**致谢**：特别感谢[ALEX](https://github.com/alexwwang)以及南京巴兰塔信息科技有限公司在我完成此项目过程中对我的帮助以及技术支持！

MNIST_GAN
-----------------------------
# 利用GAN训练一个生成MNIST图片的神经网络

## 数据

数据来自[tensorflow](https://github.com/tensorflow/tensorflow)中自带的
[read_data_sets](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py)函数,
如果指定的路径下没有相应文件则会自动下载。也可以从[Yann Lecun](http://yann.lecun.com/exdb/mnist/)的主页直接获取，
这里的**MNIST_data**就是我使用上述函数直接下载的

## 用法

直接运行整个脚本就会自动载入数据并且开始训练过程，每个epoch会自动存储一下模型权重，并且生成一些数字0-9的图片以供检验训练效果，这个训练过程在CPU上
较慢，我在i7-6700上训练一个epoch并完成测试大概需要150分钟，不过由于模型主体是卷积网络所以在GPU上的性能一定会有大幅提升。从目前的效果来看5个epoch
以后能看出数字的轮廓，从第十个epoch以后基本稳定，虽然也会随着输入随机数的变化有一些扰动，不过整体上趋于稳定，我训练到35个epoch，后期变化都不大，偶尔会
出现一些噪点。

## 原理简述

完整的原理当然要看这篇的Ian Goodfellow的[论文](https://arxiv.org/abs/1406.2661)，不仅详细的介绍了模型思想，求解方法，还证明了其收敛性。
简要的来说，他提供了一种生成模型的新思路，与传统的图像生成所使用的PixelRNN或PixelCNN抑或是Variational Autoencoders不同，GAN不仅学习原始数据的
分布模式，而且在模型中实现一种左右互搏的结构，生成器负责根据输入的一个随机数以及图片类别生成一副图片，然后把这些图片和真实的训练数据混合，交给判别器，
判别器需要同时分辨图片的类别，以及他的“真假”。在这个过程中，生成器的训练目标是尽可能的欺骗判别器，而判别器的训练目标是尽可能分辨出生成器生成的“假图片”，
这样就出现了一个对抗的训练过程，所以取名Adversarial。在训练的过程中一般会控制两个网络的更新周期不同改善训练效果。最终就得到了两个训练好的网络，一个
用来完成生成任务，另一个则可以做分类任务。在这里我做的是用于训练MNIST的网络，当然也可以自然的调整结构用于Imagenet的数据(有GPU的高玩可以去试试)，并且是
因为是图像任务所以我自然的选择使用了卷积网络，其实从代码中可以看到构建网络的过程就是平凡的卷及网络。
