
[# AI芯片涉及哪些知识？【AI芯片】内容简介](https://github.com/chenzomi12/AISystem/tree/main)
[深入GPU原理：线程和缓存关系【AI芯片】GPU原理01](https://www.bilibili.com/video/BV1bm4y1m7Ki?spm_id_from=333.1245.0.0)
https://github.com/chenzomi12/DeepLearningSystem/tree/main/Hardware


GPU :  Graphic Processing Units 原本用于图像视频处理等。


GPU vs CPU：
GPU的设计目标是 最大化**吞吐量（Throughout）**，比单任务执行快慢，更关心


# AI计算模式
## 深度学习计算模式

AI 三大范式：
- 监督学习
- 无监督学习
- 强化学习


![[Pasted image 20240330153015.png|600]]



神经网络中主要的计算模式：权重求和
muliply and accumulate (MAC) > 90% computation

主要就是乘法和加法---->化为权重求和的过程

权重求和与激活函数结合



## 主流网络模型结构

- 全连接 Fully connected layer
	- feed forword, fully connected 
	- Multilayer Perceptron (MLP)

- 卷积层 Convolutional layer
	- Feed forward， sparsely- connected， weight shading
	- Convolutional Neural Network (CNN)
	- Typically used for images

- 循环网络 Recurrent layer
	- Feedback
	- Recurrent Neural Network (RNN/LSTM)
	- Typically used for sequential data (e.g. speech, language)
- 注意力机制
	- Attention (matrix multiply) + Feed forward, fully connected
	- Foundation models
	- Transformer 

卷积计算：

主要还是乘加运算

![[Pasted image 20240330155458.png|600]]


大部分通过矩阵乘法和操作



卷积神经网络的特点是有很多channel，Multi-Input Channel and Output channel

![[Pasted image 20240330161822.png|600]]



除了channel多之外，还有非常大的batchsize（N×C×H×W）

![[Pasted image 20240330203657.png|675]]

dynamic shape

![[Pasted image 20240330203857.png|675]]


### AI 计算模式的思考（1）

1. 需要支持神经网络模型的计算逻辑
	- 权重数据共享，便于对神经元的权重值进行求和
	- 除了卷积/全连接计算，需要支持激活等Vector计算
2. 能支持高位的张量存储和计算
	- 内存Mem地址随机/自动索引
	- 大Channel和大Feature Map高效加载
3. 支持常用神经网络模型结构
	- Conv，MatMul， Transformer等高校矩阵乘法
	- 快速对应新的AI算法与结构


## 量化压缩和网络剪枝


- 网络剪枝研究模型权重中的冗余，并尝试删除/修剪冗余和非关键的权重
- 模型量化是通过减少权重表示或激活所需的比特数来压缩模型

![[Pasted image 20240330204639.png|575]]


### 低比特量化特征

1. 参数压缩
2. 提升速度
3. 降低内存
4. 功耗降低
5. 提升芯片面积

模型量化相关的研究热点

- 感知量化训练 Quantization training
	- 8-bit with stochastic rounding 混合bit 量化
- 减少计算比特位 Reduce number  of bits
	- binary Nets 二值化网络模型
- 非线性量化 Non-Linear Quantization
	- Log-net
- 减少权重计算 Reduce number of unique weights and activations
	- ADD Nets 加法网络
	- XNOR-Net 异或网络模型

剪枝 pruning make wights sparse

- 训练：训练过参数化模型，得到最佳网络性能，以此为基准
- 剪枝：根据算法对模型剪枝，调整网络结构中通道或层数，得到剪枝后的网络结构
- 微调：在原数据集上进行微调，用于重新弥补因为剪枝后的稀疏模型丢失的精度性能

![[Pasted image 20240330213228.png|575]]


分类：
- 非结构化剪枝 Unstructured Pruning：随机对独立的权重或神经元连接进行剪枝
- 结构化剪枝 Structured pruning: 对filter/channel/layer进行剪枝

![[Pasted image 20240330213420.png|550]]


### AI计算模式思考（2）

1. 提供不同的bit位数
	- 对于低比特量化的相关研究落地提供int8/int4甚至更低的精度
	- 在M-bits和E-bits之间权衡 Tradeoff（如TF32/BF16）
2. 利用硬件提供稀疏计算
	- 硬件上减少0值得重复计算
	- 减少网络模型对内存得需求，稀疏化网络模型结构





## 轻量级模型

经典的轻量级模型
CNN系列
1. SqueezeNet系列(20l6)
2. ShuffleNet系列(20I7)
3. MobileNet系列(20I7)
4. ESPnet系列(2018)
5. FBNet系列(2018)
6. EfficientNet系列（20I9)
7. GhostNet系列(20I9）
Transformer系列
1. MobileViT (2021）
2. Mobile-Former（2021）
3. EfficientFormer 2022

轻量化网络 Efficient DNN Models的主要方法
- 通过改变网络模型不同层的layer shape 或卷积方式
- 通过Neural Architecture Search(NAC)来搜索更轻量化的网络模型

![[Pasted image 20240331005735.png|550]]

把传统大卷积核改为小卷积核的组合（MobileNet）
减少卷积里channel的层数



