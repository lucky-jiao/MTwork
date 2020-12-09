基于LSTM的语言模型
数据集
PTB
开发环境
Windows系统 Python3.6 tensorflow1.8.0
实现
1. ptb_reader.py：处理ptb.train.txt、ptb.valid.txt、ptb.test.txt三个文件，将数据集分别转成字典的方法，每个单词映射到
0-9999的整数，根据批尺寸batch_size和时间步数num_steps对数据集切分，返回输入inputs和标签targets
2. main.py:定义LSTM模型，包括初始化LSTM模型、反向传播参数、学习率更新等，run_epoch函数定义了一个epoch的训练，main函数训练
模型，并输出训练集、验证集、测试集上的perplexity
3. configSettings.py：模型的配置，包括batch_size、num_steps、hiddendimension、numlayers、learning-rate等
结果
nun_epoches=10

Epoch: 10 Training Perplexity: 93.818 (Cost: 4.541)
0.0% Perplexity: 113.962 (Cost: 165.755) 
95.2% Perplexity: 99.404 (Cost: 165.343) 
Epoch: 10 Valid Perplexity: 99.273 (Cost: 4.598)
Test Perplexity: 96.182 (Cost: 4.566)

