#定义一个config对象
class Config(object):
    batch_size = 20   #批尺寸
    num_steps = 35  # number of unrolled time steps 展开的时间步数
    hidden_size = 450 # number of blocks in an LSTM cell  神经元个数
    vocab_size = 10000
    max_grad_norm = 5 # maximum gradient for clipping 梯度裁剪
    init_scale = 0.05 # scale between -0.1 and 0.1 for all random initialization
    keep_prob = 0.5 # dropout probability 让某个神经元以p的概率失效
    num_layers = 2 # number of LSTM layers
    learning_rate = 1.0
    lr_decay = 0.8  #学习率衰减因子
    lr_decay_epoch_offset = 6 # don't decay until after the Nth epoch
