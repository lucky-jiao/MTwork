from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import numpy as np
import tensorflow as tf
import os
import shutil  #提供了对os中文件操作的补充----移动 复制 打包 压缩 解压
import sys
from configSettings import Config
import zipfile  #压缩、解压文件
import ptb_reader

# Importing different files for different version of python.
if sys.version_info[0] == 2:
  import urllib2
else:
  import urllib.request as urllib2
  '''
#tensorflow中flags用于接收命令行传递参数，可以全局的更改代码中的参数
flags = tf.flags
#(1,2,3) 1是参数名称，2是参数默认值，3是描述
flags.DEFINE_string("test", None, "Path for explicit test file")
FLAGS = flags.FLAGS
'''
class LSTM(object):
  def __init__(self, is_training, config):
    # Initialize the parameter values from the config.
    self.batch_size = config.batch_size 
    self.num_steps = config.num_steps
    dropout_probability = config.keep_prob
    hidden_dimension = config.hidden_size
    vocab_size = config.vocab_size
    batch_size = self.batch_size
    num_steps = self.num_steps
    #placeholder()函数是在神经网络构建graph的时候在模型中的占位，数据类型，数据形状
    self.input_data = tf.placeholder(tf.int32, shape = [batch_size, num_steps])
    
    # Convert input matrix to embedding matrix.变量名称"embedding",变量维度，uniform 均匀分布
    #tf.get_variable 根据变量的名称来获取变量或者创建变量
    embedding = tf.get_variable("embedding", [vocab_size, hidden_dimension], 
      initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale))
    #选取一个张量中对应索引的元素
    input_vector_embedding = tf.nn.embedding_lookup(embedding, self.input_data)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dimension)
    
    # Applying dropout over the input to LSTM cell i.e. from input at time t 
    # and from hidden state at time t - 1. 
    if is_training:
      lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = dropout_probability)
      input_vector_embedding = tf.nn.dropout(input_vector_embedding, config.keep_prob)

    # Stacking of RNN with two layers of it.
    stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 2)
    #生成全0的初始状态
    self.initial_state = stacked_lstm_cell.zero_state(batch_size, tf.float32)
    output_vector_embedding = []
    hidden_cell_states = []
    current_state = self.initial_state

    # Recurrent unit for simulating RNN.
    with tf.variable_scope("LangaugeModel",
      initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)):
      for iter_ in range(num_steps):
        if (iter_ > 0):
          tf.get_variable_scope().reuse_variables()
        current_cell_input = input_vector_embedding[:, iter_, :]
        (current_cell_output, current_state) = stacked_lstm_cell(current_cell_input, current_state)
        output_vector_embedding.append(current_cell_output)
        hidden_cell_states.append(current_state)

    self.final_state = hidden_cell_states[-1]
    output = tf.reshape(tf.concat(output_vector_embedding, 1), [-1, hidden_dimension])
    self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    targets = tf.reshape(self.targets, [-1])

    #weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
    #bias = tf.get_variable("bias", [VOCAB_SIZE])
    #logits = tf.matmul(output, weight) + bias
    #weights = tf.get_variable("weights",[hidden_dimension,vocab_size])
    weights=tf.transpose(embedding)
    bias=tf.get_variable("bias",[vocab_size],dtype=tf.float32)
    logits = tf.matmul(output, weights)+bias
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones([batch_size * num_steps])], vocab_size)
    self.cost = tf.reduce_sum(loss)/ batch_size
    #反向传播更新参数
    if is_training:
      #获取所有可训练的向量
      self.lr=tf.Variable(0.0,trainable=False)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
      optimizer = tf.train.GradientDescentOptimizer(self.lr)
      #optimizer = tf.train.AdamOptimizer(0.9)
      #optimizer = tf.train.RMSPropOptimizer(0.9)
      self.train_op = optimizer.apply_gradients(zip(grads, tvars))
      self.new_lr=tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
      self.lr_update=tf.assign(self.lr,self.new_lr)
    else:
      self.train_op = tf.no_op()

  def assign_lr(self,session,lr_value):
      session.run(self.lr_update,feed_dict={self.new_lr:lr_value})
      
def run_epoch(sess,model,data):
  epoch_size=((len(data)//model.batch_size) - 1)//model.num_steps
  saver=tf.train.Saver()
  #初始化模型参数
  state = sess.run(model.initial_state)
  total_cost=0
  iterations=0
  for step,(x, y) in enumerate(ptb_reader.ptb_iterator(data, model.batch_size, model.num_steps)):
    cost, state, _ = sess.run([model.cost, model.final_state, model.train_op], 
      feed_dict={model.input_data: x,model.targets: y,model.initial_state: state})
    total_cost += cost
    iterations += model.num_steps
    perplexity = np.exp(total_cost / iterations)
    if step % 100 == 0:
        progress = (step *1.0/ epoch_size) * 100
        print("%.1f%% Perplexity: %.3f (Cost: %.3f) " % (progress, perplexity, cost))
  save_path=saver.save(sess,"./saved_model_rnn/lstm-model.ckpt")
  return (total_cost / iterations), perplexity

def test_epoch(sess,model,data):
  saver=tf.train.Saver()
  saver.restore(sess, "./saved_model_rnn/lstm-model.ckpt")
  state = sess.run(model.initial_state)
  total_cost=0
  iterations=0
  epoch_size=((len(data)//model.batch_size) - 1)//model.num_steps
  for step,(x, y) in enumerate(ptb_reader.ptb_iterator(data, model.batch_size, model.num_steps)):
    cost, state = sess.run([model.cost, model.final_state], 
      feed_dict={model.input_data: x,model.targets: y,model.initial_state: state})
    total_cost += cost
    iterations += model.num_steps
    perplexity = np.exp(total_cost / iterations)
  return (total_cost / iterations), perplexity


def main(_):
  train_config = Config()
  eval_config = Config()
  eval_config.num_steps = 1
  eval_config.batch_size=1
  num_epochs = 10

  #if not FLAGS.test:
    #print(FLAGS.test)
  train_data, valid_data, test_data, _ = ptb_reader.ptb_raw_data("../data")
  with tf.Graph().as_default() and tf.Session() as session:
    with tf.variable_scope("Model", reuse=None):
      train_model = LSTM(is_training=True, config=train_config)
    with tf.variable_scope("Model",reuse=True):
      valid_model=LSTM(is_training=False,config=train_config)
    with tf.variable_scope("Model",reuse=True):
      test_model=LSTM(is_training=False,config=eval_config)

    if not os.path.exists('saved_model_rnn'):
      os.makedirs('saved_model_rnn')
    else:
      shutil.rmtree('saved_model_rnn')
      os.makedirs('saved_model_rnn')
    session.run(tf.global_variables_initializer())
    
    for i in range(num_epochs):
      lr_decay=train_config.lr_decay**max(i+1-4,0.0)
      train_model.assign_lr(session,train_config.lr_decay*lr_decay)
      train_cost, train_perp = run_epoch(session, train_model, train_data)
      print("Epoch: %i Training Perplexity: %.3f (Cost: %.3f)" % (i + 1, train_perp, train_cost))
      valid_cost, valid_perp = run_epoch(session, valid_model, valid_data)
      print("Epoch: %i Valid Perplexity: %.3f (Cost: %.3f)" % (i + 1, valid_perp, valid_cost))
      
    test_cost, test_perp = test_epoch(session, test_model, test_data)
    print("Test Perplexity: %.3f (Cost: %.3f)" % (test_perp, test_cost))

if __name__ == "__main__":
  tf.app.run()
