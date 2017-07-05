# -*- coding: utf-8 -*-

import tensorflow as tf
import codecs
import os
import time
import numpy as np
import reader

class LMInput(object):
    """The input data"""
    
    def __init__(self, config, data, name=None):
        batch_size = config.batch_size
        num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) //num_steps
        self.input_data, self.targets = reader.lm_data_producer(data, batch_size, num_steps, name=name)


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class RNNLMModel(object):
    """RNN LM Model"""
    def __init__(self, is_training, config, input_= None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self._is_training = is_training
        self.epoch_size = 0 # for training
        
        if not input_:
            self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
            self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        else:
            self._input_data = input_.input_data
            self._targets = input_.targets
            self.epoch_size = input_.epoch_size
            self._is_training = True
        
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            # dropout to the output of lstm cell
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*config.num_layers, state_is_tuple=True)
        
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)
        
        if is_training and config.keep_prob <1:
            # add a dropout to the input
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0 : 
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # outputs = num_steps, batch_size , size
        # concat(1, outputs) = batch_size , size*num_steps
        # output = batch_size*num_steps, size
        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        # self._prob = tf.nn.softmax(logits)
        
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size*num_steps], dtype=tf.float32)])
        
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        
        self.logits = logits
        self.probs = tf.nn.softmax(logits)
        
        if not is_training:
            return
        
        # gradient clipping
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
    
    
    @property
    def input_data(self):
        return self._input_data
    
    @property
    def targets(self):
        return self._targets
    
    @property
    def initial_state(self):
        return self._initial_state
    
    @property
    def cost(self):
        return self._cost
    
    @property
    def final_state(self):
        return self._final_state
    
    @property
    def lr(self):
        return self._lr
    
    @property
    def train_op(self):
        return self._train_op
    
    @property
    def is_training(self):
        return self._is_training
    
    #@property
    #def probability(self):
    #    return self._prob
    


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    if not model.is_training:
        raise ValueError("epoch for training") 
    
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    
    for step in range(model.epoch_size):
        feed_dict = {}
        for i, (c,h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
    
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.num_steps
    
        if verbose and step % (model.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                (step * 1.0 / model.epoch_size, np.exp(costs / iters),
                 iters * model.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def run_epoch2(session, model, data, id_to_word, prob=False):    
    """Runs the model on the given data."""
    logits = None
    state = session.run(model.initial_state)
    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, 
                                                      model.num_steps)):
        if prob :
            fetches = {
            "probs": model.probs,
            "final_state": model.final_state,
            }
        else:
            fetches = {
                "logits": model.logits,
                "final_state": model.final_state,
            }

        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        
        for i, (c,h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = session.run(fetches, feed_dict)
        
        if prob:
            result = vals["probs"]
        else:
            result = vals["logits"]
        #decodeWordId = int(np.argmax(logits))
        #print("-----")
        #print(x)
        #print(id_to_word[decodeWordId])
        #print(y)
        #print("====")
        state = vals["final_state"]

    return result


