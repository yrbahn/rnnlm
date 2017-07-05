#!/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from konlpy.tag import Twitter
import sys  
import codecs
import os

def build_vocab(file_name, vocab_size):
    word_to_freq = {}
    word_to_id = {'<eos>': 0, '<unk>': 1}
    gen_word_id = 2
    
    tagger = Twitter()
    f = codecs.open(file_name, "r", encoding='utf8')
    for line in f:
        #print(line)
        tagged_sent = tagger.pos(line)
        for tagged_morph in tagged_sent:
            morph = tagged_morph[0]
            if morph not in word_to_freq:
                word_to_freq[morph] = 1
            else:
                word_to_freq[morph] = word_to_freq[morph] + 1
    
    for w in sorted(word_to_freq.items(), key=lambda x: -x[1]):
        word_to_id[w[0]] = gen_word_id
        gen_word_id = gen_word_id + 1
        if vocab_size-1 < gen_word_id:
            break
            
    return word_to_id    

def file_to_word_ids(file_name, word_to_id):
    tagger = Twitter()
    f = codecs.open(file_name, "r", encoding='utf8')
    for line in f:
        tagged_sent = tagger.pos(line)
        sent = [w[0] for w in tagged_sent]
        sent.append('<eos>')
        for morph in sent:
            if morph in word_to_id:
                yield word_to_id[morph]
            else:
                yield word_to_id['<unk>']
        
def sent_to_word_ids(sent, word_to_id):
    m_sent = []
    tagger = Twitter()
    tagged_sent = tagger.pos(sent)
    for tagged_morph in tagged_sent:
        morph = tagged_morph[0]
        print(morph)
        if morph in word_to_id:
            m_sent.append(word_to_id[morph])
        else:
            m_sent.append(word_to_id['<unk>'])
    m_sent.append(word_to_id['<eos>'])
    return m_sent

def lm_raw_sentence(sent, word_to_id, prob=False):
    sent = list(sent_to_word_ids(sent, word_to_id))
    if not prob:
        return sent, -1
    else:
        return sent[:-2], sent[-2]
       
def lm_raw_data(data_path='', vocab_size=10000):
    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")
    
    word_to_id = build_vocab(train_path, vocab_size)
    train_data = list(file_to_word_ids(train_path, word_to_id))
    valid_data = list(file_to_word_ids(valid_path, word_to_id))
    test_data = list(file_to_word_ids(test_path, word_to_id))
    
    return train_data, valid_data, test_data, word_to_id
   
def ptb_iterator(raw_data, batch_size, num_steps, name=None):
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len / batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) / num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        print(x)
        print(y)
        yield (x, y)
    
    
def lm_data_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len -1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size =0, descrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")
        
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(data, [0, i*num_steps], [batch_size, num_steps])
        y = tf.slice(data, [0, i*num_steps +1], [batch_size, num_steps])
        return x, y
