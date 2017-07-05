#-*- coding: utf-8 -*-
import tensorflow as tf
import os, sys

from configs import *
from reader import *
from rnnlm import *
from six.moves import cPickle

flags = tf.flags
logging = tf.logging
flags.DEFINE_string("save_path", None, "save_dir")
flags.DEFINE_string("sent", None, "sentence")
flags.DEFINE_bool("prob", False, "get prob of a sentence")
FLAGS = flags.FLAGS

def main(_):
    if not FLAGS.save_path:
        raise ValueError("Must set --data_path to PTB data directory")
    
    word_to_id = None
    with open(os.path.join(FLAGS.save_path, 'word_to_id.pkl'), 'rb') as f:
        word_to_id = cPickle.load(f)
        id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
        
    with open(os.path.join(FLAGS.save_path, 'config.pkl'), 'rb') as f:
        config = cPickle.load(f)
    config.batch_size = 1
    config.num_steps = 1

    sent = u"전화번호"
    if FLAGS.sent:
        sent = FLAGS.sent.decode('utf-8')
    sent, last_word = lm_raw_sentence(sent, word_to_id, FLAGS.prob)
    
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope("Model", initializer=initializer):
            m = RNNLMModel(is_training=False, config=config)
        
        saver = tf.train.Saver()
        tf.initialize_all_variables().run()
        
        ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("loading a model")
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print "no model"
            return 
        
        result = run_epoch2(session, m, sent, id_to_word, FLAGS.prob)
        if FLAGS.prob:
            print(result[0][last_word])
        else:
            print(id_to_word[np.argmax(result)])

if __name__ == "__main__":
    tf.app.run()
