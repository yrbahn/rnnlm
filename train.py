import tensorflow as tf
import os
from configs import *
from reader import *
from rnnlm import *
from six.moves import cPickle

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model",
    "medium", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string(
    "data_path",
    None,
    "The path to correctly-formatted data.")
flags.DEFINE_string("save_path", None, "Model output directory.")
flags.DEFINE_boolean(
    "tensorboard",
    False,
    "Whether to write data to a serialized TensorBoard summary.")
flags.DEFINE_integer("vocab_size", None, "vocab size")

FLAGS = flags.FLAGS

def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        return ValueError("Invalid model: %s", FLAGS.model)

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    config = get_config()
    if FLAGS.vocab_size:
        config.vocab_size = FLAGS.vocab_size
        
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    
    print(config.vocab_size)
    train_data, valid_data, test_data, word_to_id = lm_raw_data(FLAGS.data_path, config.vocab_size)
    print(len(word_to_id))
    
    if FLAGS.save_path:
        with open(os.path.join(FLAGS.save_path, 'word_to_id.pkl'), 'wb') as f:
            cPickle.dump(word_to_id, f)
        with open(os.path.join(FLAGS.save_path, 'config.pkl'), 'wb') as f:
            cPickle.dump(config, f)
            
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        with tf.name_scope("Train"):
            train_input = LMInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = RNNLMModel(is_training=True, config=config, input_=train_input)
            tf.scalar_summary("Training Loss", m.cost)
            tf.scalar_summary("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = LMInput(config=config, data=valid_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = RNNLMModel(is_training=False, config=config, input_=valid_input)
            tf.scalar_summary("Validation Loss", m.cost)

        with tf.name_scope("Test"):
            test_input = LMInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = RNNLMModel(is_training=False, config=eval_config, input_=test_input)
        
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        
            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
    
            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step = sv.global_step)

    print("training complete!!")

if __name__ == "__main__":
	tf.app.run()
