import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq

import numpy as np


class CharRNN:
    def __init__(self, args, infer=False):
        self.args = args

        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("Unknown model {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable('softmax_w', [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable('softmax_b', [args.vocab_size])

            with tf.device('/cpu:0'):
                embedding = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell,
                                                  loop_function=loop if infer else None, scope='rnnlm')

        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        # compute loss only for non-zero entries
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [tf.reshape(self.targets, [-1])],
                                                [tf.ones([args.batch_size * args.seq_length])])
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.cost,
                                           aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    def encode(self, sess, seq):
        state = self.cell.zero_state(1, tf.float32).eval()
        x = np.zeros((1, 1))
        for t in seq:
            x[0, 0] = t
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        return state

    def sample(self, sess, prime, num=200, sampling_type=1):
        state = self.cell.zero_state(1, tf.float32).eval()
        x = np.zeros((1, 1))
        print('PRIME: ', prime)
        for t in prime:
            x[0, 0] = t
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return int(np.searchsorted(t, np.random.rand(1) * s))

        def temperature_pick(weights, T):
            w = np.exp(np.log(weights) / T)
            s = np.sum(w)
            r = int(np.searchsorted(np.cumsum(w), np.random.rand(1) * s))
            return r

        ret = []
        t = prime[-1]

        x = np.zeros((1, 1))
        for n in range(num):
            x[0, 0] = t
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                sample = temperature_pick(p, 0.6)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            t = sample
            ret.append(t)

        return ret
