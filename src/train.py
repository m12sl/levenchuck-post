import numpy as np
import tensorflow as tf

import argparse
import os
import pickle
import json
from tqdm import tqdm
import errno

from data import DataGenerator
from model import CharRNN as Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')

    parser.add_argument('--vocab', type=str, default='../data/processed/vocab.json',
                        help='path to vocab.json file')

    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
            'config.pkl'        : configuration;
            'chars_vocab.pkl'   : vocabulary definitions;
            'checkpoint'        : paths to model file(s) (created by tf).
                                  Note: this file contains absolute paths, be careful when moving files around;
            'model.ckpt-*'      : file(s) with model definition (created by tf)
            """)

    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=16,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')

    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')

    args = parser.parse_args()
    train(args)


def train(args):
    path = args.save_dir
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

    with open(args.vocab, 'r') as fin:
        vocab = json.load(fin)
        args.vocab_size = len(vocab)

    args.docs_looped = True
    train = DataGenerator(args, '../data/processed/train.npy')
    args.docs_looped = False
    val = DataGenerator(args, '../data/processed/test.npy')

    args.iterations_per_epoch = int(train.samples / args.batch_size)
    args.iterations_per_val = int(val.samples / args.batch_size)

    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from), \
            " %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, "config.pkl")), \
            "config.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl')) as f:
            saved_model_args = pickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme] == vars(args)[checkme], \
                "Command line argument and saved model disagree on '%s' " % checkme

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    model = Model(args)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())

        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** e)))
            train.reset()

            state = model.initial_state.eval()
            pbar = tqdm(range(args.iterations_per_epoch))
            for b in pbar:
                x, y = next(train)

                feed = {model.input_data: x,
                        model.targets: y,
                        model.initial_state: state}

                train_loss, state, _ = sess.run([model.cost,
                                                 model.final_state,
                                                 model.train_op],
                                                feed)

                if b % 10 == 0:
                    pbar.set_description('train_loss: {:.3f}'
                                         .format(train_loss))

            print('Checkpoint')
            checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path,
                       global_step=e * args.iterations_per_epoch)

            print('Validation...')
            # we can restore training states
            # train_states = state[:]

            val_loss = 0.0
            state = model.initial_state.eval()

            val.reset()
            pbar = tqdm(range(args.iterations_per_val))
            for b in pbar:
                x, y = next(val)

                feed = {model.input_data: x,
                        model.targets: y,
                        model.initial_state: state}

                (loss, state) = sess.run([model.cost,
                                          model.final_state, ],
                                         feed)

                pbar.set_description('val_loss: {:.3f}'
                                     .format(loss))

                val_loss += loss

            val_loss /= args.iterations_per_val
            print('Mean val_loss is {:.3f}'.format(val_loss))


if __name__ == '__main__':
    main()
