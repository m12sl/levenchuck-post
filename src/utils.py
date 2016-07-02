import os
import numpy as np
import re
from collections import Counter

from tqdm import tqdm


def load_dumped(path='tmp/allj.txt'):
    data = []
    with open(path, 'r') as fin:
        tmp = []
        for line in tqdm(fin):
            if re.match('<post', line):
                title = line[line.find('>') + 1:]
                continue
            if re.match('<\/post>', line):
                data.append((title, ''.join(tmp)))
                tmp = []
                continue
            tmp.append(line)
    return data


def generate_charset():
    rus = [chr(_) for _ in range(ord('а'), ord('я') + 1)]
    eng = [chr(_) for _ in range(ord('a'), ord('z') + 1)]
    alphabet = rus + eng + [_.upper() for _ in rus + eng]
    nums = list('0123456789')
    punkt = list(' !?,.\"\':-()><\n')
    return alphabet + nums + punkt + ['oov']


def generate_vocab():
    charset = generate_charset()
    vocab = {k: i + 1 for i, k in enumerate(charset)}
    vocab[''] = 0
    return vocab


def encode_text(text, vocab):
    dtype = np.dtype(np.uint8)

    encoded = np.zeros((len(text), ), dtype=dtype)
    oov = vocab.get('oov', 0)

    idx = 0
    last = 0
    for x in tqdm(text):
        e = vocab.get(x, oov)
        if e == oov and last == oov:
            continue
        last = e
        encoded[idx] = e
        idx += 1

    return encoded[:idx]


def alphabet(data):
    titles = Counter()
    texts = Counter()
    for x in data:
        titles.update(x[0])
        texts.update(x[1])
    print('Title stats:', titles)
    print('Texts stats:', texts)


def prepare_dataset(path):
    data = load_dumped(path)

    idx = np.arange(len(data))
    np.random.seed(19)
    np.random.shuffle(idx)

    # shuffle data
    data = [data[_] for _ in idx]


    test_len = 10
    test_posts, train_posts = data[:test_len], data[test_len:]

    # drop titles
    test_text = "".join(_[1] for _ in test_posts)
    train_text = "".join(_[1] for _ in train_posts)

    vocab = generate_vocab()
    test = encode_text(test_text, vocab)
    train = encode_text(train_text, vocab)

    np.save('test', test)
    np.save('train', test)


if __name__ == "__main__":
    data = load_dumped()

    alphabet(data)



            


