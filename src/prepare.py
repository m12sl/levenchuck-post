import utils
import numpy as np
from tqdm import tqdm
import json


def main():
    print('Load raw data')
    data = utils.load_dumped('../data/raw/dump.txt')

    print('Filter text')
    content = [utils.filter_text(_[1]) for _ in tqdm(data)]

    idx = np.arange(len(content))
    np.random.seed(19)
    np.random.shuffle(idx)

    test_len = int(0.1 * len(idx))

    print('Split into train/test')
    test = "".join(content[_] for _ in tqdm(idx[:test_len]))
    train = "".join(content[_] for _ in tqdm(idx[test_len:]))

    vocab = utils.generate_vocab()
    with open('../data/processed/vocab.json', 'w') as fout:
        json.dump(vocab, fout)

    print('Encoding test')
    test = utils.encode_text(test, vocab)
    np.save('../data/processed/test', test)

    print('Encoding train')
    train = utils.encode_text(train, vocab)
    np.save('../data/processed/train', train)




if __name__ == "__main__":
    main()
