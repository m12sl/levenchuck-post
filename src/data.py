import numpy as np
import itertools


class DataGenerator:
    def __init__(self, args, path):
        self.args = args
        seq_length = args.seq_length
        batch_size = args.batch_size

        data = np.load(path)
        self.data = data

        self.offset = int(len(data) / batch_size)
        self.samples = int(self.offset / seq_length)

        self.x = np.zeros((batch_size, seq_length))
        self.y = np.zeros((batch_size, seq_length))

        self.base_row_iter = [None] * seq_length
        self.reset()

    def reset(self):
        if self.args.docs_looped:
            just_range_iterator = iter(range(0, len(self.data)))
            pointer = itertools.cycle(just_range_iterator)
        else:
            pointer = iter(range(0, self.offset))

        self.base_row_iter = zip(*[pointer] * self.args.seq_length)

    def next(self):
        base = np.array(next(self.base_row_iter))
        self.x.fill(0)
        self.y.fill(0)

        L = len(self.data)
        for i in range(0, self.args.batch_size):
            self.x[i, :] = self.data[(base + i * self.offset) % L]
            self.y[i, :] = self.data[(base + i * self.offset + 1) % L]

        return (self.x, self.y)

    __next__ = next

    def __iter__(self):
        return self
