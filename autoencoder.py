from simplenet import SimpleNet

__all__ = ['Autoencoder', 'StackedAutoencoder']

import numpy

from base import *
from nlins import *
from errors import *

class Autoencoder(SimpleNet):
    def __init__(self, ninputs, noutputs, nlin=tanh, noisyness=0.0, corrupt_value=0.0, alpha=0.01, lmbd=0.0, dtype=numpy.float32):
        SimpleNet.__init__(self, ninputs, noutputs, ninputs, hnlin=nlin, error=mse, onlin=none, alpha=alpha, lmbd=lmbd, dtype=dtype)
        self.noisyness = noisyness
        self.corrupt_value = corrupt_value

    def _save_(self, file):
        file.write('AE1')
        pickle.dump((self.noisyness, self.corrupt_value), file)

    def _load_(self, file):
        s = file.read(3)
        if s == 'AE1':
            self.noisyness, self.corrupt_value = pickle.load(file)
        else:
            raise ValueError('Not a valid Autoencoder save file')

    def test(self, x):
        return self._test(x, x)

    def code(self, x):
        return self._fprop(x)[1]

    def epoch_bprop(self, x):
        return self._epoch_bprop(x, x)

    def _epoch_bprop(self, x, y):
        if self.noisyness != 0.0:
            x[numpy.random.random(x.shape) < self.noisyness] = self.corrupt_value
        SimpleNet._epoch_bprop(self, x, y)

    def train_loop(self, x, epochs=100):
        return self._train_loop(x, x, epochs)


class StackedAutoencoder(BaseObject):
    def __init__(self, sizes, nlins, alpha=0.01, lmbd=0.0, dtype=numpy.float32):
        self.nets = map(lambda s, n: 1, [])
