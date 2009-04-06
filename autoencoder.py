from simplenet import SimpleNet

import numpy

from nlins import *
from errors import *

class Autoencoder(SimpleNet):
    def __init__(self, ninputs, noutputs, nlin=tanh, alpha=0.01, lmbd=0.0, dtype=numpy.float32):
        SimpleNet.__init__(ninputs, noutputs, ninputs, hnlin=nlin, error=mse, onlin=none, alpha=alpha, lmbd=lmbd, dtype=dtype)

    def _save_(self, file):
        file.write('AE1')

    def _load_(self, file):
        s = file.read(3)
        if s != 'AE1':
            raise ValueError('Not an Autoencoder save file')

    def test(self, x):
        return SimpleNet.test(self, x, x)

    def code(self, x):
        return self.fprop(x)[1]

    def grad(self, x):
        return SimpleNet.grad(self, x, x)

    def epoch_bprop(self, x):
        return SimpleNet.epoch_bprop(self, x, x)

    def train_loop(self, x, epochs=100):
        return SimpleNet.train_loop(self, x, x, epochs=epochs)
