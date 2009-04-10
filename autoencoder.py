from base import *

from net import *
from simplenet import *
from nlins import *
from errors import *
from trainers import *
from trainers import Trainer

__all__ = ['Autoencoder', 'StackedAutoencoder']

class auto_trainer(Trainer):
    def __init__(self, trainer_type, x, noisyness=0.0, corrupt_value=0.0, trainer_opts={}):
        self.trainer = trainer_type(x, x, **trainer_opts)
        self.noisyness = noisyness
        self.corrupt_value = corrupt_value

    def reset(self):
        self.trainer.reset()
        self.trainer.x = self.trainer.y

    def epoch(self, nnet):
        if self.noisyness > 0.0:
            self.trainer.x = self.trainer.y.copy()
            self.trainer.x[numpy.random.random(x.shape) < self.noisyness] = self.corrupt_value
        self.trainer.epoch(nnet)

class Autoencoder(SimpleNet):
    r"""
    Simple autoencoder with only one hidden layer.
    """
    def __init__(self, ninputs, noutputs, nlin=tanh, noisyness=0.0, corrupt_value=0.0, dtype=numpy.float32):
        SimpleNet.__init__(self, ninputs, noutputs, ninputs, hnlin=nlin, error=mse, onlin=nlin, dtype=dtype)
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
        x = numpy.atleast_2d(x)
        return self._test(x, x)

    def code(self, x):
        x = numpy.atleast_2d(x)
        return self._fprop(x)[1]

    def train_loop(self, x, trainer_type=bprop, epochs=100, **trainer_opts):
        return self._train_loop(auto_trainer(trainer_type, x, self.noisyness, self.corrupt_value, trainer_opts), epochs)

def layer_encoder(W1, b1, nlin):
    W2 = W1.T.copy()
    b2 = numpy.zeros(W2.shape[1])
    return NNet.virtual([layer(W1, b1), layer(W2, b2)], nlins=(nlin, nlin), error=mse)

class StackedAutoencoder(NNet):
    def __init__(self, sizes, nlins, noisyness=0.0, corrupt_value=0.0, dtype=numpy.float32):
        NNet.__init__(self, sizes + sizes[-2::-1], nlins + nlins[::-1], dtype=dtype)
        self.noisyness = noisyness
        self.corrupt_value = corrupt_value

    def _save_(self, file):
        file.write('SAE1')
        pickle.dump((self.noisyness, self.corrupt_value), file)
    
    def _load_(self, file):
        s = file.read(4)
        if s == 'SAE1':
            self.noisyness, self.corrupt_value = pickle.load(file)
        else:
            raise ValueError('Not a valid StackedAutoencoder save file')

    def code(self, x):
        x = numpy.atleast_2d(x)
        return self._fprop(x).outs[len(self.layers)/2-1]
    
    def test(self, x):
        x = numpy.atleast_2d(x)
        return self._test(x, x)

    def grad(self, x, lmbd):
        x = numpy.atleast_2d(x)
        return self._grad(x, x, lmbd)

    def pretrain(self, x, trainer_type=bprop, loops=10, **trainer_opts):
        x = numpy.atleast_2d(x)
        return self._pretrain(x, trainer_type, loops, trainer_opts)
    
    def _pretrain(self, x, trainer_type, loops, trainer_opts):
        x = numpy.atleast_2d(x)
        tnet = layer_encoder(self.layers[0].W, self.layers[0].b, nlin=self.nlins[0])
        tnet.train_loop(auto_trainer(trainer_type, x, self.noisyness, self.corrupt_value, trainer_opts), epochs=loops)
        
        for i in xrange(1, len(self.layers)):
            v = self._fprop(x).outs[i-1]
            tnet = layer_encoder(self.layers[i].W, self.layers[i].b, nlin=self.nlins[i])
            tnet.train_loop(auto_trainer(trainer_type, v, self.noisyness, self.corrupt_value, *trainer_opts), epochs=loops)
    
    def train_loop(self, x, trainer_type=bprop, epochs=100, **trainer_opts):
        return self._train_loop(auto_trainer(trainer_type, x, self.noisyness, self.corrupt_value, trainer_opts), epochs)
