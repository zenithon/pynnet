from base import *

from net import *
from simplenet import *
from nlins import *
from errors import *
from trainers import *
from trainers import Trainer

__all__ = ['Autoencoder', 'StackedAutoencoder']

class auto_trainer(Trainer):
    r"""Specialized trainer for auto-encoding layers."""
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

    WARNING: this class is poorly documented and poorly tested, use
    `StackedAutoencoder` instead.
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
        x = self._convert(x)
        return self._test(x, x)

    def code(self, x):
        x = self._convert(x)
        return self._fprop(x).outs[1]

    def train_loop(self, x, trainer_type=bprop, epochs=100, **trainer_opts):
        return self._train_loop(auto_trainer(trainer_type, x, self.noisyness, self.corrupt_value, trainer_opts), epochs)

def layer_encoder(W1, b1, nlin):
    r"""create a virtual net out of a layer for pretraining"""
    W2 = W1.T.copy()
    b2 = numpy.zeros(W2.shape[1])
    return NNet.virtual([layer(W1, b1), layer(W2, b2)], nlins=(nlin, nlin), error=mse)

class StackedAutoencoder(NNet):
    def __init__(self, sizes, nlins, noisyness=0.0, corrupt_value=0.0, dtype=numpy.float32):
        r"""
        Class to represent a deep autoencoder.

        Parameters:
        sizes -- the sizes of the encoding from input to output
        nlins -- the nonlinearities at each layer
        noisyness -- (default: 0.0) the probability of corrupting input values
        corrupt_value -- (default: 0.0) the value to put in place of the corrupted values
        dtype -- (default: numpy.float32) the datatype of the coefficients matrices

        EXAMPLES::
        
        >>> StackedAutoencoder([4, 2], nlins=[tanh]) # a 4-2-4 autoencoder
        <pynnet.autoencoder.StackedAutoencoder object at ...>
        >>> StackedAutoencoder([8, 3], nlins=[sigmoid], dtype=numpy.float64) # a 8-3-8 autoencoder
        <pynnet.autoencoder.StackedAutoencoder object at ...>
        >>> StackedAutoencoder([100, 20, 5], nlins=[sigmoid, tanh], noisyness=0.4)
        <pynnet.autoencoder.StackedAutoencoder object at ...>

        TESTS::
        
        >>> sae = StackedAutoencoder([4, 2], nlins=[sigmoid])
        >>> len(sae.layers)
        2
        >>> sae.layers[1].W.shape
        (2, 4)
        >>> sae.save('/tmp/pynnet-test-sae')
        >>> sae2 = StackedAutoencoder.load('/tmp/pynnet-test-sae')
        >>> type(sae2)
        <class 'pynnet.autoencoder.StackedAutoencoder'>
        >>> len(sae2.layers)
        2
        >>> sae2.layers[1].W.shape
        (2, 4)
        """
        NNet.__init__(self, sizes + sizes[-2::-1], nlins + nlins[::-1], dtype=dtype)
        self.noisyness = noisyness
        self.corrupt_value = corrupt_value

    def _save_(self, file):
        r"""save the state"""
        file.write('SAE1')
        pickle.dump((self.noisyness, self.corrupt_value), file)
    
    def _load_(self, file):
        r"""load the state"""
        s = file.read(4)
        if s == 'SAE1':
            self.noisyness, self.corrupt_value = pickle.load(file)
        else:
            raise ValueError('Not a valid StackedAutoencoder save file')

    def code(self, x):
        r"""
        Return the coded version of the inputs.

        Parameters:
        x -- the inputs

        EXAMPLES::
        
        >>> sae = StackedAutoencoder([4, 2], nlins=[sigmoid])
        >>> x = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        >>> sae.code(x).shape
        (4, 2)
        """
        x = self._convert(x)
        return self._fprop(x).outs[len(self.layers)/2-1]
    
    def test(self, x):
        r"""
        Special autoencoder version of `NNet.test()`.

        Parameters:
        x -- the inputs (and targets)

        EXAMPLES::

        >>> sae = StackedAutoencoder([4, 2], nlins=[sigmoid])
        >>> x = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        >>> error = sae.test(x)
        >>> sae.train_loop(x, epochs=20)
        >>> error > sae.test(x)
        True
        """
        x = self._convert(x)
        return self._test(x, x)

    def grad(self, x, lmbd=0.0):
        r"""
        Special autoencoder version of `NNet.test()`.

        Parameters:
        x -- the inputs (and targets)
        lmbd -- (default: 0.0) the lambda factor for weight decay

        EXAMPLES::
        
        >>> sae = StackedAutoencoder([4, 2], nlins=[sigmoid])
        >>> x = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        >>> G = sae.grad(x)
        >>> len(G)
        2
        >>> G[1].W.shape
        (2, 4)
        """
        x = self._convert(x)
        return self._grad(x, x, lmbd)

    def pretrain(self, x, trainer_type=conj, loops=10, **trainer_opts):
        r"""
        Do some rounds of per-layer training.

        Parameters:
        x -- the inputs (and targets)
        trainer_type -- (default: bprop) a class of trainer
        loops -- (default: 10) the number of loop per layer
        
        NOTE: You may add any other keyword options and they will be
        passed on to the trainer instance that is built.

        EXAMPLES::
        
        >>> sae = StackedAutoencoder([4, 3, 2], nlins=[sigmoid])
        >>> x = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        >>> sae.pretrain(x)
        """
        x = self._convert(x)
        return self._pretrain(x, trainer_type, loops, trainer_opts)
    
    def _pretrain(self, x, trainer_type, loops, trainer_opts):
        r"""private implementation of `StackedAutoencoder.pretrain()`"""
        tnet = layer_encoder(self.layers[0].W, self.layers[0].b, nlin=self.nlins[0])
        tnet.train_loop(auto_trainer(trainer_type, x, self.noisyness, self.corrupt_value, trainer_opts), epochs=loops)
        
        for i in xrange(1, len(self.layers)):
            v = self._fprop(x).outs[i-1]
            tnet = layer_encoder(self.layers[i].W, self.layers[i].b, nlin=self.nlins[i])
            tnet.train_loop(auto_trainer(trainer_type, v, self.noisyness, self.corrupt_value, *trainer_opts), epochs=loops)
    
    def train_loop(self, x, trainer_type=conj, epochs=100, **trainer_opts):
        r"""
        Special autoencoder version of `NNet.train_loop()`
        
        Parameters:
        x -- the inputs (and targets)
        trainer_type -- (default: conj) a class of trainer
        epochs -- (default: 100) the number of epochs to do
        
        NOTE: You may add any other keyword options and they will be
        passed on to the trainer instance that is built.
        
        EXAMPLES::
        
        >>> sae = StackedAutoencoder([4, 2], nlins=[sigmoid])
        >>> x = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        >>> error = sae.test(x)
        >>> sae.pretrain(x) # optional, but recommended for deep networks
        >>> sae.train_loop(x, epochs=10)
        >>> error > sae.test(x)
        True
        """
        return self._train_loop(auto_trainer(trainer_type, x, self.noisyness, self.corrupt_value, trainer_opts), epochs)
