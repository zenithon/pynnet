from base import *

from nlins import *
from errors import *

__all__ = ['NNet']

class NNet(BaseObject):
    def __init__(self, layers, nlins=(tanh, none), error=mse, dtype=numpy.float32):
        r"""
        Create a new fully-connected neural network with the given parameters.
        
        Arguements:
        layers -- A list of layer widths.  The first and last value
                  indicate the number of inputs and outputs
                  respectively.
        nlins -- (default: (tanh, none)) A list of non-linearities for
                 each layer.  It should contain one less element than
                 the `layers` list since no non-linearity is applied
                 to the input.  If the list contains just two
                 elements, the first one is used for all layers but
                 the last and the second is used for the last layer.
                 If the value is not a list it is taken to be a single
                 nonlinearity that is applied to all the layers.
        error -- (default: mse) The error function to use.
        dtype -- (default: numpy.float32) the numpy datatype to use
                 for the weight matrices.  The default is
                 `numpy.float32` to save space, but if you require
                 more precision, you can use `numpy.float64`.  Maybe
                 others will work but they are not tested.
        
        The weight matrices are created and initialized with uniform
        random weights.
        
        EXAMPLES::

        >>> NNet([2, 2, 1])  # a xor net
        <pynnet.net.NNet object at ...>
        >>> NNet([5, 2]) # a net with no hidden layers
        <pynnet.net.NNet object at ...>
        >>> NNet([20, 50, 10, 1], nlins=(tanh, sigmoid, sigmoid), error=nll, dtype=numpy.float64) # a complex net
        <pynnet.net.NNet object at ...>
        
        TESTS::
        
        >>> net = NNet([4, 5, 2])
        >>> net.save('/tmp/pynnet_test_save_nnet')
        >>> net2 = NNet.load('/tmp/pynnet_test_save_nnet')
        >>> type(net2)
        <class 'pynnet.net.NNet'>
        >>> type(net2.layers[0])
        <class 'pynnet.base.layer'>
        """
        lim = 1/numpy.sqrt(layers[0])
        makelayer = lambda i, o: layer(numpy.random.uniform(low=-lim, high=lim, size=(i, o)).astype(dtype), numpy.zeros(o, dtype=dtype))
        self.layers = [makelayer(i, o) for i, o in zip(layers, layers[1:])]
        if isinstance(nlins, (tuple, list)):
            if len(nlins) == len(layers) - 1:
                self.nlins = tuple(nlins)
            elif len(nlins) == 2:
                
                self.nlins = tuple(nlins[0:1] * (len(layers)-2)) + (nlins[1],)
            else:
                raise ValueError("not enough non-linearities for the number of layers")
        else:
            self.nlins = (nlins,)*(len(layers)-1)
        self.err = error

    @classmethod
    def virtual(cls, layers, nlins, error):
        r"""
        Build a virtual net using the supplied structure.

        This intended for use in pre-training.  The objects returned
        by this method are exactly like normal neural nets except that
        their weight matrices are tied to pre-existing ones and they
        cannot be saved.
        """
        net = NNet.__new__(cls)
        net._virtual = True
        net.layers = layers
        net.nlins = nlins
        net.err = error
        return net

    def _save_(self, file):
        r"""save state to a file"""
        file.write('NN1')
        pickle.dump(len(self.layers), file)
        for l in self.layers:
            self._save_layer(l, file)
        pickle.dump((self.nlins, self.err), file)
        
    def _load_(self, file):
        r"""load state from a file"""
        s = file.read(3)
        if s != 'NN1':
            raise ValueError('File is not a SimpleNet save')
        l = pickle.load(file)
        self.layers = [self._load_layer(file) for _ in xrange(l)]
        self.nlins, self.err = pickle.load(file)

    def _save_layer(self, l, file):
        r"""implementation detail, do not use"""
        numpy.save(file, l.W)
        numpy.save(file, l.b)

    def _load_layer(self, file):
        r"""implementation detail, do not use"""
        return layer(numpy.load(file), numpy.load(file))

    def fprop(self, x):
        r"""
        Return the activations and outputs at each layer.

        Parameters:
        x -- A matrix of inputs (numpy 2d array or a python list of lists)

        Returns:
        a named tuple of (acts, outs)

        Use this method if you need to get the activation or output
        values in the network for some purpose.  If you just need to
        get the result of applying the neural net to a set of inputs,
        use `NNet.eval()`.

        EXAMPLES::
        
        >>> net = NNet([2, 2, 1])
        >>> res = net.fprop([[0, 0], [1, 1], [0, 1], [1, 0]])
        >>> res.outs[-1].shape # this is the same as eval
        (4, 1)

        TESTS::

        >>> type(res)
        <class 'pynnet.base.propres'>
        >>> acts, outs = res
        >>> len(acts) == 2
        True
        >>> len(outs) == 2
        True
        >>> acts[0].shape
        (4, 2)
        >>> outs[0].shape
        (4, 2)
        >>> acts[1].shape
        (4, 1)
        >>> outs[1].shape
        (4, 1)
        """
        x = numpy.atleast_2d(x)
        return self._fprop(x)

    def _fprop(self, x):
        r"""private implementation of `NNet.fprop()`"""
        acts = [None] * (len(self.nlins))
        outs = [None] * (len(self.nlins))
        outs[-1] = x
        for i in xrange(len(self.layers)):
            acts[i] = numpy.dot(outs[i-1], self.layers[i].W) + self.layers[i].b
            outs[i] = self.nlins[i](acts[i])

        return propres(acts, outs)

    def test(self, x, y):
        r"""
        Return the mean of the error value for the examples.

        Parameters:
        x -- the input values
        y -- the target values

        The inputs and targets can be either a numpy 2d array, a numpy
        1d array, a python list or a python list of lists.  In each of
        the 2d cases, the inputs are laid row-wise.

        EXAMPLES::
        
        We build a xor net
        >>> net = NNet([2, 2, 1])
        >>> x = [[0, 0], [1, 1], [0, 1], [1, 0]]
        >>> y = [[0], [0], [1], [1]]

        measure its error
        >>> error = net.test(x, y)

        train it a bit
        >>> net.train_loop(trainers.bprop(x, y), epochs=10)

        and verify that the error improved
        >>> error > net.test(x, y)
        True
        """
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._test(x, y)

    def _test(self, x, y):
        r"""private implementation of `NNet.test()`"""
        return self.err(self._eval(x), y)

    def eval(self, x):
        r"""
        Return the output of the network for the specified inputs.

        Parameters:
        x -- the input values

        The inputs can be either a numpy 2d array, a numpy 1d array, a
        python list or a python list of lists.  In each of the 2d
        cases, the inputs are laid row-wise.

        EXAMPLES::
        
        >>> net = NNet([2, 2, 1])
        >>> x = [[0, 0], [1, 1], [0, 1], [1, 0]]
        >>> net.eval(x).shape
        (4, 1)
        """
        x = numpy.atleast_2d(x)
        return self._eval(x)

    def _eval(self, x):
        r"""private implementation of `NNet.eval()`"""
        return self._fprop(x).outs[-1]

    def grad(self, x, y, lmbd=0.0):
        r"""
        Computes the error gradient for the examples given.
        
        Parameters:
        x -- the inputs
        y -- the targets
        lmbd -- (default: 0.0) the lambda factor if using weight decay

        EXAMPLES::
        >>> net = NNet([2, 2, 1])
        >>> x = [[0, 0], [1, 1], [0, 1], [1, 0]]
        >>> y = [[0], [0], [1], [1]]
        >>> GWs, Gbs = net.grad(x, y)
        >>> len(GWs)
        2
        >>> len(Gbs)
        2
        >>> GWs[0].shape
        (2, 2)
        >>> GWs[1].shape
        (2, 1)
        >>> Gbs[0].shape
        (2,)
        >>> Gbs[1].shape
        (1,)
        """
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._grad(x, y, lmbd)

    def _grad(self, x, y, lmbd):
        r"""private implementation of `NNet.grad()`"""
        acts, outs = self._fprop(x)
        
        Gouts = [None] * len(outs)
        Gacts = [None] * len(acts)
        
        Gouts[-1] = self.err._(outs[-1], y, self.err(outs[-1], y))
        Gacts[-1] = Gouts[-1] * self.nlins[-1]._(acts[-1], outs[-1])
        
        for i in xrange(-2, -len(outs) - 1, -1):
            Gouts[i] = numpy.dot(self.layers[i+1].W, Gacts[i+1].T).T
            Gacts[i] = Gouts[i] * self.nlins[i]._(acts[i], outs[i])
        
        GWs = [None] * len(self.layers)
        Gbs = [None] * len(self.layers)
        
        for i in xrange(-1,-len(self.layers), -1):
            GWs[i] = numpy.dot(outs[i-1].T, Gacts[i]) + 2.0 * lmbd * self.layers[i].W
            Gbs[i] = Gacts[i].sum(axis=0)

        GWs[0] = numpy.dot(x.T, Gacts[0]) + 2.0 * lmbd * self.layers[0].W
        Gbs[0] = Gacts[0].sum(axis=0)

        return GWs, Gbs#, Gacts, Gouts
    
    def _estim_grad(self, x, y, eps):
        r"""private implementation of the finite differences for `NNet.test_grad()`"""
        v = self._test(x, y)

        GWs = [numpy.empty(l.W.shape, l.W.dtype) for l in self.layers]
        Gbs = [numpy.empty(l.b.shape, l.b.dtype) for l in self.layers]
        
        for (W, b), GW, Gb in zip(self.layers, GWs, Gbs):

            for i, w in numpy.ndenumerate(W):
                W[i] += eps
                GW[i] = (self._test(x,y) - v)/eps
                W[i] = w

            for i, w in numpy.ndenumerate(b):
                b[i] += eps
                Gb[i] = (self._test(x,y) - v)/eps
                b[i] = w

        return GWs, Gbs

    def test_grad(self, x, y, verbose=True, eps=1e-4):
        r"""
        Test that the gradient computation is good.

        Parameters:
        x -- sample inputs
        y -- sample targets
        verbose -- (default: True) print the ratio matrices?
        eps -- (default: 1e-4) the epsilon to use in the finite difference

        Compares the symbolic gradient computed against the gradients
        computed using the finite differences method and checks that
        the ratio between the two is reasonable.

        EXAMPLES::
        
        >>> net = NNet([2, 2, 1])
        >>> x = [[0, 0], [1, 1], [0, 1], [1, 0]]
        >>> y = [[0], [0], [1], [1]]
        >>> net.test_grad(x, y, verbose=False)
        """
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._test_grad(x, y, verbose, eps)

    def _test_grad(self, x, y, verbose, eps):
        r"""private implementation of `NNet.test_grad()`"""
        GcWs, Gcbs = self._grad(x,y, 0.0)
        GeWs, Gebs = self._estim_grad(x, y, eps)

        wrong = False

        for GWc, GWe in zip(GcWs, GeWs):
            rW = GWc/GWe
            if verbose:
                print rW
            if rW.max() > 1.01 or rW.min() < 0.99:
                wrong = True

        for Gbc, Gbe in zip(Gcbs, Gebs):
            rb = Gbc/Gbe
            if verbose:
                print rb
            if rb.max() > 1.01 or rb.min() < 0.99:
                wrong = True

        if wrong:
            raise ValueError("Wrong gradient(s) detected")

    def _apply_dir(self, GWs, Gbs, alpha=1.0):
        r"""
        Private API for the trainers.

        (look at the code if you are implementing a trainer)

        TESTS::
        
        >>> net = NNet([2, 2, 1])
        >>> x = [[0, 0], [1, 1], [0, 1], [1, 0]]
        >>> y = [[0], [0], [1], [1]]
        >>> GWs, Gbs = net.grad(x, y)
        >>> error = net.test(x, y)
        >>> net._apply_dir(GWs, Gbs, -0.2)
        >>> error > net.test(x, y)
        True
        """
        for (W, b), GW, Gb in zip(self.layers, GWs, Gbs):
            W += alpha * GW
            b += alpha * Gb

    def train_loop(self, trainer, epochs = 100):
        r"""
        Do `epochs` rounds of training with the given trainer.
        
        Parameters:
        trainer -- A `trainers.Trainer` instance
        epochs -- (default: 100) the number of epochs to make.  What
                  an epoch represents may vary from trainer to
                  trainer.

        EXAMPLES::
        >>> net = NNet([2, 2, 1])
        >>> x = [[0, 0], [1, 1], [0, 1], [1, 0]]
        >>> y = [[0], [0], [1], [1]]
        >>> trainer = trainers.conj(x, y)
        >>> error = net.test(x, y)
        >>> net.train_loop(trainer, epochs=10)
        >>> error > net.test(x, y)
        True
        """
        return self._train_loop(trainer, epochs)

    def _train_loop(self, trainer, epochs):
        r"""private implementation of `NNet.train_loop()`"""
        for _ in xrange(epochs):
            trainer.epoch(self)

