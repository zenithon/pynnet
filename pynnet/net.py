from base import *

from nlins import *
from errors import *

__all__ = ['NNet']

class AbstractNet(BaseObject):
    r"""
    Base class for neural networks.

    Defines a couple of methods that must be implemented for the rest
    of this package to be able to work with your class.

    These methods are::
    - `_test()`
    - `_grad()`
    - `_apply_dir()`

    In addition to these methods, you must define an attribute or
    property named `layers` that needs only to be readable.  It must
    return a list of `layer` named tuples which represents the
    connection weights of the network when read.

    Of course, for the network to be really usable, you should define
    some other methods, but those are left unspecified to allow for
    experimentation.

    Also, the `test_grad()` method along with some helpers are
    provided to help you test your implementation of the `_grad()`
    method.
    """

    def _test(self, x, y):
        r"""
        Test the neural net on the provided input-target pairs.

        Parameters:
        x -- the inputs
        y -- the targets

        Returns a single value that is the average of the error over
        each example.
        """
        raise NotImplementedError

    def _grad(self, x, y, lmbd):
        r"""
        Compute the gradient of the error for the given examples.

        Parameters:
        x -- the inputs
        y -- the outputs
        lmbd -- lambda parameter for weight penality (set to 0.0 to disable)
        
        Returns the gradient of the error for each weight in the
        network as a layer structure with the same dimensions as the
        `layers` property of self.
        """
        raise NotImplementedError

    def _apply_dir(self, G, alpha=1.0):
        r"""
        Move the weights of the network along the given direction.

        Parameters:
        G -- direction (same format as the `layers` property)
        alpha -- (default: 1.0) the amount of movement to make

        Adds to the weights of the network the directions given
        multiplied by the `alpha` parameter.
        """
        raise NotImplementedError

    def _estim_grad(self, x, y, eps):
        r"""private implementation of the finite differences for `AbstractNet.test_grad()`"""
        v = self._test(x, y)
        
        G = [None] * len(self.layers)
        
        for k, (W, b) in enumerate(self.layers):
            
            GW = numpy.empty(W.shape, W.dtype)
            Gb = numpy.empty(b.shape, b.dtype)
            
            for i, w in numpy.ndenumerate(W):
                W[i] += eps
                GW[i] = (self._test(x,y) - v)/eps
                W[i] = w

            for i, w in numpy.ndenumerate(b):
                b[i] += eps
                Gb[i] = (self._test(x,y) - v)/eps
                b[i] = w

            G[k] = layer(GW, Gb)

        return G

    def _test_grad(self, x, y, verbose, eps):
        r"""private implementation of `AbstractNet.test_grad()`"""
        Gc = self._grad(x,y, 0.0)
        Ge = self._estim_grad(x, y, eps)

        wrong = False

        for Glc, Gle in zip(Gc, Ge):
            rW = Glc.W/Gle.W
            rb = Glc.b/Gle.b
            if verbose:
                print rW
                print rb
            if rb.max() > 1.01 or rb.min() < 0.99 \
                    or rW.max() > 1.01 or rW.min() < 0.99:
                wrong = True
        
        if wrong:
            raise ValueError("Wrong gradient(s) detected")

    def _convert(self, a):
        return numpy.asarray(a, dtype=self.layers[0].W.dtype)

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
        """
        x = self._convert(x)
        y = self._convert(y)
        return self._test_grad(x, y, verbose, eps)

class NNet(AbstractNet):
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
        
        >>> net = NNet([2, 2, 1])
        >>> net.save('/tmp/pynnet_test_save_nnet')
        >>> net2 = NNet.load('/tmp/pynnet_test_save_nnet')
        >>> type(net2)
        <class 'pynnet.net.NNet'>
        >>> type(net2.layers[0])
        <class 'pynnet.base.layer'>
        >>> x = [[0, 0], [1, 1], [0, 1], [1, 0]]
        >>> y = [[0], [0], [1], [1]]
        >>> net.test_grad(x, y, verbose=False)
        >>> net2.test_grad(x, y, verbose=False)
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
        >>> acts[0].dtype
        dtype('float32')
        >>> outs[0].shape
        (4, 2)
        >>> outs[0].dtype
        dtype('float32')
        >>> acts[1].shape
        (4, 1)
        >>> outs[1].shape
        (4, 1)
        """
        x = self._convert(x)
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
        x = self._convert(x)
        y = self._convert(y)
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
        x = self._convert(x)
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
        >>> G = net.grad(x, y)
        >>> len(G)
        2
        >>> G[0].W.shape
        (2, 2)
        >>> G[1].W.shape
        (2, 1)
        >>> G[0].b.shape
        (2,)
        >>> G[1].b.shape
        (1,)
        >>> type(G[0])
        <class 'pynnet.base.layer'>
        """
        x = self._convert(x)
        y = self._convert(y)
        return self._grad(x, y, lmbd)

    def _grad(self, x, y, lmbd):
        r"""private implementation of `NNet.grad()`"""
        acts, outs = self._fprop(x)

        G = [None] * len(self.layers)
        
        Gacts = self.err._(outs[-1], y, self.err(outs[-1], y)) * self.nlins[-1]._(acts[-1], outs[-1])

        for i in xrange(-1,-len(self.layers), -1):
            G[i] = layer(numpy.dot(outs[i-1].T, Gacts) + 2.0 * lmbd * self.layers[i].W, \
                             Gacts.sum(axis=0))
            Gacts = numpy.dot(self.layers[i].W, Gacts.T).T * self.nlins[i-1]._(acts[i-1], outs[i-1])

        G[0] = layer(numpy.dot(x.T, Gacts) + 2.0 * lmbd * self.layers[0].W, \
                         Gacts.sum(axis=0))

        return G

    def _apply_dir(self, G, alpha=1.0):
        r"""
        Private API for the trainers.

        (look at the code if you are implementing a trainer)

        TESTS::
        
        >>> net = NNet([2, 2, 1])
        >>> x = [[0, 0], [1, 1], [0, 1], [1, 0]]
        >>> y = [[0], [0], [1], [1]]
        >>> G = net.grad(x, y)
        >>> error = net.test(x, y)
        >>> net._apply_dir(G, -0.2)
        >>> error > net.test(x, y)
        True
        """
        for (W, b), (GW, Gb) in zip(self.layers, G):
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

