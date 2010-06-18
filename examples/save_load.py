# In this example we will extend the normalization layer of the
# new_layer.py example.

from pynnet import *
# this is for BaseLayer and other utilities
from pynnet.layers.base import *
import theano

class NormalizationLayer(BaseLayer):
    # We take the desired min and max value as parameters.  Some
    # sensible defaults are set since we want the layer to be as easy
    # to use as possible.  If you are coding for yourself only, you
    # can ignore this.
    def __init__(self, min=0, max=1):
        BaseLayer.__init__(self, None)
        self.min = min
        self.max = max
        
    # Since we have parameters that depend on the instance we have to
    # properly define a _save_ and a _load_ method.

    # The save method get a single argument which is a file-like
    # object to which you should write the data for your layer.  You
    # must not call the _save_ method of any base or descendant class.
    # However if your layer contains others, you should call their
    # savef (and not _save_ directly) method to save them.
    def _save_(self, file):
        # This uses pickle to save the passed in object.  Use this for
        # most data except numpy matrices.
        psave((self.min, self.max), file)

    # Here we define the _load_ method.  Note that the method is
    # actually called _load1_.  This is to use a trick in the save and
    # load system to get versioning of the save data.  If you ever
    # update the class, you can create a new _load2_ method to load
    # the new format and update the _load1_ to load from the old
    # format and set any new attributes to default values or ignore
    # old attributes.
    def _load1_(self, file):
        self.min, self.max = pload(file)

    # You need this line to tell the save system which is the current
    # load method
    _load_ = _load1_

    def build(self, input, input_shape=None):
        # keep a reference to the input expression
        self.input = input
        # our output is of the same size as the input
        self.output_shape = input_shape
        # we have no parameter that should be adjusted with training
        self.params = []
        # finally the work we do (this will only work with matrices)
        temp = self.input - self.input.min().min()
        self.output = (temp / temp.max().max())*(self.max-self.min)+self.min

    # As-is, this layer is usable.  It can also be saved and loaded
    # correctly since it does not have any instance-specific
    # parameters.

# Now we can build networks using our new layer (and existing ones of course)

n = NNet([NormalizationLayer(min=-2, max=2)], error=errors.mse)

# This saves the net to an instance of StringIO and loads it back
# returning the loaded copy.  It avoids using temp files for tests.
normnet = test_saveload(n)

x = theano.tensor.matrix()
y = theano.tensor.matrix()

normnet.build(x, y)

eval = theano.function([x], normnet.output)

# This should print:
# [[-2.         -1.33333333 -0.66666667]
#  [ 0.         -0.66666667  0.66666667]
#  [ 2.         -0.66666667 -1.33333333]]
print eval([[0, 1, 2], [3, 2, 4], [6, 2, 1]])


