# In this example we will make a simple normalization layer.
# All output values will be in the 0-1 range.

from pynnet import *
from pynnet.layers.base import BaseLayer
import theano

# All layers should inherit from BaseLayer (this is not actually enforced).
class NormalizationLayer(BaseLayer):
    # For the __init__ method, we should take any parameters that we will
    # need while building the graph (apart from the input expression).
    # Here we do not need anything.
    def __init__(self):
        # I know some python advice will tell you to use super().
        # It is only required in the multiple inheritance diamond case.
        # Since the code is engineered to avoid this case, we don't need it.

        # Initialize the base class with a name.  Here we pass None to
        # let BaseLayer come up with an apporpriate unique name based on
        # the class name.  We could add a name parameter to the __init__
        # method to let users choose their own name.
        BaseLayer.__init__(self, None)

    # This method has a forced signature which is the one below.
    # In the method we have to set the required layer attributes.
    def build(self, input, input_shape=None):
        # keep a reference to the input expression
        self.input = input
        # our output is of the same size as the input
        self.output_shape = input_shape
        # we have no parameter that should be adjusted with training
        self.params = []
        # finally the work we do (this will only work with matrices)
        temp = self.input - self.input.min().min()
        self.output = temp / temp.max().max()

    # As-is, this layer is usable, but networks using it will not be saveable.
    # To allow the saving of network using this layer we would have to define
    # appropritate _save_ and _load_ methods.  This will not be discussed here.

# Now we can build networks using our new layer (and existing ones of course)

normnet = NNet([NormalizationLayer()], error=errors.mse)

x = theano.tensor.matrix()
y = theano.tensor.matrix()

normnet.build(x, y)

eval = theano.function([x], normnet.output)

# This should print:
# [[ 0.          0.16666667  0.33333333]
#  [ 0.5         0.33333333  0.66666667]
#  [ 1.          0.33333333  0.16666667]]
print eval([[0, 1, 2], [3, 2, 4], [6, 2, 1]])


