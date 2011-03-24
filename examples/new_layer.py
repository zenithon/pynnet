# In this example we will make a simple normalization node.
# All output values will be in the 0-1 range.

from pynnet import *
from pynnet.nodes.base import BaseNode
from pynnet.nodes import errors
import theano

# All nodes should inherit from BaseNode (this is not actually
# enforced).
class NormalizationNode(BaseNode):
    # For the __init__ method, we should take any parameters that we
    # will need while building the graph.  Here we do not need
    # anything aside from the input expression.  You should always put
    # any required inputs as the first arguments and put other
    # required arguments after.
    def __init__(self, input):
        # I know some python advice will tell you to use super().  It
        # is only required in the multiple inheritance diamond case.
        # Since the code is engineered to avoid this case, we don't
        # need it.

        # Initialize the base class with a name and inputs.  Here we
        # pass None to let BaseNode come up with an appropriate unique
        # name based on the class name.  We could add a name parameter
        # to the __init__ method to let users choose their own name.

        # The input is the one we collected form our parameter and is
        # the value upon which we will work.
        BaseNode.__init__(self, [input], None)

        # Aside from the parameters we pass to BaseNode, there are
        # special attributes which are used in the interface.  One of
        # those is local_params which should be set to the list of
        # parameters that should be adjusted by the gradient for this
        # node.  Since we don't have any we can ignore it as the
        # default is no parameters.

        # This is also the place to set up any attributes that could
        # be required by the transform() method below.  They can be
        # collected from additional arguments in the constructor.

    # This method has a simple signature which takes all of our
    # inputs, splitted apart.  In this case we only have one input.
    # The inputs recieved here are theano expressions.
    def transform(self, input):
        # We return a theano expression that is the transformed
        # version of what we want to do with the input.  We could
        # refer to instance variables of our node if to provide
        # customization if needed.
        temp = input - input.min(-1).min(-1)
        return temp / temp.max(-1).max(-1)
        # The (-1) are to avoid theano warnings about a change in
        # behavior for .max() and .min()

# Now we can build networks using our new node (and existing ones of course)

x = theano.tensor.matrix()

nn = NormalizationNode(x)

eval = theano.function([x], nn.output)

# This should print:
# [[ 0.          0.16666667  0.33333333]
#  [ 0.5         0.33333333  0.66666667]
#  [ 1.          0.33333333  0.16666667]]
print eval([[0, 1, 2], [3, 2, 4], [6, 2, 1]])


